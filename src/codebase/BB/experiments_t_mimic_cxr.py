import builtins
import os
import pickle
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

import utils
from BB.models.BB_DenseNet121 import DenseNet121
from BB.models.VIT_densenet import CONFIGS, VisionTransformer
from BB.models.t import Logistic_Regression_t
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_mimic_cxr import MIMICCXRDataset

warnings.filterwarnings("ignore")


def test_t(args):
    print("###############################################")

    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_abnorm_obs_concepts = len(args.abnorm_obs_concepts)
    args.N_labels = len(args.labels)

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss1}_" \
           f"flattening_type_{args.flattening_type}_layer_{args.layer}"
    chkpt_folder_bb = args.bb_chkpt_folder
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", chkpt_folder_bb, args.arch, disease_folder)
    chk_pt_path_t = os.path.join(args.checkpoints, args.dataset, "t", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "t", root, args.arch, disease_folder)

    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_test_configs.pkl"), "wb"))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        """
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker_test, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker_test(args.gpu, ngpus_per_node, args, chk_pt_path_bb, chk_pt_path_t, output_path)


def main_worker_test(gpu, ngpus_per_node, args, chk_pt_path_bb, chk_pt_path_t, output_path):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )

    # create model
    print("=> Creating model '{}'".format(args.arch))
    model_bb = get_bb(args)
    t_ip = get_input_size_t(args, model_bb)

    t = Logistic_Regression_t(
        ip_size=t_ip, op_size=args.N_landmarks_spec + args.N_abnorm_obs_concepts, flattening_type=args.flattening_type
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_bb.cuda(args.gpu)
            t.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_bb = torch.nn.parallel.DistributedDataParallel(model_bb, device_ids=[args.gpu])
            t = torch.nn.parallel.DistributedDataParallel(t, device_ids=[args.gpu])
        else:
            model_bb.cuda()
            t.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_bb = torch.nn.parallel.DistributedDataParallel(model_bb)
            t = torch.nn.parallel.DistributedDataParallel(t)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_bb = model_bb.cuda(args.gpu)
        t = t.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model_bb.features = torch.nn.DataParallel(model_bb.features)
            t.features = torch.nn.DataParallel(t.features)
            model_bb.cuda()
            t.cuda()
        else:
            model_bb = torch.nn.DataParallel(model_bb).cuda()
            t = torch.nn.DataParallel(t).cuda()

    # optionally resume from a checkpoint
    best_auroc = 0
    cudnn.benchmark = True
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
    arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75

    start = time.time()
    test_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode='test',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ]),
        model_type="t",
        network_type=args.network_type,
        feature_path=args.feature_path
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
        drop_last=True
    )

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    start = time.time()
    model_chk_pt = torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
    if args.arch == "densenet121":
        model_bb.load_state_dict(model_chk_pt['state_dict'])
    elif args.arch == "ViT-B_32_densenet":
        model_bb.load_state_dict(model_chk_pt["model"])

    model_chk_pt_t = torch.load(os.path.join(chk_pt_path_t, args.checkpoint_t))
    if "state_dict" in model_chk_pt_t:
        t.load_state_dict(model_chk_pt_t['state_dict'])
    else:
        t.load_state_dict(model_chk_pt_t)
    done = time.time()
    elapsed = done - start
    print("Time to load the BB: " + str(elapsed) + " secs")
    if args.save_concepts:
        start = time.time()
        train_dataset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(args.resize),
                    # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
                    transforms.CenterCrop(args.resize),
                    # transforms.RandomRotation(args.degree),
                    # transforms.RandomCrop(args.crop),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # convert pixel value to [0, 1]
                    normalize,
                ]
            ),
            model_type="t",
            network_type=args.network_type,
            feature_path=args.feature_path
        )

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
            pin_memory=True, sampler=train_sampler, drop_last=True
        )

        val_dataset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode='valid',
            transform=transforms.Compose([
                transforms.Resize(args.resize),
                transforms.CenterCrop(args.resize),
                transforms.ToTensor(),  # convert pixel value to [0, 1]
                normalize
            ]),
            model_type="t",
            network_type=args.network_type,
            feature_path=args.feature_path
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
            drop_last=True
        )

        done = time.time()
        elapsed = done - start
        print("Time to load the dataset: " + str(elapsed) + " secs")
        output_path_t_dataset_g = os.path.join(output_path, "dataset_g")
        os.makedirs(output_path_t_dataset_g, exist_ok=True)

        print("=============>> Saving concepts for test set")
        save_concepts(
            args,
            test_loader,
            model_bb,
            t,
            output_path_t_dataset_g,
            mode="test"
        )
        print("=============>> Saving concepts for training set")
        save_concepts(
            args,
            train_loader,
            model_bb,
            t,
            output_path_t_dataset_g,
            mode="train"
        )
        print("=============>> Saving concepts for val set")
        save_concepts(
            args,
            val_loader,
            model_bb,
            t,
            output_path_t_dataset_g,
            mode="val"
        )

    print("Validation tensors are saving")
    validate(args, model_bb, t, test_loader, output_path)


def save_concepts(
        args,
        loader,
        bb,
        t_model,
        output_path,
        mode
):
    feature_path = os.path.join(output_path, f"{mode}_features")
    transformed_img_path = os.path.join(output_path, f"{mode}_transformed_images")
    raw_img_path = os.path.join(output_path, f"{mode}_raw_images")
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(transformed_img_path, exist_ok=True)
    os.makedirs(raw_img_path, exist_ok=True)
    bb.eval()
    t_model.eval()

    # features_phi = torch.FloatTensor()
    bb_logits_y = torch.FloatTensor().cuda()
    logits_concepts_x = torch.FloatTensor().cuda()
    proba_concepts_x = torch.FloatTensor().cuda()
    class_labels = torch.FloatTensor().cuda()
    attributes = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    raw_image,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    vit_features = vit_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    densenet_features = densenet_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                features, bb_logits = get_phi_x(image, bb, args.arch, vit_features)
                # bb_logits, part_tokens = bb(densenet_features)

                if args.layer == "features_denseblock4":
                    logits_concepts = t_model(features)
                elif args.layer == "features_denseblock3":
                    logits_concepts = t_model(bb.feature_store["features_transition3"])
                y_hat = torch.sigmoid(logits_concepts)

                utils.save_tensor(
                    path=os.path.join(feature_path, f"features_{batch_id}.pth.tar"),
                    tensor_to_save=features.cpu()
                )
                utils.save_tensor(
                    path=os.path.join(transformed_img_path, f"transformed_img_{batch_id}.pth.tar"),
                    tensor_to_save=image.cpu()
                )
                utils.save_tensor(
                    path=os.path.join(raw_img_path, f"raw_img_{batch_id}.pth.tar"),
                    tensor_to_save=raw_image.cpu()
                )

                # features_phi = torch.cat((features_phi, features.cpu()), dim=0)
                bb_logits_y = torch.cat((bb_logits_y, bb_logits), dim=0)
                logits_concepts_x = torch.cat((logits_concepts_x, logits_concepts), dim=0)
                proba_concepts_x = torch.cat((proba_concepts_x, y_hat), dim=0)
                class_labels = torch.cat((class_labels, selected_obs_label_gt), dim=0)
                attributes = torch.cat((attributes, gt), dim=0)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    bb_logits_y = bb_logits_y.cpu()
    logits_concepts_x = logits_concepts_x.cpu()
    proba_concepts_x = proba_concepts_x.cpu()
    class_labels = class_labels.cpu()
    attributes = attributes.cpu()

    # print(f"====> Saved features phi size: {features_phi.size()}")
    print(f"====> Saved logits bb_y size: {bb_logits_y.size()}")
    print(f"====> Saved logits concepts_x size: {logits_concepts_x.size()}")
    print(f"====> Saved proba concepts_x size: {proba_concepts_x.size()}")
    print(f"====> Saved class_labels size: {class_labels.size()}")
    print(f"====> Saved attributes size: {attributes.size()}")

    # utils.save_tensor(path=os.path.join(output_path, f"{mode}_features_phi.pt"), tensor_to_save=features_phi)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_logits_bb.pt"), tensor_to_save=bb_logits_y)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_logits_concepts.pt"), tensor_to_save=logits_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_proba_concepts.pt"), tensor_to_save=proba_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_class_labels.pt"), tensor_to_save=class_labels)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_attributes.pt"), tensor_to_save=attributes)

    # print(f"====> features phi saved at {os.path.join(output_path, f'{mode}_features_phi.pt')}")
    print(f"====> Logits BB saved at {os.path.join(output_path, f'{mode}_logits_bb.pt')}")
    print(f"====> Logits Concepts saved at {os.path.join(output_path, f'{mode}_logits_concepts.pt')}")
    print(f"====> Proba Concepts saved at {os.path.join(output_path, f'{mode}_proba_concepts.pt')}")
    print(f"====> Class labels saved at {os.path.join(output_path, f'{mode}_class_labels.pt')}")
    print(f"====> Attributes labels saved at {os.path.join(output_path, f'{mode}_attributes.pt')}")


def validate(args, bb, t_model, loader, output_path):
    bb.eval()
    t_model.eval()

    concept_names = args.landmark_names_spec + args.abnorm_obs_concepts

    out_prob_arr_bb = []
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    _,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    vit_features = vit_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    densenet_features = densenet_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                features, bb_logits = get_phi_x(image, bb, args.arch, vit_features)
                if args.layer == "features_denseblock4":
                    logits_concepts = t_model(features)
                elif args.layer == "features_denseblock3":
                    logits_concepts = t_model(bb.feature_store["features_transition3"])
                y_hat = torch.sigmoid(logits_concepts)

                out_put_predict = torch.cat((out_put_predict, y_hat), dim=0)
                out_put_GT = torch.cat((out_put_GT, gt), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

    # cls_report = {}
    # for i, concept_name in enumerate(concept_names):
    #     cls_report[concept_name] = {}
    # for i, concept_name in enumerate(concept_names):
    #     cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #
    # cls_report["accuracy_overall"] = (y_pred == out_put_GT_np).sum() / (out_put_GT_np.shape[0] * out_put_GT_np.shape[1])
    # for i, concept_name in enumerate(concept_names):
    #     print(f"{concept_name}: {cls_report[concept_name]}")
    #
    # print(f"Overall Accuracy: {cls_report['accuracy_overall']}")

    out_AUROC = utils.compute_AUROC(
        out_put_GT,
        out_put_predict,
        len(concept_names)
    )

    auroc_mean = np.array(out_AUROC).mean()
    print("<<< Model Test Results: AUROC >>>")
    print("MEAN", ": {:.4f}".format(auroc_mean))

    for i in range(0, len(out_AUROC)):
        print(concept_names[i], ': {:.4f}'.format(out_AUROC[i]))
    print("------------------------")

    # utils.dump_in_pickle(output_path=output_path, file_name="cls_report.pkl", stats_to_dump=cls_report)
    utils.dump_in_pickle(output_path=output_path, file_name="AUC_ROC.pkl", stats_to_dump=out_AUROC)

    # print(f"Classification report is saved at {output_path}/cls_report.pkl")
    print(f"AUC-ROC report is saved at {output_path}/AUC_ROC.pkl")


def train_t(args):
    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_abnorm_obs_concepts = len(args.abnorm_obs_concepts)
    args.N_labels = len(args.labels)

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss1}_" \
           f"flattening_type_{args.flattening_type}_layer_{args.layer}"
    chkpt_folder_bb = args.bb_chkpt_folder
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", chkpt_folder_bb, args.arch, disease_folder)
    chk_pt_path_t = os.path.join(args.checkpoints, args.dataset, "t", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "t", root, args.arch, disease_folder)
    tb_logs_path = os.path.join(args.logs, args.dataset, "t", f"{root}_{disease_folder}")

    os.makedirs(chk_pt_path_t, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_train_configs.pkl"), "wb"))
    print(os.path.join(output_path, "MIMIC_train_configs.pkl"))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        """
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, chk_pt_path_bb, chk_pt_path_t, output_path, tb_logs_path)


def main_worker(gpu, ngpus_per_node, args, chk_pt_path_bb, chk_pt_path_t, output_path, tb_logs_path):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )

    # create model
    print("=> Creating model '{}'".format(args.arch))

    model_bb = get_bb(args)
    t_ip = get_input_size_t(args, model_bb)

    t = Logistic_Regression_t(
        ip_size=t_ip, op_size=args.N_landmarks_spec + args.N_abnorm_obs_concepts, flattening_type=args.flattening_type
    )
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_bb.cuda(args.gpu)
            t.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_bb = torch.nn.parallel.DistributedDataParallel(model_bb, device_ids=[args.gpu])
            t = torch.nn.parallel.DistributedDataParallel(t, device_ids=[args.gpu])
        else:
            model_bb.cuda()
            t.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_bb = torch.nn.parallel.DistributedDataParallel(model_bb)
            t = torch.nn.parallel.DistributedDataParallel(t)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_bb = model_bb.cuda(args.gpu)
        t = t.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model_bb.features = torch.nn.DataParallel(model_bb.features)
            t.features = torch.nn.DataParallel(t.features)
            model_bb.cuda()
            t.cuda()
        else:
            model_bb = torch.nn.DataParallel(model_bb).cuda()
            t = torch.nn.DataParallel(t).cuda()

    optimizer = torch.optim.SGD(
        [
            {'params': list(t.linear.parameters()), 'lr': args.lr,
             'weight_decay': args.weight_decay, 'momentum': args.momentum
             }
        ])

    # optionally resume from a checkpoint
    best_auroc = 0
    if args.resume:
        # TODO: this needs to be changed as per GLT
        ckpt_path = os.path.join(chk_pt_path_t, args.resume)
        if os.path.isfile(ckpt_path):
            config_path = os.path.join(output_path, 'MIMIC_train_configs.pkl')
            args = pickle.load(open(config_path, "rb"))
            args.distributed = False
            t = Logistic_Regression_t(
                ip_size=t_ip, op_size=args.N_landmarks_spec + args.N_abnorm_obs_concepts,
                flattening_type=args.flattening_type
            )
            checkpoint = torch.load(ckpt_path)
            args.start_epoch = checkpoint['epoch']
            best_auroc = checkpoint['best_auroc']
            t.load_state_dict(checkpoint['state_dict'])
            t = t.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
    arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75

    start = time.time()
    train_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode="train",
        transform=transforms.Compose(
            [
                transforms.Resize(args.resize),
                # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
                transforms.CenterCrop(args.resize),
                # transforms.RandomRotation(args.degree),
                # transforms.RandomCrop(args.crop),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # convert pixel value to [0, 1]
                normalize,
            ]
        ),
        model_type="t",
        network_type=args.network_type,
        feature_path=args.feature_path
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode='test',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ]),
        model_type="t",
        network_type=args.network_type,
        feature_path=args.feature_path
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=True
    )

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    print("########### Paths ###########")
    print(chk_pt_path_t)
    print(tb_logs_path)
    print(output_path)
    print("########### Paths ###########")
    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_MIMIC_CXR(
        1, best_auroc, args.start_epoch, chk_pt_path_t, tb_logs_path, output_path, train_loader, val_loader,
        args.N_landmarks_spec + args.N_abnorm_obs_concepts, model_type="t"
    )

    start = time.time()
    model_chk_pt = torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
    print("bb loaded from ================>>>")
    print(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
    if args.arch == "densenet121":
        model_bb.load_state_dict(model_chk_pt['state_dict'])
    elif args.arch == "ViT-B_32_densenet":
        model_bb.load_state_dict(model_chk_pt["model"])
    done = time.time()
    elapsed = done - start
    print("Time to load the BB: " + str(elapsed) + " secs")
    fit(args, model_bb, t, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id)


def fit(args, model_bb, t_model, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id):
    model_bb.eval()
    run_manager.begin_run(run_id)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        run_manager.begin_epoch()
        # switch to train mode
        t_model.train()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    _,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    vit_features = vit_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    # densenet_features = densenet_features.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)
                    landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
                    full_obs_inverse_weight = full_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                weights = torch.cat((landmarks_spec_inverse_weight, full_obs_inverse_weight), dim=1)

                features, bb_logits = get_phi_x(image, model_bb, args.arch, vit_features)
                if args.layer == "features_denseblock4":
                    logits_concepts = t_model(features)
                elif args.layer == "features_denseblock3":
                    logits_concepts = t_model(model_bb.feature_store["features_transition3"])

                train_loss = compute_loss(args, logits_concepts, gt, weights)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts), gt)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        t_model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (
                        dicom_id,
                        image,
                        vit_features,
                        densenet_features,
                        _,
                        adj_mtx,
                        chexpert_label,
                        _,
                        landmark_spec_label,
                        landmarks_spec_inverse_weight,
                        landmark_spec_label_pnu,
                        selected_obs_label_gt,
                        selected_obs_inverse_weight,
                        selected_obs_label_pnu,
                        full_obs_label_gt,
                        full_obs_inverse_weight,
                        full_obs_label_pnu,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = data

                    if args.gpu is not None:
                        image = image.cuda(args.gpu, non_blocking=True)

                    if torch.cuda.is_available():
                        landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                        vit_features = vit_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                        full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)
                        landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
                        full_obs_inverse_weight = full_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

                    gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                    weights = torch.cat((landmarks_spec_inverse_weight, full_obs_inverse_weight), dim=1)

                    features, bb_logits = get_phi_x(image, model_bb, args.arch, vit_features)
                    if args.layer == "features_denseblock4":
                        logits_concepts = t_model(features)
                    elif args.layer == "features_denseblock3":
                        logits_concepts = t_model(model_bb.feature_store["features_transition3"])

                    val_loss = compute_loss(args, logits_concepts, gt, weights)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_multilabel_per_epoch(
                        torch.sigmoid(logits_concepts), gt
                    )
                    run_manager.track_val_bb_outputs(
                        out_class=torch.sigmoid(logits_concepts),
                        val_y=gt
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss)
                    )
                    t.update()

        run_manager.end_epoch(t_model, optimizer, multi_label=True)

        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_auroc: {round(run_manager.best_auroc, 4)} (0-1) "
              f"Val_Accuracy: {round(run_manager.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(run_manager.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(run_manager.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 6 epochs"""
    lr = args.lr * (0.33 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_loss(args, logits, y, weights):
    if args.loss1 == 'BCE':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logits), y, reduction='mean')
    elif args.loss1 == 'BCE_W':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logits), y, weight=weights, reduction='mean')
    else:
        raise Exception('Invalid loss 1 type.')

    return loss1


def get_bb(args):
    if args.arch == "densenet121":
        return DenseNet121(args, layer=args.layer)
    elif args.arch == "ViT-B_32_densenet":
        return VisionTransformer(
            CONFIGS[args.arch], args.resize, zero_head=True, num_classes=len(args.labels),
            smoothing_value=args.smoothing_value, pretrained="n"
        )


def get_input_size_t(args, model_bb):
    if args.arch == "densenet121":
        if args.flattening_type == "flatten":
            return (
                       model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" else
                       int(model_bb.fc1.weight.shape[1] / 2)
                   ) * 16 * 16
        elif args.flattening_type == "adaptive":
            return model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(model_bb.fc1.weight.shape[1] / 2)
        elif args.flattening_type == "max_pool":
            return model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(model_bb.fc1.weight.shape[1] / 2)
    elif args.arch == "ViT-B_32_densenet":
        # return 1024
        return 17 * 1024


def get_phi_x(image, model_bb, arch, vit_features):
    if arch == "densenet121":
        features, _, logits = model_bb(image)
        return features, logits
    # elif arch == "ViT-B_32_densenet":
    #     return torch.flatten(vit_features, start_dim=1)
