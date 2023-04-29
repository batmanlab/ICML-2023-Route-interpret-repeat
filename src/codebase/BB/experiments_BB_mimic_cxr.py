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
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_mimic_cxr import MIMICCXRDataset

warnings.filterwarnings("ignore")


def test(args):
    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_labels = len(args.labels)
    device = utils.get_device()
    print(f"Device: {device}")

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

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
        main_worker_test(args.gpu, ngpus_per_node, args, disease_folder)


def main_worker_test(gpu, ngpus_per_node, args, disease_folder):
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
    model = DenseNet121(args)
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss}"
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch, disease_folder)
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch, disease_folder)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_test_configs.pkl"), "wb"))
    model_chk_pt = torch.load(os.path.join(chk_pt_path, args.checkpoint_bb))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

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
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=True
    )

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    start = time.time()
    if "state_dict" in model_chk_pt:
        model.load_state_dict(model_chk_pt['state_dict'])
    else:
        model.load_state_dict(model_chk_pt)
    done = time.time()
    elapsed = done - start
    print("Time to load the BB: " + str(elapsed) + " secs")

    start = time.time()
    validate(args, model, test_loader, output_path)
    done = time.time()
    elapsed = done - start
    print("Time to run validation: " + str(elapsed) + " secs")


def validate(args, model, loader, output_path):
    model.eval()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for idx, data in enumerate(loader):
                (dicom_id,
                 image,
                 adj_mtx, _, _,
                 landmark_spec_label,
                 landmarks_spec_inverse_weight,
                 landmark_spec_label_pnu,
                 selected_obs_label_gt,
                 selected_obs_inverse_weight,
                 selected_obs_label_pnu, _, _, _, _, _
                 ) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)

                features, pooled_features, logits = model(image)
                out_put_predict = torch.cat((out_put_predict, logits), dim=0)
                out_put_GT = torch.cat((out_put_GT, selected_obs_label_gt), dim=0)

                t.set_postfix(batch_id='{0}'.format(idx + 1))
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    # y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

    # print(y_pred)
    # print(y_pred.shape)
    print(out_put_predict.size())
    print(out_put_GT.size())

    # cls_report = utils.cal_classification_report(
    #     label=out_put_GT_np, out=y_pred, labels=args.selected_obs
    # )
    #
    # out_AUROC, out_AUPRC = utils.compute_AUCs(
    #     out_put_GT,
    #     out_put_predict,
    #     len(args.selected_obs)
    # )
    #
    # auroc_mean = np.array(out_AUROC).mean()
    #
    # print(cls_report)
    # print("<<< Model Test Results: AUROC >>>")
    # print("MEAN AUROC", ": {:.4f}".format(auroc_mean))
    #
    # for i in range(0, len(out_AUROC)):
    #     print(args.selected_obs[i], ': {:.4f}'.format(out_AUROC[i]), ': {:.4f}'.format(out_AUPRC[i]))
    # print("------------------------")

    torch.save(out_put_GT.cpu(), os.path.join(output_path, "GT.pth.tar"))
    torch.save(out_put_predict.cpu(), os.path.join(output_path, "predictions.pth.tar"))

    # df = pd.DataFrame(cls_report).transpose()
    # df.to_csv(f"{os.path.join(output_path, 'classification_report.csv')}")
    # utils.dump_in_pickle(output_path=output_path, file_name="cls_report.pkl", stats_to_dump=cls_report)
    # utils.dump_in_pickle(output_path=output_path, file_name="AUC_ROC.pkl", stats_to_dump=out_AUROC)
    # utils.dump_in_pickle(output_path=output_path, file_name="AUC_RPC.pkl", stats_to_dump=out_AUPRC)

    # print(f"GT shape: {out_put_GT.size()}")
    # print(f"Predictions shape: {out_put_predict.size()}")
    #
    # print(f"Classification report is saved at {output_path}/cls_report.pkl")
    # print(f"AUC-ROC report is saved at {output_path}/AUC_ROC.pkl")


def train(args):
    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_labels = len(args.labels)
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss}"
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch, disease_folder)
    tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}_{disease_folder}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_train_configs.pkl"), "wb"))

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
        main_worker(args.gpu, ngpus_per_node, args, chk_pt_path, output_path, tb_logs_path, disease_folder)


def main_worker(gpu, ngpus_per_node, args, chk_pt_path, output_path, tb_logs_path, disease_folder):
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
    model = DenseNet121(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        [{'params': list(model.backbone.parameters()), 'lr': args.lr,
          'weight_decay': args.weight_decay, 'momentum': args.momentum},
         {'params': list(model.fc1.parameters()), 'lr': args.lr,
          'weight_decay': args.weight_decay, 'momentum': args.momentum}
         ])

    # optionally resume from a checkpoint
    best_auroc = 0
    if args.resume:
        ckpt_path = os.path.join(chk_pt_path, args.resume)
        print(ckpt_path)
        if os.path.isfile(ckpt_path):
            config_path = os.path.join(output_path, 'MIMIC_train_configs.pkl')
            args = pickle.load(open(config_path, "rb"))
            args.distributed = False
            # args.batch_size = 8
            model = DenseNet121(args)
            checkpoint = torch.load(ckpt_path)
            args.start_epoch = checkpoint['epoch']
            best_auroc = checkpoint['best_auroc']
            # best_auroc = checkpoint['best_auroc']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(args.resume, disease_folder)))

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
        mode='train',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
            transforms.CenterCrop(args.resize),
            # transforms.RandomRotation(args.degree),
            # transforms.RandomCrop(args.crop),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ])
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
        mode='valid',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ])
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

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_MIMIC_CXR(
        1, best_auroc, args.start_epoch, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader,
        args.N_labels, model_type="bb"
    )

    fit(args, model, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id)


def fit(args, model, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id):
    run_manager.begin_run(run_id)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        run_manager.begin_epoch()
        # switch to train mode
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (dicom_id,
                 image,
                 adj_mtx, _, _,
                 landmark_spec_label,
                 landmarks_spec_inverse_weight,
                 landmark_spec_label_pnu,
                 selected_obs_label_gt,
                 selected_obs_inverse_weight,
                 selected_obs_label_pnu, _, _, _, _, _) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)
                    selected_obs_inverse_weight = selected_obs_inverse_weight.cuda(args.gpu, non_blocking=True)
                    selected_obs_label_gt = selected_obs_label_gt.view(-1)
                    selected_obs_inverse_weight = selected_obs_inverse_weight.view(-1)

                features, pooled_features, logits = model(image)
                train_loss = compute_loss(args, logits, selected_obs_label_gt, selected_obs_inverse_weight)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(logits, selected_obs_label_gt)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (dicom_id,
                     image,
                     adj_mtx, _, _,
                     landmark_spec_label,
                     landmarks_spec_inverse_weight,
                     landmark_spec_label_pnu,
                     selected_obs_label_gt,
                     selected_obs_inverse_weight,
                     selected_obs_label_pnu, _, _, _, _, _) = data

                    if args.gpu is not None:
                        image = image.cuda(args.gpu, non_blocking=True)

                    if torch.cuda.is_available():
                        selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)
                        selected_obs_inverse_weight = selected_obs_inverse_weight.cuda(args.gpu, non_blocking=True)
                        selected_obs_label_gt = selected_obs_label_gt.view(-1)
                        selected_obs_inverse_weight = selected_obs_inverse_weight.view(-1)

                    features, pooled_features, logits = model(image)
                    val_loss = compute_loss(args, logits, selected_obs_label_gt, selected_obs_inverse_weight)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(logits, selected_obs_label_gt)
                    run_manager.track_val_bb_outputs(out_class=logits, val_y=selected_obs_label_gt)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss)
                    )
                    t.update()

        run_manager.end_epoch(model, optimizer, multi_label=False)

        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_AUROC: {round(run_manager.best_auroc, 4)}  "
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


def compute_loss(args, logits, target, weights):
    target = target.to(torch.long)
    weight_tensor = torch.Tensor([0.0375, 0.9625]).cuda(args.gpu, non_blocking=True)
    if args.loss == 'CE':
        loss = F.cross_entropy(logits, target, reduction='mean')
    elif args.loss == 'CE_W':
        loss = F.cross_entropy(logits, target, weight=weight_tensor, reduction='mean')
    else:
        raise Exception('Invalid loss type.')
    return loss
