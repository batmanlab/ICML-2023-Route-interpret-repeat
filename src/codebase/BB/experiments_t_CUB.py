import os
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import dataset.utils_dataset as utils_dataset
import jupyter_notebook.stats_t as stats
import utils
from BB.models.BB_ResNet import ResNet
from BB.models.BB_ResNet50_metanorm import BB_ResNet50_metanorm
from BB.models.VIT import VisionTransformer, CONFIGS
from BB.models.VIT_p import VisionTransformer_projected
from BB.models.focal_loss import FocalLoss
from BB.models.t import Logistic_Regression_t
from Logger.logger_cubs import Logger_CUBS

warnings.filterwarnings("ignore")


def train_t(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}"
    )
    tb_logs_path_t = os.path.join(
        args.logs,
        args.dataset,
        "t",
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}"
    )
    chk_pt_path_t = os.path.join(
        args.checkpoints,
        args.dataset,
        "t",
        root,
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}"
    )

    os.makedirs(tb_logs_path_t, exist_ok=True)
    os.makedirs(chk_pt_path_t, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")

    print("##################### Paths #####################")
    print(tb_logs_path_t)
    print(chk_pt_path_t)
    print(output_path)
    print("#################################################")
    bb = get_bb(args, chk_pt_path_bb, device)
    bb.eval()

    # phi = Phi(bb, args.layer)
    input_size_t = get_input_size_t(args, bb)
    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=len(args.concept_names), flattening_type=args.flattening_type
    ).to(device)
    start = time.time()
    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    train_loader, val_loader = utils_dataset.get_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        train_transform,
        val_transform,
        attribute_file=args.attribute_file_name
    )
    done = time.time()
    elapsed = done - start

    print("Time to load train-test-val datasets: " + str(elapsed) + " secs")
    criterion = get_loss(args.loss_LR)
    optimizer = get_optim(args.solver_LR, t.parameters())
    final_parameters = OrderedDict(
        lr=[args.lr],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_CUBS(1, chk_pt_path_t, tb_logs_path_t, output_path, train_loader, val_loader, len(args.labels))
    run_manager.set_n_attributes(len(args.concept_names))

    fit_t(
        args.projected,
        args.spurious_waterbird_landbird,
        args.arch,
        bb,
        t,
        args.epochs_LR,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        args.flattening_type,
        args.dataset,
        args.layer,
        run_manager,
        run_id,
        args.concept_names[0: 112] if args.spurious_waterbird_landbird else args.concept_names,
        device
    )


def test_t(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args.data_root)

    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path_t_stats = os.path.join(
        args.output,
        args.dataset,
        "t",
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}",
        "stats"
    )
    output_path_t_dataset_g = os.path.join(
        args.output,
        args.dataset,
        "t",
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}",
        "dataset_g"
    )
    chk_pt_path_t = os.path.join(
        args.checkpoints,
        args.dataset,
        "t",
        root,
        f"{root}_{args.arch}_{args.layer}_{args.flattening_type}_{args.solver_LR}_{args.loss_LR}"
    )

    os.makedirs(output_path_t_stats, exist_ok=True)
    os.makedirs(output_path_t_dataset_g, exist_ok=True)

    print("#########################################")
    print("Paths")
    print(chk_pt_path_t)
    print(output_path_t_stats)
    print(output_path_t_dataset_g)
    print("#########################################")

    device = utils.get_device()
    print(f"Device: {device}")
    bb = get_bb(args, chk_pt_path_bb, device)
    bb.eval()

    # phi = Phi(bb, args.layer)
    input_size_t = get_input_size_t(args, bb)
    print(len(args.concept_names))
    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=len(args.concept_names), flattening_type=args.flattening_type
    ).to(device)
    t.load_state_dict(torch.load(os.path.join(chk_pt_path_t, args.checkpoint_file_t)))
    t.eval()

    start = time.time()
    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    train_loader, val_loader = utils_dataset.get_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        train_transform,
        val_transform,
        train_shuffle=False,
        attribute_file=args.attribute_file_name
    )
    test_loader = utils_dataset.get_test_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        val_transform,
        attribute_file=args.attribute_file_name
    )

    done = time.time()
    elapsed = done - start
    print("Time to load train-test-val datasets: " + str(elapsed) + " secs")
    predict_t(
        args.projected,
        args.spurious_waterbird_landbird,
        args.arch,
        bb,
        t,
        test_loader,
        args.flattening_type,
        args.dataset,
        args.layer,
        args.concept_names,
        output_path_t_stats,
        device,
    )

    print(chk_pt_path_t)
    print(output_path_t_stats)
    print(output_path_t_dataset_g)

    if args.save_concepts:
        print("Saving spurious concepts for training set")
        print("Saving concepts for training set")
        save_concepts(
            args.projected,
            args.arch,
            train_loader,
            bb,
            t,
            args.flattening_type,
            args.dataset,
            args.layer,
            output_path_t_dataset_g,
            mode="train",
            device=device
        )
        print("Saving concepts for val set")
        save_concepts(
            args.projected,
            args.arch,
            val_loader,
            bb,
            t,
            args.flattening_type,
            args.dataset,
            args.layer,
            output_path_t_dataset_g,
            mode="val",
            device=device
        )
        print("Saving concepts for test set")
        save_concepts(
            args.projected,
            args.arch,
            test_loader,
            bb,
            t,
            args.flattening_type,
            args.dataset,
            args.layer,
            output_path_t_dataset_g,
            mode="test",
            device=device
        )


def save_concepts(
        projected,
        arch,
        loader,
        bb,
        t_model,
        flattening_type,
        dataset,
        layer,
        output_path,
        mode,
        device
):
    image_tensor = torch.FloatTensor()
    logits_concepts_x = torch.FloatTensor().cuda()
    proba_concepts_x = torch.FloatTensor().cuda()
    class_labels = torch.FloatTensor().cuda()
    attributes = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, (image, label, attribute) in enumerate(loader):
                image = image.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                attribute = attribute.to(device, dtype=torch.float)
                batch_size = image.size(0)
                X_batch = np.zeros((batch_size, 4))
                X_batch[:, 0:4] = attribute[:, 108:112].cpu().detach().numpy()
                with torch.no_grad():
                    cfs = torch.nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)

                feature_x = get_phi_x(image, bb, arch, layer, cfs, projected)
                logits_concepts = t_model(feature_x)
                y_hat = torch.sigmoid(logits_concepts)

                logits_concepts_x = torch.cat((logits_concepts_x, logits_concepts), dim=0)
                proba_concepts_x = torch.cat((proba_concepts_x, y_hat), dim=0)
                class_labels = torch.cat((class_labels, label), dim=0)
                attributes = torch.cat((attributes, attribute), dim=0)
                image_tensor = torch.cat((image_tensor, image.cpu()), dim=0)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    logits_concepts_x = logits_concepts_x.cpu()
    proba_concepts_x = proba_concepts_x.cpu()
    class_labels = class_labels.cpu()
    attributes = attributes.cpu()

    print(f"Saved logits concepts_x size: {logits_concepts_x.size()}")
    print(f"Saved proba concepts_x size: {proba_concepts_x.size()}")
    print(f"Saved class_labels size: {class_labels.size()}")
    print(f"Saved attributes size: {attributes.size()}")
    print(f"Saved image_tensor size: {image_tensor.size()}")

    utils.save_tensor(path=os.path.join(output_path, f"{mode}_logits_concepts.pt"), tensor_to_save=logits_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_proba_concepts.pt"), tensor_to_save=proba_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_class_labels.pt"), tensor_to_save=class_labels)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_attributes.pt"), tensor_to_save=attributes)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_image_tensor.pt"), tensor_to_save=image_tensor)

    print(f"Logits Concepts saved at {os.path.join(output_path, f'{mode}_logits_concepts.pt')}")
    print(f"Proba Concepts saved at {os.path.join(output_path, f'{mode}_proba_concepts.pt')}")
    print(f"Class labels saved at {os.path.join(output_path, f'{mode}_class_labels.pt')}")
    print(f"Attributes labels saved at {os.path.join(output_path, f'{mode}_attributes.pt')}")
    print(f"Image_tensor saved at {os.path.join(output_path, f'{mode}_image_tensor.pt')}")


def predict_t(
        projected,
        spurious_waterbird_landbird,
        arch,
        bb,
        t_model,
        loader,
        flattening_type,
        dataset,
        layer,
        concept_names,
        output_path,
        device
):
    print(concept_names)
    print(len(concept_names))
    out_prob_arr_bb = []
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data_tuple in enumerate(loader):
                image, attribute = utils.get_image_attributes(data_tuple, spurious_waterbird_landbird, dataset)
                image = image.to(device, dtype=torch.float)
                attribute = attribute.to(device, dtype=torch.float)
                batch_size = image.size(0)
                X_batch = np.zeros((batch_size, 4))
                X_batch[:, 0:4] = attribute[:, 108:112].cpu().detach().numpy()
                with torch.no_grad():
                    cfs = torch.nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)

                feature_x = get_phi_x(image, bb, arch, layer, cfs, projected)
                # feature_x = get_phi_x(image, bb, arch, layer)
                logits_concepts = t_model(feature_x)
                y_hat = torch.sigmoid(logits_concepts)

                out_put_predict = torch.cat((out_put_predict, y_hat), dim=0)
                out_put_GT = torch.cat((out_put_GT, attribute), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

    print(len(concept_names))
    print(out_put_GT_np.shape)
    print(out_put_predict_np.shape)
    cls_report = {}
    for i, concept_name in enumerate(concept_names):
        cls_report[concept_name] = {}
    for i, concept_name in enumerate(concept_names):
        cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
        cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
        cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
        cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])

    cls_report["accuracy_overall"] = (y_pred == out_put_GT_np).sum() / (out_put_GT_np.shape[0] * out_put_GT_np.shape[1])
    for i, concept_name in enumerate(concept_names):
        print(f"{concept_name}: {cls_report[concept_name]}")

    print(f"Overall Accuracy: {cls_report['accuracy_overall']}")

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

    # _dict_scores = stats.get_dict_CI(
    #     output_path, concept_names, out_put_GT_np, out_put_predict_np
    # )
    #  utils.dump_in_pickle(output_path=output_path, file_name="cls_report.pkl", stats_to_dump=cls_report)
    # with open(os.path.join(path, "CI_concepts.pkl"), "rb") as input_file:
    #     _dict_scores = pickle.load(input_file)

    utils.dump_in_pickle(output_path=output_path, file_name="cls_report.pkl", stats_to_dump=cls_report)
    utils.dump_in_pickle(output_path=output_path, file_name="AUC_ROC.pkl", stats_to_dump=out_AUROC)

    print(f"Classification report is saved at {output_path}/cls_report.pkl")
    print(f"AUC-ROC report is saved at {output_path}/AUC_ROC.pkl")


def fit_t(
        projected,
        spurious_waterbird_landbird,
        arch,
        bb,
        t_model,
        epochs,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        flattening_type,
        dataset_name,
        layer,
        run_manager,
        run_id,
        concept_names,
        device
):
    run_manager.begin_run(run_id)
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    for epoch_id in range(epochs):
        run_manager.begin_epoch()
        running_loss = 0
        t_model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                image, attribute = utils.get_image_attributes(data_tuple, spurious_waterbird_landbird, dataset_name)
                image = image.to(device, dtype=torch.float)
                attribute = attribute.to(device, dtype=torch.float)
                batch_size = image.size(0)
                X_batch = np.zeros((batch_size, 4))
                X_batch[:, 0:4] = attribute[:, 108:112].cpu().detach().numpy()
                with torch.no_grad():
                    cfs = torch.nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)
                feature_x = get_phi_x(image, bb, arch, layer, cfs, projected)
                logits_concepts = t_model(feature_x)
                loss = criterion(logits_concepts, attribute)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run_manager.track_train_loss(loss.item())
                run_manager.track_total_train_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts), attribute)

                running_loss += loss.item()
                t.set_postfix(epoch='{0}'.format(epoch_id), training_loss='{:05.3f}'.format(running_loss))
                t.update()

        t_model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    image, attribute = utils.get_image_attributes(data_tuple, spurious_waterbird_landbird, dataset_name)
                    image = image.to(device, dtype=torch.float)
                    attribute = attribute.to(device, dtype=torch.float)
                    batch_size = image.size(0)
                    X_batch = np.zeros((batch_size, 4))
                    X_batch[:, 0:4] = attribute[:, 108:112].cpu().detach().numpy()
                    with torch.no_grad():
                        cfs = torch.nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)
                    feature_x = get_phi_x(image, bb, arch, layer, cfs, projected)
                    logits_concepts = t_model(feature_x)
                    val_loss = criterion(logits_concepts, attribute)
                    y_hat = torch.sigmoid(logits_concepts)

                    out_put_predict = torch.cat((out_put_predict, y_hat), dim=0)
                    out_put_GT = torch.cat((out_put_GT, attribute), dim=0)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts),
                                                                             attribute)
                    t.set_postfix(
                        epoch='{0}'.format(epoch_id),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        print()
        print("####################### Statistics on dev set #########################")
        out_put_GT_np = out_put_GT.cpu().numpy()
        out_put_predict_np = out_put_predict.cpu().numpy()
        y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

        print(len(concept_names))
        print(out_put_GT_np.shape)
        print(out_put_predict_np.shape)
        cls_report = {}
        for i, concept_name in enumerate(concept_names):
            cls_report[concept_name] = {}
        for i, concept_name in enumerate(concept_names):
            cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
            cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
            cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
            cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])

        cls_report["accuracy_overall"] = (y_pred == out_put_GT_np).sum() / (
                out_put_GT_np.shape[0] * out_put_GT_np.shape[1])
        for i, concept_name in enumerate(concept_names):
            print(f"{concept_name}: {cls_report[concept_name]}")

        print(f"Overall Accuracy: {cls_report['accuracy_overall']}")

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
        print("####################### Statistics on dev set #########################")

        run_manager.end_epoch(t_model, multi_label=True)
        print(f"Epoch: [{epoch_id + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()


def get_flattened_x(features, flattening_type):
    if flattening_type == "max_pooled":
        return utils.flatten_cnn_activations_using_max_pooled(
            features,
            kernel_size=features.size(3),
            stride=1
        )
    elif flattening_type == "flatten":
        return utils.flatten_cnn_activations_using_activations(
            features
        )
    elif flattening_type == "avg_pooled":
        return utils.flatten_cnn_activations_using_avg_pooled(
            features,
            kernel_size=features.size(3),
            stride=1
        )


def get_loss(loss_type):
    if loss_type == "focal":
        return FocalLoss(gamma=0)
    elif loss_type == "BCE":
        return torch.nn.BCEWithLogitsLoss()


def get_optim(solver_type, params):
    if solver_type == "sgd":
        return torch.optim.SGD(params, lr=1e-2)
    elif solver_type == "adam":
        return torch.optim.Adam(params, lr=1e-2)


def get_bb(args, chk_pt_path_bb, device):
    if (args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152") and args.projected == "n":
        bb = ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels), model_choice=args.arch,
            layer=args.layer
        ).to(device)
        print(f"BB is loaded from {os.path.join(chk_pt_path_bb, args.checkpoint_file)}")
        bb.load_state_dict(torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_file)))
        return bb
    elif (args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152") and args.projected == "y":
        dataset_path = os.path.join(
            args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g"
        )
        attributes_train = torch.load(os.path.join(dataset_path, "train_attributes.pt")).numpy()
        N = attributes_train.shape[0]
        X = np.zeros((N, 4))
        X[:, 0:4] = attributes_train[:, 108:112]
        XTX = np.transpose(X).dot(X)
        kernel = np.linalg.inv(XTX)
        cf_kernel = torch.nn.Parameter(torch.tensor(kernel).float().to(device), requires_grad=False)

        bb = BB_ResNet50_metanorm(args, dataset_size=N, kernel=cf_kernel, train=False).to(device)
        chkpt = os.path.join(chk_pt_path_bb, args.checkpoint_file)

        print(f"Projected CNN BB is loaded from {chkpt}")
        bb.load_state_dict(torch.load(chkpt))
        return bb
    elif args.arch == "ViT-B_16":
        config = CONFIGS[args.arch]
        print(f"BB is loaded from {os.path.join(chk_pt_path_bb, args.checkpoint_file)}")
        bb = VisionTransformer(
            config, args.img_size, zero_head=True, num_classes=len(args.labels),
            smoothing_value=args.smoothing_value
        ).to(device)
        bb.load_state_dict(torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_file))["model"])
        return bb
    elif args.arch == "ViT-B_16_projected":
        config = CONFIGS["ViT-B_16"]
        # chkpt = os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb)
        chkpt = os.path.join(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-specific-classes/cub/explainer/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.95_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none/iter1/explainer_projected",
            "VIT_CUBS_8000_checkpoint.bin"
        )
        print(f"==>> Loading projected VIT BB from : {chkpt}")
        bb = VisionTransformer_projected(
            config, args.img_size, zero_head=True, num_classes=len(args.labels), smoothing_value=args.smoothing_value,
            get_phi=True
        ).to(device)
        bb.load_state_dict(torch.load(chkpt)["model"])
        return bb


def get_phi_x(image, bb, arch, layer, cfs, projected="n"):
    if (arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152") and projected == "n":
        _ = bb(image)
        # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
        feature_x = bb.feature_store[layer]
        return feature_x
    elif (arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152") and projected == "y":
        feature_x, _ = bb(image, cfs, get_phi=True)
        return feature_x
    elif arch == "ViT-B_16" and projected == "n":
        logits, tokens = bb(image)
        return tokens[:, 0]
    # elif arch == "ViT-B_16_projected":
    #     phi_r_x, _ = bb(image, concepts, scale, sigma, device)
    #     return phi_r_x


def get_input_size_t(args, bb):
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        return 2048
    elif args.arch == "ViT-B_16" or args.arch == "ViT-B_16_projected":
        return 768
