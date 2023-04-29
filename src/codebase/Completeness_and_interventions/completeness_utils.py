import copy
import json
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from BB.models.BB_Inception_V3 import get_model, get_BB_model_isic
from BB.models.t import Logistic_Regression_t
from Explainer.models.G import G
from Explainer.models.residual import Residual
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_completeness import Dataset_completeness_features, Dataset_completeness_mimic_cxr


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_initial_setups(args, model_type):
    start = time.time()
    if args.dataset == "mimic_cxr":
        dataset_path = os.path.join(args.output, args.dataset)
        train_dataset = Dataset_completeness_mimic_cxr(
            dataset_path, args.disease, args.seed, args.topK, model_type, transform=None, mode="train"
        )
        val_dataset = Dataset_completeness_mimic_cxr(
            dataset_path, args.disease, args.seed, args.topK, model_type, transform=None, mode="test"
        )
        g_chk_pt_path = os.path.join(
            args.checkpoints, args.dataset, "completeness", f"seed_{args.seed}", "dataset", model_type, args.disease,
            f"concepts_topK_{args.topK}"
        )
        g_tb_logs_path = os.path.join(
            args.logs, args.dataset, "completeness", f"seed_{args.seed}", "dataset", model_type, args.disease,
            f"concepts_topK_{args.topK}"
        )
        g_output_path = os.path.join(
            args.output, args.dataset, "completeness", f"seed_{args.seed}", "dataset", model_type, args.disease,
            f"concepts_topK_{args.topK}"
        )

        results_path = os.path.join(
            args.output, args.dataset, "completeness", f"seed_{args.seed}", "dataset", model_type, args.disease
        )
    else:
        dataset_path = os.path.join(
            args.output, args.dataset, "completeness", args.arch, "dataset", model_type, f"concepts_topK_{args.topK}"
        )
        train_dataset = Dataset_completeness_features(dataset_path, transform=None, mode="train")
        val_dataset = Dataset_completeness_features(dataset_path, transform=None, mode="test")
        g_chk_pt_path = os.path.join(
            args.checkpoints, args.dataset, "completeness", args.arch, model_type, f"concepts_topK_{args.topK}"
        )
        g_tb_logs_path = os.path.join(
            args.logs, args.dataset, "completeness", args.arch, model_type, f"concepts_topK_{args.topK}"
        )
        g_output_path = os.path.join(
            args.output, args.dataset, "completeness", args.arch, model_type, f"concepts_topK_{args.topK}"
        )

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")

    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)
    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))

    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print(g_tb_logs_path)
    print("############# Paths ############# ")

    return train_loader, val_loader, g_chk_pt_path, g_output_path, g_tb_logs_path


def setup_pretrained_models_cv(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(args.json_file) as _file:
        paths = json.load(_file)

    checkpoint_t_path = paths[f"{args.dataset}_{args.arch}"]["t"]["checkpoint_t_path"]
    checkpoint_file = paths[f"{args.dataset}_{args.arch}"]["t"]["checkpoint_file"]
    args.root_bb = paths[f"{args.dataset}_{args.arch}"]["bb"]["root_bb"]
    args.checkpoint_bb = paths[f"{args.dataset}_{args.arch}"]["bb"]["checkpoint_bb"]
    args.smoothing_value = paths[f"{args.dataset}_{args.arch}"]["bb"]["smoothing_value"]
    args.img_size = paths[f"{args.dataset}_{args.arch}"]["bb"]["img_size"]
    args.layer = paths[f"{args.dataset}_{args.arch}"]["bb"]["layer"]
    args.labels = get_n_labels_concepts(args)["labels"]
    args.projected = "n"
    args.pretrained = True

    bb_model = utils.get_model_explainer(args, device)
    bb_model.eval()

    checkpoint_t = os.path.join(args.checkpoints, args.dataset, "t", checkpoint_t_path, checkpoint_file)
    input_size_t = get_input_size_t(args, bb_model)
    op_size = len(get_n_labels_concepts(args)["concepts"])
    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=op_size, flattening_type=args.flattening_type
    ).to(device)

    print(f"Device: {device}")
    print(f"===>>> t is loaded from: {checkpoint_t}")

    t.load_state_dict(torch.load(checkpoint_t))
    t.eval()

    cav = t.linear.weight.detach()
    print(f"Size of CAV: {cav.size()}")
    g_completeness = G(True, args.arch, dataset=args.dataset, hidden_nodes=1000).to(device)
    classifier = Residual(args.dataset, True, len(args.labels), args.arch).to(device)
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        classifier.fc.weight = copy.deepcopy(bb_model.base_model.fc.weight)
        classifier.fc.bias = copy.deepcopy(bb_model.base_model.fc.bias)
    elif args.arch == "ViT-B_16":
        classifier.fc.weight = copy.deepcopy(bb_model.part_head.weight)
        classifier.fc.bias = copy.deepcopy(bb_model.part_head.bias)

    classifier.eval()
    print(t)
    return g_completeness, classifier, cav, device


def setup_pretrained_models_skin(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(args.json_file) as _file:
        paths = json.load(_file)

    concept_path = os.path.join(args.output, args.dataset, "t", args.arch)
    bb_model, bb_model_bottom, bb_model_top = None, None, None
    bb_dir = os.path.join(args.checkpoints, args.dataset, "BB", args.arch)
    model_name = paths[f"{args.dataset}_{args.arch}"]["bb"]["model_name"]
    concept_file_name = paths[f"{args.dataset}_{args.arch}"]["bb"]["concept_file_name"]

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(bb_dir, model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(bb_dir, model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, concept_file_name), "rb")
    )
    cavs = []
    for key in concepts_dict.keys():
        cavs.append(concepts_dict[key][0][0].tolist())
    cavs = np.array(cavs)
    print(f"cavs size: {cavs.shape}")
    classifier = copy.deepcopy(bb_model_top)
    classifier.eval()

    g_completeness = G(True, args.arch, dataset=args.dataset, hidden_nodes=1000).to(device)
    cav = torch.from_numpy(cavs).to(device, dtype=torch.float32)
    return g_completeness, classifier, cav, device


def setup_pretrained_models_mimic_cxr(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.json_file)
    with open(args.json_file) as _file:
        paths = json.load(_file)

    checkpoint_t_path = paths[f"{args.disease}"]["t"]["checkpoint_t_path"]
    checkpoint_file = paths[f"{args.disease}"]["t"]["checkpoint_file"]
    args.root_bb = paths[f"{args.disease}"]["bb"]["root_bb"]
    args.checkpoint_bb = paths[f"{args.disease}"]["bb"]["checkpoint_bb"]
    args.pool1 = paths[f"{args.disease}"]["bb"]["pool1"]
    args.layer = paths[f"{args.disease}"]["bb"]["layer"]
    args.labels = paths[f"{args.disease}"]["params"]["labels"]
    args.disease_folder = args.disease
    args.pretrained = True

    bb_model = utils.get_model_explainer(args, device)
    bb_model = bb_model.to(device)
    bb_model.eval()

    checkpoint_t = os.path.join(
        args.checkpoints, args.dataset, "t", checkpoint_t_path, args.arch, args.disease, checkpoint_file
    )
    input_size_t = get_input_size_t(args, bb_model)
    op_size = 107
    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=op_size, flattening_type=args.flattening_type
    ).to(device)

    print(f"Device: {device}")
    print(f"===>>> t is loaded from: {checkpoint_t}")
    t.load_state_dict(torch.load(checkpoint_t)['state_dict'])
    t.eval()

    cav = t.linear.weight.detach()
    cav_select_mask_path = os.path.join(
        args.output, args.dataset, "t", checkpoint_t_path, args.arch, args.disease, "dataset_g",
        "torch_concepts_mask_auroc.pt"
    )
    cav_select_mask = torch.load(cav_select_mask_path).unsqueeze(dim=1).repeat(1, cav.size(1)).to(device)
    cav_select = torch.masked_select(cav, cav_select_mask.bool()).reshape(-1, cav.size(1))
    args.N_concepts = cav_select.size(0)
    print(f"Selected CAV size: {cav_select.size()}")

    g_completeness = G(True, args.arch, dataset=args.dataset, hidden_nodes=1000, concept_size=args.N_concepts).to(
        device)
    classifier = Residual(args.dataset, True, len(args.labels), args.arch).to(device)
    classifier.fc.weight = copy.deepcopy(bb_model.fc1.weight)
    classifier.fc.bias = copy.deepcopy(bb_model.fc1.bias)
    classifier = classifier.to(device)
    classifier.eval()

    print(classifier)
    return g_completeness, classifier, cav_select, device


def setup_pretrained_models(args):
    if args.dataset == "cub" or args.dataset == "awa2":
        return setup_pretrained_models_cv(args)
    elif args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        return setup_pretrained_models_skin(args)
    elif args.dataset == "mimic_cxr":
        return setup_pretrained_models_mimic_cxr(args)


def train_g_for_completeness_score(
        args, g_completeness, classifier, cav, g_chk_pt_path, g_tb_logs_path, g_output_path,
        train_loader, val_loader, device, num_classes
):
    optimizer = torch.optim.Adam(g_completeness.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    logger = Logger_CUBS(
        1, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
        len(get_n_labels_concepts(args)["labels"]), device
    )

    run_id = "g_train"
    logger.begin_run(run_id)

    for epoch in range(args.epochs):
        logger.begin_epoch()
        g_completeness.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                train_features, train_y, train_mask = data_tuple
                train_features, train_y, train_mask = train_features.to(device), train_y.to(torch.long).to(
                    device), train_mask.to(device)

                bs = train_features.size(0)
                norm_vc = get_normalized_vc(
                    train_features,
                    cav,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type=args.flattening_type,
                    per_iter_completeness=args.per_iter_completeness,
                    train_mask=train_mask
                )
                # train_y = train_y.squeeze(dim=1)
                concept_to_act = g_completeness(norm_vc)

                if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "densenet121":
                    concept_to_act = concept_to_act.reshape(
                        bs, train_features.size(1), train_features.size(2), train_features.size(3)
                    )

                if args.dataset == "awa2" or args.dataset == "cub" or args.dataset == "mimic_cxr":
                    completeness_logits = classifier(concept_to_act)
                if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
                    completeness_logits = classifier(concept_to_act)[0]
                optimizer.zero_grad()
                train_loss = criterion(completeness_logits, train_y)

                train_loss.backward()
                optimizer.step()

                logger.track_train_loss(train_loss.item())
                logger.track_total_train_correct_per_epoch(completeness_logits, train_y)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        g_completeness.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    val_features, val_y, val_mask = data_tuple
                    val_features, val_y, val_mask = val_features.to(device), val_y.to(torch.long).to(
                        device), val_mask.to(device)
                    bs = val_features.size(0)

                    norm_vc = get_normalized_vc(
                        val_features,
                        cav,
                        th=0,
                        val_after_th=0,
                        cav_flattening_type=args.flattening_type,
                        per_iter_completeness=args.per_iter_completeness,
                        train_mask=val_mask
                    )
                    # val_y = val_y.squeeze(dim=1)
                    concept_to_act = g_completeness(norm_vc)
                    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "densenet121":
                        concept_to_act = concept_to_act.reshape(
                            bs, val_features.size(1), val_features.size(2), val_features.size(3)
                        )

                    if args.dataset == "awa2" or args.dataset == "cub" or args.dataset == "mimic_cxr":
                        completeness_logits = classifier(concept_to_act)
                    if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
                        completeness_logits = classifier(concept_to_act)[0]

                    val_loss = criterion(completeness_logits, val_y)

                    logger.track_val_loss(val_loss.item())
                    logger.track_total_val_correct_per_epoch(completeness_logits, val_y)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        logger.end_epoch(g_completeness)
        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(logger.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(logger.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(logger.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(logger.get_epoch_duration(), 4)}")
    logger.end_run()

    return g_completeness


def compute_completeness_score(
        args, g_completeness, classifier, cav, g_output_path, test_loader, device, num_classes
):
    g_completeness.eval()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_completeness = torch.FloatTensor().cuda()
    print("\n Computing the completeness score: \n")
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, data_tuple in enumerate(test_loader):
                val_features, val_y, val_mask = data_tuple
                val_features, val_y, val_mask = val_features.to(device), val_y.to(torch.long).to(
                    device), val_mask.to(device)
                bs = val_features.size(0)
                norm_vc = get_normalized_vc(
                    val_features,
                    cav,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type=args.flattening_type,
                    per_iter_completeness=args.per_iter_completeness,
                    train_mask=val_mask
                )
                # val_y = val_y.squeeze(dim=1)
                concept_to_act = g_completeness(norm_vc)
                if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "densenet121":
                    concept_to_act = concept_to_act.reshape(
                        bs, val_features.size(1), val_features.size(2), val_features.size(3)
                    )

                if args.dataset == "awa2" or args.dataset == "cub" or args.dataset == "mimic_cxr":
                    completeness_logits = classifier(concept_to_act)
                    bb_logits = classifier(val_features)
                elif args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
                    completeness_logits = classifier(concept_to_act)[0]
                    bb_logits = classifier(val_features)[0]

                out_put_predict_bb = torch.cat((out_put_predict_bb, bb_logits), dim=0)
                out_put_predict_completeness = torch.cat((out_put_predict_completeness, completeness_logits), dim=0)
                out_put_GT = torch.cat((out_put_GT, val_y), dim=0)
                t.set_postfix(iteration=f"{batch_id}")
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1).numpy()
    y_hat_completeness_bb = out_put_predict_completeness.cpu().argmax(dim=1).numpy()

    results = None
    a_r = 1 / num_classes
    if args.dataset == "awa2" or args.dataset == "cub":
        acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
        acc_completeness = utils.cal_accuracy(out_put_GT_np, y_hat_completeness_bb)
        completeness_score = (acc_completeness - a_r) / (acc_bb - a_r)

        print(f"Top concepts: {args.topK}")
        print(f"Accuracy of the bb: {acc_bb * 100} (%)")
        print(f"Accuracy using the completeness: {acc_completeness * 100} (%)")
        print(f"Completeness_score: {completeness_score}")
        results = {
            "concepts": args.topK,
            "acc_bb": acc_bb * 100,
            "acc_completeness": acc_completeness * 100,
            "completeness_score": completeness_score
        }
    elif args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC" or args.dataset == "mimic_cxr":
        y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1).numpy()
        y_hat_completeness_bb = out_put_predict_completeness.cpu().argmax(dim=1).numpy()
        proba_bb = torch.nn.Softmax(dim=1)(out_put_predict_bb)[:, 1]
        proba_completeness = torch.nn.Softmax(dim=1)(out_put_predict_completeness)[:, 1]
        val_auroc_bb, val_aurpc_bb = utils.compute_AUC(out_put_GT, pred=proba_bb)
        val_auroc_completeness, val_aurpc_completeness = utils.compute_AUC(out_put_GT, pred=proba_completeness)
        completeness_score_auroc = val_auroc_completeness / val_auroc_bb

        print(f"Top concepts: {args.topK}")
        print(f"Auroc of the bb: {val_auroc_bb}")
        print(f"Auroc using the completeness: {val_auroc_completeness}")
        print(f"Completeness_score: {completeness_score_auroc}")

        acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
        acc_completeness = utils.cal_accuracy(out_put_GT_np, y_hat_completeness_bb)
        completeness_score = (acc_completeness - a_r) / (acc_bb - a_r)

        print(f"Top concepts: {args.topK}")
        print(f"Accuracy of the bb: {acc_bb * 100} (%)")
        print(f"Accuracy using the completeness: {acc_completeness * 100} (%)")
        print(f"Completeness_score: {completeness_score}")

        results = {
            "concepts": args.topK,
            "auroc_bb": val_auroc_bb,
            "auroc_completeness": val_auroc_completeness,
            "completeness_score_auroc": completeness_score_auroc,
            "acc_bb": acc_bb * 100,
            "acc_g": acc_completeness * 100,
            "completeness_score_acc": completeness_score
        }

    np.save(os.path.join(g_output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(
        out_put_predict_completeness.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_completeness.pt")
    )
    torch.save(y_hat_bb, os.path.join(g_output_path, f"out_put_predict_bb.pt"))
    print(os.path.join(g_output_path, f"out_put_predict_bb.pt"))

    return results


def get_input_size_t(args, bb):
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        return 2048
    elif args.arch == "ViT-B_16" or args.arch == "ViT-B_16_projected":
        return 768
    elif args.arch == "densenet121":
        t_ip = 0
        if args.flattening_type == "flatten":
            t_ip = (
                       bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" else
                       int(bb.fc1.weight.shape[1] / 2)
                   ) * 16 * 16
        elif args.flattening_type == "adaptive":
            t_ip = bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(bb.fc1.weight.shape[1] / 2)
        elif args.flattening_type == "max_pool":
            t_ip = bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(bb.fc1.weight.shape[1] / 2)
        return t_ip


def get_n_labels_concepts(args):
    if args.dataset == "cub":
        return {"labels": range(200), "concepts": range(108)}
    elif args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        return {"labels": range(2), "concepts": range(8)}
    elif args.dataset == "awa2":
        return {"labels": range(50), "concepts": range(85)}
    elif args.dataset == "mimic_cxr":
        return {"labels": range(2), "concepts": range(args.N_concepts)}


def get_normalized_vc(
        activations,
        torch_concept_vector,
        th,
        val_after_th,
        cav_flattening_type,
        per_iter_completeness=False,
        train_mask=None
):
    if cav_flattening_type == "max_pooled" or cav_flattening_type == "avg_pooled" or cav_flattening_type == "adaptive":
        return get_normalized_vc_using_pooling(
            activations,
            torch_concept_vector,
            th,
            val_after_th,
            per_iter_completeness,
            train_mask
        )
    elif cav_flattening_type == "flattened" or cav_flattening_type == "flatten" or cav_flattening_type == "VIT":
        return get_normalized_vc_using_flattening(
            activations,
            torch_concept_vector,
            th,
            val_after_th,
            per_iter_completeness,
            train_mask
        )


def get_normalized_vc_using_pooling(
        activations,
        torch_concept_vector,
        th,
        val_after_th,
        per_iter_completeness,
        mask
):
    bs, ch = activations.size(0), activations.size(1)
    vc = torch.matmul(
        activations.reshape((bs, ch, -1)).permute((0, 2, 1)),
        torch_concept_vector.T
    ).reshape((bs, -1))
    th_fn = torch.nn.Threshold(threshold=th, value=val_after_th)
    th_vc = th_fn(vc)
    norm_vc = torch.nn.functional.normalize(th_vc, p=2, dim=1)
    if per_iter_completeness:
        norm_vc = norm_vc.reshape((bs, -1, torch_concept_vector.size(0)))
        mask_ = mask.reshape((bs, -1, mask.size(-1))).expand(norm_vc.size())
        norm_vc = (norm_vc * mask_).reshape((bs, -1))
    return norm_vc


def get_normalized_vc_using_flattening(
        activations,
        torch_concept_vector,
        th,
        val_after_th,
        per_iter_completeness,
        mask
):
    bs = activations.size(0)
    vc = torch.matmul(
        activations.reshape((bs, -1)),
        torch_concept_vector.T
    )
    th_fn = torch.nn.Threshold(threshold=th, value=val_after_th)
    th_vc = th_fn(vc)
    norm_vc = torch.nn.functional.normalize(th_vc, p=2, dim=1)

    if per_iter_completeness:
        norm_vc = (norm_vc * mask)
    return norm_vc
