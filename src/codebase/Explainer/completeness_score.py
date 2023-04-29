import copy
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
from Explainer.utils_explainer import get_normalized_vc, ConceptBank, get_data_loaders, \
    get_data_loaders_per_iter_completeness, get_data_tuple, get_data_loaders_per_iter_completeness_baseline
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_awa2 import Dataset_awa2_for_explainer
from dataset.dataset_cubs import Dataset_cub_for_explainer
from dataset.dataset_ham10k import load_ham_data, load_isic
from dataset.utils_dataset import get_dataset_with_image_and_attributes


def cal_completeness_score_per_iter(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.time()
    train_loader, val_loader = get_data_loaders_per_iter_completeness(args)
    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")
    g_chk_pt_path = os.path.join(
        args.checkpoints, args.dataset, "completeness", args.arch, "g_per_iter"
    )
    g_tb_logs_path = os.path.join(args.logs, args.dataset, "completeness", args.arch, "g_per_iter")
    g_output_path = os.path.join(args.output, args.dataset, "completeness", args.arch, "g_per_iter")
    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)
    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")
    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print(g_tb_logs_path)
    print("############# Paths ############# ")
    if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        do_cal_completeness_ham_isic(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
            per_iter_completeness=True
        )
    else:
        do_cal_completeness_cub_awa2(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
            per_iter_completeness=True
        )

def cal_completeness_stats_per_iter(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.time()
    _, test_loader = get_data_loaders_per_iter_completeness(args)
    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")
    g_chk_pt_path = os.path.join(
        args.checkpoints, args.dataset, "completeness", args.arch, "g_per_iter"
    )
    g_tb_logs_path = os.path.join(args.logs, args.dataset, "completeness", args.arch, "g_per_iter")
    g_output_path = os.path.join(args.output, args.dataset, "completeness", args.arch, "g_per_iter")
    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)
    pickle.dump(args, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")
    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print(g_tb_logs_path)
    print("############# Paths ############# ")
    if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        do_test_cal_completeness_ham_isic(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader,
            args.g_checkpoint,
            per_iter_completeness=True
        )
    else:
        do_test_cal_completeness_cub_awa2(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader,
            args.g_checkpoint,
            per_iter_completeness=True
        )


def cal_completeness_stats(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    test_loader = get_test_loaders(args)
    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")

    g_chk_pt_path = os.path.join(args.checkpoints, args.dataset, "completeness", args.arch, "g_all")
    g_tb_logs_path = os.path.join(args.logs, args.dataset, "completeness", args.arch, "g_all")
    g_output_path = os.path.join(args.output, args.dataset, "completeness", args.arch, "g_all")

    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")

    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print(g_tb_logs_path)
    print("############# Paths ############# ")
    if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        do_test_cal_completeness_ham_isic(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader, args.g_checkpoint
        )
    else:
        do_test_cal_completeness_cub_awa2(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader, args.g_checkpoint
        )


def cal_completeness_score(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    train_loader, val_loader = get_data_loaders(args)
    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")

    g_all = f"g_all/{args.disease_folder}" if args.dataset == "mimic_cxr" else "g_all"
    g_chk_pt_path = os.path.join(args.checkpoints, args.dataset, "completeness", args.arch, g_all)
    g_tb_logs_path = os.path.join(args.logs, args.dataset, "completeness", args.arch, g_all)
    g_output_path = os.path.join(args.output, args.dataset, "completeness", args.arch, g_all)

    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")

    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print(g_tb_logs_path)
    print("############# Paths ############# ")

    if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        do_cal_completeness_ham_isic(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader
        )
    else:
        do_cal_completeness_cub_awa2(
            args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader
        )


def do_test_cal_completeness_ham_isic(
        args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader, checkpoint,
        per_iter_completeness=True
):
    concept_path = os.path.join(args.output, args.dataset, "t", args.arch)
    bb_model, bb_model_bottom, bb_model_top = None, None, None

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    cavs = []
    for key in concepts_dict.keys():
        cavs.append(concepts_dict[key][0][0].tolist())
    cavs = np.array(cavs)
    print(f"cavs size: {cavs.shape}")
    concept_bank = ConceptBank(concepts_dict, device)
    residual = copy.deepcopy(bb_model_top)
    residual.eval()

    g = G(args.pretrained, args.arch, dataset=args.dataset, hidden_nodes=args.hidden_nodes).to(device)
    g.load_state_dict(torch.load(os.path.join(g_chk_pt_path, checkpoint)))
    g.eval()

    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_completeness = torch.FloatTensor().cuda()
    torch_concept_vector = torch.from_numpy(cavs).to(device, dtype=torch.float32)
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, data_tuple in enumerate(test_loader):
                if not per_iter_completeness:
                    # for all concepts
                    val_images, val_y = data_tuple
                    val_images = val_images.to(device)
                    val_y = val_y.to(device)
                    val_mask = None
                else:
                    # for selected concepts by experts
                    val_images, val_y, val_mask = data_tuple
                    val_images = val_images.to(device)
                    val_y = val_y.to(device)
                    val_mask = val_mask.to(device)

                with torch.no_grad():
                    bb_logits = bb_model(val_images)
                    feature_x = bb_model_bottom(val_images)
                norm_vc = get_normalized_vc(
                    feature_x,
                    torch_concept_vector,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type="flattened",
                    per_iter_completeness=per_iter_completeness,
                    train_mask=val_mask
                )
                concept_to_act = g(norm_vc)
                completeness_logits = residual(concept_to_act)[0]

                out_put_predict_bb = torch.cat((out_put_predict_bb, bb_logits), dim=0)
                out_put_predict_completeness = torch.cat((out_put_predict_completeness, completeness_logits), dim=0)
                out_put_GT = torch.cat((out_put_GT, val_y), dim=0)
                t.set_postfix(iteration=f"{batch_id}")
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.cpu().numpy()
    out_put_predict_completeness_np = out_put_predict_completeness.cpu().numpy()

    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1).numpy()
    y_hat_completeness_bb = out_put_predict_completeness.cpu().argmax(dim=1).numpy()
    proba_bb = torch.nn.Softmax(dim=1)(out_put_predict_bb)[:, 1]
    proba_completeness = torch.nn.Softmax(dim=1)(out_put_predict_completeness)[:, 1]

    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    val_auroc_bb, val_aurpc_bb = utils.compute_AUC(out_put_GT, pred=proba_bb)

    acc_completeness = utils.cal_accuracy(out_put_GT_np, y_hat_completeness_bb)
    val_auroc_completeness, val_aurpc_completeness = utils.compute_AUC(out_put_GT, pred=proba_completeness)
    completeness_score_acc = (acc_completeness - 0.5) / (acc_bb - 0.5)
    completeness_score_auroc = (val_auroc_completeness) / (val_auroc_bb)

    print(f"Accuracy of the bb: {acc_bb * 100} (%)")
    print(f"Accuracy using the completeness: {acc_completeness * 100} (%)")
    print(f"Completeness_score based on accuracy: {completeness_score_acc}")
    print(f"Auroc of the bb: {val_auroc_bb}")
    print(f"Auroc using the completeness: {val_auroc_completeness}")
    print(f"Completeness_score based on auroc: {completeness_score_auroc}")

    np.save(os.path.join(g_output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(
        out_put_predict_completeness.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_completeness.pt")
    )
    torch.save(y_hat_bb, os.path.join(g_output_path, f"out_put_predict_bb.pt"))

    print(os.path.join(g_output_path, f"out_put_predict_bb.pt"))


def do_cal_completeness_ham_isic(
        args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
        per_iter_completeness=False
):
    concept_path = os.path.join(args.output, args.dataset, "t", args.arch)
    bb_model, bb_model_bottom, bb_model_top = None, None, None

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    cavs = []
    for key in concepts_dict.keys():
        cavs.append(concepts_dict[key][0][0].tolist())
    cavs = np.array(cavs)
    print(f"cavs size: {cavs.shape}")
    concept_bank = ConceptBank(concepts_dict, device)
    residual = copy.deepcopy(bb_model_top)
    residual.eval()

    g = G(args.pretrained, args.arch, dataset=args.dataset, hidden_nodes=args.hidden_nodes).to(device)
    optimizer = torch.optim.Adam(g.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    logger = Logger_CUBS(
        1, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader, len(args.labels), device
    )
    run_id = "g_train"
    fit_g_skin(
        args.epochs, bb_model, bb_model_bottom, residual, cavs, g, optimizer, train_loader, val_loader, criterion,
        logger,
        args.arch, args.dataset, run_id, device, per_iter_completeness
    )


def fit_g_skin(
        epochs,
        bb_model,
        bb_model_bottom,
        residual,
        cav,
        g,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        logger,
        arch,
        dataset,
        run_id,
        device,
        per_iter_completeness=False
):
    torch_concept_vector = torch.from_numpy(cav).to(device, dtype=torch.float32)
    logger.begin_run(run_id)
    for epoch in range(epochs):
        logger.begin_epoch()
        g.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if per_iter_completeness:
                    train_images, train_y, train_mask = data_tuple
                    train_images = train_images.to(device)
                    train_y = train_y.to(torch.long).to(device)
                    train_mask = train_mask.to(device)
                else:
                    train_images, train_y = data_tuple
                    train_images = train_images.to(device)
                    train_y = train_y.to(torch.long).to(device)

                bs = train_images.size(0)
                with torch.no_grad():
                    bb_logits = bb_model(train_images)
                    feature_x = bb_model_bottom(train_images)

                norm_vc = get_normalized_vc(
                    feature_x,
                    torch_concept_vector,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type="flattened",
                    per_iter_completeness=per_iter_completeness,
                    train_mask=train_mask
                )
                concept_to_act = g(norm_vc)
                completeness_logits = residual(concept_to_act)[0]
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

        g.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if per_iter_completeness:
                        val_images, val_y, val_mask = data_tuple
                        val_images = val_images.to(device)
                        val_y = val_y.to(torch.long).to(device)
                        val_mask = val_mask.to(device)
                    else:
                        val_images, val_y = data_tuple
                        val_images = val_images.to(device)
                        val_y = val_y.to(torch.long).to(device)

                    with torch.no_grad():
                        bb_logits = bb_model(val_images)
                        feature_x = bb_model_bottom(val_images)

                    norm_vc = get_normalized_vc(
                        feature_x,
                        torch_concept_vector,
                        th=0,
                        val_after_th=0,
                        cav_flattening_type="flattened",
                        per_iter_completeness=per_iter_completeness,
                        train_mask=val_mask
                    )
                    concept_to_act = g(norm_vc)
                    completeness_logits = residual(concept_to_act)[0]
                    val_loss = criterion(completeness_logits, val_y)

                    logger.track_val_loss(val_loss.item())
                    logger.track_total_val_correct_per_epoch(completeness_logits, val_y)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        logger.end_epoch(g)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(logger.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(logger.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(logger.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(logger.get_epoch_duration(), 4)}")
    logger.end_run()


def do_test_cal_completeness_cub_awa2(
        args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, test_loader, g_checkpoint,
        per_iter_completeness=False
):
    bb = utils.get_model_explainer(args, device)
    bb.eval()
    print(" ################ BB loaded ################")

    chk_pt_path_t = os.path.join(
        args.checkpoints,
        args.dataset,
        "t",
        args.checkpoint_t_path
    )

    print(f"t is loaded from: {chk_pt_path_t}")
    input_size_t = get_input_size_t(args, bb)
    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=len(args.concept_names), flattening_type=args.flattening_type
    ).to(device)
    t.load_state_dict(torch.load(os.path.join(chk_pt_path_t, args.checkpoint_file_t)))
    t.eval()

    cav = t.linear.weight.detach()
    print(f"Size of CAV: {cav.size()}")
    g = G(args.pretrained, args.arch, dataset=args.dataset, hidden_nodes=args.hidden_nodes).to(device)
    g.load_state_dict(torch.load(os.path.join(g_chk_pt_path, g_checkpoint)))
    g.eval()

    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        residual.fc.weight = copy.deepcopy(bb.base_model.fc.weight)
        residual.fc.bias = copy.deepcopy(bb.base_model.fc.bias)
    elif args.arch == "ViT-B_16":
        residual.fc.weight = copy.deepcopy(bb.part_head.weight)
        residual.fc.bias = copy.deepcopy(bb.part_head.bias)
    residual.eval()

    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_completeness = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, data_tuple in enumerate(test_loader):
                if not per_iter_completeness:
                    # for all concepts
                    val_images, val_y = get_data_tuple(data_tuple, args.dataset, device, per_iter_completeness)
                else:
                    # for selected concepts by experts
                    val_images, val_y, val_mask = get_data_tuple(
                        data_tuple, args.dataset, device, per_iter_completeness
                    )
                bs = val_images.size(0)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(val_images, bb, args.arch, args.layer)
                norm_vc = get_normalized_vc(
                    feature_x,
                    cav,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type=args.flattening_type,
                    per_iter_completeness=per_iter_completeness,
                    train_mask=val_mask
                )
                concept_to_act = g(norm_vc)
                if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "densenet121":
                    concept_to_act = concept_to_act.reshape(
                        bs, feature_x.size(1), feature_x.size(2), feature_x.size(3)
                    )
                completeness_logits = residual(concept_to_act)

                out_put_predict_bb = torch.cat((out_put_predict_bb, bb_logits), dim=0)
                out_put_predict_completeness = torch.cat((out_put_predict_completeness, completeness_logits), dim=0)
                out_put_GT = torch.cat((out_put_GT, val_y), dim=0)
                t.set_postfix(iteration=f"{batch_id}")
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.cpu().numpy()
    out_put_predict_completeness_np = out_put_predict_completeness.cpu().numpy()

    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1).numpy()
    y_hat_completeness_bb = out_put_predict_completeness.cpu().argmax(dim=1).numpy()
    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    acc_completeness = utils.cal_accuracy(out_put_GT_np, y_hat_completeness_bb)
    completeness_score = (acc_completeness - 0.5) / (acc_bb - 0.5)

    print(f"Accuracy of the bb: {acc_bb * 100} (%)")
    print(f"Accuracy using the completeness: {acc_completeness * 100} (%)")
    print(f"Completeness_score: {completeness_score}")

    np.save(os.path.join(g_output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(
        out_put_predict_completeness.cpu(), os.path.join(g_output_path, f"out_put_predict_logits_completeness.pt")
    )
    torch.save(y_hat_bb, os.path.join(g_output_path, f"out_put_predict_bb.pt"))

    print(os.path.join(g_output_path, f"out_put_predict_bb.pt"))


def do_cal_completeness_cub_awa2(
        args, device, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
        per_iter_completeness=False
):
    bb = utils.get_model_explainer(args, device)
    bb.eval()
    print(" ################ BB loaded ################")

    chk_pt_path_t = os.path.join(
        args.checkpoints,
        args.dataset,
        "t",
        args.checkpoint_t_path
    )

    print(f"t is loaded from: {chk_pt_path_t}")
    input_size_t = get_input_size_t(args, bb)

    op_size = len(args.landmark_names_spec) + len(args.abnorm_obs_concepts) if args.dataset == "mimic_cxr" else len(
        args.concept_names)

    t = Logistic_Regression_t(
        ip_size=input_size_t, op_size=op_size, flattening_type=args.flattening_type
    ).to(device)
    if args.dataset == "mimic_cxr":
        t.load_state_dict(torch.load(os.path.join(chk_pt_path_t, args.checkpoint_file_t))['state_dict'])
    else:
        t.load_state_dict(torch.load(os.path.join(chk_pt_path_t, args.checkpoint_file_t)))
    t.eval()

    cav = t.linear.weight.detach()
    print(f"Size of CAV: {cav.size()}")
    g = G(args.pretrained, args.arch, dataset=args.dataset, hidden_nodes=args.hidden_nodes).to(device)
    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        residual.fc.weight = copy.deepcopy(bb.base_model.fc.weight)
        residual.fc.bias = copy.deepcopy(bb.base_model.fc.bias)
    elif args.arch == "ViT-B_16":
        residual.fc.weight = copy.deepcopy(bb.part_head.weight)
        residual.fc.bias = copy.deepcopy(bb.part_head.bias)
    elif args.arch == "densenet121":
        residual.fc.weight = copy.deepcopy(bb.fc1.weight)
        residual.fc.bias = copy.deepcopy(bb.fc1.bias)
        residual = residual.to(device)

    residual.eval()
    optimizer = torch.optim.Adam(g.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    logger = Logger_CUBS(
        1, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader, len(args.labels), device
    )

    print(t)
    run_id = "g_train"
    fit_g(
        args.epochs, bb, residual, cav, g, optimizer, train_loader, val_loader, criterion, logger, args.layer,
        args.arch, args.dataset, run_id, device, args.flattening_type, per_iter_completeness
    )


def fit_g(
        epochs,
        bb,
        residual,
        cav,
        g,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        logger,
        layer,
        arch,
        dataset,
        run_id,
        device,
        flattening_type,
        per_iter_completeness=False
):
    logger.begin_run(run_id)
    for epoch in range(epochs):
        logger.begin_epoch()
        g.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if not per_iter_completeness:
                    # for all concepts
                    train_images, train_y = get_data_tuple(data_tuple, dataset, device, per_iter_completeness)
                    train_mask = None
                else:
                    # for selected concepts by experts
                    train_images, train_y, train_mask = get_data_tuple(
                        data_tuple, dataset, device, per_iter_completeness
                    )
                bs = train_images.size(0)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)

                norm_vc = get_normalized_vc(
                    feature_x,
                    cav,
                    th=0,
                    val_after_th=0,
                    cav_flattening_type=flattening_type,
                    per_iter_completeness=per_iter_completeness,
                    train_mask=train_mask
                )
                # train_y = train_y.squeeze(dim=1)
                concept_to_act = g(norm_vc)
                if arch == "ResNet50" or arch == "ResNet101" or arch == "densenet121":
                    concept_to_act = concept_to_act.reshape(bs, feature_x.size(1), feature_x.size(2), feature_x.size(3))

                completeness_logits = residual(concept_to_act)
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

        g.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if not per_iter_completeness:
                        # for all concepts
                        val_images, val_y = get_data_tuple(data_tuple, dataset, device, per_iter_completeness)
                        val_mask = None
                    else:
                        # for selected concepts by experts
                        val_images, val_y, val_mask = get_data_tuple(
                            data_tuple, dataset, device, per_iter_completeness
                        )
                    bs = val_images.size(0)
                    with torch.no_grad():
                        bb_logits, feature_x = get_phi_x(val_images, bb, arch, layer)

                    norm_vc = get_normalized_vc(
                        feature_x,
                        cav,
                        th=0,
                        val_after_th=0,
                        cav_flattening_type=flattening_type,
                        per_iter_completeness=per_iter_completeness,
                        train_mask=val_mask
                    )
                    # val_y = val_y.squeeze(dim=1)
                    concept_to_act = g(norm_vc)
                    if arch == "ResNet50" or arch == "ResNet101" or arch == "densenet121":
                        concept_to_act = concept_to_act.reshape(
                            bs, feature_x.size(1), feature_x.size(2), feature_x.size(3)
                        )
                    completeness_logits = residual(concept_to_act)
                    val_loss = criterion(completeness_logits, val_y)

                    logger.track_val_loss(val_loss.item())
                    logger.track_total_val_correct_per_epoch(completeness_logits, val_y)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        logger.end_epoch(g)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(logger.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(logger.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(logger.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(logger.get_epoch_duration(), 4)}")
    logger.end_run()


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


def get_phi_x(image, bb, arch, layer):
    if arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152":
        bb_logits = bb(image)
        # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
        feature_x = bb.feature_store[layer]
        return bb_logits, feature_x
    elif arch == "ViT-B_16":
        logits, tokens = bb(image)
        return logits, tokens[:, 0]
    if arch == "densenet121":
        return None, image.squeeze(dim=1)


def get_test_loaders(args):
    if args.dataset == "cub":
        dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
        test_data, test_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="test",
            attribute_file=args.attribute_file_name,
        )
        transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
        test_transform = transforms["val_transform"]

        val_dataset = Dataset_cub_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt", test_data,
            test_transform
        )
        test_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader

    elif args.dataset == 'awa2':
        dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
        transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
        test_transform = transforms["val_transform"]
        test_dataset = Dataset_awa2_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt",
            "test_image_names.pkl",
            test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader

    elif args.dataset == 'HAM10k':
        from torchvision import transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize
            ]
        )
        train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx)
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        return val_loader
    elif args.dataset == "SIIM-ISIC":
        from torchvision import transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize
            ]
        )
        train_loader, val_loader, idx_to_class = load_isic(args, transform, mode="train")
        return val_loader
