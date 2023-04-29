import copy
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import utils
from BB.models.BB_Inception_V3 import get_model, get_BB_model_isic
from Explainer.loss_F import Selective_Distillation_Loss, entropy_loss, KD_Residual_Loss
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.explainer import Explainer
from Explainer.profiler import fit_g_ham_profile, fit_residual_ham_profiler
from Explainer.utils_explainer import get_previous_pi_vals, get_glts_for_HAM10k, ConceptBank, \
    get_glts_for_HAM10k_soft_seed
from Logger.logger_cubs import Logger_CUBS
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_ham10k import load_ham_data

warnings.filterwarnings("ignore")


def test(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}" \
           f"_input-size-pi_{args.input_size_pi}" \
           f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}" \
           f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}" \
           f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"

    concept_path = os.path.join(args.output, args.dataset, "t", args.arch)
    print(root)

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
    train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx, mode="save")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(idx_to_class)

    iteration = args.iter
    cov = args.cov[iteration - 1]
    lr_explainer = args.lr[iteration - 1]
    print(args.seed)
    for seed in range(args.seed):
        print("==============================================")
        print(f"seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("==============================================")
        if args.expert_to_train == "explainer":
            print("Testing explainer")
            test_explainer(args, seed, cov, lr_explainer, root, iteration, concept_path, train_loader, val_loader)
        elif args.expert_to_train == "residual":
            print("Testing for residual")
            test_residual(args, seed, root, iteration, concept_path, train_loader, val_loader)


def test_residual(args, seed, root, iteration, concept_path, train_loader, val_loader):
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    chk_pt_residual = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer")
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        chk_pt_residual = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(
            output_path_explainer, f"iter{iteration}", "explainer", "accuracy"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"iter{iteration}", "bb", "accuracy"
        )
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"iter{iteration}", "bb", "accuracy")
    else:
        g_chk_pt_path = os.path.join(
            chk_pt_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "explainer", "accuracy"
        )
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "explainer",
            "accuracy"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "bb", "accuracy"
        )
        residual_chk_pt_path = os.path.join(
            chk_pt_residual, f"cov_{args.cov[-1]}", f"iter{iteration}", "bb", "accuracy"
        )

    residual_tb_logs_path = log_path_explainer
    # residual_output_path = os.path.join(args.output, args.dataset, "explainer", root, f"iter{iteration}", "bb")
    output_path_model_outputs = os.path.join(residual_output_path, "model_outputs")
    output_path_residual_outputs = os.path.join(residual_output_path, "residual_outputs")

    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_residual_outputs, exist_ok=True)

    os.makedirs(residual_chk_pt_path, exist_ok=True)
    os.makedirs(residual_tb_logs_path, exist_ok=True)
    os.makedirs(residual_output_path, exist_ok=True)

    print("################### Paths ###################")
    print(residual_chk_pt_path)
    print(residual_output_path)
    print("################### Paths ###################  ")

    pickle.dump(args, open(os.path.join(residual_output_path, "train_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    print(concepts_dict)
    concept_bank = ConceptBank(concepts_dict, device)
    lambda_lens = args.lambda_lens

    glt_list = []
    prev_residual = None
    glt = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens
    ).to(device)
    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    # glt.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    glt.load_state_dict(torch.load(glt_chk_pt))
    glt.eval()

    if iteration > 1:
        prev_residual_chk_pt = os.path.join(
            args.prev_explainer_chk_pt_folder[-1], "bb", "accuracy", args.checkpoint_residual[-1]
        )
        print(f"=======>> BB (residual) is loaded from {prev_residual_chk_pt}")
        glt_list = get_glts_for_HAM10k(iteration, args, device)

    cur_residual_chkpt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
    print(f"===> Latest residual checkpoint is loaded for iteration {iteration}: {cur_residual_chkpt}")
    residual = copy.deepcopy(bb_model_top)
    residual.load_state_dict(torch.load(cur_residual_chkpt))
    residual.eval()

    print("!! Saving val loader for residual expert !!")
    save_results_selected_residual_by_pi(
        args.arch,
        iteration,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        concept_bank,
        glt,
        glt_list,
        residual,
        val_loader,
        output_path_residual_outputs,
        args.selection_threshold,
        device,
        mode="test"
    )
    print(f"######### {output_path_residual_outputs} #########")

    print("!! Saving train loader for residual expert !!")
    save_results_selected_residual_by_pi(
        args.arch,
        iteration,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        concept_bank,
        glt,
        glt_list,
        residual,
        val_loader,
        output_path_residual_outputs,
        args.selection_threshold,
        device,
        mode="test"
    )

    print("Saving the results for the overall dataset")
    predict_residual(
        bb_model,
        bb_model_bottom,
        bb_model_top,
        concept_bank,
        residual,
        val_loader,
        output_path_model_outputs,
        device
    )


def save_results_selected_residual_by_pi(
        arch,
        iteration,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        concept_bank,
        glt,

        glt_list,
        residual,
        loader,
        output_path,
        selection_threshold,
        device,
        mode
):
    tensor_images = torch.FloatTensor()
    tensor_concepts = torch.FloatTensor().cuda()
    tensor_preds_residual = torch.FloatTensor().cuda()
    tensor_preds_bb = torch.FloatTensor().cuda()
    tensor_y = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, (images, raw_image, target) in enumerate(loader):
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    concepts = compute_dist(concept_bank, phi)

                residual_student_logits, _ = residual(phi)
                out_class, out_select, out_aux = glt(concepts)
                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, concepts)

                arr_sel_indices = get_selected_idx_for_residual(
                    iteration, out_select, selection_threshold, device, pi_list
                )
                if arr_sel_indices.size(0) > 0:
                    residual_images = images[arr_sel_indices, :, :, :]
                    residual_concepts = concepts[arr_sel_indices, :]
                    residual_preds = residual_student_logits[arr_sel_indices, :]
                    bb_preds = bb_logits[arr_sel_indices, :]
                    residual_y = target[arr_sel_indices]

                    tensor_images = torch.cat((tensor_images, residual_images.cpu()), dim=0)
                    tensor_concepts = torch.cat((tensor_concepts, residual_concepts), dim=0)
                    tensor_preds_bb = torch.cat((tensor_preds_bb, bb_preds), dim=0)
                    tensor_preds_residual = torch.cat((tensor_preds_residual, residual_preds), dim=0)
                    tensor_y = torch.cat((tensor_y, residual_y), dim=0)

                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    tensor_concepts = tensor_concepts.cpu()
    tensor_preds_bb = tensor_preds_bb.cpu()
    tensor_preds_residual = tensor_preds_residual.cpu()
    tensor_y = tensor_y.cpu()

    print("Output sizes: ")
    print(f"tensor_images size: {tensor_images.size()}")
    print(f"tensor_concepts size: {tensor_concepts.size()}")
    print(f"tensor_preds_bb size: {tensor_preds_bb.size()}")
    print(f"tensor_preds_residual size: {tensor_preds_residual.size()}")
    print(f"tensor_y size: {tensor_y.size()}")

    print("------------------- Metrics ---------------------")
    labels = ['0 (Benign)', '1 (Malignant)']
    proba = torch.nn.Softmax(dim=1)(tensor_preds_residual)[:, 1]
    val_auroc, val_aurpc = utils.compute_AUC(tensor_y, pred=proba)
    acc_bb = utils.cal_accuracy(
        tensor_y.cpu().numpy(), tensor_preds_residual.cpu().argmax(dim=1).numpy()
    )
    print(f"Accuracy of the network: {acc_bb * 100} (%)")
    print(f"Val AUROC of the network: {val_auroc} (0-1)")
    print("------------------- Metrics ---------------------")

    acc_residual = torch.sum(tensor_preds_residual.argmax(dim=1) == tensor_y) / tensor_y.size(0)
    acc_bb = torch.sum(tensor_preds_bb.argmax(dim=1) == tensor_y) / tensor_y.size(0)
    print(f"Accuracy of Residual: {acc_residual * 100} || Accuracy of BB: {acc_bb * 100}")

    cov = tensor_y.size(0) / 2003
    print(f"Scaled Accuracy of Residual: {acc_residual * cov * 100}")
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_images.pt"), tensor_to_save=tensor_images
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concepts.pt"), tensor_to_save=tensor_concepts
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds_bb.pt"), tensor_to_save=tensor_preds_bb
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds_residual.pt"), tensor_to_save=tensor_preds_residual
    )
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_tensor_y.pt"), tensor_to_save=tensor_y)


def get_selected_idx_for_residual(iteration, selection_out, selection_threshold, device, prev_selection_outs=None):
    if iteration == 1:
        return torch.nonzero(
            selection_out < selection_threshold, as_tuple=True
        )[0]
    else:
        condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        for proba in prev_selection_outs:
            condition = condition & (proba < selection_threshold)
        return torch.nonzero(
            (condition & (selection_out < selection_threshold)), as_tuple=True
        )[0]


def predict_residual(
        bb_model,
        bb_model_bottom,
        bb_model_top,
        concept_bank,
        residual,
        loader,
        output_path_model_outputs,
        device
):
    out_put_preds_residual = torch.FloatTensor().cuda()
    out_put_preds_bb = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, (images, raw_image, target) in enumerate(loader):
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    concepts = compute_dist(concept_bank, phi)

                residual_student_logits, _ = residual(phi)
                out_put_preds_bb = torch.cat((out_put_preds_bb, bb_logits), dim=0)
                out_put_preds_residual = torch.cat((out_put_preds_residual, residual_student_logits), dim=0)
                out_put_target = torch.cat((out_put_target, target), dim=0)

                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    out_put_preds_bb = out_put_preds_bb.cpu()
    out_put_preds_residual = out_put_preds_residual.cpu()
    out_put_target = out_put_target.cpu()

    print(f"out_put_preds_bb size: {out_put_preds_bb.size()}")
    print(f"out_put_preds_residual size: {out_put_preds_residual.size()}")
    print(f"out_put_target size: {out_put_target.size()}")

    print(
        f"BB Accuracy: "
        f"{(out_put_preds_bb.argmax(dim=1).eq(out_put_target).sum() / out_put_preds_bb.size(0)) * 100}"
    )
    print(
        f"Residual Accuracy: "
        f"{(out_put_preds_residual.argmax(dim=1).eq(out_put_target).sum() / out_put_preds_bb.size(0)) * 100}"
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"test_out_put_preds_bb.pt"),
        tensor_to_save=out_put_preds_bb
    )
    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"test_out_put_preds_residual.pt"),
        tensor_to_save=out_put_preds_residual
    )
    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"test_out_put_target.pt"),
        tensor_to_save=out_put_target
    )


def test_explainer(args, seed, cov, lr_explainer, root, iteration, concept_path, train_loader, val_loader):
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer")
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(
            output_path_explainer, f"iter{iteration}", "explainer", "accuracy"
        )
    else:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"cov_{cov}", f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}", f"iter{iteration}", "explainer", "accuracy"
        )

    # g_tb_logs_path = os.path.join(args.logs, args.dataset, "explainer", root)
    # chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", root)
    # g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer")
    # g_output_path = os.path.join(args.output, args.dataset, "explainer", root, f"iter{iteration}", "explainer")

    output_path_model_outputs = os.path.join(g_output_path, "model_outputs")
    output_path_g_outputs = os.path.join(g_output_path, "g_outputs")
    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_g_outputs, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))

    device = utils.get_device()
    print(f"Device: {device}")

    bb_model, bb_model_bottom, bb_model_top = None, None, None
    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    print(concepts_dict)
    concept_bank = ConceptBank(concepts_dict, device)
    glt_list = []
    residual = None
    if iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1], "bb", "accuracy", args.checkpoint_residual[-1]
        )
        print(f"=======>> BB (residual) is loaded from {residual_chk_pt_path}")
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        residual = copy.deepcopy(bb_model_top)
        # residual.load_state_dict(torch.load(residual_chk_pt_path)["state_dict"])
        residual.load_state_dict(torch.load(residual_chk_pt_path))
    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(residual_chk_pt_path))
    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(residual_chk_pt_path))

    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi, args.concept_names, args.labels, args.hidden_nodes, args.conceptizator,
        args.temperature_lens,
    ).to(device)

    # model.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    print("Save overall whole model outputs")
    predict(
        bb_model,
        bb_model_bottom,
        bb_model_top,
        model,
        residual,
        val_loader,
        concept_bank,
        output_path_model_outputs,
        device,
    )

    print("!! Saving val loader only selected by g!!")
    save_results_selected_by_pi(
        iteration,
        model,
        bb_model,
        bb_model_bottom,
        residual,
        val_loader,
        concept_bank,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="val",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )

    print("!! Saving train loader only selected by g!!")
    save_results_selected_by_pi(
        iteration,
        model,
        bb_model,
        bb_model_bottom,
        residual,
        train_loader,
        concept_bank,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="train",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )


def get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, prev_selection_outs=None):
    if iteration == 1:
        return torch.nonzero(
            selection_out >= selection_threshold, as_tuple=True
        )[0]
    else:
        condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        for proba in prev_selection_outs:
            condition = condition & (proba < selection_threshold)
        return torch.nonzero(
            (condition & (selection_out >= selection_threshold)), as_tuple=True
        )[0]


def save_results_selected_by_pi(
        iteration,
        model,
        bb_model,
        bb_model_bottom,
        residual,
        loader,
        concept_bank,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params
):
    glt_list = None
    residual = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]
        residual = higher_iter_params["residual"]

    tensor_images = torch.FloatTensor()
    tensor_features = torch.FloatTensor()
    tensor_concepts = torch.FloatTensor().cuda()
    tensor_preds = torch.FloatTensor().cuda()
    tensor_preds_bb = torch.FloatTensor().cuda()
    tensor_y = torch.FloatTensor().cuda()
    tensor_conceptizator_concepts = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch in enumerate(loader):
                batch_idx, (images, raw_image, target) = batch
                images = images.to(device)
                raw_image = raw_image.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    val_bb_logits = None
                    if iteration == 1:
                        val_bb_logits = bb_logits
                    elif iteration > 1:
                        val_bb_logits, _ = residual(phi)

                    concepts = compute_dist(concept_bank, phi)
                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, concepts)

                prediction_out, selection_out, auxiliary_out, concept_mask, \
                alpha, alpha_norm, conceptizator = model(concepts.to(device), test=True)

                arr_sel_indices = get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, pi_list)
                if arr_sel_indices.size(0) > 0:
                    g_images = raw_image[arr_sel_indices, :, :, :]
                    g_features = phi[arr_sel_indices, :]
                    g_concepts = concepts[arr_sel_indices, :]
                    g_preds = prediction_out[arr_sel_indices, :]
                    g_preds_bb = val_bb_logits[arr_sel_indices, :]
                    g_y = target[arr_sel_indices]
                    g_conceptizator_concepts = conceptizator.concepts[:, arr_sel_indices, :]

                    # tensor_images = torch.cat((tensor_images, g_images.cpu()), dim=0)
                    tensor_features = torch.cat((tensor_features, g_features.cpu()), dim=0)
                    tensor_concepts = torch.cat((tensor_concepts, g_concepts), dim=0)
                    tensor_preds = torch.cat((tensor_preds, g_preds), dim=0)
                    tensor_preds_bb = torch.cat((tensor_preds_bb, g_preds_bb), dim=0)
                    tensor_y = torch.cat((tensor_y, g_y), dim=0)
                    tensor_conceptizator_concepts = torch.cat(
                        (tensor_conceptizator_concepts, g_conceptizator_concepts), dim=1
                    )

                # tensor_conceptizator_threshold = conceptizator.threshold
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm
                t.set_postfix(batch_id="{0}".format(batch_idx))
                t.update()

    # tensor_images = tensor_images.cpu()

    tensor_concepts = tensor_concepts.cpu()
    tensor_preds = tensor_preds.cpu()
    tensor_y = tensor_y.cpu()
    tensor_conceptizator_concepts = tensor_conceptizator_concepts.cpu()
    # tensor_conceptizator_threshold = tensor_conceptizator_threshold.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    print("Output sizes: ")
    print(f"tensor_images size: {tensor_images.size()}")
    print(f"tensor_features size: {tensor_features.size()}")
    print(f"tensor_concepts size: {tensor_concepts.size()}")
    print(f"tensor_preds size: {tensor_preds.size()}")
    print(f"tensor_preds_bb size: {tensor_preds_bb.size()}")
    print(f"tensor_y size: {tensor_y.size()}")
    print(f"tensor_conceptizator_concepts size: {tensor_conceptizator_concepts.size()}")

    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    print("------------------- Metrics ---------------------")
    labels = ['0 (Benign)', '1 (Malignant)']
    proba = torch.nn.Softmax(dim=1)(tensor_preds)[:, 1]
    val_auroc, val_aurpc = utils.compute_AUC(tensor_y, pred=proba)
    acc_bb = utils.cal_accuracy(tensor_y.cpu().numpy(), tensor_preds.cpu().argmax(dim=1).numpy())
    print(f"Accuracy of the network: {acc_bb * 100} (%)")
    print(f"Val AUROC of the network: {val_auroc} (0-1)")
    print("------------------- Metrics ---------------------")

    # utils.save_tensor(
    #     path=os.path.join(output_path, f"{mode}_tensor_images.pt"), tensor_to_save=tensor_images
    # )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_features.pt"), tensor_to_save=tensor_features
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concepts.pt"), tensor_to_save=tensor_concepts
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds.pt"), tensor_to_save=tensor_preds
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds_bb.pt"), tensor_to_save=tensor_preds_bb
    )
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_tensor_y.pt"), tensor_to_save=tensor_y)
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_conceptizator_concepts.pt"),
        tensor_to_save=tensor_conceptizator_concepts
    )

    # utils.save_tensor(
    #     path=os.path.join(output_path, f"{mode}_tensor_conceptizator_threshold.pt"),
    #     tensor_to_save=tensor_conceptizator_threshold
    # )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concept_mask.pt"),
        tensor_to_save=tensor_concept_mask
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha.pt"),
        tensor_to_save=tensor_alpha
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha_norm.pt"), tensor_to_save=tensor_alpha_norm
    )

    print(output_path)


def predict(
        bb_model,
        bb_model_bottom,
        bb_model_top,
        model,
        residual,
        loader,
        concept_bank,
        output_path,
        device,
):
    out_put_sel_proba = torch.FloatTensor().cuda()
    out_put_class = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()
    concepts = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch in enumerate(loader):
                batch_idx, (images, _, target) = batch
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    phi = bb_model_bottom(images)
                    val_concepts = compute_dist(concept_bank, phi)
                out_class, out_select, out_aux = model(val_concepts)
                out_put_sel_proba = torch.cat((out_put_sel_proba, out_select), dim=0)
                out_put_class = torch.cat((out_put_class, out_class), dim=0)
                out_put_target = torch.cat((out_put_target, target), dim=0)
                concepts = torch.cat((concepts, val_concepts), dim=0)

                t.set_postfix(batch_id="{0}".format(batch_idx))
                t.update()

    out_put_sel_proba = out_put_sel_proba.cpu()
    out_put_class_pred = out_put_class.cpu()
    out_put_target = out_put_target.cpu()
    concepts = concepts.cpu()

    print(f"out_put_sel_proba size: {out_put_sel_proba.size()}")
    print(f"out_put_class_pred size: {out_put_class_pred.size()}")
    print(f"out_put_target size: {out_put_target.size()}")
    print(f"concepts size: {concepts.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"test_out_put_sel_proba.pt"),
        tensor_to_save=out_put_sel_proba
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"test_out_put_class_pred.pt"),
        tensor_to_save=out_put_class_pred
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"test_out_put_target.pt"),
        tensor_to_save=out_put_target
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"test_concepts_target.pt"),
        tensor_to_save=concepts
    )


def train(args):
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    root = f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}" \
           f"_input-size-pi_{args.input_size_pi}" \
           f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}" \
           f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}" \
           f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"

    concept_path = os.path.join(args.output, args.dataset, "t", args.arch)
    print(root)

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
    print(idx_to_class)

    iteration = args.iter
    cov = args.cov[iteration - 1]
    lr_explainer = args.lr[iteration - 1]
    print(f"iteration: {iteration}========================>>")
    for seed in range(args.seed):
        print("==============================================")
        print(f"seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("==============================================")
        if args.expert_to_train == "explainer":
            train_explainer(
                args, seed, cov, lr_explainer, root, iteration, concept_path, train_loader, val_loader
            )
        elif args.expert_to_train == "residual":
            train_residual(args, seed, root, iteration, concept_path, train_loader, val_loader)


def train_residual(args, seed, root, iteration, concept_path, train_loader, val_loader):
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    chk_pt_residual = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer")
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        chk_pt_residual = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(
            output_path_explainer, f"iter{iteration}", "explainer", "accuracy"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"iter{iteration}", "bb", "accuracy"
        )
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"iter{iteration}", "bb", "accuracy")
    else:
        g_chk_pt_path = os.path.join(
            chk_pt_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "explainer", "accuracy"
        )
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "explainer",
            "accuracy"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"cov_{args.cov[-1]}", f"iter{iteration}", "bb", "accuracy"
        )
        residual_chk_pt_path = os.path.join(
            chk_pt_residual, f"cov_{args.cov[-1]}", f"iter{iteration}", "bb", "accuracy"
        )

    residual_tb_logs_path = log_path_explainer
    # residual_output_path = os.path.join(args.output, args.dataset, "explainer", root, f"iter{iteration}", "bb")

    os.makedirs(residual_chk_pt_path, exist_ok=True)
    os.makedirs(residual_tb_logs_path, exist_ok=True)
    os.makedirs(residual_output_path, exist_ok=True)

    print("################### Paths ###################")
    print(residual_chk_pt_path)
    print(residual_output_path)
    print("################### Paths ###################  ")

    pickle.dump(args, open(os.path.join(residual_output_path, "train_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")

    bb_model, bb_model_bottom, bb_model_top = None, None, None

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")

    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    print(concepts_dict)
    concept_bank = ConceptBank(concepts_dict, device)
    lambda_lens = args.lambda_lens

    glt_list = []
    prev_residual = None
    glt = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens
    ).to(device)
    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    # glt.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    glt.load_state_dict(torch.load(glt_chk_pt))
    glt.eval()

    residual = copy.deepcopy(bb_model_top)
    if iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        prev_residual_chk_pt = os.path.join(
            args.prev_explainer_chk_pt_folder[-1], "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        # residual.load_state_dict(torch.load(prev_residual_chk_pt)["state_dict"])
        residual.load_state_dict(torch.load(prev_residual_chk_pt))
        print(f"=======>> BB (residual) is loaded from {prev_residual_chk_pt}")
        prev_residual = copy.deepcopy(bb_model_top)
        # prev_residual.load_state_dict(torch.load(prev_residual_chk_pt)["state_dict"])
        residual.load_state_dict(torch.load(prev_residual_chk_pt))

    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        prev_residual_chk_pt = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb" "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual.load_state_dict(torch.load(prev_residual_chk_pt))
        print(f"=======>> BB (residual) is loaded from {prev_residual_chk_pt}")
        prev_residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(prev_residual_chk_pt))

    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        prev_residual_chk_pt = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual.load_state_dict(torch.load(prev_residual_chk_pt))
        print(f"=======>> BB (residual) is loaded from {prev_residual_chk_pt}")
        prev_residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(prev_residual_chk_pt))

    optimizer = torch.optim.Adam(residual.parameters(), lr=args.lr_residual)
    schedule = utils.get_scheduler(optimizer, args)
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    kd_Loss = KD_Residual_Loss(iteration, CE, KLDiv, T_KD=args.temperature_KD, alpha_KD=args.alpha_KD)

    best_auroc = 0
    n_class = 2
    start_epoch = 0
    # logger = Logger_MIMIC_CXR(
    #     iteration, best_auroc, start_epoch, residual_chk_pt_path, residual_tb_logs_path, residual_output_path,
    #     train_loader, val_loader, n_class, model_type="g", device=device
    # )

    logger = Logger_CUBS(
        iteration, residual_chk_pt_path, residual_tb_logs_path, residual_output_path, train_loader, val_loader,
        len(args.labels), device
    )
    if args.profile == "n":
        fit_residual(
            iteration,
            args.epochs_residual,
            concept_bank,
            bb_model,
            bb_model_bottom,
            bb_model_top,
            glt,
            glt_list,
            prev_residual,
            residual,
            optimizer,
            schedule,
            train_loader,
            val_loader,
            kd_Loss,
            logger,
            os.path.join(root, f"iter{iteration}", "bb"),
            args.selection_threshold,
            device
        )

    else:
        fit_residual_ham_profiler(
            iteration,
            args.epochs_residual,
            concept_bank,
            bb_model,
            bb_model_bottom,
            bb_model_top,
            glt,
            glt_list,
            prev_residual,
            residual,
            optimizer,
            schedule,
            train_loader,
            val_loader,
            kd_Loss,
            logger,
            os.path.join(root, f"iter{iteration}", "bb"),
            args.selection_threshold,
            device
        )
    print("done")


def fit_residual(
        iteration,
        epochs,
        concept_bank,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        glt,
        glt_list,
        prev_residual,
        residual,
        optimizer,
        schedule,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        run_id,
        selection_threshold,
        device
):
    logger.begin_run(run_id)
    for epoch in range(epochs):
        logger.begin_epoch()
        residual.train()

        with tqdm(total=len(train_loader)) as t:
            for batch in enumerate(train_loader):
                batch_idx, (images, target) = batch
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    train_bb_logits = None
                    if iteration == 1:
                        train_bb_logits = bb_logits
                    elif iteration > 1:
                        train_bb_logits, _ = prev_residual(phi)
                    train_concepts = compute_dist(concept_bank, phi)

                out_class, out_select, out_aux = glt(train_concepts)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                residual_student_logits, _ = residual(phi)
                residual_teacher_logits = train_bb_logits - (out_select * out_class)

                loss_dict = kd_Loss(
                    student_preds=residual_student_logits,
                    teacher_preds=residual_teacher_logits,
                    target=target,
                    selection_weights=out_select,
                    prev_selection_outs=pi_list
                )

                train_distillation_risk = loss_dict["distillation_risk"]
                train_CE_risk = loss_dict["CE_risk"]
                train_KD_risk = loss_dict["KD_risk"]

                total_train_loss = train_KD_risk
                optimizer.zero_grad()
                total_train_loss.backward()
                optimizer.step()

                # logger.track_train_loss(total_train_loss.item())
                # logger.track_total_train_correct_per_epoch(residual_student_logits, target)
                logger.track_train_loss(total_train_loss.item())
                logger.track_total_train_correct_per_epoch(residual_student_logits, target)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch in enumerate(val_loader):
                    batch_idx, (images, target) = batch
                    images = images.to(device)
                    target = target.to(device)
                    with torch.no_grad():
                        bb_logits = bb_model(images)
                        phi = bb_model_bottom(images)
                        val_bb_logits = None
                        if iteration == 1:
                            val_bb_logits = bb_logits
                        elif iteration > 1:
                            val_bb_logits, _ = prev_residual(phi)
                        val_concepts = compute_dist(concept_bank, phi)

                    out_class, out_select, out_aux = glt(val_concepts)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_concepts)

                    residual_student_logits, _ = residual(phi)
                    residual_teacher_logits = val_bb_logits - (out_select * out_class)

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=target,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    total_val_loss = loss_dict["KD_risk"]

                    # logger.track_val_loss(total_val_loss.item())
                    # logger.track_val_outputs(out_select, residual_student_logits, target, val_bb_logits)
                    # logger.track_total_val_correct_per_epoch(residual_student_logits, target)

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, residual_student_logits, target)
                    logger.track_total_val_correct_per_epoch(residual_student_logits, target)

                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # # evaluate residual for correctly selected samples (pi < 0.5)
        # # should be higher
        # logger.evaluate_g_correctly(selection_threshold, expert="residual")
        # #
        # # # evaluate residual for correctly rejected samples (pi >= 0.5)
        # # # should be lower
        # logger.evaluate_g_incorrectly(selection_threshold, expert="residual")
        # logger.evaluate_coverage_stats(selection_threshold, expert="residual")
        # logger.end_epoch(residual, optimizer, track_explainer_loss=False, save_model_wrt_g_performance=True)

        # evaluate residual for correctly selected samples (pi < 0.5)
        # should be higher
        logger.evaluate_g_correctly(selection_threshold, expert="residual")
        #
        # # evaluate residual for correctly rejected samples (pi >= 0.5)
        # # should be lower
        logger.evaluate_g_incorrectly(selection_threshold, expert="residual")
        logger.evaluate_coverage_stats(selection_threshold, expert="residual")
        logger.end_epoch(
            residual, track_explainer_loss=False, save_model_wrt_g_performance=True, model_type="residual"
        )

        # print(
        #     f"Epoch: [{epoch + 1}/{epochs}] || "
        #     f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
        #     f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
        #     f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
        #     f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
        #     f"Val_Auroc (Entire set): {round(logger.val_auroc, 4)} || "
        #     f"Val_residual_Accuracy (pi < 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
        #     f"Val_residual_Auroc (pi < 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
        #     f"Val_BB_Auroc (pi < 0.5): {round(logger.val_bb_auroc, 4)} || "
        #     f"Val_residual_Incorrect_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)}(%) || "
        #     f"Val_residual_Incorrect_Auroc (pi >= 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
        #     f"Val_BB_Incorrect_Auroc (pi >= 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
        #     f"Best_residual_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
        #     f"Best_Epoch: {logger.get_best_epoch_id()} || "
        #     f"n_selected: {logger.get_n_selected()} || "
        #     f"n_rejected: {logger.get_n_rejected()} || "
        #     f"coverage: {round(logger.get_coverage(), 4)} || "
        #     f"n_pos_g: {logger.get_n_pos_g()} || "
        #     f"n_pos_bb: {logger.get_n_pos_bb()}"
        # )

        print(
            f"Epoch: [{epoch + 1}/{epochs}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_residual_Accuracy: {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_residual_Incorrect_Accuracy: {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
            f"Best_residual_Val_Accuracy: {round(logger.get_final_best_G_val_accuracy(), 4)} (%)  || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)}"
        )
    logger.end_run()


def train_explainer(args, seed, cov, lr_explainer, root, iteration, concept_path, train_loader, val_loader):
    print(f"Training the explainer for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer")
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")


    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "explainer",
                                     "accuracy")
    else:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"cov_{cov}", f"iter{iteration}", "explainer", "accuracy")
        g_output_path = os.path.join(output_path_explainer, f"cov_{cov}", f"iter{iteration}", "explainer", "accuracy")
    g_tb_logs_path = os.path.join(log_path_explainer, root)

    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))

    device = utils.get_device()
    print(f"Device: {device}")

    bb_model, bb_model_bottom, bb_model_top = None, None, None

    if args.dataset == "HAM10k":
        bb_model, bb_model_bottom, bb_model_top = get_model(args.bb_dir, args.model_name)
    elif args.dataset == "SIIM-ISIC":
        bb_model, bb_model_bottom, bb_model_top = get_BB_model_isic(args.bb_dir, args.model_name, args.dataset)

    print("BB is loaded successfully")
    concepts_dict = pickle.load(
        open(os.path.join(concept_path, args.concept_file_name), "rb")
    )
    print(concepts_dict)
    concept_bank = ConceptBank(concepts_dict, device)

    lambda_lens = args.lambda_lens
    glt_list = []
    residual = None
    if iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1], "bb", "accuracy", args.checkpoint_residual[-1]
        )
        print(f"=======>> BB (residual) is loaded from {residual_chk_pt_path}")
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        residual = copy.deepcopy(bb_model_top)
        # residual.load_state_dict(torch.load(residual_chk_pt_path)["state_dict"])
        residual.load_state_dict(torch.load(residual_chk_pt_path))
    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(residual_chk_pt_path))
    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb", "accuracy", args.checkpoint_residual[-1]
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = copy.deepcopy(bb_model_top)
        residual.load_state_dict(torch.load(residual_chk_pt_path))

    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_explainer, momentum=0.9, weight_decay=5e-4)
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    selective_KD_loss = Selective_Distillation_Loss(
        iteration, CE, KLDiv, T_KD=args.temperature_KD, alpha_KD=args.alpha_KD,
        selection_threshold=args.selection_threshold, coverage=cov, lm=args.lm, dataset=args.dataset
    )

    best_auroc = 0
    n_class = 2
    start_epoch = 0
    logger = Logger_CUBS(
        iteration, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader, len(args.labels), device
    )

    # logger = Logger_MIMIC_CXR(
    #     iteration, best_auroc, start_epoch, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
    #     n_class, model_type="g", device=device
    # )

    if args.profile == "n":
        fit_g(
            iteration,
            concept_bank,
            args.epochs,
            args.alpha,
            args.temperature_KD,
            args.alpha_KD,
            bb_model, bb_model_bottom, bb_model_top,
            model,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            selective_KD_loss,
            logger,
            lambda_lens,
            os.path.join(root, f"iter{iteration}", "explainer"),
            args.selection_threshold,
            device
        )
    else:
        fit_g_ham_profile(
            iteration,
            concept_bank,
            args.epochs,
            args.alpha,
            args.temperature_KD,
            args.alpha_KD,
            bb_model, bb_model_bottom, bb_model_top,
            model,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            selective_KD_loss,
            logger,
            lambda_lens,
            os.path.join(root, f"iter{iteration}", "explainer"),
            args.selection_threshold,
            device
        )


def fit_g(
        iteration,
        concept_bank,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        model,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        logger,
        lambda_lens,
        run_id,
        selection_threshold,
        device
):
    logger.begin_run(run_id)

    for epoch in range(epochs):
        logger.begin_epoch()
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch in enumerate(train_loader):
                batch_idx, (images, target) = batch
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    train_bb_logits = None
                    if iteration == 1:
                        train_bb_logits = bb_logits
                    elif iteration > 1:
                        train_bb_logits, _ = residual(phi)
                    train_concepts = compute_dist(concept_bank, phi)

                out_class, out_select, out_aux = model(train_concepts)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                entropy_loss_elens = entropy_loss(model.explainer)
                loss_dict = selective_KD_loss(
                    out_class,
                    out_select,
                    target,
                    train_bb_logits,
                    entropy_loss_elens,
                    lambda_lens,
                    epoch,
                    device,
                    pi_list
                )
                train_selective_loss = loss_dict["selective_loss"]
                train_emp_coverage = loss_dict["emp_coverage"]
                train_distillation_risk = loss_dict["distillation_risk"]
                train_CE_risk = loss_dict["CE_risk"]
                train_KD_risk = loss_dict["KD_risk"]
                train_entropy_risk = loss_dict["entropy_risk"]
                train_emp_risk = loss_dict["emp_risk"]
                train_cov_penalty = loss_dict["cov_penalty"]

                train_selective_loss *= alpha
                aux_distillation_loss = torch.nn.KLDivLoss()(
                    F.log_softmax(out_aux / temperature_KD, dim=1),
                    F.softmax(train_bb_logits / temperature_KD, dim=1)
                )
                aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, target)
                aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                              (1. - alpha_KD) * aux_ce_loss

                aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                train_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                train_aux_loss *= (1.0 - alpha)

                total_train_loss = train_selective_loss + train_aux_loss
                optimizer.zero_grad()
                total_train_loss.backward()
                optimizer.step()

                logger.track_train_loss(total_train_loss.item())
                logger.track_train_losses_wrt_g(
                    train_emp_coverage.item(), train_distillation_risk.item(), train_CE_risk.item(),
                    train_KD_risk.item(), train_entropy_risk.item(), train_emp_risk.item(),
                    train_cov_penalty.item(), train_selective_loss.item(), train_aux_loss.item()
                )
                logger.track_total_train_correct_per_epoch(out_class, target)

                # logger.track_train_loss(total_train_loss.item())
                # logger.track_train_losses_wrt_g(
                #     train_emp_coverage.item(), train_distillation_risk.item(), train_CE_risk.item(),
                #     train_KD_risk.item(), train_entropy_risk.item(), train_emp_risk.item(),
                #     train_cov_penalty.item(), train_selective_loss.item(), train_aux_loss.item()
                # )
                # logger.track_total_train_correct_per_epoch(out_class, target)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch in enumerate(val_loader):
                    batch_idx, (images, target) = batch
                    images = images.to(device)
                    target = target.to(device)
                    with torch.no_grad():
                        bb_logits = bb_model(images)
                        phi = bb_model_bottom(images)
                        val_bb_logits = None
                        if iteration == 1:
                            val_bb_logits = bb_logits
                        elif iteration > 1:
                            val_bb_logits, _ = residual(phi)
                        val_concepts = compute_dist(concept_bank, phi)

                    out_class, out_select, out_aux = model(val_concepts)
                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_concepts)

                    entropy_loss_elens = entropy_loss(model.explainer)
                    loss_dict = selective_KD_loss(
                        out_class,
                        out_select,
                        target,
                        val_bb_logits,
                        entropy_loss_elens,
                        lambda_lens,
                        epoch,
                        device,
                        pi_list
                    )

                    val_selective_loss = loss_dict["selective_loss"]
                    val_emp_coverage = loss_dict["emp_coverage"]
                    val_distillation_risk = loss_dict["distillation_risk"]
                    val_CE_risk = loss_dict["CE_risk"]
                    val_KD_risk = loss_dict["KD_risk"]
                    val_entropy_risk = loss_dict["entropy_risk"]
                    val_emp_risk = loss_dict["emp_risk"]
                    val_cov_penalty = loss_dict["cov_penalty"]

                    val_selective_loss *= alpha

                    aux_distillation_loss = torch.nn.KLDivLoss()(
                        F.log_softmax(out_aux / temperature_KD, dim=1),
                        F.softmax(val_bb_logits / temperature_KD, dim=1)
                    )
                    aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, target)
                    aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                                  (1. - alpha_KD) * aux_ce_loss

                    aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                    val_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                    val_aux_loss *= (1.0 - alpha)

                    total_val_loss = val_selective_loss + val_aux_loss

                    # logger.track_val_loss(total_val_loss.item())
                    # logger.track_val_losses_wrt_g(
                    #     val_emp_coverage.item(), val_distillation_risk.item(), val_CE_risk.item(),
                    #     val_KD_risk.item(), val_entropy_risk.item(), val_emp_risk.item(),
                    #     val_cov_penalty.item(), val_selective_loss.item(), val_aux_loss.item()
                    # )
                    # logger.track_val_outputs(out_select, out_class, target, val_bb_logits)
                    # logger.track_total_val_correct_per_epoch(out_class, target)

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_losses_wrt_g(
                        val_emp_coverage.item(), val_distillation_risk.item(), val_CE_risk.item(),
                        val_KD_risk.item(), val_entropy_risk.item(), val_emp_risk.item(),
                        val_cov_penalty.item(), val_selective_loss.item(), val_aux_loss.item()
                    )
                    logger.track_val_outputs(out_select, out_class, target)
                    logger.track_val_bb_outputs(val_bb_logits)
                    logger.track_total_val_correct_per_epoch(out_class, target)

                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # # evaluate g for correctly selected samples (pi >= 0.5)
        # # should be higher
        # logger.evaluate_g_correctly(selection_threshold, expert="explainer")
        #
        # # evaluate g for correctly rejected samples (pi < 0.5)
        # # should be lower
        # logger.evaluate_g_incorrectly(selection_threshold, expert="explainer")
        # logger.evaluate_coverage_stats(selection_threshold)
        # # logger.end_epoch(model, track_explainer_loss=True, save_model_wrt_g_performance=True, model_type="g")
        # logger.end_epoch(model, optimizer, track_explainer_loss=True, save_model_wrt_g_performance=True)

        # evaluate g for correctly selected samples (pi >= 0.5)
        # should be higher
        logger.evaluate_g_correctly(selection_threshold, expert="explainer")
        logger.evaluate_g_correctly_auroc(selection_threshold, expert="explainer")

        # evaluate g for correctly rejected samples (pi < 0.5)
        # should be lower
        logger.evaluate_g_incorrectly(selection_threshold, expert="explainer")
        logger.evaluate_coverage_stats(selection_threshold)
        logger.end_epoch(model, track_explainer_loss=True, save_model_wrt_g_performance=True, model_type="g")
        # print(
        #     f"Epoch: [{epoch + 1}/{epochs}] || "
        #     f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
        #     f"Train_KD_loss: {round(logger.get_final_train_KD_loss(), 4)} || "
        #     f"Train_entropy_loss: {round(logger.get_final_train_entropy_loss(), 4)} || "
        #     f"Train_aux_loss: {round(logger.get_final_train_aux_loss(), 4)} || "
        #     f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
        #     f"Val_KD_loss: {round(logger.get_final_val_KD_loss(), 4)} || "
        #     f"Val_entropy_loss: {round(logger.get_final_val_entropy_loss(), 4)} || "
        #     f"Val_aux_loss: {round(logger.get_final_val_aux_loss(), 4)} || "
        #     f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
        #     f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
        #     # f"Val_Auroc (Entire set by G): {round(logger.val_auroc, 4)} || "
        #     f"Val_G_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
        #     # f"Val_G_Auroc (pi >= 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
        #     # f"Val_BB_Auroc (pi >= 0.5): {round(logger.val_bb_auroc, 4)} || "
        #     f"Val_G_Incorrect_Accuracy (pi < 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
        #     # f"Val_G_Incorrect_Auroc (pi < 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
        #     # f"Val_BB_Incorrect_Auroc (pi < 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
        #     # f"Best_G_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
        #     f"Best_Epoch: {logger.get_best_epoch_id()} || "
        #     f"n_selected: {logger.get_n_selected()} || "
        #     f"n_rejected: {logger.get_n_rejected()} || "
        #     f"coverage: {round(logger.get_coverage(), 4)} || "
        #     # f"n_pos_g: {logger.get_n_pos_g()} || "
        #     # f"n_pos_bb: {logger.get_n_pos_bb()}"
        # )
        print(
            f"Epoch: [{epoch + 1}/{epochs}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"Train_KD_loss: {round(logger.get_final_train_KD_loss(), 4)} || "
            f"Train_entropy_loss: {round(logger.get_final_train_entropy_loss(), 4)} || "
            f"Train_aux_loss: {round(logger.get_final_train_aux_loss(), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"Val_KD_loss: {round(logger.get_final_val_KD_loss(), 4)} || "
            f"Val_entropy_loss: {round(logger.get_final_val_entropy_loss(), 4)} || "
            f"Val_aux_loss: {round(logger.get_final_val_aux_loss(), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_G_Accuracy: {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_G_Auroc (pi >= 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
            f"Val_BB_Auroc (pi >= 0.5): {round(logger.get_final_BB_val_auroc(), 4)} || "
            f"Val_G_Incorrect_Accuracy: {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
            f"Best_G_Val_Accuracy: {round(logger.get_final_best_G_val_accuracy(), 4)} (%)  || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)}"
        )

        # print(f"Epoch: [{epoch + 1}/{epochs}] || "
        #       f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
        #       f"Train_KD_loss: {round(logger.get_final_train_KD_loss(), 4)} || "
        #       f"Train_entropy_loss: {round(logger.get_final_train_entropy_loss(), 4)} || "
        #       f"Train_aux_loss: {round(logger.get_final_train_aux_loss(), 4)} || "
        #       f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
        #       f"Val_KD_loss: {round(logger.get_final_val_KD_loss(), 4)} || "
        #       f"Val_entropy_loss: {round(logger.get_final_val_entropy_loss(), 4)} || "
        #       f"Val_aux_loss: {round(logger.get_final_val_aux_loss(), 4)} || "
        #       f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
        #       f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
        #       f"Val_G_Accuracy: {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
        #       f"Val_G_Incorrect_Accuracy: {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
        #       f"Best_G_Val_Accuracy: {round(logger.get_final_best_G_val_accuracy(), 4)} (%)  || "
        #       f"Best_Epoch: {logger.get_best_epoch_id()} || "
        #       f"n_selected: {logger.get_n_selected()} || "
        #       f"n_rejected: {logger.get_n_rejected()} || "
        #       f"coverage: {round(logger.get_coverage(), 4)}")

    logger.end_run()


def compute_dist(concept_bank, phi):
    margins = (torch.matmul(concept_bank.vectors, phi.T) + concept_bank.intercepts) / concept_bank.norms
    return margins.T
