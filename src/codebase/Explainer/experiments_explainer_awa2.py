import copy
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from Explainer.loss_F import entropy_loss, Selective_Distillation_Loss, KD_Residual_Loss
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.explainer import Explainer
from Explainer.models.residual import Residual
from Explainer.profiler import fit_g_awa2_profiler, fit_awa2_residual
from Explainer.utils_explainer import get_residual, get_previous_pi_vals, get_glts_for_HAM10k, \
    get_glts_for_HAM10k_soft_seed
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_awa2 import Dataset_awa2_for_explainer
from dataset.dataset_mnist import Dataset_mnist_for_explainer
from dataset.utils_dataset import get_dataset_with_image_and_attributes

warnings.filterwarnings("ignore")

def test_glt(args):
    explainer_init = "none"
    use_concepts_as_pi_input = True if args.use_concepts_as_pi_input == "y" else False
    # float

    root = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    print(root)
    dataset_path = os.path.join(
        args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g"
    )

    start = time.time()

    if args.dataset == "awa2":
        transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
        train_transform = transforms["train_transform"]
        val_transform = transforms["val_transform"]
        print(transforms)
        train_dataset = Dataset_awa2_for_explainer(
            dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt",
            "train_image_names.pkl", train_transform
        )
        val_dataset = Dataset_awa2_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt",
            "test_image_names.pkl",
            val_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(args)

    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")

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
            print("Testing explainer")
            test_explainer(
                args, seed, cov, lr_explainer, root, iteration, use_concepts_as_pi_input, train_loader, test_loader
            )
        elif args.expert_to_train == "residual":
            # needs rewrite to incorporate Hard/Soft concepts and seeds like others
            test_residual(args, seed, root, iteration, use_concepts_as_pi_input, train_loader, test_loader)


def test_residual(args, seed, root, iteration, use_concepts_as_pi_input, train_loader, test_loader):
    print(f"Testing the residual for iteration: {iteration}")
    chk_pt_explainer = None
    chk_pt_residual = None
    output_path_explainer = None
    log_path_explainer = None

    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer",
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        chk_pt_residual = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", args.arch, root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer")
        g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "explainer")
        residual_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "bb")
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"iter{iteration}", "bb")
    else:
        cov = args.cov[iteration - 1]
        lr_explainer = args.lr[iteration - 1]
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer")
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}",
            "bb"
        )
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "bb")

    output_path_model_outputs = os.path.join(residual_output_path, "model_outputs")
    output_path_residual_outputs = os.path.join(residual_output_path, "residual_outputs")

    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_residual_outputs, exist_ok=True)

    pickle.dump(args, open(os.path.join(residual_output_path, "test_configs.pkl"), "wb"))

    device = utils.get_device()
    print(f"Device: {device}")

    bb = utils.get_model_explainer(args, device)
    bb.eval()

    glt_list = []
    if iteration > 1:
        # residual_chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[-1], "bb")
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        # residual = get_residual(iteration, args, residual_chk_pt_path, device)

    cur_glt_chkpt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G checkpoint is loaded for iteration {iteration}: {cur_glt_chkpt}")
    glt = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input
    ).to(device)
    glt.load_state_dict(torch.load(cur_glt_chkpt))
    glt.eval()

    cur_residual_chkpt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
    print(f"===> Latest residual checkpoint is loaded for iteration {iteration}: {cur_residual_chkpt}")
    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    residual.load_state_dict(torch.load(cur_residual_chkpt))
    residual.eval()

    print("!! Saving test loader for residual expert !!")
    save_results_selected_residual_by_pi(
        args.arch,
        iteration,
        bb,
        glt,
        glt_list,
        residual,
        test_loader,
        args.layer,
        use_concepts_as_pi_input,
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
        bb,
        glt,
        glt_list,
        residual,
        train_loader,
        args.layer,
        use_concepts_as_pi_input,
        output_path_residual_outputs,
        args.selection_threshold,
        device,
        mode="train"
    )

    print("Saving the results for the overall dataset")
    predict_residual(
        args.arch,
        bb,
        residual,
        test_loader,
        args.layer,
        output_path_model_outputs,
        device
    )


def save_results_selected_residual_by_pi(
        arch,
        iteration,
        bb,
        glt,
        glt_list,
        residual,
        loader,
        layer,
        use_concepts_as_pi_input,
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
            for batch_id, (
                    images, concepts, attrs, image_names, y, y_one_hot
            ) in enumerate(loader):
                images, concepts, y, y_one_hot = images.to(device), \
                                                 concepts.to(device), \
                                                 y.to(torch.long).to(device), \
                                                 y_one_hot.to(device)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(images, bb, arch, layer)

                if use_concepts_as_pi_input:
                    _, out_select, _ = glt(concepts)
                else:
                    _, out_select, _ = glt(concepts, feature_x.to(device))

                residual_student_logits = residual(feature_x)

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
                    residual_y = y[arr_sel_indices]

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

    acc_residual = torch.sum(tensor_preds_residual.argmax(dim=1) == tensor_y) / tensor_y.size(0)
    acc_bb = torch.sum(tensor_preds_bb.argmax(dim=1) == tensor_y) / tensor_y.size(0)
    print(f"Accuracy of Residual: {acc_residual * 100} || Accuracy of BB: {acc_bb * 100}")

    cov = tensor_y.size(0) / 7465
    print(f"Scaled Accuracy of Residual (cov): {acc_residual * cov * 100} ({cov})")

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
        arch,
        bb,
        residual,
        test_loader,
        layer,
        output_path_model_outputs,
        device
):
    out_put_preds_residual = torch.FloatTensor().cuda()
    out_put_preds_bb = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (
                    test_images, test_concepts, test_attrs, test_image_names, test_y, test_y_one_hot
            ) in enumerate(test_loader):
                test_images, test_concepts, test_y, test_y_one_hot = test_images.to(device), \
                                                                     test_concepts.to(device), \
                                                                     test_y.to(torch.long).to(device), \
                                                                     test_y_one_hot.to(device)
                with torch.no_grad():
                    bb_logits, test_feature_x = get_phi_x(test_images, bb, arch, layer)

                residual_student_logits = residual(test_feature_x)

                out_put_preds_bb = torch.cat((out_put_preds_bb, bb_logits), dim=0)
                out_put_preds_residual = torch.cat((out_put_preds_residual, residual_student_logits), dim=0)
                out_put_target = torch.cat((out_put_target, test_y), dim=0)

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
        f"BB Accuracy: {out_put_preds_bb.argmax(dim=1).eq(out_put_target).sum() / out_put_preds_bb.size(0)}"
    )
    print(
        f"Residual Accuracy: {out_put_preds_residual.argmax(dim=1).eq(out_put_target).sum() / out_put_preds_bb.size(0)}"
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


def test_explainer(
        args, seed, cov, lr_explainer, root, iteration, use_concepts_as_pi_input, train_loader, test_loader
):
    print(f"Testing the explainer for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer",
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", args.arch, root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer")
        g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "explainer")
    else:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer")
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer"
        )

    output_path_model_outputs = os.path.join(g_output_path, "model_outputs")
    output_path_g_outputs = os.path.join(g_output_path, "g_outputs")

    os.makedirs(g_output_path, exist_ok=True)
    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_g_outputs, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))

    device = utils.get_device()
    print(f"Device: {device}")
    print("############# Paths ############# ")
    print(g_chk_pt_path)
    print(g_output_path)
    print("############# Paths ############# ")
    args.projected = "n"
    bb = utils.get_model_explainer(args, device).to(device)
    bb.eval()

    glt_list = []
    residual = None
    if iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        residual_chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[-1], "bb")
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)
    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb")
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)
    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb"
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)

    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi, args.concept_names, args.labels, args.hidden_nodes, args.conceptizator,
        args.temperature_lens, use_concepts_as_pi_input,
    ).to(device)

    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    print("!! Saving test loader only selected by g!!")
    save_results_selected_by_pi(
        args.dataset,
        iteration,
        bb,
        model,
        args.arch,
        args.layer,
        test_loader,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="test",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )

    print("!! Saving train loader only selected by g!!")

    save_results_selected_by_pi(
        args.dataset,
        iteration,
        bb,
        model,
        args.arch,
        args.layer,
        train_loader,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="train",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        },
        save_image=False
    )

    print("Save overall whole model outputs")
    predict(
        args.dataset,
        bb,
        model,
        test_loader,
        args.arch,
        args.layer,
        use_concepts_as_pi_input,
        output_path_model_outputs,
        device,
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
        dataset,
        iteration,
        bb,
        model,
        arch,
        layer,
        data_loader,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params,
        save_image=False
):
    glt_list = None
    residual = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]
        residual = higher_iter_params["residual"]

    tensor_images = torch.FloatTensor()
    tensor_features = torch.FloatTensor()
    tensor_concepts = torch.FloatTensor().cuda()
    tensor_attributes = torch.FloatTensor().cuda()
    tensor_preds = torch.FloatTensor().cuda()
    tensor_preds_bb = torch.FloatTensor().cuda()
    tensor_y = torch.FloatTensor().cuda()
    tensor_conceptizator_concepts = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()
    image_name_tensor = []

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            image_names = None
            for batch_id, data_tuple in enumerate(data_loader):
                if dataset == "awa2":
                    images, concepts, attributes, image_names, y, y_one_hot = data_tuple
                elif dataset == "mnist":
                    images, concepts, attributes, y, y_one_hot = data_tuple

                images, concepts, attributes, y, y_one_hot = (
                    images.to(device),
                    concepts.to(device),
                    attributes.to(device),
                    y.to(torch.long).to(device),
                    y_one_hot.to(device),
                )
                bb_logits, feature_x = get_phi_x(images, bb, arch, layer)
                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, concepts)

                prediction_out, selection_out, auxiliary_out, concept_mask, \
                alpha, alpha_norm, conceptizator = model(concepts.to(device), test=True)

                arr_sel_indices = get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, pi_list)
                arr_sel_indices_list = arr_sel_indices.tolist()
                if arr_sel_indices.size(0) > 0:
                    if image_names is not None:
                        for indx in arr_sel_indices_list:
                            image_name_tensor.append(image_names[indx])

                    if arch == "ResNet101" or arch == "ResNet50":
                        g_features = feature_x[arr_sel_indices, :, :, :]
                    else:
                        g_features = feature_x[arr_sel_indices, :]

                    # g_images = images[arr_sel_indices, :, :, :]
                    g_concepts = concepts[arr_sel_indices, :]
                    g_attributes = attributes[arr_sel_indices, :]
                    g_preds = prediction_out[arr_sel_indices, :]
                    g_preds_bb = bb_logits[arr_sel_indices, :]
                    g_y = y[arr_sel_indices]
                    g_conceptizator_concepts = conceptizator.concepts[:, arr_sel_indices, :]
                    if save_image:
                        tensor_images = torch.cat((tensor_images, g_images.cpu()), dim=0)
                    tensor_concepts = torch.cat((tensor_concepts, g_concepts), dim=0)
                    tensor_features = torch.cat((tensor_features, g_features.cpu()), dim=0)
                    tensor_preds = torch.cat((tensor_preds, g_preds), dim=0)
                    tensor_preds_bb = torch.cat((tensor_preds_bb, g_preds_bb), dim=0)
                    tensor_y = torch.cat((tensor_y, g_y), dim=0)
                    tensor_attributes = torch.cat((tensor_attributes, g_attributes), dim=0)
                    tensor_conceptizator_concepts = torch.cat(
                        (tensor_conceptizator_concepts, g_conceptizator_concepts), dim=1
                    )

                # tensor_conceptizator_threshold = conceptizator.threshold
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm
                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    # tensor_images = tensor_images.cpu()
    tensor_concepts = tensor_concepts.cpu()
    tensor_preds = tensor_preds.cpu()
    tensor_preds_bb = tensor_preds_bb.cpu()
    tensor_y = tensor_y.cpu()
    tensor_attributes = tensor_attributes.cpu()
    tensor_conceptizator_concepts = tensor_conceptizator_concepts.cpu()
    # tensor_conceptizator_threshold = tensor_conceptizator_threshold.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    acc_g = torch.sum(tensor_preds.argmax(dim=1) == tensor_y) / tensor_y.size(0)
    print(f"Accuracy: {acc_g}")
    print("Output sizes: ")
    print(f"tensor_images size: {tensor_images.size()}")
    print(f"tensor_features size: {tensor_features.size()}")
    print(f"tensor_concepts size: {tensor_concepts.size()}")
    print(f"tensor_attributes size {tensor_attributes.size()}")
    print(f"tensor_preds size: {tensor_preds.size()}")
    print(f"tensor_preds bb size {tensor_preds_bb.size()}")
    print(f"tensor_y size: {tensor_y.size()}")
    print(f"tensor_conceptizator_concepts size: {tensor_conceptizator_concepts.size()}")

    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")
    print(f"Saved image_name_tensor size: {len(image_name_tensor)}")

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
        path=os.path.join(output_path, f"{mode}_tensor_attributes.pt"), tensor_to_save=tensor_attributes
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

    utils.dump_in_pickle(output_path=output_path, file_name=f"{mode}_image_names.pkl", stats_to_dump=image_name_tensor)

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


def predict(
        dataset,
        bb,
        model,
        test_loader,
        arch,
        layer,
        use_concepts_as_pi_input,
        output_path,
        device
):
    out_put_sel_proba = torch.FloatTensor().cuda()
    out_put_class = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, data_tuple in enumerate(test_loader):
                if dataset == "awa2":
                    test_images, test_concepts, test_attrs, test_image_names, test_y, test_y_one_hot = data_tuple
                elif dataset == "mnist":
                    test_images, test_concepts, test_attrs, test_y, test_y_one_hot = data_tuple

                test_images, test_concepts, test_y, test_y_one_hot = (
                    test_images.to(device),
                    test_concepts.to(device),
                    test_y.to(torch.long).to(device),
                    test_y_one_hot.to(device),
                )

                bb_logits, feature_x = get_phi_x(test_images, bb, arch, layer)

                if use_concepts_as_pi_input:
                    out_class, out_select, out_aux = model(test_concepts)
                else:
                    out_class, out_select, out_aux = model(
                        test_concepts, feature_x.to(device)
                    )
                out_put_sel_proba = torch.cat((out_put_sel_proba, out_select), dim=0)
                out_put_class = torch.cat((out_put_class, out_class), dim=0)
                out_put_target = torch.cat((out_put_target, test_y), dim=0)

                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    out_put_sel_proba = out_put_sel_proba.cpu()
    out_put_class_pred = out_put_class.cpu()
    out_put_target = out_put_target.cpu()

    print(f"out_put_sel_proba size: {out_put_sel_proba.size()}")
    print(f"out_put_class_pred size: {out_put_class_pred.size()}")
    print(f"out_put_target size: {out_put_target.size()}")

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


def get_mnist_loaders(args):
    train_set, train_attributes = get_dataset_with_image_and_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="mnist",
        mode="train",
        attribute_file="attributes.npy"
    )

    test_set, test_attributes = get_dataset_with_image_and_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="mnist",
        mode="test",
        attribute_file="attributes.npy"
    )

    # train_transform = get_transforms(size=224)
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
    train = Dataset_mnist_for_explainer(
        train_set, dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt", transform
    )

    test = Dataset_mnist_for_explainer(
        test_set, dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt", transform
    )

    train_loader = DataLoader(train, batch_size=10, shuffle=True)
    test_loader = DataLoader(test, batch_size=10, shuffle=True)
    return train_loader, test_loader


def train_glt(args):
    explainer_init = "none"
    use_concepts_as_pi_input = True if args.use_concepts_as_pi_input == "y" else False

    root = f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}" \
           f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}" \
           f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}" \
           f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}" \
           f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}" \
           f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
    print(root)

    start = time.time()
    if args.dataset == "awa2":
        transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
        train_transform = transforms["train_transform"]
        val_transform = transforms["val_transform"]
        print(transforms)
        train_dataset = Dataset_awa2_for_explainer(
            dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt",
            "train_image_names.pkl", train_transform
        )
        val_dataset = Dataset_awa2_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt",
            "test_image_names.pkl",
            val_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == "mnist":
        train_loader, val_loader = get_mnist_loaders(args)

    done = time.time()
    elapsed = done - start
    print("Time to the full datasets: " + str(elapsed) + " secs")

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
                args, seed, cov, lr_explainer, root, iteration, use_concepts_as_pi_input, train_loader, val_loader
            )
        elif args.expert_to_train == "residual":
            train_residual(args, seed, root, iteration, use_concepts_as_pi_input, train_loader, val_loader)
    # iter 1
    # print("Training G")


def train_residual(args, seed, root, iteration, use_concepts_as_pi_input, train_loader, val_loader):
    chk_pt_explainer = None
    chk_pt_residual = None
    output_path_explainer = None
    log_path_explainer = None

    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer",
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        chk_pt_residual = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        chk_pt_residual = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", args.arch, root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer")
        g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "explainer")
        residual_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "bb")
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"iter{iteration}", "bb")
    else:
        cov = args.cov[iteration - 1]
        lr_explainer = args.lr[iteration - 1]
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer")
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer"
        )
        residual_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}",
            "bb"
        )
        residual_chk_pt_path = os.path.join(chk_pt_residual, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "bb")

    residual_tb_logs_path = log_path_explainer

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
    args.projected = "n"
    bb = utils.get_model_explainer(args, device)
    bb.eval()

    glt_list = []
    prev_residual = None
    glt = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input
    ).to(device)
    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    glt.load_state_dict(torch.load(glt_chk_pt))
    glt.eval()

    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if iteration == 1:
        if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
            residual.fc.weight = copy.deepcopy(bb.base_model.fc.weight)
            residual.fc.bias = copy.deepcopy(bb.base_model.fc.bias)
        elif args.arch == "ViT-B_16":
            residual.fc.weight = copy.deepcopy(bb.part_head.weight)
            residual.fc.bias = copy.deepcopy(bb.part_head.bias)
    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        prev_residual_chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[-1], "bb")
        prev_residual = get_residual(iteration, args, prev_residual_chk_pt_path, device)
        residual.fc.weight = copy.deepcopy(prev_residual.fc.weight)
        residual.fc.bias = copy.deepcopy(prev_residual.fc.bias)

    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        prev_residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb")
        prev_residual = get_residual(iteration, args, prev_residual_chk_pt_path, device)
        residual.fc.weight = copy.deepcopy(prev_residual.fc.weight)
        residual.fc.bias = copy.deepcopy(prev_residual.fc.bias)
    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        prev_residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb"
        )
        prev_residual = get_residual(iteration, args, prev_residual_chk_pt_path, device)
        residual.fc.weight = copy.deepcopy(prev_residual.fc.weight)
        residual.fc.bias = copy.deepcopy(prev_residual.fc.bias)

    print(prev_residual)
    optimizer = torch.optim.SGD(
        residual.parameters(), lr=args.lr_residual, momentum=args.momentum_residual,
        weight_decay=args.weight_decay_residual
    )
    schedule = utils.get_scheduler(optimizer, args)
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    kd_Loss = KD_Residual_Loss(iteration, CE, KLDiv, T_KD=args.temperature_KD,
                               alpha_KD=args.alpha_KD if args.dataset == "awa2" else 0.7)
    logger = Logger_CUBS(
        iteration, residual_chk_pt_path, residual_tb_logs_path, residual_output_path, train_loader, val_loader,
        len(args.labels), device
    )

    if args.profile == "n":
        fit_residual(
            args.dataset,
            iteration,
            args.epochs_residual,
            bb,
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
            args.layer,
            args.arch,
            os.path.join(root, f"iter{iteration}", "bb"),
            args.selection_threshold,
            use_concepts_as_pi_input,
            device
        )
    else:
        fit_awa2_residual(
            args.dataset,
            iteration,
            args.epochs_residual,
            bb,
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
            args.layer,
            args.arch,
            os.path.join(root, f"iter{iteration}", "bb"),
            args.selection_threshold,
            use_concepts_as_pi_input,
            device
        )
    print("done")


def fit_residual(
        dataset,
        iteration,
        epochs,
        bb,
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
        layer,
        arch,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    logger.begin_run(run_id)

    for epoch in range(epochs):
        logger.begin_epoch()
        residual.train()

        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if dataset == "awa2":
                    train_images, train_concepts, train_attrs, train_image_names, train_y, train_y_one_hot = data_tuple
                elif dataset == "mnist":
                    train_images, train_concepts, train_attrs, train_y, train_y_one_hot = data_tuple

                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)

                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)

                    print(torch.sum(bb_logits.argmax(dim=1) == train_y) / train_y.size(0))
                    train_bb_logits = bb_logits if iteration == 1 else prev_residual(feature_x)

                if use_concepts_as_pi_input:
                    out_class, out_select, out_aux = glt(train_concepts)
                else:
                    out_class, out_select, out_aux = glt(train_concepts, feature_x.to(device))

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                residual_student_logits = residual(feature_x)
                residual_teacher_logits = train_bb_logits - out_class

                loss_dict = kd_Loss(
                    student_preds=residual_student_logits,
                    teacher_preds=residual_teacher_logits,
                    target=train_y,
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

                logger.track_train_loss(total_train_loss.item())
                logger.track_total_train_correct_per_epoch(residual_student_logits, train_y)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if dataset == "awa2":
                        val_images, val_concepts, val_attrs, val_image_names, val_y, val_y_one_hot = data_tuple
                    elif dataset == "mnist":
                        val_images, val_concepts, val_attrs, val_y, val_y_one_hot = data_tuple

                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)
                    with torch.no_grad():
                        bb_logits, val_feature_x = get_phi_x(val_images, bb, arch, layer)
                        val_bb_logits = bb_logits if iteration == 1 else prev_residual(val_feature_x)

                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = glt(val_concepts)
                    else:
                        out_class, out_select, out_aux = glt(val_concepts, val_feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_concepts)

                    residual_student_logits = residual(val_feature_x)
                    residual_teacher_logits = val_bb_logits - out_class

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=val_y,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    total_val_loss = loss_dict["KD_risk"]

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, residual_student_logits, val_y)
                    logger.track_total_val_correct_per_epoch(residual_student_logits, val_y)

                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

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

        print(f"Epoch: [{epoch + 1}/{epochs}] || "
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
              f"coverage: {round(logger.get_coverage(), 4)}")
    logger.end_run()


def train_explainer(args, seed, cov, lr_explainer, root, iteration, use_concepts_as_pi_input, train_loader, val_loader):
    print(f"Training the explainer for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    if args.with_seed.lower() == 'y' and args.soft == 'y':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "soft_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "soft_concepts", f"seed_{seed}", "explainer",
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", "explainer",
        )
    elif args.with_seed.lower() == 'y' and args.soft == 'n':
        chk_pt_explainer = os.path.join(
            args.checkpoints, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        output_path_explainer = os.path.join(
            args.output, args.dataset, "hard_concepts", f"seed_{seed}", "explainer", args.arch, root
        )
        log_path_explainer = os.path.join(
            args.logs, args.dataset, "hard_concepts", f"seed_{seed}", "explainer"
        )
    elif args.with_seed.lower() == 'n' and args.soft == 'y':
        chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", args.arch, root)
        output_path_explainer = os.path.join(args.output, args.dataset, "explainer", args.arch, root)
        log_path_explainer = os.path.join(args.logs, args.dataset, "explainer")

    if iteration == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "explainer")
        g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "explainer")
        g_tb_logs_path = os.path.join(log_path_explainer, f"iter{iteration}", args.arch, root)
    else:
        g_chk_pt_path = os.path.join(
            chk_pt_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer"
        )
        g_output_path = os.path.join(
            output_path_explainer, f"cov_{cov}_lr_{lr_explainer}", f"iter{iteration}", "explainer"
        )
        g_tb_logs_path = os.path.join(
            output_path_explainer, f"iter{iteration}", args.arch, f"cov_{cov}_lr_{lr_explainer}-explainer"
        )

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
    args.projected = "n"
    bb = utils.get_model_explainer(args, device)
    bb.eval()
    print(" ################ BB loaded ################")

    lambda_lens = args.lambda_lens
    glt_list = []
    residual = None
    if iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'y'):
        residual_chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[-1], "bb")
        glt_list = get_glts_for_HAM10k(iteration, args, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)
    elif iteration > 1 and (args.with_seed.lower() == 'n' and args.soft == 'n'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter),
            "bb")
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)
    elif iteration > 1 and (args.with_seed.lower() == 'y' and args.soft == 'y'):
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        residual_chk_pt_path = os.path.join(
            args.prev_explainer_chk_pt_folder[-1].format(soft_hard_filter=soft_hard_filter, seed=seed),
            "bb"
        )
        glt_list = get_glts_for_HAM10k_soft_seed(iteration, args, seed, device)
        residual = get_residual(iteration, args, residual_chk_pt_path, device)

    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_explainer, momentum=0.9, weight_decay=5e-4)
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    selective_KD_loss = Selective_Distillation_Loss(
        iteration, CE, KLDiv, T_KD=args.temperature_KD, alpha_KD=args.alpha_KD,
        selection_threshold=args.selection_threshold, coverage=cov, arch=args.arch
    )

    logger = Logger_CUBS(
        iteration, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader, len(args.labels), device
    )
    if args.profile == "n":
        fit_g(
            iteration,
            args.dataset,
            args.arch,
            args.epochs,
            args.alpha,
            args.temperature_KD,
            args.alpha_KD,
            bb,
            model,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            selective_KD_loss,
            logger,
            lambda_lens,
            args.layer,
            os.path.join(root, f"iter{iteration}", "explainer"),
            args.selection_threshold,
            use_concepts_as_pi_input,
            device
        )
    else:
        fit_g_awa2_profiler(
            iteration,
            args.dataset,
            args.arch,
            args.epochs,
            args.alpha,
            args.temperature_KD,
            args.alpha_KD,
            bb,
            model,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            selective_KD_loss,
            logger,
            lambda_lens,
            args.layer,
            os.path.join(root, f"iter{iteration}", "explainer"),
            args.selection_threshold,
            use_concepts_as_pi_input,
            device
        )


def fit_g(
        iteration,
        dataset,
        arch,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        bb,
        model,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        logger,
        lambda_lens,
        layer,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    logger.begin_run(run_id)

    for epoch in range(epochs):
        logger.begin_epoch()
        model.train()
        out_put_class_bb = torch.FloatTensor().cuda()
        out_put_target_bb = torch.FloatTensor().cuda()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if dataset == "awa2":
                    train_images, train_concepts, train_attrs, train_image_names, train_y, train_y_one_hot = data_tuple
                elif dataset == "mnist":
                    train_images, train_concepts, train_attrs, train_y, train_y_one_hot = data_tuple

                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)
                    train_bb_logits = bb_logits if iteration == 1 else residual(feature_x)

                if use_concepts_as_pi_input:
                    out_class, out_select, out_aux = model(train_concepts)
                else:
                    out_class, out_select, out_aux = model(train_concepts, feature_x.to(device))

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                entropy_loss_elens = entropy_loss(model.explainer)
                loss_dict = selective_KD_loss(
                    out_class,
                    out_select,
                    train_y,
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
                aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, train_y)
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
                logger.track_total_train_correct_per_epoch(out_class, train_y)
                out_put_class_bb = torch.cat((out_put_class_bb, train_bb_logits), dim=0)
                out_put_target_bb = torch.cat((out_put_target_bb, train_y), dim=0)
                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        print(
            f"Train Acc_bb: {torch.sum(out_put_class_bb.argmax(dim=1) == out_put_target_bb) / out_put_class_bb.size(0) * 100}"
        )
        out_put_class_bb = torch.FloatTensor().cuda()
        out_put_target_bb = torch.FloatTensor().cuda()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if dataset == "awa2":
                        val_images, val_concepts, val_attrs, val_image_names, val_y, val_y_one_hot = data_tuple
                    elif dataset == "mnist":
                        val_images, val_concepts, val_attrs, val_y, val_y_one_hot = data_tuple

                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)
                    with torch.no_grad():
                        bb_logits, feature_x = get_phi_x(val_images, bb, arch, layer)
                        val_bb_logits = bb_logits if iteration == 1 else residual(feature_x)

                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = model(val_concepts)
                    else:
                        out_class, out_select, out_aux = model(val_concepts, feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_concepts)

                    entropy_loss_elens = entropy_loss(model.explainer)
                    loss_dict = selective_KD_loss(
                        out_class,
                        out_select,
                        val_y,
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
                    aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, val_y)
                    aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                                  (1. - alpha_KD) * aux_ce_loss

                    aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                    val_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                    val_aux_loss *= (1.0 - alpha)

                    total_val_loss = val_selective_loss + val_aux_loss

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_losses_wrt_g(
                        val_emp_coverage.item(), val_distillation_risk.item(), val_CE_risk.item(),
                        val_KD_risk.item(), val_entropy_risk.item(), val_emp_risk.item(),
                        val_cov_penalty.item(), val_selective_loss.item(), val_aux_loss.item()
                    )
                    logger.track_val_outputs(out_select, out_class, val_y)
                    logger.track_total_val_correct_per_epoch(out_class, val_y)

                    out_put_class_bb = torch.cat((out_put_class_bb, val_bb_logits), dim=0)
                    out_put_target_bb = torch.cat((out_put_target_bb, val_y), dim=0)

                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # evaluate g for correctly selected samples (pi >= 0.5)
        # should be higher
        logger.evaluate_g_correctly(selection_threshold, expert="explainer")

        # evaluate g for correctly rejected samples (pi < 0.5)
        # should be lower
        logger.evaluate_g_incorrectly(selection_threshold, expert="explainer")
        logger.evaluate_coverage_stats(selection_threshold)
        logger.end_epoch(model, track_explainer_loss=True, save_model_wrt_g_performance=True, model_type="g")
        print(out_put_class_bb.size())
        print(out_put_target_bb.size())
        print(
            f"Val Acc_bb: {torch.sum(out_put_class_bb.argmax(dim=1) == out_put_target_bb) / out_put_class_bb.size(0) * 100}"
        )
        print(f"Epoch: [{epoch + 1}/{epochs}] || "
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
              f"Val_G_Incorrect_Accuracy: {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
              f"Best_G_Val_Accuracy: {round(logger.get_final_best_G_val_accuracy(), 4)} (%)  || "
              f"Best_Epoch: {logger.get_best_epoch_id()} || "
              f"n_selected: {logger.get_n_selected()} || "
              f"n_rejected: {logger.get_n_rejected()} || "
              f"coverage: {round(logger.get_coverage(), 4)}")

    logger.end_run()


def get_phi_x(image, bb, arch, layer):
    if arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152":
        bb_logits = bb(image)
        # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
        feature_x = bb.feature_store[layer]
        return bb_logits, feature_x
    elif arch == "ViT-B_16":
        logits, tokens = bb(image)
        return logits, tokens[:, 0]


def get_fc_weight(arch, bb):
    if arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152":
        return bb.base_model.fc.weight
    elif arch == "ViT-B_16":
        return bb.part_head.weight


def get_fc_bias(arch, bb):
    if arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152":
        return bb.base_model.fc.bias
    elif arch == "ViT-B_16":
        return bb.part_head.bias
