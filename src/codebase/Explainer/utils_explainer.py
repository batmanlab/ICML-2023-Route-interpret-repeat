import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import utils
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.residual import Residual
from dataset.dataset_awa2 import Dataset_awa2_for_explainer
from dataset.dataset_completeness import Dataset_completeness
from dataset.dataset_cubs import Dataset_cub_for_explainer
from dataset.dataset_ham10k import load_isic, load_ham_data
from dataset.dataset_mimic_cxr import Dataset_mimic_for_explainer
from dataset.utils_dataset import get_dataset_with_image_and_attributes


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


def get_glts(iteration, args, device, disease_folder, dataset="CUB"):
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(chk_pt_explainer, f"iter{i + 1}", "g", args.checkpoint_model[i])
        if dataset == "mimic_cxr":
            if i == 0:
                chk_pt_path = os.path.join(
                    args.checkpoints, args.dataset, "explainer", disease_folder, args.prev_chk_pt_explainer_folder[i],
                    f"iter{i + 1}", "g", "selected", args.metric, args.checkpoint_model[i]
                )
            elif i == 1:
                chk_pt_path = os.path.join(
                    args.checkpoints, args.dataset, "explainer", disease_folder,
                    args.prev_chk_pt_explainer_folder[i],
                    f"iter{i + 1}", "g", "selected", args.metric, args.checkpoint_model[i]
                )

        else:
            chk_pt_path = os.path.join(
                args.checkpoints, args.dataset, "explainer", disease_folder, args.prev_chk_pt_explainer_folder[i],
                f"iter{i + 1}", "g", args.checkpoint_model[i]
            )
        print(f"===> G for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        if dataset == "CUB":
            glt.load_state_dict(torch.load(chk_pt_path))
        elif dataset == "mimic_cxr":
            glt.load_state_dict(torch.load(chk_pt_path)["state_dict"])
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_glts_soft_seed(iteration, args, seed, device, dataset="CUB"):
    soft_hard_filter = "soft" if args.soft == 'y' else "hard"
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(chk_pt_explainer, f"iter{i + 1}", "g", args.checkpoint_model[i])
        if dataset == "mimic_cxr" or dataset == "stanford_cxr":
            prev_chk_pt_path = args.prev_chk_pt_explainer_folder[i].format(soft_hard_filter=soft_hard_filter, seed=seed)
            chk_pt_path = os.path.join(
                args.checkpoints, dataset, prev_chk_pt_path, f"iter{i + 1}", "g", "selected", args.metric,
                args.checkpoint_model[i]
            )
            split_arr = args.prev_chk_pt_explainer_folder[i].split("_")
            hidden_node = split_arr[split_arr.index("hidden-layers") + 1]
            hidden_nodes = [int(hidden_node[i:i + 2]) for i in range(0, len(hidden_node), 2)]

        print(f"=======>> G for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        if dataset == "CUB":
            glt.load_state_dict(torch.load(chk_pt_path))
        elif dataset == "mimic_cxr":
            glt.load_state_dict(torch.load(chk_pt_path)["state_dict"])
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_glts_soft_seed_domain_transfer(iteration, args, seed, device, dataset="CUB"):
    soft_hard_filter = "soft" if args.soft == 'y' else "hard"
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(chk_pt_explainer, f"iter{i + 1}", "g", args.checkpoint_model[i])
        if dataset == "mimic_cxr" or dataset == "stanford_cxr":
            prev_chk_pt_path = args.prev_chk_pt_explainer_folder[i].format(soft_hard_filter=soft_hard_filter, seed=seed)
            chk_pt_path = os.path.join(
                args.checkpoints, dataset, prev_chk_pt_path, f"iter{i + 1}", "g", "selected", args.metric
            )
            if args.cov != 0:
                chk_pt_path = f"{chk_pt_path}_cov_{args.prev_covs[i]}"
            if args.initialize_w_mimic == "y":
                chk_pt_path = f"{chk_pt_path}_initialize_w_mimic_{args.initialize_w_mimic}"

            chk_pt_path = os.path.join(chk_pt_path, args.checkpoint_model[i])
            split_arr = args.prev_chk_pt_explainer_folder[i].split("_")
            hidden_node = split_arr[split_arr.index("hidden-layers") + 1]
            hidden_nodes = [int(hidden_node[i:i + 2]) for i in range(0, len(hidden_node), 2)]

        print(f"=======>> G for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        if dataset == "CUB":
            glt.load_state_dict(torch.load(chk_pt_path))
        elif dataset == "mimic_cxr":
            glt.load_state_dict(torch.load(chk_pt_path)["state_dict"])
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_residual(iteration, args, residual_chk_pt_path, device, dataset="CUB"):
    prev_residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if dataset == "mimic_cxr":
        residual_chk_pt = os.path.join(
            residual_chk_pt_path, "selected", args.metric, args.checkpoint_residual[-1]
        )
    else:
        residual_chk_pt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
    print(f"=======>> Residual loaded from: {residual_chk_pt}")
    # iteration - 2 = because we need to fetch the last residual, i.e (iteration -1)th residual and
    # the index of the array starts with 0. For example if iteration = 2, we need to fetch the 1st residual.
    # However, the residual_array index starts with 0, so we have to subtract 2 from current iteration.
    if dataset == "CUB" or dataset == "CIFAR10":
        prev_residual.load_state_dict(torch.load(residual_chk_pt))
    elif dataset == "mimic_cxr":
        prev_residual.load_state_dict(torch.load(residual_chk_pt)["state_dict"])

    prev_residual.eval()
    return prev_residual


def get_previous_pi_vals(iteration, glt_list, concepts):
    pi = []
    for i in range(iteration - 1):
        _, out_select, _ = glt_list[i](concepts)
        pi.append(out_select)

    return pi


def get_glts_for_all(iteration, args, device, g_chk_pt_path):
    glt_list = []
    for i in range(iteration - 1):
        chk_pt_path = os.path.join(g_chk_pt_path, f"iter{i + 1}", "explainer", args.checkpoint_model[i])
        print(f"===> G for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        model_chk_pt = torch.load(chk_pt_path)
        if "state_dict" in model_chk_pt:
            glt.load_state_dict(model_chk_pt['state_dict'])
        else:
            glt.load_state_dict(model_chk_pt)
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_glts_for_HAM10k(iteration, args, device):
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[i], "explainer", args.checkpoint_model[i])
        if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i], "explainer", "accuracy", args.checkpoint_model[i]
            )
        else:
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i], "explainer", args.checkpoint_model[i]
            )
        print(f"=======>> glt for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        model_chk_pt = torch.load(chk_pt_path)
        if "state_dict" in model_chk_pt:
            glt.load_state_dict(model_chk_pt['state_dict'])
        else:
            glt.load_state_dict(model_chk_pt)
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_glts_for_HAM10k_soft_seed(iteration, args, seed, device):
    soft_hard_filter = "soft" if args.soft == 'y' else "hard"

    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[i], "explainer", args.checkpoint_model[i])
        if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i].format(soft_hard_filter=soft_hard_filter, seed=seed),
                "explainer", "accuracy", args.checkpoint_model[i]
            ) if args.with_seed.lower() == "y" else os.path.join(
                args.prev_explainer_chk_pt_folder[i].format(soft_hard_filter=soft_hard_filter),
                "explainer", "accuracy", args.checkpoint_model[i]
            )
        else:
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i].format(soft_hard_filter=soft_hard_filter, seed=seed),
                "explainer", args.checkpoint_model[i]
            ) if args.with_seed.lower() == "y" else os.path.join(
                args.prev_explainer_chk_pt_folder[i].format(soft_hard_filter=soft_hard_filter),
                "explainer", args.checkpoint_model[i]
            )

        print(f"=======>> glt for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        model_chk_pt = torch.load(chk_pt_path)
        if "state_dict" in model_chk_pt:
            glt.load_state_dict(model_chk_pt['state_dict'])
        else:
            glt.load_state_dict(model_chk_pt)
        glt.eval()
        glt_list.append(glt)

    return glt_list


# def get_residual_for_all(iteration, args, residual_chk_pt_path, device, dataset="CUB"):
#     prev_residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
#     residual_chk_pt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
#     print(f"---> Residual loaded from: {residual_chk_pt}")
#     # iteration - 2 = because we need to fetch the last residual, i.e (iteration -1)th residual and
#     # the index of the array starts with 0. For example if iteration = 2, we need to fetch the 1st residual.
#     # However, the residual_array index starts with 0, so we have to subtract 2 from current iteration.
#     if dataset == "CUB":
#         prev_residual.load_state_dict(torch.load(residual_chk_pt))
#     elif dataset == "mimic_cxr":
#         prev_residual.load_state_dict(torch.load(residual_chk_pt)["state_dict"])
#
#     prev_residual.eval()
#     return prev_residual


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


def get_details_cubs_experiment(i, arch, alpha_KD, temperature_lens, layer):
    lr = 0.01
    cov = 0.2
    iteration = f"iter{i}"
    if i == 1:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/{arch}/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1_layer_{layer}_explainer_init_none",
                iteration, "explainer", "test_explainer_configs.pkl",
            ), "rb",
        )
    else:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/{arch}/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1_layer_{layer}_explainer_init_none",
                "cov_0.2_lr_0.01", iteration, "explainer", "test_explainer_configs.pkl",
            ), "rb",
        )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    x_to_bool = 0.5
    top_k_explanations = 50
    concept_names = args.concept_names
    print("########################")
    print(args.lr)
    print(args.cov)
    print(args.arch)
    explainer_init = "none"
    use_concepts_as_pi_input = True if args.use_concepts_as_pi_input == "y" else False
    root = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    chk_pt_explainer = os.path.join(
        args.checkpoints, args.dataset, "explainer", args.arch, root
    )

    use_concepts_as_pi_input = True
    explainer_init = "none"

    experiment_folder = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/"
    experiment = f"explainer/{args.arch}/{experiment_folder}"
    expert_type = "explainer"
    output = "g_outputs"

    if i == 1:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # test
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "test_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "test_tensor_y.pt")
        )

        # val
        val_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        val_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        val_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        val_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        val_tensor_concepts_bool = (val_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    else:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # test
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                "cov_0.2_lr_0.01",
                iteration,
                expert_type,
                output,
                "test_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "test_tensor_y.pt")
        )

        # val
        val_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                "cov_0.2_lr_0.01",
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        val_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        val_tensor_preds = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        val_tensor_y = torch.load(
            os.path.join(root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                "cov_0.2_lr_0.01",
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        val_tensor_concepts_bool = (val_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print("\n\n << Test sizes >>")
    print(test_tensor_concepts.size())
    print(test_tensor_concepts_bool.size())
    print(test_tensor_preds.size())
    print(test_tensor_y.size())
    print(test_tensor_conceptizator_concepts.size())

    print("\n\n << Val sizes >>")
    print(val_tensor_concepts.size())
    print(val_tensor_concepts_bool.size())
    print(val_tensor_preds.size())
    print(val_tensor_y.size())
    print(val_tensor_conceptizator_concepts.size())

    print("\n\n << Train sizes >>")
    print(train_tensor_concepts.size())
    print(train_tensor_concepts_bool.size())
    print(train_tensor_preds.size())
    print(train_tensor_y.size())
    print(train_tensor_conceptizator_concepts.size())
    output_path = None
    if i == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"{iteration}", "explainer")
        output_path = os.path.join(root, experiment, iteration, expert_type, output)
    else:
        g_chk_pt_path = os.path.join(chk_pt_explainer, "cov_0.2_lr_0.01", f"{iteration}", "explainer")
        output_path = os.path.join(root, experiment, "cov_0.2_lr_0.01", iteration, expert_type, output)

    if i == 6 and args.arch == "ViT-B_16":
        glt_chk_pt = os.path.join(g_chk_pt_path, "model_g_best_model_epoch_204.pth.tar")
    else:
        glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    device = utils.get_device()
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input,
    ).to(device)
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "val_tensor_concepts": val_tensor_concepts,
        "val_tensor_concepts_bool": val_tensor_concepts_bool,
        "val_tensor_preds": val_tensor_preds,
        "val_tensor_y": val_tensor_y,
        "val_tensor_conceptizator_concepts": val_tensor_conceptizator_concepts,
        "train_tensor_concepts": train_tensor_concepts,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "output_path": output_path,
        "glt": model,
        "labels": args.labels
    }


def get_details_awa2_experiment(i, arch, alpha_KD, temperature_lens, lr, cov, layer, prev_path):
    iteration = f"iter{i}"
    if i == 1:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/{arch}/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1_layer_{layer}_explainer_init_none",
                iteration, "explainer", "test_explainer_configs.pkl",
            ), "rb",
        )
    else:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/{arch}/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1_layer_{layer}_explainer_init_none",
                prev_path, iteration, "explainer", "test_explainer_configs.pkl",
            ), "rb",
        )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    x_to_bool = 0.5
    top_k_explanations = 50
    concept_names = args.concept_names
    print("########################")
    print(args.lr)
    print(args.cov)
    print(args.arch)
    explainer_init = "none"
    use_concepts_as_pi_input = True if args.use_concepts_as_pi_input == "y" else False
    root = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    chk_pt_explainer = os.path.join(
        args.checkpoints, args.dataset, "explainer", args.arch, root
    )

    use_concepts_as_pi_input = True
    explainer_init = "none"

    experiment_folder = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/"
    experiment = f"explainer/{args.arch}/{experiment_folder}"
    expert_type = "explainer"
    output = "g_outputs"

    if i == 1:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # test
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "test_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "test_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    else:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "test_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "test_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "test_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # test
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "test_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "test_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "test_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "test_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print("\n\n << Test sizes >>")
    print(test_tensor_concepts.size())
    print(test_tensor_concepts_bool.size())
    print(test_tensor_preds.size())
    print(test_tensor_y.size())
    print(test_tensor_conceptizator_concepts.size())

    print("\n\n << Train sizes >>")
    print(train_tensor_concepts.size())
    print(train_tensor_concepts_bool.size())
    print(train_tensor_preds.size())
    print(train_tensor_y.size())
    print(train_tensor_conceptizator_concepts.size())
    output_path = None
    if i == 1:
        g_chk_pt_path = os.path.join(chk_pt_explainer, f"{iteration}", "explainer")
        output_path = os.path.join(root, experiment, iteration, expert_type, output)
    else:
        g_chk_pt_path = os.path.join(chk_pt_explainer, prev_path, f"{iteration}", "explainer")
        output_path = os.path.join(root, experiment, prev_path, iteration, expert_type, output)

    if i == 6 and args.arch == "ViT-B_16":
        glt_chk_pt = os.path.join(g_chk_pt_path, "model_g_best_model_epoch_80.pth.tar")
    else:
        glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    device = utils.get_device()
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input,
    ).to(device)
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "train_tensor_concepts": train_tensor_concepts,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "output_path": output_path,
        "glt": model,
        "labels": args.labels
    }


def get_details_ham10k_experiment(i, arch, alpha_KD, temperature_lens, lr, cov, prev_path):
    iteration = f"iter{i}"
    if i == 1:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/explainer/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1",
                iteration, "explainer", "accuracy", "test_explainer_configs.pkl",
            ), "rb",
        )
    else:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/explainer/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1",
                prev_path, iteration, "explainer", "accuracy", "test_explainer_configs.pkl",
            ), "rb",
        )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    x_to_bool = 0.5
    top_k_explanations = 50
    concept_names = args.concept_names
    print("########################")
    print(args.lr)
    print(args.cov)
    print(args.arch)

    experiment_folder = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
    )

    root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/"
    experiment = f"explainer/{experiment_folder}"
    expert_type = "explainer/accuracy"
    output = "g_outputs"

    if i == 1:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # val
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    else:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # val
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print("\n\n << Test sizes >>")
    print(test_tensor_concepts.size())
    print(test_tensor_concepts_bool.size())
    print(test_tensor_preds.size())
    print(test_tensor_y.size())
    print(test_tensor_conceptizator_concepts.size())

    print("\n\n << Train sizes >>")
    print(train_tensor_concepts.size())
    print(train_tensor_concepts_bool.size())
    print(train_tensor_preds.size())
    print(train_tensor_y.size())
    print(train_tensor_conceptizator_concepts.size())
    output_path = None
    if i == 1:
        glt_chk_pt = os.path.join(
            args.checkpoints,
            args.dataset,
            "explainer",
            experiment_folder,
            f"{iteration}",
            "explainer",
            "accuracy",
            args.checkpoint_model[-1]
        )
        output_path = os.path.join(root, experiment, iteration, expert_type, output)
    else:
        glt_chk_pt = os.path.join(
            args.checkpoints,
            args.dataset,
            "explainer",
            experiment_folder,
            prev_path,
            f"{iteration}",
            "explainer",
            "accuracy",
            args.checkpoint_model[-1]
        )
        output_path = os.path.join(root, experiment, prev_path, iteration, expert_type, output)

    device = utils.get_device()
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "train_tensor_concepts": train_tensor_concepts,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "output_path": output_path,
        "glt": model,
        "labels": args.labels
    }


def get_details_isic_experiment(i, arch, alpha_KD, temperature_lens, lr, cov, prev_path):
    iteration = f"iter{i}"
    if i == 1:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/SIIM-ISIC/explainer/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1",
                iteration, "explainer", "accuracy", "test_explainer_configs.pkl",
            ), "rb",
        )
    else:
        pickle_in = open(
            os.path.join(
                f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/SIIM-ISIC/explainer/lr_{lr}_epochs_500_temperature-lens_{temperature_lens}_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{alpha_KD}_temperature-KD_10.0_hidden-layers_1",
                prev_path, iteration, "explainer", "accuracy", "test_explainer_configs.pkl",
            ), "rb",
        )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    x_to_bool = 0.5
    top_k_explanations = 50
    concept_names = args.concept_names
    print("########################")
    print(args.lr)
    print(args.cov)
    print(args.arch)

    experiment_folder = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
    )

    root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/SIIM-ISIC/"
    experiment = f"explainer/{experiment_folder}"
    expert_type = "explainer/accuracy"
    output = "g_outputs"

    if i == 1:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # val
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    else:
        tensor_alpha = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_alpha.pt"
            )
        )

        tensor_alpha_norm = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_alpha_norm.pt"
            )
        )

        tensor_concept_mask = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_concept_mask.pt"
            )
        )

        conceptizator_threshold = 0.5
        # val
        test_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "val_tensor_conceptizator_concepts.pt",
            )
        )

        test_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_concepts.pt"
            )
        )

        test_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "val_tensor_preds.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "val_tensor_y.pt")
        )

        # train
        train_tensor_conceptizator_concepts = torch.load(
            os.path.join(
                root,
                experiment,
                prev_path,
                iteration,
                expert_type,
                output,
                "train_tensor_conceptizator_concepts.pt",
            )
        )

        train_tensor_concepts = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_concepts.pt"
            )
        )

        train_tensor_preds = torch.load(
            os.path.join(
                root, experiment, prev_path, iteration, expert_type, output, "train_tensor_preds.pt"
            )
        )

        train_tensor_y = torch.load(
            os.path.join(root, experiment, prev_path, iteration, expert_type, output, "train_tensor_y.pt")
        )

        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print("\n\n << Test sizes >>")
    print(test_tensor_concepts.size())
    print(test_tensor_concepts_bool.size())
    print(test_tensor_preds.size())
    print(test_tensor_y.size())
    print(test_tensor_conceptizator_concepts.size())

    print("\n\n << Train sizes >>")
    print(train_tensor_concepts.size())
    print(train_tensor_concepts_bool.size())
    print(train_tensor_preds.size())
    print(train_tensor_y.size())
    print(train_tensor_conceptizator_concepts.size())
    output_path = None
    if i == 1:
        glt_chk_pt = os.path.join(
            args.checkpoints,
            args.dataset,
            "explainer",
            experiment_folder,
            f"{iteration}",
            "explainer",
            "accuracy",
            args.checkpoint_model[-1]
        )
        output_path = os.path.join(root, experiment, iteration, expert_type, output)
    else:
        glt_chk_pt = os.path.join(
            args.checkpoints,
            args.dataset,
            "explainer",
            experiment_folder,
            prev_path,
            f"{iteration}",
            "explainer",
            "accuracy",
            args.checkpoint_model[-1]
        )
        output_path = os.path.join(root, experiment, prev_path, iteration, expert_type, output)

    device = utils.get_device()
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "train_tensor_concepts": train_tensor_concepts,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "output_path": output_path,
        "glt": model,
        "labels": args.labels
    }


def get_data_loaders_per_iter_completeness_baseline(args):
    if args.dataset == "awa2":
        dataset_path = args.dataset_path
        train_dataset = Dataset_completeness(dataset_path, transform=None, mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=None, mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader
    elif args.dataset == "cub":
        dataset_path = args.dataset_path
        train_dataset = Dataset_completeness(dataset_path, transform=None, mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=None, mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader


def get_data_loaders_per_iter_completeness(args):
    if args.dataset == "awa2":
        dataset_path = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/completeness/{args.arch}/dataset"
        train_dataset = Dataset_completeness(dataset_path, transform=None, mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=None, mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader
    elif args.dataset == "cub" and (args.arch == "ResNet50" or args.arch == "ResNet101"):
        from torchvision import transforms
        _transforms = {
            "train_transform": transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]),
            "val_transform": transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )

            ])
        }
        dataset_path = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/completeness/{args.arch}/dataset"
        train_dataset = Dataset_completeness(dataset_path, transform=_transforms["train_transform"], mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=_transforms["val_transform"], mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    elif args.dataset == "cub" and (args.arch == "ViT-B_16" or args.arch == "ViT-B_16_projected"):
        from torchvision import transforms
        _transforms = {
            "train_transform": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((448, 448), Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]),
            "val_transform": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((448, 448), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ])
        }
        dataset_path = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/completeness/{args.arch}/dataset"
        train_dataset = Dataset_completeness(dataset_path, transform=_transforms["train_transform"], mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=_transforms["val_transform"], mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    elif args.dataset == "HAM10k":
        from torchvision import transforms
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        dataset_path = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/completeness/{args.arch}/dataset"
        train_dataset = Dataset_completeness(dataset_path, transform=transform, mode="train")
        val_dataset = Dataset_completeness(dataset_path, transform=transform, mode="test")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader


def get_test_loaders_per_iter_completeness(i, args):
    iteration = f"iter{i}"
    if args.dataset == "cub" or args.dataset == "awa2":
        from torchvision import transforms
        _transforms = transforms.Compose(
            [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ) if args.dataset == "cub" else None

        lr = 0.001 if args.dataset == "awa2" and args.arch == "ResNet50" else 0.01
        cov = 0.4 if args.dataset == "awa2" and args.arch == "ResNet50" else 0.2
        prev_path = "cov_0.4_lr_0.001" if args.dataset == "awa2" and args.arch == "ResNet50" else "cov_0.2_lr_0.01"
        dataset_path = os.path.join(
            f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/explainer/{args.arch}/lr_{lr}_"
            f"epochs_500_temperature-lens_{args.temperature_lens}_"
            f"use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_"
            f"alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{args.alpha_KD}_"
            f"temperature-KD_10.0_hidden-layers_1_layer_{args.layer}_explainer_init_none",
            iteration, "explainer", "g_outputs"
        ) if i == 1 else os.path.join(
            f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/explainer/{args.arch}/lr_{lr}_"
            f"epochs_500_temperature-lens_{args.temperature_lens}_"
            f"use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_"
            f"alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_{args.alpha_KD}_"
            f"temperature-KD_10.0_hidden-layers_1_layer_{args.layer}_explainer_init_none",
            prev_path, iteration, "explainer", "g_outputs"
        )
        test_dataset = Dataset_completeness(dataset_path, transform=_transforms, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader
    elif args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
        from torchvision import transforms
        base_lr = 0.01
        base_cov = 0.2
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        dataset_path = os.path.join(
            f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/explainer/lr_{base_lr}_"
            f"epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_{base_cov}_"
            f"alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            iteration,
            "explainer",
            "accuracy",
            "g_outputs"
        ) if i == 1 else os.path.join(
            f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/{args.dataset}/explainer/lr_{base_lr}_"
            f"epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_{base_cov}_"
            f"alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "cov_0.2",
            iteration,
            "explainer",
            "accuracy",
            "g_outputs"
        )
        test_dataset = Dataset_completeness(dataset_path, transform=transform, mode="val")
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader


def get_data_loaders(args):
    if args.dataset == "cub":
        dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
        train_data, train_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="train",
            attribute_file=args.attribute_file_name,
        )

        test_data, test_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="test",
            attribute_file=args.attribute_file_name,
        )
        transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
        train_transform = transforms["train_transform"]
        val_transform = transforms["val_transform"]

        train_dataset = Dataset_cub_for_explainer(
            dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt",
            train_data, train_transform
        )
        val_dataset = Dataset_cub_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt", test_data,
            val_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    elif args.dataset == 'awa2':
        dataset_path = os.path.join(args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g")
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
        return train_loader, val_loader

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
        return train_loader, val_loader
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
        return train_loader, val_loader
    elif args.dataset == "mimic_cxr":
        # from torchvision import transforms
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        #
        # arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
        # arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75
        # args.N_landmarks_spec = len(args.landmark_names_spec)
        # args.N_selected_obs = len(args.selected_obs)
        # args.N_labels = len(args.labels)
        # train_dataset = MIMICCXRDataset(
        #     args=args,
        #     radgraph_sids=arr_rad_graph_sids,
        #     radgraph_adj_mtx=arr_rad_graph_adj,
        #     mode='train',
        #     transform=transforms.Compose([
        #         transforms.Resize(args.resize),
        #         # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
        #         transforms.CenterCrop(args.resize),
        #         # transforms.RandomRotation(args.degree),
        #         # transforms.RandomCrop(args.crop),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),  # convert pixel value to [0, 1]
        #         normalize
        #     ])
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, shuffle=True,
        #     num_workers=args.workers, pin_memory=True, drop_last=True
        # )
        # val_dataset = MIMICCXRDataset(
        #     args=args,
        #     radgraph_sids=arr_rad_graph_sids,
        #     radgraph_adj_mtx=arr_rad_graph_adj,
        #     mode='valid',
        #     transform=transforms.Compose([
        #         transforms.Resize(args.resize),
        #         transforms.CenterCrop(args.resize),
        #         transforms.ToTensor(),  # convert pixel value to [0, 1]
        #         normalize
        #     ])
        # )
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        #     drop_last=True
        # )
        dataset_path = os.path.join(
            args.output,
            args.dataset,
            "t",
            args.dataset_folder_concepts,
            args.arch,
            args.disease_folder,
            "dataset_g",
        )
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )

        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )

        return train_loader, val_loader


def get_data_tuple(tuple, dataset, device, per_iter_completeness=False):
    if dataset == "cub" and not per_iter_completeness:
        train_images, train_concepts, _, train_y, train_y_one_hot = tuple
        return train_images.to(device), train_y.to(torch.long).to(device)
    if dataset == "mimic_cxr" and not per_iter_completeness:
        # (dicom_id,
        #  image,
        #  adj_mtx, _, _,
        #  landmark_spec_label,
        #  landmarks_spec_inverse_weight,
        #  landmark_spec_label_pnu,
        #  selected_obs_label_gt,
        #  selected_obs_inverse_weight,
        #  selected_obs_label_pnu, _, _, _, _, _) = tuple
        # image = image.to(device)
        # selected_obs_label_gt = selected_obs_label_gt.to(device)
        # selected_obs_label_gt = selected_obs_label_gt.view(-1)
        (
            _,
            _,
            train_features_phi,
            train_bb_logits,
            _,
            train_proba_concept_x,
            train_y,
            y_one_hot,
            concepts
        ) = tuple
        return train_features_phi.to(device), train_y.to(torch.long).to(device)
    elif dataset == "awa2" and not per_iter_completeness:
        train_images, train_concepts, train_attrs, train_image_names, train_y, train_y_one_hot = tuple
        return train_images.to(device), train_y.to(torch.long).to(device)
    elif (dataset == "cub" or dataset == "awa2") and per_iter_completeness:
        train_images, train_y, train_mask = tuple
        return train_images.to(device), train_y.to(torch.long).to(device), train_mask.to(device)


def get_details_cubs_experiment_baseline(path):
    pickle_in = open(
        os.path.join(
            path, "test_explainer_configs.pkl"
        ), "rb",
    )
    args = pickle.load(pickle_in)
    x_to_bool = 0.5
    mode = "test"
    tensor_alpha = torch.load(
        os.path.join(
            path, f"{mode}_tensor_alpha.pt"
        )
    )

    tensor_alpha_norm = torch.load(
        os.path.join(
            path, f"{mode}_tensor_alpha_norm.pt"
        )
    )

    tensor_concept_mask = torch.load(
        os.path.join(
            path, f"{mode}_tensor_concept_mask.pt"
        )
    )

    test_images = torch.load(
        os.path.join(
            path, f"{mode}_tensor_images.pt"
        )
    )

    test_tensor_conceptizator_concepts = torch.load(
        os.path.join(
            path,
            f"{mode}_tensor_conceptizator_concepts.pt",
        )
    )

    test_tensor_concepts = torch.load(
        os.path.join(
            path, f"{mode}_tensor_concepts.pt"
        )
    )
    test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    test_tensor_preds = torch.load(
        os.path.join(
            path, f"{mode}_tensor_preds.pt"
        )
    )

    test_tensor_y = torch.load(
        os.path.join(path, f"{mode}_tensor_y.pt")
    )

    mode = "train"
    train_images = torch.load(
        os.path.join(
            path, f"{mode}_tensor_images.pt"
        )
    )

    train_tensor_conceptizator_concepts = torch.load(
        os.path.join(
            path,
            f"{mode}_tensor_conceptizator_concepts.pt",
        )
    )

    train_tensor_concepts = torch.load(
        os.path.join(
            path, f"{mode}_tensor_concepts.pt"
        )
    )

    train_tensor_preds = torch.load(
        os.path.join(
            path, f"{mode}_tensor_preds.pt"
        )
    )

    train_tensor_y = torch.load(
        os.path.join(path, f"{mode}_tensor_y.pt")
    )

    train_tensor_concepts_bool = (train_tensor_concepts.cpu() > x_to_bool).to(torch.float)

    mode = "val"
    val_images = torch.load(
        os.path.join(
            path, f"{mode}_tensor_images.pt"
        )
    )

    val_tensor_conceptizator_concepts = torch.load(
        os.path.join(
            path,
            f"{mode}_tensor_conceptizator_concepts.pt",
        )
    )

    val_tensor_concepts = torch.load(
        os.path.join(
            path, f"{mode}_tensor_concepts.pt"
        )
    )

    val_tensor_preds = torch.load(
        os.path.join(
            path, f"{mode}_tensor_preds.pt"
        )
    )

    val_tensor_y = torch.load(
        os.path.join(path, f"{mode}_tensor_y.pt")
    )

    val_tensor_concepts_bool = torch.load(
        os.path.join(
            path, f"{mode}_tensor_concepts.pt"
        )
    )

    val_tensor_concepts_bool = (val_tensor_concepts_bool.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print("\n\n << Val sizes >>")
    # print(val_ori_images.size())
    print(val_images.size())
    print(val_tensor_concepts.size())
    print(val_tensor_concepts_bool.size())
    print(val_tensor_preds.size())
    print(val_tensor_y.size())
    print(val_tensor_conceptizator_concepts.size())
    output_path = path

    n_classes = len(args.labels)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("1. Checkpoint path: ===========>>>>>")
    print(f"{args.g_chk_pt_path}/{args.g_checkpoint}")
    cur_glt_chkpt = os.path.join(args.g_chk_pt_path, args.g_checkpoint)
    g = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)
    g.load_state_dict(torch.load(cur_glt_chkpt))
    g.eval()

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "val_tensor_concepts": val_tensor_concepts,
        "val_tensor_concepts_bool": val_tensor_concepts_bool,
        "val_tensor_preds": val_tensor_preds,
        "val_tensor_y": val_tensor_y,
        "val_tensor_conceptizator_concepts": val_tensor_conceptizator_concepts,
        "train_tensor_concepts": train_tensor_concepts,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "output_path": output_path,
        "glt": g,
        "labels": args.labels
    }


def get_cov_weights_mimic(disease):
    if disease == "effusion":
        return 0.23459593951702118
    elif disease == "cardiomegaly":
        return 0.13967624306678772
    elif disease == "pneumothorax":
        return 0.037894297391176224
    elif disease == "pneumonia":
        return 0.034385863691568375
    elif disease == "edema":
        return 0.11261117458343506
