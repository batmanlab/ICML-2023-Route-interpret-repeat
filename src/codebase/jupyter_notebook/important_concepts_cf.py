import copy
import os
import pickle
import sys

from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)
import numpy as np
import torch
import utils


def show_results_cub_vit(
        idx,
        args,
        test_tensor_y,
        test_tensor_preds,
        test_tensor_preds_bb,
        tensor_alpha_norm,
        test_tensor_concepts_bool,
        test_tensor_concepts,
        model,
        n_concept_to_retain
):
    print(f"{idx}: ==================================================>")
    device = utils.get_device()
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    print("----------------------------")
    print(f"Ground Truth class_label: {args.labels[target_class]} ({target_class})")
    print(f"Predicted(g) class_label: {args.labels[y_hat]} ({y_hat})")
    print(f"Check wrt GT: {target_class == y_hat}")

    top_concepts = torch.topk(tensor_alpha_norm[y_hat], n_concept_to_retain)[1]
    concepts = test_tensor_concepts[idx]
    concepts[top_concepts] = 0
    y_pred_ex, _, _ = model(concepts.unsqueeze(0).to(device))
    print(y_pred_ex.argmax(dim=1))
    y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)[0]
    return y_pred_ex.item()


def counterfactual_preds(n_experts, path, prev_path, n_concept_to_retain):
    _dict = {}
    for i in range(n_experts):
        i += 1
        print(f"<<<<<<< Expert : {i} >>>>>>>> ")
        full_path = path if i == 1 else os.path.join(path, prev_path)
        pickle_in = open(
            os.path.join(
                full_path,
                f"iter{i}",
                "explainer",
                "test_explainer_configs.pkl",
            ),
            "rb",
        )
        args = pickle.load(pickle_in)
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
        experiment = f"explainer/{args.arch}/{experiment_folder}" if i == 1 else \
            f"explainer/{args.arch}/{experiment_folder}/{prev_path}"
        iteration = f"iter{i}"
        expert_type = "explainer"
        output = "g_outputs"
        # model specific
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

        test_images = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_images.pt"
            )
        )

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

        test_tensor_preds_bb = torch.load(
            os.path.join(
                root, experiment, iteration, expert_type, output, "test_tensor_preds_bb.pt"
            )
        )

        test_tensor_y = torch.load(
            os.path.join(root, experiment, iteration, expert_type, output, "test_tensor_y.pt")
        )
        x_to_bool = 0.5
        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)
        print("\n\n << Test sizes >>")
        print(test_images.size())
        print(test_tensor_concepts.size())
        print(test_tensor_concepts_bool.size())
        print(test_tensor_preds.size())
        print(test_tensor_y.size())
        print(test_tensor_conceptizator_concepts.size())

        percentile_selection = 85
        args.flattening_type = "VIT"

        consistency_pred_target = 0
        consistency_pred_explainer = 0

        print(args.checkpoints)
        chkpt = os.path.join(
            args.checkpoints, args.dataset, "explainer", args.arch, experiment_folder, f"{iteration}", "explainer",
            args.checkpoint_model[-1]
        ) if i == 1 else os.path.join(
            args.checkpoints, args.dataset, "explainer", args.arch, experiment_folder, prev_path,
            f"{iteration}", "explainer",
            args.checkpoint_model[-1]
        )

        device = utils.get_device()
        print(f"---> Latest G for iteration {iteration} is loaded from {chkpt}")
        model = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens,
            use_concepts_as_pi_input,
        ).to(device)
        model.load_state_dict(torch.load(chkpt))
        model.eval()

        y_unique = torch.unique(test_tensor_preds.argmax(dim=1))
        concept_dict = {}
        for y in y_unique:
            concept_dict[int(y.item())] = []

        _feature_names = [f"feature{j:010}" for j in range(test_tensor_concepts_bool.size(1))]
        concept_label = []
        num_concepts_ex = []
        results_arr = []
        y_pred = []
        test_tensor_y_arr = []
        test_tensor_preds_g_arr = []
        for _idx in range(test_tensor_concepts_bool.size(0)):
            y_pred.append(
                show_results_cub_vit(
                    _idx,
                    args,
                    test_tensor_y,
                    test_tensor_preds,
                    test_tensor_preds_bb,
                    tensor_alpha_norm,
                    test_tensor_concepts_bool,
                    test_tensor_concepts,
                    model,
                    n_concept_to_retain
                )
            )
            test_tensor_y_arr.append(test_tensor_y[_idx].item())
            test_tensor_preds_g_arr.append(test_tensor_preds[_idx].argmax(dim=0).item())

        _dict[i] = {
            "y_pred_cf": y_pred,
            "test_tensor_y": test_tensor_y_arr,
            "test_tensor_preds_g": test_tensor_preds_g_arr
        }

    return _dict


if __name__ == '__main__':
    lr = 0.01
    cov = 0.2
    path = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/ViT-B_16/lr_{lr}_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_{cov}_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none"
    n_experts = 6
    _dict = counterfactual_preds(n_experts, path)
