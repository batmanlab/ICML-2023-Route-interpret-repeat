import os
# sys.path.append(
#     os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
# )
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from Explainer.models.Gated_Logic_Net import Gated_Logic_Net


# from jupyter_notebook import ipython_utils as ipy
def replace_names(explanation: str, concept_names) -> str:
    """
    Replace names of concepts in a formula.

    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation


def show_results(
        args,
        idx,
        val_ori_images,
        val_tensor_y,
        val_tensor_concepts_bool,
        val_tensor_concepts,
        val_tensor_preds,
        feature_names,
        tensor_alpha_norm,
        concept_names,
        model,
):
    print(f"{idx}: ==================================================>")
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    inv_tensor = inv_normalize(val_ori_images[idx])
    im = inv_tensor.permute(1, 2, 0).numpy()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    target_class = val_tensor_y[idx].to(torch.int32)
    y_hat = val_tensor_preds[idx].argmax(dim=0)
    print("----------------------------")
    print(f"Ground Truth class_label: {args.labels[target_class]} ({target_class})")
    print(f"Predicted(g) class_label: {args.labels[y_hat]} ({y_hat})")
    print(f"Check wrt GT: {target_class == y_hat}")
    percentile_selection = 99
    ps = 100
    while True:
        if percentile_selection == 0:
            print(f"percentile_selection: {percentile_selection}")
            percentile_selection = 90
            ps = 0
        percentile_val = np.percentile(tensor_alpha_norm[y_hat], percentile_selection)
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        #         print(percentile_val)
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = val_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = val_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        if ps == 0:
            break
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    #     print(mask_indxs)
    #     print(imp_concepts)

    dict_sample_concept = {}
    for concept in args.concept_names:
        dict_sample_concept[concept] = 0

    dict_sample_concept["y_GT"] = 0
    dict_sample_concept["y_G"] = 0

    concepts = []
    for indx in mask_indxs:
        dict_sample_concept[args.concept_names[indx]] = 1
        concepts.append(args.concept_names[indx])
    dict_sample_concept["y_GT"] = target_class.item()
    dict_sample_concept["y_G"] = y_hat.item()
    dict_sample_concept["correctly_predicted_wrt_GT"] = (
            target_class.item() == y_hat.item()
    )

    explanations = ""
    for m_idx in mask_indxs.tolist():
        if explanations:
            explanations += " & "

        if val_tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif val_tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = replace_names(explanations, concept_names)

    print("Concept Explanations: =======================>>>>")
    print(f"{args.labels[y_hat]} ({y_hat}) <=> {explanation_complete}")

    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        "concept_dict_key": int(y_hat.item()),
        #         "concept_dict_key": int(y_hat.item())
        #         if (target_class.item() == y_hat.item())
        #         else -1,
        "concept_dict_val": explanation_complete,
        "im": im,
        "test_tensor_concepts": val_tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
    }


def get_test_set(path, mode="test"):
    x_to_bool = 0.5
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

    val_tensor_concepts_bool = (val_tensor_concepts.cpu() > x_to_bool).to(torch.float)

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

    return tensor_alpha, tensor_alpha_norm, tensor_concept_mask, val_images, val_tensor_conceptizator_concepts, \
           val_tensor_concepts, val_tensor_preds, val_tensor_y, val_tensor_concepts_bool


def cal_performance(val_tensor_preds, val_tensor_y):
    val_acc_g1 = torch.sum(
        val_tensor_preds.argmax(dim=1).eq(val_tensor_y)
    ) / val_tensor_preds.size(0)

    proba = torch.nn.Softmax(dim=1)(val_tensor_preds)[:, 1]
    print("############### Performance ##############")
    print(f"Val Accuracy: {val_acc_g1 * 100} (%)")
    print("########################################")


def get_explainer_baseline(args):
    if args.dataset == "cub" or  args.dataset == "awa2":
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
        return g


def get_explanations(path, show_image=True, mode="test", get_explanation=True):
    lr = 0.01
    cov = 0.2
    pickle_in = open(
        os.path.join(
            path,
            "test_explainer_configs.pkl",
        ),
        "rb",
    )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    top_k_explanations = 50
    concept_names = args.concept_names
    args.input_size_pi = 2048
    args.bs = 2
    print("########################################")
    print(f"Getting explanations for the dataset: {args.dataset}")
    print("########################################")

    # get test results
    tensor_alpha, tensor_alpha_norm, tensor_concept_mask, val_images, val_tensor_conceptizator_concepts, \
    val_tensor_concepts, val_tensor_preds, val_tensor_y, val_tensor_concepts_bool = get_test_set(path, mode)

    # calculate performance
    cal_performance(val_tensor_preds, val_tensor_y)

    # get explanations
    y_unique = torch.unique(val_tensor_preds.argmax(dim=1))
    concept_dict = {}
    for y in y_unique:
        concept_dict[int(y.item())] = []

    _feature_names = [f"feature{j:010}" for j in range(val_tensor_concepts_bool.size(1))]
    concept_label = []
    num_concepts_ex = []
    results_arr = []
    g = get_explainer_baseline(args)
    output_path = os.path.join(args.g_output_path)
    print("2. output_path: ===========>>>>>")
    print(f"Output path: {output_path}")

    print("########################################")
    print(f"Explanations")
    print("########################################")

    if get_explanation:
        for _idx in range(val_tensor_concepts_bool.size(0)):
            results = show_results(
                args,
                _idx,
                val_images,
                val_tensor_y,
                val_tensor_concepts_bool,
                val_tensor_concepts,
                val_tensor_preds,
                _feature_names,
                tensor_alpha_norm,
                concept_names,
                g,
            )
            results_arr.append(results)
            concept_label.append(results["dict_sample_concept"])
            num_concepts_ex.append(results["num_concepts"])
            print(f"concept_dict_key: {results['concept_dict_key']}")
            #     if results["concept_dict_key"] != -1:
            concept_dict[results["concept_dict_key"]].append(results["concept_dict_val"])
            if show_image:
                plt.imshow(results["im"])
                plt.axis("off")
                plt.show()

        pickle.dump(
            results_arr, open(os.path.join(
                output_path, f"{args.dataset}-baseline_explanations.pkl"
            ), "wb"))


        return concept_label, num_concepts_ex, concept_dict, output_path, tensor_alpha, tensor_alpha_norm, tensor_concept_mask, val_images, \
               val_tensor_conceptizator_concepts, \
               val_tensor_concepts, val_tensor_preds, val_tensor_y, val_tensor_concepts_bool, args, g
    else:
        return concept_label, num_concepts_ex, concept_dict, output_path, tensor_alpha, tensor_alpha_norm, tensor_concept_mask, val_images, \
               val_tensor_conceptizator_concepts, \
               val_tensor_concepts, val_tensor_preds, val_tensor_y, val_tensor_concepts_bool, args, g



if __name__ == '__main__':
    path = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ViT-B_16/explainer"
    get_explanations(path)
