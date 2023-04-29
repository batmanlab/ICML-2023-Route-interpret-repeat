import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)
import pandas as pd
import numpy as np
import utils


def compute_explanations_per_sample(
        iteration,
        idx,
        feature_names,
        test_tensor_preds,
        test_tensor_y,
        test_tensor_concepts_bool,
        tensor_alpha_norm,
        percentile_selection,
        concept_names,
        model,
        test_tensor_concepts_proba,
        device,
        model_type="explainer"
):
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    ps = 100
    while True:
        if percentile_selection == 0:
            percentile_selection = 80
            ps = 0
        percentile_val = np.percentile(tensor_alpha_norm[y_hat], percentile_selection)
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        mask = mask_alpha_norm
        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = test_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = test_tensor_concepts_proba[idx] * mask_alpha_norm
        if model_type == "baseline":
            y_pred_ex = model(imp_concept_vector.unsqueeze(0).to(device))
        else:
            y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        if ps == 0:
            break
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    concept_names_in_explanations = []
    explanations = ""
    for m_idx in mask_indxs.tolist():
        concept_names_in_explanations.append(concept_names[m_idx])
        if explanations:
            explanations += " & "

        if test_tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif test_tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = utils.replace_names(explanations, concept_names)

    return {
        "idx": idx,
        "all_concept_proba": test_tensor_concepts_proba[idx].tolist(),
        "concept_ids_in_explanations": mask_indxs.tolist(),
        "concept_names_in_explanations": concept_names_in_explanations,
        "g_pred": y_hat.item(),
        "g_pred_logit": test_tensor_preds[idx].tolist(),
        "ground_truth": target_class.item(),
        "expert_id": iteration,
        "raw_explanations": explanations,
        "actual_explanations": explanation_complete,
    }


def build_FOLs(_dict, args):
    tensor_alpha_norm = _dict["tensor_alpha_norm"]
    test_tensor_concepts = _dict["test_tensor_concepts"]
    test_tensor_y = _dict["test_tensor_y"]
    test_tensor_preds = _dict["test_tensor_preds"]
    test_tensor_concepts_bool = _dict["test_tensor_concepts_bool"]

    moIE = _dict["moIE"]
    pkl = _dict["pkl"]
    device = _dict["device"]
    _feature_names = [f"feature{j:010}" for j in range(test_tensor_concepts_bool.size(1))]
    percentile_selection = 99
    results_arr = []
    for _idx in range(test_tensor_concepts_bool.size(0)):
        results = compute_explanations_per_sample(
            args.cur_iter,
            _idx,
            _feature_names,
            test_tensor_preds,
            test_tensor_y,
            test_tensor_concepts_bool,
            tensor_alpha_norm,
            percentile_selection,
            pkl.concept_names,
            moIE,
            test_tensor_concepts,
            device,
            model_type="explainer"
        )
        results_arr.append(results)
        print(
            f" {[results['idx']]}, "
            f"predicted: {pkl.labels[results['g_pred']]}, "
            f"target: {pkl.labels[results['ground_truth']]}"
        )
        print(f" {pkl.labels[results['g_pred']]} <=> {results['actual_explanations']}")

    fol_path = Path(f"{args.output}/{args.dataset}/FOLs/{args.arch}")
    print(fol_path)
    os.makedirs(fol_path, exist_ok=True)
    test_results_df = pd.DataFrame.from_dict(results_arr, orient='columns')
    pickle.dump(args, open(fol_path / f"configs_{args.cur_iter}.pkl", "wb"))
    test_results_df.to_csv(fol_path / f"test_results_expert_{args.cur_iter}.csv")

    print(f"******************* saved at: {fol_path} *******************")
