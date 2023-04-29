import copy
import warnings

from BB.models.BB_DenseNet121 import DenseNet121
from Explainer.models.residual import Residual

warnings.filterwarnings("ignore")
import random
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from Explainer.loss_F import entropy_loss, KD_Residual_Loss, \
    Selective_Distillation_Loss
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_mimic_cxr import Dataset_mimic_for_explainer
from Explainer.utils_explainer import get_glts, get_previous_pi_vals, get_residual, get_cov_weights_mimic

import json
import os
import pickle
import sys

import torch

from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)
import pandas as pd
import numpy as np
import utils


def compute_cumulative_performance(
        out_put_target_g, out_put_preds_g, out_put_preds_bb, out_put_target_total, out_put_preds_total,
        out_put_preds_bb_total
):
    prediction_g = out_put_preds_g.argmax(dim=1)
    proba_g = torch.nn.Softmax(dim=1)(out_put_preds_g)
    prediction_bb_g = out_put_preds_bb.argmax(dim=1)
    proba_bb_g = torch.nn.Softmax(dim=1)(out_put_preds_bb)

    prediction_tot = out_put_preds_total.argmax(dim=1)
    proba_tot = torch.nn.Softmax(dim=1)(out_put_preds_total)
    prediction_bb_tot = out_put_preds_bb_total.argmax(dim=1)
    proba_bb_tot = torch.nn.Softmax(dim=1)(out_put_preds_bb_total)

    moie_emp_coverage = prediction_g.size(0) / prediction_tot.size(0)
    moie_acc = utils.cal_accuracy(label=out_put_target_g, out=prediction_g)
    moie_auroc, moie_aurpc = utils.compute_AUC(gt=out_put_target_g, pred=proba_g[:, 1])
    moie_recall = utils.cal_recall_multiclass(label=out_put_target_g.numpy(), out=proba_g.argmax(dim=1).numpy())
    moie_bb_acc = utils.cal_accuracy(label=out_put_target_g, out=prediction_bb_g)
    moie_bb_auroc, moie_bb_aurpc = utils.compute_AUC(gt=out_put_target_g, pred=proba_bb_g[:, 1])
    moie_bb_recall = utils.cal_recall_multiclass(label=out_put_target_g.numpy(), out=proba_bb_g.argmax(dim=1).numpy())

    moie_r_acc = utils.cal_accuracy(label=out_put_target_total, out=prediction_tot)
    moie_r_auroc, moie_r_aurpc = utils.compute_AUC(gt=out_put_target_total, pred=proba_tot[:, 1])
    moie_r_recall = utils.cal_recall_multiclass(label=out_put_target_total.numpy(), out=proba_tot.argmax(dim=1).numpy())
    moie_r_bb_acc = utils.cal_accuracy(label=out_put_target_total, out=prediction_bb_tot)
    moie_r_bb_auroc, moie_r_bb_aurpc = utils.compute_AUC(gt=out_put_target_total, pred=proba_bb_tot[:, 1])
    moie_r_bb_recall = utils.cal_recall_multiclass(
        label=out_put_target_total.numpy(), out=proba_bb_tot.argmax(dim=1).numpy()
    )

    return {
        "moie_emp_coverage": moie_emp_coverage,
        "moie_acc": moie_acc,
        "moie_auroc": moie_auroc,
        "moie_aurpc": moie_aurpc,
        "moie_recall": moie_recall,
        "moie_bb_acc": moie_bb_acc,
        "moie_bb_auroc": moie_bb_auroc,
        "moie_bb_aurpc": moie_bb_aurpc,
        "moie_bb_recall": moie_bb_recall,
        "moie_r_acc": moie_r_acc,
        "moie_r_auroc": moie_r_auroc,
        "moie_r_aurpc": moie_r_aurpc,
        "moie_r_recall": moie_r_recall,
        "moie_r_bb_acc": moie_r_bb_acc,
        "moie_r_bb_auroc": moie_r_bb_auroc,
        "moie_r_bb_aurpc": moie_r_bb_aurpc,
        "moie_r_bb_recall": moie_r_bb_recall
    }


def compute_performance_metrics(
        out_put_class_pred, out_put_class_bb_pred, out_put_target, mask_by_pi, args, get_masked_output=False
):
    prediction_result = out_put_class_pred.argmax(dim=1)
    h_rjc = torch.masked_select(prediction_result, mask_by_pi.bool())
    t_rjc = torch.masked_select(out_put_target, mask_by_pi.bool())
    positive_samples_gt = (t_rjc == 1).sum(dim=0)
    postive_samples_pred = (t_rjc == 1).sum(dim=0)
    idx = (t_rjc == 1).nonzero(as_tuple=True)[0]
    postive_samples_pred_correct = (t_rjc[idx] == h_rjc[idx]).sum()

    t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
    f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum()
              )

    acc = float(t / (t + f + 1e-12)) * 100

    s = mask_by_pi.view(-1, 1)
    sel = torch.cat((s, s), dim=1)
    h_rjc = torch.masked_select(out_put_class_pred, sel.bool()).view(-1, 2)
    proba_g = torch.nn.Softmax(dim=1)(h_rjc)
    t_rjc = torch.masked_select(out_put_target, mask_by_pi.bool())
    g_auroc, g_aurpc = utils.compute_AUC(gt=t_rjc, pred=proba_g[:, 1])
    recall = utils.cal_recall_multiclass(label=t_rjc.numpy(), out=proba_g.argmax(dim=1).numpy())

    out_put_bb_logits_rjc = torch.masked_select(out_put_class_bb_pred, sel.bool()).view(
        -1, 2
    )
    proba_bb = torch.nn.Softmax(dim=1)(out_put_bb_logits_rjc)
    bb_auroc, bb_aurpc = utils.compute_AUC(gt=t_rjc, pred=proba_bb[:, 1])

    condition_true = prediction_result == out_put_target
    condition_false = prediction_result != out_put_target
    condition_acc = mask_by_pi == torch.ones_like(mask_by_pi)
    condition_rjc = mask_by_pi == torch.zeros_like(mask_by_pi)
    ta = float(torch.where(
        condition_true & condition_acc, torch.ones_like(prediction_result), torch.zeros_like(prediction_result),
    ).sum())
    tr = float(torch.where(
        condition_true & condition_rjc, torch.ones_like(prediction_result), torch.zeros_like(prediction_result),
    ).sum())
    fa = float(
        torch.where(
            condition_false & condition_acc,
            torch.ones_like(prediction_result),
            torch.zeros_like(prediction_result),
        ).sum()
    )
    fr = float(
        torch.where(
            condition_false & condition_rjc,
            torch.ones_like(prediction_result),
            torch.zeros_like(prediction_result),
        ).sum()
    )

    rejection_rate = float((tr + fr) / (ta + tr + fa + fr + 1e-12))
    n_rejected = tr + fr
    n_selected = out_put_class_pred.size(0) - (tr + fr)
    coverage = 1 - rejection_rate
    results = {
        "test_g_acc": acc,
        "test_g_auroc": g_auroc,
        "test_g_aurpc": g_aurpc,
        "test_g_recall": recall,
        "test_bb_auroc": bb_auroc,
        "test_selected": n_selected,
        "test_rejected": n_rejected,
        "test_emperical_coverage": coverage,
        "test_true_coverage": args.cov,
        "test_positive_samples_gt": positive_samples_gt,
        "test_postive_samples_pred": postive_samples_pred,
        "test_postive_samples_pred_correct": postive_samples_pred_correct
    }
    print(results)
    if get_masked_output:
        return t_rjc, proba_g, proba_bb
    else:
        return results


def setup(path, residual=False):
    device = utils.get_device()
    print(f"Device: {device}")
    if residual:
        pickle_in = open(os.path.join(path, "test_configs.pkl"), "rb", )
    else:
        pickle_in = open(os.path.join(path, "test_explainer_configs.pkl"), "rb", )
    args = pickle.load(pickle_in)
    all_concept_names = args.concept_names
    print("########################")
    print(all_concept_names)
    print(len(all_concept_names))
    print(args.lr)
    print(args.cov)
    print(args.prev_chk_pt_explainer_folder)
    print(args.checkpoint_model)
    print("########################")
    return device, args


def load_master_csv(args, mode="test"):
    df_master = pd.read_csv(
        "/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/mimic-cxr-chexpert.csv"
    )
    arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)
    df_bbox = pd.read_csv(args.nvidia_bounding_box_file)

    df_master["study_id"] = df_master["study_id"].apply(str)
    subject_ids = np.array(df_master["subject_id"].unique())  # ~65K patients
    np.random.seed(0)
    np.random.shuffle(subject_ids)
    k1 = int(len(subject_ids) * 0.8)
    k2 = int(len(subject_ids) * 0.9)
    train_subject_ids = list(subject_ids[:k1])
    valid_subject_ids = list(subject_ids[k1:k2])
    test_subject_ids = list(subject_ids[k2:])
    idx1 = df_master["ViewPosition"].isin(["AP", "PA"])  # frontal view CXR only
    idx2 = df_master["study_id"].isin(list(arr_rad_graph_sids))  # must have RadGraph labels
    idx3 = df_master["dicom_id"].isin(
        list(df_bbox["dicom_id"].unique())
    )  # leave NVIDIA bbox out for evaluation

    idx = None
    if mode == 'train':
        idx4 = df_master['subject_id'].isin(train_subject_ids)  # is in training dataset
        idx = idx1 & idx2 & (~idx3) & idx4
    elif mode == 'val':
        idx4 = df_master['subject_id'].isin(valid_subject_ids)  # is in validate dataset
        idx = idx1 & idx2 & (~idx3) & idx4
    elif mode == 'test':
        idx4 = df_master['subject_id'].isin(test_subject_ids)  # is in test dataset
        idx = idx1 & idx2 & (~idx3) & idx4

    df_master_sel = df_master[idx]
    return df_master_sel


def get_outputs(iteration, args, output_path, dataset_path, mode="test"):
    g_outputs = "g_outputs"
    model_outputs = "model_outputs"
    print("############################################")
    print(f"ite: {iteration} ==========================>")
    print("############################################")
    tensor_alpha = torch.load(os.path.join(output_path, g_outputs, "test_tensor_alpha.pt"))
    tensor_alpha_norm = torch.load(os.path.join(output_path, g_outputs, "test_tensor_alpha_norm.pt"))
    tensor_concept_mask = torch.load(os.path.join(output_path, g_outputs, "test_tensor_concept_mask.pt"))
    test_mask_by_pi = torch.load(os.path.join(output_path, g_outputs, f"{mode}_mask_by_pi.pt")).view(-1)
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"{mode}_mask_by_pi size: {test_mask_by_pi.size()}")

    out_put_g_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_class_pred.pt"))
    out_put_bb_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_class_bb_pred.pt"))
    out_put_target = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_target.pt"))
    proba_concepts = torch.load(os.path.join(output_path, model_outputs, f"{mode}_proba_concept.pt"))
    ground_truth_concepts = torch.load(os.path.join(output_path, model_outputs, f"{mode}_attributes_gt.pt"))
    concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print("\n")

    print(f"{mode}_out_put_target size: {out_put_target.size()}")
    print(f"{mode}_out_put_class_pred size: {out_put_g_pred.size()}")
    print(f"{mode}_out_put_class_bb_pred size: {out_put_bb_pred.size()}")
    print(f"{mode}_concepts size: {proba_concepts.size()}")
    print(f"{mode}_ground_truth_concepts size: {ground_truth_concepts.size()}")
    print(f"# concepts: {len(concept_names)}")
    print("\n")

    return (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, test_mask_by_pi, out_put_g_pred,
        out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts, concept_names
    )


def get_outputs_residual(iteration, output_path):
    g_outputs = "residual_outputs"
    model_outputs = "model_outputs"
    print("############################################")
    print(f"ite: {iteration} ==========================>")
    print("############################################")

    test_mask_by_pi = torch.load(os.path.join(output_path, g_outputs, "test_mask_by_pi.pt")).view(-1)
    print(f"test_mask_by_pi size: {test_mask_by_pi.size()}")

    test_out_put_g_pred = torch.load(os.path.join(output_path, model_outputs, "test_out_put_preds_residual.pt"))
    test_out_put_bb_pred = torch.load(os.path.join(output_path, model_outputs, "test_out_put_preds_bb.pt"))
    test_out_put_target = torch.load(os.path.join(output_path, model_outputs, "test_out_put_target.pt"))
    print("\n")

    print(f"test_out_put_target size: {test_out_put_target.size()}")
    print(f"test_out_put_class_pred size: {test_out_put_g_pred.size()}")
    print(f"test_out_put_class_bb_pred size: {test_out_put_bb_pred.size()}")
    print("\n")

    return test_mask_by_pi, test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target


def get_moie(args, concept_names, iteration, disease, root, device):
    moie = Gated_Logic_Net(
        args.input_size_pi,
        concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)
    chk_pt_explainer = os.path.join(
        args.checkpoints, args.dataset, "explainer", disease, root, f"iter{iteration}",
        "g", "selected", "auroc", args.checkpoint_model[-1]
    )
    moie.load_state_dict(torch.load(chk_pt_explainer)["state_dict"])
    moie.eval()
    return moie


def compute_explanations_per_sample(
        iteration,
        idx,
        df_master_sel,
        feature_names,
        test_tensor_preds,
        test_tensor_preds_bb,
        test_tensor_y,
        test_tensor_concepts_bool,
        tensor_alpha_norm,
        percentile_selection,
        concept_names,
        model,
        test_tensor_concepts_proba,
        test_ground_truth_concepts,
        device,
):
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    y_hat_bb = test_tensor_preds_bb[idx].argmax(dim=0)
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
        "all_concept_gt": test_ground_truth_concepts[idx].tolist(),
        "concept_ids_in_explanations": mask_indxs.tolist(),
        "concept_names_in_explanations": concept_names_in_explanations,
        "g_pred": y_hat.item(),
        "g_pred_logit": test_tensor_preds[idx].tolist(),
        "bb_pred": y_hat_bb.item(),
        "ground_truth": target_class.item(),
        "expert_id": iteration,
        "raw_explanations": explanations,
        "actual_explanations": explanation_complete,
        "impression_from_report": df_master_sel.iloc[idx]["impression"],
        "findings_from_report": df_master_sel.iloc[idx]["findings"]
    }


def get_g_outputs(iterations, json_file, output, disease, _seed, dataset_path):
    out_put_target_g = torch.FloatTensor()
    out_put_preds_g = torch.FloatTensor()
    out_put_preds_bb = torch.FloatTensor()

    for iteration in range(iterations):
        iteration += 1

        with open(json_file) as _file:
            paths = json.load(_file)
        root = paths[disease]["MoIE_paths"][f"iter{iteration}"]["base_path"]
        print(root)
        output_path = f"{output}/mimic_cxr/explainer/{disease}/{root}/iter{iteration}/g/selected/auroc"
        device, configs = setup(output_path)
        (
            _, _, _, test_mask_by_pi, test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, _, _, _
        ) = get_outputs(iteration, configs, output_path, dataset_path)
        target_selected, g_pred_selected, bb_pred_selected = compute_performance_metrics(
            test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, test_mask_by_pi, configs,
            get_masked_output=True
        )
        print(f"Iteration: {iteration}, Target: {target_selected.size()}, "
              f"Pred_g: {g_pred_selected.size()}, Pred bb: {bb_pred_selected.size()}")
        out_put_target_g = torch.cat((out_put_target_g, target_selected), dim=0)
        out_put_preds_g = torch.cat((out_put_preds_g, g_pred_selected), dim=0)
        out_put_preds_bb = torch.cat((out_put_preds_bb, bb_pred_selected), dim=0)

    final_iteration = iterations
    with open(json_file) as _file:
        paths = json.load(_file)
    root_residual = paths[disease]["MoIE_paths"][f"iter{final_iteration}"]["base_path"]
    output_residual = f"{output}/mimic_cxr/seed_{_seed}/explainer/{disease}/{root_residual}/iter{final_iteration}/residual/selected/auroc"
    print(output_residual)
    device, configs_residual = setup(output_residual, residual=True)
    (
        test_mask_by_pi, test_out_put_residual_pred, test_out_put_bb_pred, test_out_put_target
    ) = get_outputs_residual(final_iteration, output_residual)
    target_residual_selected, g_pred_residual_selected, bb_pred_residual_selected = compute_performance_metrics(
        test_out_put_residual_pred, test_out_put_bb_pred, test_out_put_target, test_mask_by_pi, configs_residual,
        get_masked_output=True
    )

    print("")
    print("Selected by residual sizes: ")
    print(f"Iteration: {final_iteration}, Target: {target_residual_selected.size()}, "
          f"Pred_g: {g_pred_residual_selected.size()}, Pred bb: {bb_pred_residual_selected.size()}")
    print("")
    out_put_target_total = torch.cat((out_put_target_g, target_residual_selected), dim=0)
    out_put_preds_total = torch.cat((out_put_preds_g, g_pred_residual_selected), dim=0)
    out_put_preds_bb_total = torch.cat((out_put_preds_bb, bb_pred_residual_selected), dim=0)

    return (
        out_put_target_g, out_put_preds_g, out_put_preds_bb, out_put_target_total, out_put_preds_total,
        out_put_preds_bb_total
    )


def get_explanations(disease, root, iteration, output_path, dataset_path, path_to_dump, mode="test"):
    x_to_bool = 0.5

    device, configs = setup(output_path)
    df_master_sel = load_master_csv(configs, mode)
    (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, mask_by_pi, out_put_g_pred,
        out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts, concept_names
    ) = get_outputs(iteration, configs, output_path, dataset_path, mode)

    results = compute_performance_metrics(
        out_put_g_pred, out_put_bb_pred, out_put_target, mask_by_pi, configs
    )
    pickle.dump(results, open(os.path.join(path_to_dump, f"{mode}_performance_metrics_expert_{iteration}.pkl"), "wb"))

    moie = get_moie(configs, concept_names, iteration, disease, root, device)
    selected_ids = (mask_by_pi == 1).nonzero(as_tuple=True)[0]
    selected_ids_list = selected_ids.tolist()
    test_tensor_concepts_bool = (proba_concepts.cpu() > x_to_bool).to(torch.float)
    _feature_names = [f"feature{j:010}" for j in range(test_tensor_concepts_bool.size(1))]

    print("Creating Explanations")
    percentile_selection = 99
    ii = 0
    results_arr = []
    for _idx in selected_ids_list:
        ii += 1
        print(f"===>> {ii} <<===")
        results = compute_explanations_per_sample(
            iteration,
            _idx,
            df_master_sel,
            _feature_names,
            out_put_g_pred,
            out_put_bb_pred,
            out_put_target,
            test_tensor_concepts_bool,
            tensor_alpha_norm,
            percentile_selection,
            concept_names,
            moie,
            proba_concepts,
            ground_truth_concepts,
            device
        )
        results_arr.append(results)
        print(
            f" {[results['idx']]}, predicted: {configs.labels[results['g_pred']]}, target: {configs.labels[results['ground_truth']]}"
        )
        print(f" {configs.labels[results['g_pred']]} <=> {results['actual_explanations']}")
    return results_arr


def get_FOL_icml(args):
    args.base_path = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"
    args.iteration = args.iter
    _disease = args.disease
    _iter = args.iteration
    _seed = args.seed
    json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr_icml.json")
    with open(json_file) as _file:
        paths = json.load(_file)
    root = paths[_disease]["MoIE_paths"][f"iter{_iter}"]["base_path"]
    print(root)
    output_path = f"{args.output}/mimic_cxr/explainer/{_disease}/{root}/iter{_iter}/g/selected/auroc"
    dataset_path = f"{args.output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
    path_to_dump = os.path.join(output_path, "FOLs")
    os.makedirs(path_to_dump, exist_ok=True)

    print("######################## Test Explanations Start ########################")
    test_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="test")
    pickle.dump(test_results_arr, open(os.path.join(path_to_dump, f"test_FOL_results_expert_{_iter}.pkl"), "wb"))
    test_results_df = pd.DataFrame.from_dict(test_results_arr, orient='columns')
    test_results_df.to_csv(os.path.join(path_to_dump, f"test_results_expert_{_iter}.csv"))
    print("######################## Test Explanations End ########################")

    print("######################## Val Explanations Start ########################")
    val_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="val")
    pickle.dump(val_results_arr, open(os.path.join(path_to_dump, f"val_FOL_results_expert_{_iter}.pkl"), "wb"))
    val_results_df = pd.DataFrame.from_dict(val_results_arr, orient='columns')
    val_results_df.to_csv(os.path.join(path_to_dump, f"val_results_expert_{_iter}.csv"))
    print("######################## Val Explanations End ########################")

    print("######################## Train Explanations Start ########################")
    train_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="train")
    pickle.dump(train_results_arr, open(os.path.join(path_to_dump, f"train_FOL_results_expert_{_iter}.pkl"), "wb"))
    train_results_df = pd.DataFrame.from_dict(train_results_arr, orient='columns')
    train_results_df.to_csv(os.path.join(path_to_dump, f"train_results_expert_{_iter}.csv"))
    print("######################## Train Explanations End ########################")
    print(path_to_dump)


def test_glt_icml(args):
    print("###" * 30)
    print("================== Testing for ICML rebuttal ================== ")
    print("###" * 30)

    args.concept_names = args.landmark_names_spec + args.abnorm_obs_concepts
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    hidden_layers = ""
    for hl in args.hidden_nodes:
        hidden_layers += str(hl)

    root = f"{args.arch}_{args.bs}_lr_{args.lr}_{args.optim}_temperature-lens_{args.temperature_lens}" \
           f"_cov_{args.cov}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}" \
           f"_lm_{args.lm}_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}" \
           f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{hidden_layers}" \
           f"_input-size-pi_{args.input_size_pi}_layer_{args.layer}"
    print(root)

    iteration = args.iter
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"iteration: {iteration}========================>>")
    if args.expert_to_train == "explainer":
        test_explainer(args, root, iteration, disease_folder)
    elif args.expert_to_train == "residual":
        test_residual(args, root, iteration, disease_folder)


def test_residual(args, root, iteration, disease_folder):
    print(f"Testing the residual for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", disease_folder, root)
    output_path_explainer = os.path.join(args.output, args.dataset, "explainer", disease_folder, root)
    log_path_explainer = os.path.join(args.logs, args.dataset, "explainer", disease_folder, args.metric)

    g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "g", "selected", args.metric)
    residual_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "residual", "selected", args.metric)
    residual_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "residual", "selected", args.metric)

    output_path_model_outputs = os.path.join(residual_output_path, "model_outputs")
    output_path_residual_outputs = os.path.join(residual_output_path, "residual_outputs")
    print(output_path_model_outputs)
    print(output_path_residual_outputs)
    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_residual_outputs, exist_ok=True)
    pickle.dump(args, open(os.path.join(residual_output_path, "test_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")
    dataset_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        args.dataset_folder_concepts,
        args.arch,
        disease_folder,
        "dataset_g",
    )
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print(f"Length concept_names: {len(args.concept_names)}")

    args.N_labels = len(args.labels)
    glt_list = []
    if iteration > 1:
        glt_list = get_glts(iteration, args, device, disease_folder=disease_folder, dataset="mimic_cxr")
    prev_residual = None

    cur_glt_chkpt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {cur_glt_chkpt}")
    glt = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens
    ).to(device)
    glt.load_state_dict(torch.load(cur_glt_chkpt)["state_dict"])
    glt.eval()

    cur_residual_chkpt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
    print(f"=======>> Latest residual checkpoint is loaded for iteration {iteration}: {cur_residual_chkpt}")
    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    residual.load_state_dict(torch.load(cur_residual_chkpt)["state_dict"])
    residual.eval()

    start = time.time()
    if args.iter == 1:
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True
        )

        test_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True
        )

        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="val",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    else:
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        bb_logits_path = os.path.join(
            args.output, args.dataset, "explainer", args.selected_obs[0], args.prev_chk_pt_explainer_folder[-1],
            f"iter{iteration - 1}", "residual", "selected", args.metric, "model_outputs"
        )

        print(f"=======>> bb_logits_path: {bb_logits_path}")

        test_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )
        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="val",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    print("======>> Save overall whole model outputs ")
    predict_residual(args, residual, train_loader, output_path_model_outputs, mode="train")
    predict_residual(args, residual, val_loader, output_path_model_outputs, mode="val")
    predict_residual(args, residual, test_loader, output_path_model_outputs, mode="test")

    print("!! Saving test loader only selected by g!!")
    save_results_selected_residual_by_pi(
        args,
        iteration,
        residual,
        glt,
        test_loader,
        args.selection_threshold,
        output_path_residual_outputs,
        device,
        mode="test",
        higher_iter_params={
            "glt_list": glt_list
        }
    )
    print(f"######### {output_path_residual_outputs} #########")
    print(f"{output_path_residual_outputs}")

    print("!! Saving train loader only selected by g!!")
    save_results_selected_residual_by_pi(
        args,
        iteration,
        residual,
        glt,
        train_loader,
        args.selection_threshold,
        output_path_residual_outputs,
        device,
        mode="train",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )

    print("!! Saving val loader only selected by g!!")
    save_results_selected_residual_by_pi(
        args,
        iteration,
        residual,
        glt,
        val_loader,
        args.selection_threshold,
        output_path_residual_outputs,
        device,
        mode="val",
        higher_iter_params={
            "glt_list": glt_list
        }
    )


def predict_residual(args, residual, loader, output_path_model_outputs, mode="train"):
    print(f"Mode: {mode}")
    out_put_preds_residual = torch.FloatTensor().cuda()
    out_put_preds_bb = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (_, _, features_phi, bb_logits, _, _, _, y, y_one_hot, concepts) = data
                if torch.cuda.is_available():
                    features_phi = features_phi.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    bb_logits = bb_logits.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                residual_student_logits = residual(features_phi)

                out_put_preds_residual = torch.cat((out_put_preds_residual, residual_student_logits), dim=0)
                out_put_preds_bb = torch.cat((out_put_preds_bb, bb_logits), dim=0)
                out_put_target = torch.cat((out_put_target, y), dim=0)

                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    out_put_preds_residual = out_put_preds_residual.cpu()
    out_put_preds_bb = out_put_preds_bb.cpu()
    out_put_target = out_put_target.cpu()

    print(f"out_put_preds_residual size: {out_put_preds_residual.size()}")
    print(f"out_put_preds_bb size: {out_put_preds_bb.size()}")
    print(f"out_put_target size: {out_put_target.size()}")

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_residual.pt"),
        tensor_to_save=out_put_preds_residual
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_bb.pt"),
        tensor_to_save=out_put_preds_bb
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_target.pt"),
        tensor_to_save=out_put_target
    )


def save_results_selected_residual_by_pi(
        args,
        iteration,
        residual,
        glt_model,
        data_loader,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params
):
    glt_list = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]

    mask_by_pi = torch.FloatTensor().cuda()
    positive_selected = 0
    negative_selected = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for batch_id, data in enumerate(data_loader):
                (_, _, features_phi, bb_logits, _, proba_concept_x, _, y, y_one_hot, concepts) = data
                if torch.cuda.is_available():
                    features_phi = features_phi.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    bb_logits = bb_logits.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                out_class, out_select, out_aux = glt_model(proba_concept_x)
                residual_student_logits = residual(features_phi)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                if iteration == 1:
                    idx_selected = (out_select < 0.5).nonzero(as_tuple=True)[0]
                else:
                    condition = torch.full(pi_list[0].size(), True).to(device)
                    for proba in pi_list:
                        condition = condition & (proba < selection_threshold)
                    idx_selected = (condition & (out_select < selection_threshold)).nonzero(as_tuple=True)[0]

                if idx_selected.size(0) > 0:
                    positive_selected += y[idx_selected].sum(dim=0).item()
                    negative_selected += y[idx_selected].size(0) - y[idx_selected].sum(dim=0).item()

                condition = get_selected_idx_for_residual(
                    iteration, out_select, selection_threshold, device, pi_list
                )
                mask = torch.where(
                    condition,
                    torch.ones_like(out_select),
                    torch.zeros_like(out_select),
                )
                mask_by_pi = torch.cat((mask_by_pi, mask), dim=0)
                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    mask_by_pi = mask_by_pi.cpu()

    print("Output sizes: ")
    print(f"mask_by_pi size: {mask_by_pi.size()}")
    print(f"Selected by Residual: {mask_by_pi.sum(dim=0)}")
    print(f"n_pos_pred_residual: {positive_selected} || n_neg_pred_residual: {negative_selected} ")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_mask_by_pi.pt"), tensor_to_save=mask_by_pi
    )


def get_selected_idx_for_residual(iteration, selection_out, selection_threshold, device, prev_selection_outs=None):
    if iteration == 1:
        return selection_out < selection_threshold
    else:
        condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        for proba in prev_selection_outs:
            condition = condition & (proba < selection_threshold)
        return condition & (selection_out < selection_threshold)


def test_explainer(args, root, iteration, disease_folder):
    print(f"Testing the explainer for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None

    chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", disease_folder, root)
    output_path_explainer = os.path.join(args.output, args.dataset, "explainer", disease_folder, root)
    log_path_explainer = os.path.join(args.logs, args.dataset, "explainer", disease_folder, args.metric, )

    g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "g", "selected", args.metric)
    g_tb_logs_path = os.path.join(log_path_explainer, root)
    g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "g", "selected", args.metric)

    output_path_model_outputs = os.path.join(g_output_path, "model_outputs")
    output_path_g_outputs = os.path.join(g_output_path, "g_outputs")

    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_g_outputs, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))

    device = utils.get_device()
    print(f"Device: {device}")

    print("######################## Paths ###########################")
    print(g_chk_pt_path)
    print(g_output_path)
    print(root)
    print("######################## Paths ###########################")
    start = time.time()
    dataset_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        args.dataset_folder_concepts,
        args.arch,
        disease_folder,
        "dataset_g",
    )
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print(f"Length concept_names: {len(args.concept_names)}")
    if args.iter == 1:
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )

        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="val",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )
        test_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )

    else:
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        bb_logits_path = os.path.join(
            args.output, args.dataset, "explainer", args.selected_obs[0], args.prev_chk_pt_explainer_folder[-1],
            f"iter{iteration - 1}", "residual", "selected", args.metric, "model_outputs"
        )

        print(f"===> bb_logits_path: {bb_logits_path}")

        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )
        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="val",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )

        test_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    glt_list = []
    residual = None
    if iteration > 1:
        glt_list = get_glts(iteration, args, device, disease_folder, dataset="mimic_cxr")

    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi, args.concept_names, args.labels, args.hidden_nodes, args.conceptizator,
        args.temperature_lens
    ).to(device)

    model.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    model.eval()

    print("======>> Save overall whole model outputs ")

    print(f"{output_path_model_outputs}")
    print(f"{output_path_g_outputs}")

    print("!! Saving test loader only selected by g!!")
    predict(args, model, test_loader, output_path_model_outputs, mode="test")
    save_results_selected_by_pi(
        args,
        iteration,
        model,
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
    predict(args, model, train_loader, output_path_model_outputs, mode="train")
    save_results_selected_by_pi(
        args,
        iteration,
        model,
        train_loader,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="train",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )

    print("!! Saving val loader only selected by g!!")
    predict(args, model, val_loader, output_path_model_outputs, mode="val")
    save_results_selected_by_pi(
        args,
        iteration,
        model,
        val_loader,
        args.selection_threshold,
        output_path_g_outputs,
        device,
        mode="val",
        higher_iter_params={
            "glt_list": glt_list,
            "residual": residual
        }
    )

    print(f"{output_path_g_outputs}")


def save_results_selected_by_pi(
        args,
        iteration,
        model,
        data_loader,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params,
        output_path_model_outputs=None
):
    glt_list = None
    residual = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]
        residual = higher_iter_params["residual"]

    mask_by_pi = torch.FloatTensor().cuda()
    gt_tensor = torch.FloatTensor().cuda()
    pred_tensor = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for batch_id, data in enumerate(data_loader):
                (
                    bb_logits,
                    logits_concept_x,
                    proba_concept_x,
                    _,
                    y,
                    y_one_hot,
                    _
                ) = data
                if torch.cuda.is_available():
                    bb_logits = bb_logits.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                prediction_out, selection_out, auxiliary_out, concept_mask, \
                alpha, alpha_norm, conceptizator = model(proba_concept_x, test=True)

                condition = get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, pi_list)
                mask = torch.where(
                    condition,
                    torch.ones_like(selection_out),
                    torch.zeros_like(selection_out),
                )
                mask_by_pi = torch.cat((mask_by_pi, mask), dim=0)
                gt_tensor = torch.cat((gt_tensor, y), dim=0)
                pred_tensor = torch.cat((pred_tensor, prediction_out), dim=0)
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm
                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    gt_tensor = gt_tensor
    pred_tensor_class = pred_tensor.argmax(dim=1)
    pred_tensor_proba = pred_tensor[:, 1]
    print(gt_tensor.size(), pred_tensor_class.size(), pred_tensor_proba.size())
    print(f"All acc: {(gt_tensor == pred_tensor_class).sum(dim=0) / gt_tensor.size(0)}")
    print(f"All auroc: {utils.compute_AUC(gt_tensor, pred_tensor_proba)}")

    sel = mask_by_pi
    print(mask_by_pi.size())
    g_gt = torch.masked_select(gt_tensor.view(-1, 1), sel.bool())
    g_pred_tensor_class = torch.masked_select(pred_tensor_class.view(-1, 1), sel.bool())
    g_pred_tensor_proba = torch.masked_select(pred_tensor_proba.view(-1, 1), sel.bool())

    # prediction_result = g_pred_tensor_class.argmax(dim=1)
    # h_rjc = torch.masked_select(prediction_result, mask_by_pi.bool())
    # t_rjc = torch.masked_select(gt_tensor, mask_by_pi.bool())
    # t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
    # f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum()
    #           )
    #
    # acc = float(t / (t + f + 1e-12)) * 100

    print(g_gt.size(), g_pred_tensor_class.size(), g_pred_tensor_proba.size())
    print(f"g acc: {(g_gt == g_pred_tensor_class).sum(dim=0) / g_gt.size(0)}")
    print(f"g auroc: {utils.compute_AUC(g_gt, g_pred_tensor_proba)}")

    mask_by_pi = mask_by_pi.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    print("Output sizes: ")
    print(f"mask_by_pi size: {mask_by_pi.size()}")

    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_mask_by_pi.pt"), tensor_to_save=mask_by_pi
    )

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
    # print("XXXXXX" * 20)
    # test_out_put_class_pred = torch.load(
    #     os.path.join(output_path_model_outputs, "test_out_put_class_pred.pt")
    # )
    # test_out_put_target = torch.load(
    #     os.path.join(output_path_model_outputs, "test_out_put_target.pt")
    # )
    # prediction_result = test_out_put_class_pred.argmax(dim=1)
    #
    # s = mask_by_pi.view(-1, 1)
    # sel = torch.cat((s, s), dim=1)
    # test_out_put_class_pred_sel = torch.masked_select(test_out_put_class_pred, sel.bool()).view(-1, 2).argmax(dim=1)
    # cov = test_out_put_class_pred_sel.size(0) / test_out_put_class_pred.size(0)
    # if test_out_put_class_pred_sel.size(0) / test_out_put_class_pred.size(0) < 0.70:
    #     print(f"cov: {cov}")
    #     print(f"# +ve samples: {torch.sum(test_out_put_class_pred_sel == 1)}/"
    #           f"{torch.sum(test_out_put_target == 1)}")
    #
    #     return torch.sum(test_out_put_class_pred_sel == 1)
    # else:
    #     return 0


def get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, prev_selection_outs=None):
    if iteration == 1:
        return selection_out >= selection_threshold
    else:
        condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        for proba in prev_selection_outs:
            condition = condition & (proba < selection_threshold)
        return condition & (selection_out >= selection_threshold)


def predict(args, model, loader, output_path, mode):
    out_put_sel_proba = torch.FloatTensor().cuda()
    out_put_class_bb = torch.FloatTensor().cuda()
    out_put_class = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()
    proba_concept = torch.FloatTensor().cuda()
    attributes_gt = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    bb_logits,
                    logits_concept_x,
                    proba_concept_x,
                    attributes_gt_x,
                    y,
                    y_one_hot,
                    concepts
                ) = data
                if torch.cuda.is_available():
                    bb_logits = bb_logits.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    attributes_gt_x = attributes_gt_x.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                out_class, out_select, out_aux = model(proba_concept_x)
                out_put_sel_proba = torch.cat((out_put_sel_proba, out_select), dim=0)
                out_put_class_bb = torch.cat((out_put_class_bb, bb_logits), dim=0)
                out_put_class = torch.cat((out_put_class, out_class), dim=0)
                out_put_target = torch.cat((out_put_target, y), dim=0)
                proba_concept = torch.cat((proba_concept, proba_concept_x), dim=0)
                attributes_gt = torch.cat((attributes_gt, attributes_gt_x), dim=0)

                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    out_put_sel_proba = out_put_sel_proba.cpu()
    out_put_class_pred = out_put_class.cpu()
    out_put_class_bb_pred = out_put_class_bb.cpu()
    out_put_target = out_put_target.cpu()
    proba_concept = proba_concept.cpu()
    attributes_gt = attributes_gt.cpu()

    print(f"out_put_sel_proba size: {out_put_sel_proba.size()}")
    print(f"out_put_class_pred size: {out_put_class_pred.size()}")
    print(f"out_put_class_bb_pred size: {out_put_class_bb_pred.size()}")
    print(f"out_put_target size: {out_put_target.size()}")
    print(f"proba_concept size: {proba_concept.size()}")
    print(f"attributes_gt size: {attributes_gt.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_out_put_sel_proba.pt"), tensor_to_save=out_put_sel_proba
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_out_put_class_bb_pred.pt"), tensor_to_save=out_put_class_bb_pred
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_out_put_class_pred.pt"), tensor_to_save=out_put_class_pred
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_out_put_target.pt"), tensor_to_save=out_put_target
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_proba_concept.pt"), tensor_to_save=proba_concept
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_attributes_gt.pt"), tensor_to_save=attributes_gt
    )


def train_glt_icml(args):
    print("###" * 30)
    print("================== Training for ICML rebuttal ================== ")
    print("###" * 30)

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    hidden_layers = ""
    for hl in args.hidden_nodes:
        hidden_layers += str(hl)

    root = f"{args.arch}_{args.bs}_lr_{args.lr}_{args.optim}_temperature-lens_{args.temperature_lens}" \
           f"_cov_{args.cov}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}" \
           f"_lm_{args.lm}_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}" \
           f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{hidden_layers}" \
           f"_input-size-pi_{args.input_size_pi}_layer_{args.layer}"

    print(root)
    device = utils.get_device()
    print(f"Device: {device}")

    iteration = args.iter
    cov = args.cov
    lr_explainer = args.lr

    print(f"iteration: {iteration}========================>>")
    # iter 1
    # print("Training G")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.expert_to_train == "explainer":
        train_explainer(args, cov, lr_explainer, root, iteration, disease_folder)
    elif args.expert_to_train == "residual":
        train_residual(args, root, iteration, disease_folder)


def train_residual(args, root, iteration, disease_folder):
    chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", disease_folder, root)
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", disease_folder, root)
    output_path_explainer = os.path.join(args.output, args.dataset, "explainer", disease_folder, root)
    log_path_explainer = os.path.join(args.logs, args.dataset, "explainer", disease_folder, args.metric, )

    g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "g", "selected", args.metric)
    residual_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "residual", "selected", args.metric)
    residual_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "residual", "selected", args.metric)
    residual_tb_logs_path = os.path.join(log_path_explainer, root)

    os.makedirs(residual_chk_pt_path, exist_ok=True)
    os.makedirs(residual_tb_logs_path, exist_ok=True)
    os.makedirs(residual_output_path, exist_ok=True)

    output_path_model_outputs = os.path.join(residual_output_path, "model_outputs")
    output_path_residual_outputs = os.path.join(residual_output_path, "residual_outputs")
    print(output_path_model_outputs)
    print(output_path_residual_outputs)
    os.makedirs(output_path_model_outputs, exist_ok=True)
    os.makedirs(output_path_residual_outputs, exist_ok=True)

    print(residual_output_path)
    pickle.dump(args, open(os.path.join(residual_output_path, "train_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")

    print("############## Paths ##############")
    print(residual_chk_pt_path)
    print(residual_tb_logs_path)
    print(residual_output_path)
    print(root)
    print("############## Paths ##############")

    args.N_labels = len(args.labels)
    bb = DenseNet121(args, layer=args.layer).to(device)
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", args.bb_chkpt_folder, args.arch, disease_folder)
    model_chk_pt = torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
    if "state_dict" in model_chk_pt:
        bb.load_state_dict(model_chk_pt['state_dict'])
    else:
        bb.load_state_dict(model_chk_pt)

    bb.eval()
    classifier_bb = list(bb.children())[-1]
    print(classifier_bb)
    glt_list = []
    prev_residual = None
    dataset_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        args.dataset_folder_concepts,
        args.arch,
        disease_folder,
        "dataset_g",
    )
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
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
    glt.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    glt.eval()

    residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if iteration == 1:
        residual.fc.weight = copy.deepcopy(bb.fc1.weight)
        residual.fc.bias = copy.deepcopy(bb.fc1.bias)
    else:
        glt_list = get_glts(iteration, args, device, disease_folder, dataset="mimic_cxr")
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        prev_chk_pt_path = args.prev_chk_pt_explainer_folder[-1]
        prev_residual_chk_pt_path = os.path.join(
            args.checkpoints, args.dataset, "explainer", args.selected_obs[0], prev_chk_pt_path, f"iter{iteration - 1}",
            "residual"
        )
        prev_residual = get_residual(iteration, args, prev_residual_chk_pt_path, device, dataset="mimic_cxr")
        residual.fc.weight = copy.deepcopy(prev_residual.fc.weight)
        residual.fc.bias = copy.deepcopy(prev_residual.fc.bias)

    optimizer = torch.optim.SGD(
        residual.parameters(), lr=args.lr_residual, momentum=args.momentum_residual,
        weight_decay=args.weight_decay_residual
    )
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    kd_Loss = KD_Residual_Loss(iteration, CE, KLDiv, T_KD=args.temperature_KD, alpha_KD=args.alpha_KD)

    start = time.time()
    if args.iter == 1:
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=args.workers, pin_memory=True
        )

        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    else:
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        bb_logits_path = os.path.join(
            args.output, args.dataset, "explainer", args.selected_obs[0], args.prev_chk_pt_explainer_folder[-1],
            f"iter{iteration - 1}", "residual", "selected", args.metric, "model_outputs"
        )
        print(f"=======>> {bb_logits_path}")
        # bb_logits_path = os.path.join(
        #     args.output, args.dataset, "explainer", root, disease_folder, f"iter{iteration - 1}", "residual",
        #     "model_outputs"
        # )
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")
    print("############### Paths ###############")
    best_auroc = 0
    n_class = 2
    logger = Logger_MIMIC_CXR(
        iteration, best_auroc, args.start_epoch, residual_chk_pt_path, residual_tb_logs_path, residual_output_path,
        train_loader, val_loader, n_class, model_type="residual", device=device
    )
    fit_residual(
        args,
        classifier_bb,
        iteration,
        glt,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        os.path.join(root, f"iter{iteration}", "residual"),
        output_path_model_outputs,
        output_path_residual_outputs,
        args.selection_threshold,
        device
    )


def fit_residual(
        args,
        classifier_bb,
        iteration,
        glt,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        run_id,
        output_path_model_outputs,
        output_path_residual_outputs,
        selection_threshold,
        device
):
    logger.begin_run(run_id)
    for epoch in range(args.epochs_residual):
        logger.begin_epoch()
        residual.train()
        avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        with tqdm(total=len(train_loader)) as t:
            # train_out_put_preds_residual = torch.FloatTensor().cuda()
            # train_out_put_preds_bb = torch.FloatTensor().cuda()
            # train_out_put_target = torch.FloatTensor().cuda()
            # train_mask_by_pi = torch.FloatTensor().cuda()
            for batch_id, data in enumerate(train_loader):
                (
                    _,
                    _,
                    train_features_phi,
                    train_bb_logits,
                    _,
                    train_proba_concept_x,
                    _,
                    train_y,
                    y_one_hot,
                    concepts
                ) = data

                if torch.cuda.is_available():
                    train_features_phi = train_features_phi.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                    train_bb_logits = train_bb_logits.cuda(args.gpu, non_blocking=True)
                    train_proba_concept_x = train_proba_concept_x.cuda(args.gpu, non_blocking=True)
                    train_y = train_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                feat_dims = train_features_phi.size(1)
                # train_bb_logits = classifier_bb(avgpool(train_features_phi).reshape(-1, feat_dims * 1 * 1))
                out_class, out_select, out_aux = glt(train_proba_concept_x)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_proba_concept_x)

                residual_student_logits = residual(train_features_phi)
                residual_teacher_logits = train_bb_logits - (out_select * out_class)
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

                # condition = get_selected_idx_for_residual(
                #     iteration, out_select, selection_threshold, device, pi_list
                # )
                # mask = torch.where(
                #     condition,
                #     torch.ones_like(out_select),
                #     torch.zeros_like(out_select),
                # )
                # train_mask_by_pi = torch.cat((train_mask_by_pi, mask), dim=0)
                # train_out_put_preds_residual = torch.cat((train_out_put_preds_residual, residual_student_logits), dim=0)
                # train_out_put_preds_bb = torch.cat((train_out_put_preds_bb, train_bb_logits), dim=0)
                # train_out_put_target = torch.cat((train_out_put_target, train_y), dim=0)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        # train_out_put_preds_residual = train_out_put_preds_residual.cpu()
        # train_out_put_preds_bb = train_out_put_preds_bb.cpu()
        # train_out_put_target = train_out_put_target.cpu()
        # train_mask_by_pi = train_mask_by_pi.cpu()
        #
        # save_outputs(
        #     train_out_put_preds_residual, train_out_put_preds_bb, train_out_put_target,
        #     train_mask_by_pi, output_path_model_outputs, output_path_residual_outputs, mode="train"
        # )

        residual.eval()
        positive_selected = 0
        negative_selected = 0
        positive_out_selected = 0
        negative_out_selected = 0

        # val_out_put_preds_residual = torch.FloatTensor().cuda()
        # val_out_put_preds_bb = torch.FloatTensor().cuda()
        # val_out_put_target = torch.FloatTensor().cuda()
        # val_mask_by_pi = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    (
                        _,
                        _,
                        val_features_phi,
                        val_bb_logits,
                        _,
                        val_proba_concept_x,
                        _,
                        val_y,
                        y_one_hot,
                        concepts
                    ) = data
                    if torch.cuda.is_available():
                        val_features_phi = val_features_phi.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
                        val_bb_logits = val_bb_logits.cuda(args.gpu, non_blocking=True)
                        val_proba_concept_x = val_proba_concept_x.cuda(args.gpu, non_blocking=True)
                        val_y = val_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                    feat_dims = val_features_phi.size(1)
                    # val_bb_logits = classifier_bb(avgpool(val_features_phi).reshape(-1, feat_dims * 1 * 1))
                    out_class, out_select, out_aux = glt(val_proba_concept_x)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_proba_concept_x)

                    residual_student_logits = residual(val_features_phi)
                    residual_teacher_logits = val_bb_logits - (out_select * out_class)

                    # condition = get_selected_idx_for_residual(
                    #     iteration, out_select, selection_threshold, device, pi_list
                    # )
                    # mask = torch.where(
                    #     condition,
                    #     torch.ones_like(out_select),
                    #     torch.zeros_like(out_select),
                    # )
                    # val_mask_by_pi = torch.cat((val_mask_by_pi, mask), dim=0)
                    # val_out_put_preds_residual = torch.cat((val_out_put_preds_residual, residual_student_logits), dim=0)
                    # val_out_put_preds_bb = torch.cat((val_out_put_preds_bb, val_bb_logits), dim=0)
                    # val_out_put_target = torch.cat((val_out_put_target, val_y), dim=0)

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=val_y,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    total_val_loss = loss_dict["KD_risk"]

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, residual_student_logits, val_y, val_bb_logits)
                    logger.track_total_val_correct_per_epoch(residual_student_logits, val_y)

                    if iteration == 1:
                        idx_selected = (out_select < 0.5).nonzero(as_tuple=True)[0]
                    else:
                        condition = torch.full(pi_list[0].size(), True).to(device)
                        for proba in pi_list:
                            condition = condition & (proba < selection_threshold)
                        idx_selected = (condition & (out_select < selection_threshold)).nonzero(as_tuple=True)[0]

                    if idx_selected.size(0) > 0:
                        positive_selected += val_y[idx_selected].sum(dim=0).item()
                        negative_selected += val_y[idx_selected].size(0) - val_y[idx_selected].sum(dim=0).item()
                        out_class_positive_batch = residual_student_logits[idx_selected].argmax(dim=1).sum(dim=0).item()
                        out_class_negative_batch = residual_student_logits[idx_selected].size(
                            0) - out_class_positive_batch
                        positive_out_selected += out_class_positive_batch
                        negative_out_selected += out_class_negative_batch
                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)
                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # evaluate residual for correctly selected samples (pi < 0.5)
        # should be higher
        logger.evaluate_g_correctly(args.selection_threshold, expert="residual")
        #
        # # evaluate residual for correctly rejected samples (pi >= 0.5)
        # # should be lower
        logger.evaluate_g_incorrectly(args.selection_threshold, expert="residual")
        logger.evaluate_coverage_stats(args.selection_threshold, expert="residual")
        logger.end_epoch(residual, optimizer, track_explainer_loss=False, save_model_wrt_g_performance=True)

        # val_out_put_preds_residual = val_out_put_preds_residual.cpu()
        # val_out_put_preds_bb = val_out_put_preds_bb.cpu()
        # val_out_put_target = val_out_put_target.cpu()
        # val_mask_by_pi = val_mask_by_pi.cpu()
        #
        # save_outputs(
        #     val_out_put_preds_residual, val_out_put_preds_bb, val_out_put_target,
        #     val_mask_by_pi, output_path_model_outputs, output_path_residual_outputs, mode="test"
        # )
        print(
            f"Epoch: [{epoch + 1}/{args.epochs_residual}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_Auroc (Entire set): {round(logger.val_auroc, 4)} || "
            f"Val_residual_Accuracy (pi < 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_residual_Auroc (pi < 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
            f"Val_BB_Auroc (pi < 0.5): {round(logger.val_bb_auroc, 4)} || "
            f"Val_residual_Incorrect_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)}(%) || "
            f"Val_residual_Incorrect_Auroc (pi >= 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
            f"Val_BB_Incorrect_Auroc (pi >= 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
            f"Best_residual_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)} || "
            f"n_pos_residual: {positive_selected} || "
            f"n_neg_residual: {negative_selected} || "
            f"n_pos_pred_residual: {positive_out_selected} ||"
            f"n_neg_pred_residual: {negative_out_selected} "
        )
    logger.end_run()


def save_outputs(
        out_put_preds_residual, out_put_preds_bb, out_put_target,
        mask_by_pi, output_path_model_outputs, output_path_residual_outputs, mode
):
    print(f"{mode} out_put_preds_residual size: {out_put_preds_residual.size()}")
    print(f"{mode} out_put_preds_bb size: {out_put_preds_bb.size()}")
    print(f"{mode} out_put_target size: {out_put_target.size()}")
    print(f"{mode} mask size: {mask_by_pi.size()}")
    print(f"Selected by Residual: {mask_by_pi.sum(dim=0)}")

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_residual.pt"),
        tensor_to_save=out_put_preds_residual
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_bb.pt"),
        tensor_to_save=out_put_preds_bb
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_target.pt"),
        tensor_to_save=out_put_target
    )
    utils.save_tensor(
        path=os.path.join(output_path_residual_outputs, f"{mode}_mask_by_pi.pt"), tensor_to_save=mask_by_pi
    )


def train_explainer(args, cov, lr_explainer, root, iteration, disease_folder):
    print(f"Training the explainer for iteration: {iteration}")
    chk_pt_explainer = None
    output_path_explainer = None
    log_path_explainer = None
    output_path_residual = None
    chk_pt_explainer = os.path.join(args.checkpoints, args.dataset, "explainer", disease_folder, root)
    output_path_explainer = os.path.join(args.output, args.dataset, "explainer", disease_folder, root)
    output_path_residual = os.path.join(args.output, args.dataset, "explainer", disease_folder)
    log_path_explainer = os.path.join(args.logs, args.dataset, "explainer", disease_folder, args.metric, )

    g_chk_pt_path = os.path.join(chk_pt_explainer, f"iter{iteration}", "g", "selected", args.metric)
    g_tb_logs_path = os.path.join(log_path_explainer, root)
    g_output_path = os.path.join(output_path_explainer, f"iter{iteration}", "g", "selected", args.metric)

    print("############### Paths ###############")
    print(g_chk_pt_path)
    print(g_tb_logs_path)
    print(g_output_path)
    print("############### Paths ###############")
    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)

    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))
    device = utils.get_device()
    print(f"Device: {device}")
    print("############## Paths ##############")
    print(g_chk_pt_path)
    print(g_tb_logs_path)
    print(g_output_path)
    print(root)
    print("############## Paths ##############")

    start = time.time()
    dataset_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        args.dataset_folder_concepts,
        args.arch,
        disease_folder,
        "dataset_g",
    )
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print("Concepts:")
    print(args.concept_names)
    print(f"Length of concepts: {len(args.concept_names)}")
    y = torch.load(os.path.join(dataset_path, "test_class_labels.pt")).squeeze().to(torch.long)
    # print(y.size())
    # print(y)
    np_y = y.numpy()
    np_class_sample_count = np.array([len(np.where(np_y == t)[0]) for t in np.unique(np_y)])
    weight = 1.0 / np_class_sample_count
    samples_weight = np.array([weight[t] for t in np_y])
    # print(np_class_sample_count)
    # print(samples_weight)
    # print(samples_weight[(y == 0).nonzero(as_tuple=True)[0].tolist()])
    # print(samples_weight[(y == 1).nonzero(as_tuple=True)[0].tolist()])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    if args.iter == 1:
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True,
            # sampler=sampler
        )

        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        soft_hard_filter = "soft" if args.soft == 'y' else "hard"
        bb_logits_path = os.path.join(
            args.output, args.dataset, "explainer", args.selected_obs[0], args.prev_chk_pt_explainer_folder[-1],
            f"iter{iteration - 1}", "residual", "selected", args.metric, "model_outputs"
        )
        print(f"=======>> bb_logits loaded from: {bb_logits_path}")
        train_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="train",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True,
            # sampler=sampler
        )
        val_dataset = Dataset_mimic_for_explainer(
            iteration=args.iter,
            mode="test",
            metric=args.metric,
            expert=args.expert_to_train,
            dataset_path=dataset_path,
            bb_logits_path=bb_logits_path
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
        )

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")
    lambda_lens = args.lambda_lens
    glt_list = []
    if iteration > 1:
        glt_list = get_glts(iteration, args, device, disease_folder, dataset="mimic_cxr")

    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)

    optimizer = None
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_explainer, momentum=0.9, weight_decay=5e-4)
    elif args.optim == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_explainer)

    CE = torch.nn.CrossEntropyLoss(reduction="none")
    print(args.selected_obs)
    cov_weight = get_cov_weights_mimic(args.selected_obs[0])
    print(f"Proportions of positive samples: {cov_weight}")
    KLDiv = torch.nn.KLDivLoss(reduction="none")
    selective_KD_loss = Selective_Distillation_Loss(
        iteration, CE, KLDiv, T_KD=args.temperature_KD, alpha_KD=args.alpha_KD,
        selection_threshold=args.selection_threshold, coverage=cov, dataset="mimic_cxr", lm=args.lm
    )
    best_auroc = 0
    n_class = 2
    logger = Logger_MIMIC_CXR(
        iteration, best_auroc, args.start_epoch, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader, val_loader,
        n_class, model_type="g", device=device
    )

    fit_g(
        args,
        cov,
        cov_weight,
        iteration,
        args.epochs,
        args.alpha,
        args.temperature_KD,
        args.alpha_KD,
        model,
        glt_list,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        logger,
        lambda_lens,
        os.path.join(root, f"iter{iteration}", "g"),
        args.selection_threshold,
        device
    )


def fit_g(
        args,
        cov,
        cov_weight,
        iteration,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        model,
        glt_list,
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
            for batch_id, data in enumerate(train_loader):
                (
                    train_bb_logits,
                    train_logits_concept_x,
                    train_proba_concept_x,
                    _,
                    train_y,
                    y_one_hot,
                    concepts
                ) = data
                if torch.cuda.is_available():
                    train_bb_logits = train_bb_logits.cuda(args.gpu, non_blocking=True)
                    train_proba_concept_x = train_proba_concept_x.cuda(args.gpu, non_blocking=True)
                    train_y = train_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                out_class, out_select, out_aux = model(train_proba_concept_x)
                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, train_proba_concept_x)

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
                    prev_selection_outs=pi_list
                )
                train_selective_loss = loss_dict["selective_loss"]
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
                logger.track_total_train_correct_per_epoch(out_class, train_y)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        model.eval()

        positive_selected = 0
        negative_selected = 0
        positive_out_selected = 0
        negative_out_selected = 0
        tot_pos = 0
        total_item = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    (
                        val_bb_logits,
                        val_logits_concept_x,
                        val_proba_concept_x,
                        _,
                        val_y,
                        y_one_hot,
                        concepts
                    ) = data

                    if torch.cuda.is_available():
                        val_bb_logits = val_bb_logits.cuda(args.gpu, non_blocking=True)
                        val_proba_concept_x = val_proba_concept_x.cuda(args.gpu, non_blocking=True)
                        val_y = val_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                    out_class, out_select, out_aux = model(val_proba_concept_x)

                    # tot_pos += out_class.argmax(dim=1).sum()
                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, val_proba_concept_x)

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
                        prev_selection_outs=pi_list
                    )

                    val_selective_loss = loss_dict["selective_loss"]

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

                    if iteration == 1:
                        idx_selected = (out_select >= 0.5).nonzero(as_tuple=True)[0]
                    else:
                        condition = torch.full(pi_list[0].size(), True).to(device)
                        for proba in pi_list:
                            condition = condition & (proba < selection_threshold)
                        idx_selected = (condition & (out_select >= selection_threshold)).nonzero(as_tuple=True)[0]

                    if idx_selected.size(0) > 0:
                        positive_selected += val_y[idx_selected].sum(dim=0).item()
                        negative_selected += val_y[idx_selected].size(0) - val_y[idx_selected].sum(dim=0).item()
                        out_class_positive_batch = out_class[idx_selected].argmax(dim=1).sum(dim=0).item()
                        out_class_negative_batch = out_class[idx_selected].size(0) - out_class_positive_batch
                        positive_out_selected += out_class_positive_batch
                        negative_out_selected += out_class_negative_batch

                    total_item += val_y.size(0)
                    total_val_loss = val_selective_loss + val_aux_loss

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, out_class, val_y, val_bb_logits)
                    logger.track_total_val_correct_per_epoch(out_class, val_y)

                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)

                    # ii = (val_y == 1).nonzero(as_tuple=True)[0]
                    # print(out_class[ii, :].argmax(dim=1).sum(dim=0))
                    # print(val_bb_logits[ii, :].argmax(dim=1).sum(dim=0))
                    # print(val_y[ii].sum(dim=0))
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
        logger.end_epoch(model, optimizer, track_explainer_loss=True, save_model_wrt_g_performance=True)
        print(
            f"Epoch: [{epoch + 1}/{epochs}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_Auroc (Entire set by G): {round(logger.val_auroc, 4)} || "
            f"Val_G_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_G_Auroc (pi >= 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
            f"Val_BB_Auroc (pi >= 0.5): {round(logger.val_bb_auroc, 4)} || "
            f"Val_G_Incorrect_Accuracy (pi < 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
            f"Val_G_Incorrect_Auroc (pi < 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
            f"Val_BB_Incorrect_Auroc (pi < 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
            f"Best_G_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)} || "
            # f"coverage_1: {(positive_selected + negative_selected) / total_item} || "
            f"n_pos_ground_truth_g: {positive_selected} || "
            f"n_neg_ground_truth_g: {negative_selected} ||"
            f"n_pos_pred_g: {positive_out_selected} ||"
            f"n_neg_pred_g: {negative_out_selected} "
        )
        print()
    logger.end_run()
