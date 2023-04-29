import json
import os
import pickle
import sys

import torch

import Completeness_and_interventions.concept_completeness_intervention_utils as cci
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)
import pandas as pd
import numpy as np
import utils


def merge_csv_from_experts(iterations, json_file, disease, fol_path_root, save_path, mode="test"):
    df_list = []
    for iteration in range(iterations):
        iteration += 1
        with open(json_file) as _file:
            paths = json.load(_file)
        root = paths[disease]["MoIE_paths"][f"iter{iteration}"]["base_path"]
        fol_path = f"{fol_path_root}/{root}/iter{iteration}/g/selected/auroc/FOLs"
        df = pd.read_csv(os.path.join(fol_path, f"{mode}_results_expert_{iteration}.csv"))
        df_list.append(df)

    df_master = pd.concat(df_list, ignore_index=True)
    df_master.to_csv(os.path.join(save_path, f"{mode}_master_FOL_results.csv"))
    print(f"Master csv is saved at: {save_path}")
    print(f"{mode}_df size: {df_master.shape}")


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


def expert_specific_outputs(
        mask_by_pi, out_put_g_pred, out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts
):
    selected_ids = (mask_by_pi == 1).nonzero(as_tuple=True)[0]
    selected_ids_list = selected_ids.tolist()
    out_put_g_pred_expert = out_put_g_pred[selected_ids_list]
    out_put_bb_pred_expert = out_put_bb_pred[selected_ids_list]
    out_put_target_expert = out_put_target[selected_ids_list]
    proba_concepts_expert = proba_concepts[selected_ids_list]
    ground_truth_concepts_expert = ground_truth_concepts[selected_ids_list]
    return (
        out_put_g_pred_expert, out_put_bb_pred_expert, out_put_target_expert, proba_concepts_expert,
        ground_truth_concepts_expert
    )


def compute_performance_metrics(
        out_put_class_pred, out_put_class_bb_pred, out_put_target, mask_by_pi, args, get_masked_output=False,
        domain_transfer=False
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

    if not domain_transfer:
        out_put_bb_logits_rjc = torch.masked_select(out_put_class_bb_pred, sel.bool()).view(
            -1, 2
        )

        proba_bb = torch.nn.Softmax(dim=1)(out_put_bb_logits_rjc)
        bb_auroc, bb_aurpc = utils.compute_AUC(gt=t_rjc, pred=proba_bb[:, 1])
    else:
        bb_auroc, bb_aurpc = 0, 0
        proba_bb = torch.randn(proba_g.size())

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
    print(f"Length all concepts: {len(all_concept_names)}")
    # print(args.lr)
    # print(args.cov)
    # print(args.prev_chk_pt_explainer_folder)
    # print(args.checkpoint_model)
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


def get_outputs(iteration, args, output_path, dataset_path, mode="test", domain_transfer=False):
    g_outputs = "g_outputs" if args.model == "MoIE" else ""
    model_outputs = "model_outputs" if args.model == "MoIE" else ""
    print("############################################")
    print(f"ite: {iteration} ==========================>")
    print("############################################")
    print(os.path.join(output_path, g_outputs))
    tensor_alpha = torch.load(os.path.join(output_path, g_outputs, "test_tensor_alpha.pt"))
    tensor_alpha_norm = torch.load(os.path.join(output_path, g_outputs, "test_tensor_alpha_norm.pt"))
    tensor_concept_mask = torch.load(os.path.join(output_path, g_outputs, "test_tensor_concept_mask.pt"))

    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print("\n")

    if args.model == "MoIE":
        test_mask_by_pi = torch.load(os.path.join(output_path, g_outputs, f"{mode}_mask_by_pi.pt")).view(-1)
        print(f"{mode}_mask_by_pi size: {test_mask_by_pi.size()}")
    else:
        test_mask_by_pi = None

    out_put_g_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_class_pred.pt"))
    out_put_target = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_target.pt"))
    proba_concepts = torch.load(os.path.join(output_path, model_outputs, f"{mode}_proba_concept.pt"))
    if domain_transfer:
        ground_truth_concepts = None
    else:
        ground_truth_concepts = torch.load(os.path.join(output_path, model_outputs, f"{mode}_attributes_gt.pt"))
        print(f"{mode}_ground_truth_concepts size: {ground_truth_concepts.size()}")
    concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))

    if (args.model == "MoIE" or args.model == "Baseline_PostHoc") and not domain_transfer:
        out_put_bb_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_class_bb_pred.pt"))
        print(f"{mode}_out_put_class_bb_pred size: {out_put_bb_pred.size()}")
    else:
        out_put_bb_pred = None

    print(f"{mode}_out_put_target size: {out_put_target.size()}")
    print(f"{mode}_out_put_class_pred size: {out_put_g_pred.size()}")
    print(f"{mode}_concepts size: {proba_concepts.size()}")
    print(f"# concepts: {len(concept_names)}")
    print("\n")
    return (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, test_mask_by_pi, out_put_g_pred,
        out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts, concept_names
    )


def get_residuals(iteration, args, output_path, dataset_path, mode="test"):
    g_outputs = "residual_outputs" if args.model == "MoIE" else ""
    model_outputs = "model_outputs" if args.model == "MoIE" else ""
    print("############################################")
    print(f"ite: {iteration} ==========================>")
    print("############################################")
    print(os.path.join(output_path, g_outputs))
    if args.model == "MoIE":
        test_mask_by_pi = torch.load(os.path.join(output_path, g_outputs, f"{mode}_mask_by_pi.pt")).view(-1)
        print(f"{mode}_mask_by_pi size: {test_mask_by_pi.size()}")
    else:
        test_mask_by_pi = None

    out_put_g_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_preds_residual.pt"))
    out_put_target = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_target.pt"))
    concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print("\n")

    if args.model == "MoIE" or args.model == "Baseline_PostHoc":
        out_put_bb_pred = torch.load(os.path.join(output_path, model_outputs, f"{mode}_out_put_preds_bb.pt"))
        print(f"{mode}_out_put_class_bb_pred size: {out_put_bb_pred.size()}")
    else:
        out_put_bb_pred = None
    print(f"{mode}_out_put_target size: {out_put_target.size()}")
    print(f"{mode}_out_put_class_pred size: {out_put_g_pred.size()}")
    print(f"# concepts: {len(concept_names)}")
    print("\n")

    return test_mask_by_pi, out_put_g_pred, out_put_bb_pred, out_put_target, concept_names


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
        args.checkpoints, args.dataset, "soft_concepts/seed_0/", "explainer", disease, root, f"iter{iteration}",
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
        model_type="explainer"
):
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    y_hat_bb = test_tensor_preds_bb[idx].argmax(dim=0) if test_tensor_preds_bb is not None else torch.tensor(0)
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
        "all_concept_gt": test_ground_truth_concepts[idx].tolist() if test_ground_truth_concepts is not None else [],
        "concept_ids_in_explanations": mask_indxs.tolist(),
        "concept_names_in_explanations": concept_names_in_explanations,
        "g_pred": y_hat.item(),
        "g_pred_logit": test_tensor_preds[idx].tolist(),
        "bb_pred": y_hat_bb.item(),
        "ground_truth": target_class.item(),
        "expert_id": iteration,
        "raw_explanations": explanations,
        "actual_explanations": explanation_complete,
        "impression_from_report": df_master_sel.iloc[idx]["impression"] if df_master_sel is not None else "Domain Transfer",
        "findings_from_report": df_master_sel.iloc[idx]["findings"] if df_master_sel is not None else "Domain Transfer",
    }


def get_g_outputs(
        iterations, json_file, output, disease, _seed, dataset_path, domain_transfer=False, tot_samples=0, cov=None,
        initialize_w_mimic="n", train_phi="n"
):
    out_put_target_g = torch.FloatTensor()
    out_put_preds_g = torch.FloatTensor()
    out_put_preds_bb = torch.FloatTensor()

    for iteration in range(iterations):
        iteration += 1

        with open(json_file) as _file:
            paths = json.load(_file)
        root = paths[disease]["MoIE_paths"][f"iter{iteration}"]["base_path"]
        print(root)

        if domain_transfer:
            output_path = f"{output}/{root}_sample_{tot_samples}_train_phi_{train_phi}/iter{iteration}/g/selected/auroc" \
                if train_phi == "y" else f"{output}/{root}_sample_{tot_samples}/iter{iteration}/g/selected/auroc"
            if len(cov) > 0:
                output_path = f"{output_path}_cov_{cov[iteration - 1]}"
            if initialize_w_mimic == "y":
                output_path = f"{output_path}_initialize_w_mimic_{initialize_w_mimic}"
        else:
            output_path = f"{output}/{root}/iter{iteration}/g/selected/auroc"
        device, configs = setup(output_path)
        configs.model = "MoIE"
        (
            _, _, _, test_mask_by_pi, test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, _, _, _
        ) = get_outputs(iteration, configs, output_path, dataset_path, domain_transfer=domain_transfer)
        target_selected, g_pred_selected, bb_pred_selected = compute_performance_metrics(
            test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, test_mask_by_pi, configs,
            get_masked_output=True, domain_transfer=domain_transfer
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
    if domain_transfer:
        output_residual = f"{output}/{root_residual}_sample_{tot_samples}_train_phi_{train_phi}/iter{final_iteration}/residual/selected/auroc" \
            if train_phi == "y" else f"{output}/{root_residual}_sample_{tot_samples}/iter{final_iteration}/residual/selected/auroc"
        if len(cov) > 0:
            output_residual = f"{output_residual}_cov_{cov[final_iteration-1]}"
        if initialize_w_mimic == "y":
            output_residual = f"{output_residual}_initialize_w_mimic_{initialize_w_mimic}"
    else:
        output_residual = f"{output}/{root_residual}/iter{final_iteration}/residual/selected/auroc"

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


def get_expert_specific_outputs(_iter, args, json_file, output, dataset_path, save_path_top_K, mode):
    all_mask_alpha = torch.BoolTensor()
    all_proba_concepts = torch.FloatTensor()
    all_ground_truth_concepts = torch.FloatTensor()
    all_ground_truth_labels = torch.FloatTensor()
    all_preds_g = torch.FloatTensor()
    all_preds_bb = torch.FloatTensor()

    for _iter in range(args.iterations):
        _iter += 1
        print(f"#####" * 20)
        print(f"iteration: {_iter} || topK: {args.topK}")
        print(f"############################### iteration: {_iter} start ###############################")
        with open(json_file) as _file:
            paths = json.load(_file)

        output_path = None
        if args.model == "MoIE":
            root = paths[args.disease]["MoIE_paths"][f"iter{_iter}"]["base_path"]
            output_path = f"{output}/{root}/iter{_iter}/g/selected/auroc"
        (
            _, tensor_alpha_norm, _, mask_by_pi, out_put_g_pred, out_put_bb_pred, out_put_target,
            proba_concepts, ground_truth_concepts, _
        ) = get_outputs(_iter, args, output_path, dataset_path, mode=mode)
        (
            out_put_g_pred_expert, out_put_bb_pred_expert, out_put_target_expert,
            proba_concepts_expert, ground_truth_concepts_expert
        ) = expert_specific_outputs(
            mask_by_pi, out_put_g_pred, out_put_bb_pred, out_put_target, proba_concepts,
            ground_truth_concepts
        )

        mask_alpha = cci.get_concept_masks_top_k(
            out_put_g_pred_expert, tensor_alpha_norm, proba_concepts_expert, args.topK
        )

        print("    Expert specific output        ")
        print(f"{mode} mask size: {mask_alpha.size()}")
        print(f"{mode} g pred size: {out_put_g_pred_expert.size()}")
        print(f"{mode} bb pred size: {out_put_bb_pred_expert.size()}")
        print(f"{mode} target size: {out_put_target_expert.size()}")
        print(f"{mode} concept proba size: {proba_concepts_expert.size()}")
        print(f"{mode} ground truth size: {ground_truth_concepts_expert.size()}")

        all_mask_alpha = torch.cat((all_mask_alpha, mask_alpha), dim=0)
        all_proba_concepts = torch.cat((all_proba_concepts, proba_concepts_expert), dim=0)
        all_ground_truth_concepts = torch.cat((all_ground_truth_concepts, ground_truth_concepts_expert), dim=0)
        all_ground_truth_labels = torch.cat((all_ground_truth_labels, out_put_target_expert), dim=0)
        all_preds_g = torch.cat((all_preds_g, out_put_g_pred_expert), dim=0)
        all_preds_bb = torch.cat((all_preds_bb, out_put_bb_pred_expert), dim=0)

    print(f"############# Total {mode} size ###############")
    print(f"{mode} mask size: {all_mask_alpha.size()}")
    print(f"{mode} g pred size: {all_preds_g.size()}")
    print(f"{mode} bb pred size: {all_preds_bb.size()}")
    print(f"{mode} target size: {all_ground_truth_labels.size()}")
    print(f"{mode} concept proba size: {all_proba_concepts.size()}")
    print(f"{mode} ground truth size: {all_ground_truth_concepts.size()}")

    torch.save(all_mask_alpha, os.path.join(save_path_top_K, f"{mode}_all_mask_alpha.pt"))
    torch.save(all_preds_g, os.path.join(save_path_top_K, f"{mode}_all_preds_g.pt"))
    torch.save(all_preds_bb, os.path.join(save_path_top_K, f"{mode}_all_preds_bb.pt"))
    torch.save(all_ground_truth_labels, os.path.join(save_path_top_K, f"{mode}_all_ground_truth_labels.pt"))
    torch.save(all_proba_concepts, os.path.join(save_path_top_K, f"{mode}_all_proba_concepts.pt"))
    torch.save(all_ground_truth_concepts, os.path.join(save_path_top_K, f"{mode}_all_ground_truth_concepts.pt"))

    print(save_path_top_K)


def get_residual_outputs(iterations, json_file, output, disease, _seed, dataset_path):
    results_arr = []
    for iteration in range(iterations):
        iteration += 1
        print(f"**************************** {iteration} ****************************")
        with open(json_file) as _file:
            paths = json.load(_file)
        root = paths[disease]["MoIE_paths"][f"iter{iteration}"]["base_path"]
        print(root)
        output_path = f"{output}/{root}/iter{iteration}/residual/selected/auroc"
        device, configs = setup(output_path, residual=True)
        configs.model = "MoIE"

        (
            test_mask_by_pi, test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, _
        ) = get_residuals(iteration, configs, output_path, dataset_path)
        results = compute_performance_metrics(
            test_out_put_g_pred, test_out_put_bb_pred, test_out_put_target, test_mask_by_pi, configs,
            get_masked_output=False
        )

        print(f"Accuracy (g): {results['test_g_acc']}")
        print(f"Auroc (g): {results['test_g_auroc']}")
        print(f"Aurpc (g): {results['test_g_aurpc']}")
        print(f"Recall (g): {results['test_g_recall']}")
        print(f"Auroc BB (g): {results['test_bb_auroc']}")
        print(f"Empirical coverage (g): {results['test_emperical_coverage']}")
        results_arr.append(results)
    return results_arr
