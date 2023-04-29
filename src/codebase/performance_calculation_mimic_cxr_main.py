import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

import MIMIC_CXR.mimic_cxr_utils as FOL_mimic

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')

    parser.add_argument('--disease', type=str, default="effusion", help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--iterations', type=int, default="1", help='total number of iteration')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE or Baseline_CBM_logic or Baseline_PCBM_logic')
    parser.add_argument('--icml', default='n', type=str, help='ICML or MICCAI')

    return parser.parse_args()


def calculate_performance(disease, iterations, output, json_file, _seed, dataset_path):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    (
        out_put_target_g, out_put_preds_g, out_put_preds_bb, out_put_target_total, out_put_preds_total,
        out_put_preds_bb_total
    ) = FOL_mimic.get_g_outputs(
        iterations, json_file, output, disease, _seed, dataset_path
    )

    print("")
    print(f"Disease: {disease}, Sizes by MoIE: ")
    print(
        f"Size Target: {out_put_target_g.size()}, Pred_g: {out_put_preds_g.size()}, Pred bb: {out_put_preds_bb.size()}"
    )
    print("Sizes by MoIE + Residual: ")
    print(
        f"Size Target: {out_put_target_total.size()}, Pred_g: {out_put_preds_total.size()}, "
        f"Pred bb: {out_put_preds_bb_total.size()}"
    )
    print("")
    print("Output csv is saved at: ")
    print(dataset_path)
    results = FOL_mimic.compute_cumulative_performance(
        out_put_target_g, out_put_preds_g, out_put_preds_bb, out_put_target_total, out_put_preds_total,
        out_put_preds_bb_total
    )
    print(f"########################### {disease} ############################")
    print("\n>>>>>>>>>>>>>>> MOIE Results: <<<<<<<<<<<<<<<<<<<<<")
    print(f"Accuracy (g): {results['moie_acc']}")
    print(f"Auroc (g): {results['moie_auroc']}")
    print(f"Aurpc (g): {results['moie_aurpc']}")
    print(f"Recall (g): {results['moie_recall']}")
    print(f"Accuracy BB (g): {results['moie_bb_acc']}")
    print(f"Auroc BB (g): {results['moie_bb_auroc']}")
    print(f"Aurpc BB (g): {results['moie_bb_aurpc']}")
    print(f"Recall BB (g): {results['moie_bb_recall']}")
    print(f"Empirical coverage (g): {results['moie_emp_coverage']}")

    print("\n>>>>>>>>>>>>>>> MOIE + Residual Results: <<<<<<<<<<<<<<<<<<<<<")
    print(f"Accuracy (with residual): {results['moie_r_acc']}")
    print(f"Auroc (with residual): {results['moie_r_auroc']}")
    print(f"Aurpc (with residual): {results['moie_r_aurpc']}")
    print(f"Recall (with residual): {results['moie_r_recall']}")

    print("\n>>>>>>>>>>>>>>> Blackbox Results: <<<<<<<<<<<<<<<<<<<<<")
    print(f"Accuracy BB: {results['moie_r_bb_acc']}")
    print(f"Auroc BB: {results['moie_r_bb_auroc']}")
    print(f"Aurpc BB: {results['moie_r_bb_aurpc']}")
    print(f"Recall BB: {results['moie_r_bb_recall']}")
    print(f"##################################################################")
    return results, dataset_path


def main():
    args = config()
    _disease = args.disease
    _iters = args.iterations
    _seed = args.seed
    _output = args.output
    if args.icml == "y":
        print("=====================>>>>> Calculating performance for ICML paper <<<<<<=====================")
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/explainer/{_disease}"
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr_icml.json")
    else:
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"{_output}/mimic_cxr/soft_concepts/seed_{_seed}/explainer/{_disease}"
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr.json")

    results, path_to_dump = calculate_performance(_disease, _iters, output, json_file, _seed, dataset_path)
    print(results)
    with open(os.path.join(path_to_dump, f"total_results_{_disease}.json"), "w") as outfile:
        json.dump(results, outfile)

    pickle.dump(results, open(os.path.join(path_to_dump, f"total_results_{_disease}.pkl"), "wb"))
    print(path_to_dump)


if __name__ == '__main__':
    main()
