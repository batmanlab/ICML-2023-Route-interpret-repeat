import argparse
import os
import pickle
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

import Completeness_and_interventions.concept_completeness_intervention_utils as cci

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')

    parser.add_argument('--arch', type=str, default="ViT-B_16", help='BB architecture')
    parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
    parser.add_argument('--iterations', default=6, type=int, help='iterations for MoIE')
    parser.add_argument('--top_K', nargs='+', default=[3, 5, 10, 15, 20, 25, 30], type=int, help='How many concepts?')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')
    return parser.parse_args()


def perform_interventions(args):
    # load data from outputs
    tensor_alpha, tensor_alpha_norm, tensor_concept_mask, test_tensor_conceptizator_concepts, \
    test_tensor_concepts, test_tensor_preds, test_tensor_y, \
    test_tensor_concepts_bool, _, _, _ = cci.load_saved_outputs(args, mode="test", get_features=False)

    # perform intervention
    alt_preds = cci.intervene_and_pred_alt_results(
        args, test_tensor_y, test_tensor_preds, tensor_alpha_norm, test_tensor_concepts
    )

    return alt_preds, test_tensor_y, test_tensor_preds


def compute_acc_and_save(pred_g_intervene, gt, pred_g, model_type):
    np_pred_g_intervene = pred_g_intervene.argmax(dim=1).numpy()
    np_gt = gt.numpy()
    np_pred_g = pred_g.argmax(dim=1).numpy()
    acc_g = accuracy_score(np_gt, np_pred_g) * 100
    acc_g_in = accuracy_score(np_gt, np_pred_g_intervene) * 100
    acc_drop = ((acc_g - acc_g_in) / acc_g) * 100
    acc = {
        "concepts": args.topK,
        "acc_g": acc_g,
        "acc_g_in": acc_g_in,
        "acc_drop": acc_drop
    }

    print(f"topk: {args.topK}, Accuracy_g: {acc['acc_g']} (%), Accuracy_g_in: {acc['acc_g_in']} (%)")
    save_path = os.path.join(args.output, args.dataset, model_type, args.arch, "intervene_concepts_results")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"concepts_{args.topK}.pkl"), 'wb') as file:
        pickle.dump(acc, file)
    print(save_path)
    print(save_path)

    return acc, save_path


def compute_auroc_and_save(pred_g_intervene, gt, pred_g, model_type):
    np_pred_g_intervene = pred_g_intervene[:, 1].numpy()
    np_gt = gt.numpy()
    np_pred_g = pred_g[:, 1].numpy()
    auroc_g = roc_auc_score(np_gt, np_pred_g)
    auroc_g_in = roc_auc_score(np_gt, np_pred_g_intervene)
    auroc_drop = ((auroc_g - auroc_g_in) / auroc_g) * 100
    results = {
        "concepts": args.topK,
        "auroc_g": auroc_g,
        "auroc_g_in": auroc_g_in,
        "auroc_drop": auroc_drop
    }

    print(f"topk: {args.topK}, Auroc_g: {results['auroc_g']}, Auroc_g_in: {results['auroc_g_in']}")
    save_path = os.path.join(args.output, args.dataset, model_type, args.arch, "intervene_concepts_results")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"concepts_{args.topK}.pkl"), 'wb') as file:
        pickle.dump(acc, file)
    print(save_path)
    print(save_path)

    return results, save_path


def intervene_moIE(args):
    args.json_file = os.path.join(args.base_path, "codebase", "Completeness_and_interventions", "paths_MoIE.json")
    gt = torch.FloatTensor()
    pred_g = torch.FloatTensor()
    pred_g_intervene = torch.FloatTensor()
    for _iter in range(args.iterations):
        _iter += 1
        print(f"#####" * 20)
        print(f"iteration: {_iter}")
        args.cur_iter = _iter
        alt_preds, test_tensor_y, test_tensor_preds = perform_interventions(args)
        print(alt_preds.size())
        pred_g_intervene = torch.cat((pred_g_intervene, alt_preds), dim=0)
        gt = torch.cat((gt, test_tensor_y), dim=0)
        pred_g = torch.cat((pred_g, test_tensor_preds), dim=0)

    # save
    return compute_acc_and_save(pred_g_intervene, gt, pred_g, model_type="explainer")


if __name__ == "__main__":
    args = config()
    acc_arr = []
    save_path = None
    for top_k in args.top_K:
        args.topK = top_k
        acc, save_path = intervene_moIE(args)
        acc_arr.append(acc)

    df = pd.DataFrame(acc_arr)
    df.to_csv(os.path.join(save_path, "intervention_important_concepts.csv"))
    print(df)
