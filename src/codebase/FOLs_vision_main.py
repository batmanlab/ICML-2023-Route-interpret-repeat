import argparse
import os
import pickle
import sys
import time

import Completeness_and_interventions.concept_completeness_intervention_utils as cci
import utils
from Explainer.Explanations_builder import build_FOLs

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
    parser.add_argument('--save_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/Plots/concepts',
                        help='path of the concepts to be saved')
    parser.add_argument('--arch', type=str, default="ViT-B_16", help='Architecture of the blackbox')
    parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
    parser.add_argument('--iterations', default=6, type=int, help='iterations for MoIE')
    return parser.parse_args()


def get_saved_outputs(args):
    # load data from outputs
    (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, train_tensor_conceptizator_concepts,
        train_tensor_concepts, train_tensor_preds, train_tensor_y, train_tensor_concepts_bool, _, _, _
    ) = cci.load_saved_outputs(args, mode="train", get_features=False)

    (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, test_tensor_conceptizator_concepts, test_tensor_concepts,
        test_tensor_preds, test_tensor_y, test_tensor_concepts_bool, _, _, _
    ) = cci.load_saved_outputs(args, mode="test", get_features=False)

    if args.dataset == "cub":
        (
            _, _, _, val_tensor_conceptizator_concepts,
            val_tensor_concepts,
            val_tensor_preds, val_tensor_y, val_tensor_concepts_bool, _, _, _
        ) = cci.load_saved_outputs(args, mode="val", get_features=False)
    else:
        val_tensor_conceptizator_concepts = test_tensor_conceptizator_concepts
        val_tensor_concepts = test_tensor_concepts
        val_tensor_preds = test_tensor_preds
        val_tensor_y = test_tensor_y
        val_tensor_concepts_bool = test_tensor_concepts_bool

    checkpoint, test_config = cci.checkpoint_paths(args)
    pkl = pickle.load(open(test_config, "rb"))
    device = utils.get_device()
    moIE = cci.get_model(pkl, args, device, checkpoint)

    return {
        "tensor_alpha": tensor_alpha,
        "tensor_alpha_norm": tensor_alpha_norm,
        "tensor_concept_mask": tensor_concept_mask,
        "train_tensor_conceptizator_concepts": train_tensor_conceptizator_concepts,
        "train_tensor_preds": train_tensor_preds,
        "train_tensor_y": train_tensor_y,
        "train_tensor_concepts_bool": train_tensor_concepts_bool,
        "train_tensor_concepts": train_tensor_concepts,
        "test_tensor_conceptizator_concepts": test_tensor_conceptizator_concepts,
        "test_tensor_concepts": test_tensor_concepts,
        "test_tensor_preds": test_tensor_preds,
        "test_tensor_y": test_tensor_y,
        "test_tensor_concepts_bool": test_tensor_concepts_bool,
        "val_tensor_conceptizator_concepts": val_tensor_conceptizator_concepts,
        "val_tensor_concepts": val_tensor_concepts,
        "val_tensor_preds": val_tensor_preds,
        "val_tensor_y": val_tensor_y,
        "val_tensor_concepts_bool": val_tensor_concepts_bool,
        "moIE": moIE,
        "pkl": pkl,
        "device": device
    }


def generate_FOL(args):
    args.json_file = os.path.join(args.base_path, "codebase", "Completeness_and_interventions", "paths_MoIE.json")
    for _iter in range(args.iterations):
        _iter += 1
        args.cur_iter = _iter
        print(
            f"******************************************* "
            f"iteration: {_iter} start"
            f"******************************************* "
        )
        args.cur_iter = _iter
        output_dict = get_saved_outputs(args)
        start = time.time()
        build_FOLs(output_dict, args)
        done = time.time()
        elapsed = done - start
        print("Time to complete this iteration: " + str(elapsed) + " secs")
        print(
            f"******************************************* "
            f"iteration: {_iter} end"
            f"******************************************* "
        )


if __name__ == "__main__":
    args = config()
    args.model = "MoIE"
    start = time.time()
    generate_FOL(args)
    done = time.time()
    elapsed = done - start
    print("Time to complete: " + str(elapsed) + " secs")
