import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))
import Completeness_and_interventions.concept_completeness_intervention_utils as cci


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')

    parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
    parser.add_argument('--arch', type=str, default="ResNet101", help='Arch')
    parser.add_argument('--iterations', type=int, default="6", help='iteration')
    parser.add_argument('--top_K', nargs='+', default=[3, 5, 10, 15, 20, 25, 30], type=int, help='How many concepts?')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')

    return parser.parse_args()


def create_dataset_for_completeness_moIE(args):
    args.json_file = os.path.join(args.base_path, "codebase", "Completeness_and_interventions", "paths_MoIE.json")
    save_path = os.path.join(
        args.output, args.dataset, "completeness", args.arch, "dataset", "moIE", f"concepts_topK_{args.topK}"
    )
    os.makedirs(save_path, exist_ok=True)

    all_train_tensor_fetaures = torch.FloatTensor()
    all_train_tensor_y = torch.FloatTensor()
    all_train_mask_alpha = torch.BoolTensor()

    all_val_tensor_fetaures = torch.FloatTensor()
    all_val_tensor_y = torch.FloatTensor()
    all_val_mask_alpha = torch.BoolTensor()

    all_test_tensor_fetaures = torch.FloatTensor()
    all_test_tensor_y = torch.FloatTensor()
    all_test_mask_alpha = torch.BoolTensor()

    for _iter in range(args.iterations):
        _iter += 1
        args.cur_iter = _iter
        print(f"#####" * 20)
        print(f"iteration: {_iter} || topK: {args.topK}")

        # Get test data
        if args.dataset == "cub" or args.dataset == "awa2" or args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC":
            (
                tensor_alpha, tensor_alpha_norm, tensor_concept_mask, test_tensor_conceptizator_concepts,
                test_tensor_concepts, test_tensor_preds, test_tensor_y,
                test_tensor_concepts_bool, test_tensor_features, _, test_full_output_path
            ) = cci.load_saved_outputs(args, mode="test")

            test_mask_alpha = cci.get_concept_masks_top_k(
                test_tensor_preds, tensor_alpha_norm, test_tensor_concepts, args.topK
            )

            all_test_tensor_fetaures = torch.cat((all_test_tensor_fetaures, test_tensor_features), dim=0)
            all_test_tensor_y = torch.cat((all_test_tensor_y, test_tensor_y), dim=0)
            all_test_mask_alpha = torch.cat((all_test_mask_alpha, test_mask_alpha), dim=0)

            print(f"Test mask size: {test_mask_alpha.size()}")
            torch.save(test_mask_alpha, os.path.join(test_full_output_path, f"test_mask_alpha_topK_{args.topK}.pt"))

        # Get val data
        if args.dataset == "cub":
            (
                tensor_alpha, tensor_alpha_norm, tensor_concept_mask, val_tensor_conceptizator_concepts,
                val_tensor_concepts, val_tensor_preds, val_tensor_y,
                val_tensor_concepts_bool, val_tensor_features, _, val_full_output_path
            ) = cci.load_saved_outputs(args, mode="val")

            val_mask_alpha = cci.get_concept_masks_top_k(
                val_tensor_preds, tensor_alpha_norm, val_tensor_concepts, args.topK
            )

            all_val_tensor_fetaures = torch.cat((all_val_tensor_fetaures, val_tensor_features), dim=0)
            all_val_tensor_y = torch.cat((all_val_tensor_y, val_tensor_y), dim=0)
            all_val_mask_alpha = torch.cat((all_val_mask_alpha, val_mask_alpha), dim=0)

            print(f"Val mask size: {val_mask_alpha.size()}")
            torch.save(val_mask_alpha, os.path.join(val_full_output_path, f"val_mask_alpha_topK_{args.topK}.pt"))
        # Get train data
        if args.dataset == "cub" or args.dataset == "HAM10k" or \
                args.dataset == "SIIM-ISIC" or args.dataset == "awa2":
            print("Getting train masks")
            (
                tensor_alpha, tensor_alpha_norm, tensor_concept_mask, train_tensor_conceptizator_concepts,
                train_tensor_concepts, train_tensor_preds, train_tensor_y,
                train_tensor_concepts_bool, train_tensor_features, _, train_full_output_path
            ) = cci.load_saved_outputs(args, mode="train")

            train_mask_alpha = cci.get_concept_masks_top_k(
                train_tensor_preds, tensor_alpha_norm, train_tensor_concepts, args.topK
            )

            all_train_tensor_fetaures = torch.cat((all_train_tensor_fetaures, train_tensor_features), dim=0)
            all_train_tensor_y = torch.cat((all_train_tensor_y, train_tensor_y), dim=0)
            all_train_mask_alpha = torch.cat((all_train_mask_alpha, train_mask_alpha), dim=0)

            print(f"Train mask size: {train_mask_alpha.size()}")
            torch.save(train_mask_alpha, os.path.join(train_full_output_path, f"train_mask_alpha_topK_{args.topK}.pt"))

    print("=====>>> Test size:")
    print(all_test_tensor_fetaures.size())
    print(all_test_tensor_y.size())
    print(all_test_mask_alpha.size())

    print("=====>>> Val size:")
    print(all_val_tensor_fetaures.size())
    print(all_val_tensor_y.size())
    print(all_val_mask_alpha.size())

    print("=====>>> Train size:")
    print(all_train_tensor_fetaures.size())
    print(all_train_tensor_y.size())
    print(all_train_mask_alpha.size())

    torch.save(all_train_tensor_fetaures, os.path.join(save_path, f"train_tensor_features.pt"))
    torch.save(all_train_tensor_y, os.path.join(save_path, f"train_tensor_y.pt"))
    torch.save(all_train_mask_alpha, os.path.join(save_path, f"train_mask_alpha.pt"))

    torch.save(all_val_tensor_fetaures, os.path.join(save_path, f"val_tensor_features.pt"))
    torch.save(all_val_tensor_y, os.path.join(save_path, f"val_tensor_y.pt"))
    torch.save(all_val_mask_alpha, os.path.join(save_path, f"val_mask_alpha.pt"))

    torch.save(all_test_tensor_fetaures, os.path.join(save_path, f"test_tensor_features.pt"))
    torch.save(all_test_tensor_y, os.path.join(save_path, f"test_tensor_y.pt"))
    torch.save(all_test_mask_alpha, os.path.join(save_path, f"test_mask_alpha.pt"))

    print(save_path)


def main():
    args = config()
    print(f"Get important concept masks: {args.dataset}")
    for top_k in args.top_K:
        args.topK = top_k
        create_dataset_for_completeness_moIE(args)


if __name__ == '__main__':
    main()
