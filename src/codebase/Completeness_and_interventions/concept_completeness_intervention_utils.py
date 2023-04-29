import json
import os
import pickle
import sys

import torch

import utils
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.explainer import Explainer

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)


def get_out_paths(args):
    with open(args.json_file) as _file:
        paths = json.load(_file)

    if args.model == "MoIE":
        base_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["base_path"]
        prev_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["prev_path"]
        output = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["output"]
        explainer = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["explainer_path"]
        files = paths[f"{args.dataset}_{args.arch}"]["files"]

        full_output_path = os.path.join(
            args.output, args.dataset, "explainer", base_path, prev_path, f"iter{args.cur_iter}",
            explainer, output
        )
        return files, full_output_path


def checkpoint_paths(args):
    with open(args.json_file) as _file:
        paths = json.load(_file)
    if args.model == "MoIE":
        base_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["base_path"]
        prev_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["prev_path"]
        checkpoint_file = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["checkpoint_g"]
        explainer = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["explainer_path"]
        checkpoint = os.path.join(
            args.checkpoints, args.dataset, "explainer", base_path, prev_path, f"iter{args.cur_iter}",
            explainer, checkpoint_file
        )
        test_config = os.path.join(
            args.output, args.dataset, "explainer", base_path, prev_path, f"iter{args.cur_iter}",
            explainer, "test_explainer_configs.pkl",
        )
        return checkpoint, test_config


def load_saved_outputs(args, mode, get_features=True, get_attrs=False):
    files, full_output_path = get_out_paths(args)
    print(f"=======>> {full_output_path}")
    x_to_bool = 0.5
    conceptizator_threshold = 0.5

    # model parameters
    tensor_alpha = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_alpha"]))
    tensor_alpha_norm = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_alpha_norm"]))
    tensor_concept_mask = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_concept_mask"]))

    # test
    tensor_conceptizator_concepts = torch.load(
        os.path.join(full_output_path, files[f"{mode}_tensor_conceptizator_concepts"]))
    tensor_concepts = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_concepts"]))
    tensor_preds = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_preds"]))
    tensor_y = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_y"]))

    if get_features:
        tensor_features = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_features"]))
    else:
        tensor_features = None

    if get_attrs:
        tensor_attrs = torch.load(os.path.join(full_output_path, files[f"{mode}_tensor_attributes"]))
    else:
        tensor_attrs = None

    tensor_concepts_bool = (tensor_concepts.cpu() > x_to_bool).to(torch.float)

    print("<< Model specific sizes >>")
    print(tensor_alpha.size())
    print(tensor_alpha_norm.size())
    print(tensor_concept_mask.size())

    print(f"\n << {mode} sizes >>")
    if tensor_features is not None:
        print(f"features: {tensor_features.size()}")

    print(f"concepts: {tensor_concepts.size()}")
    print(f"concepts_bool: {tensor_concepts_bool.size()}")
    print(f"preds (g): {tensor_preds.size()}")
    print(f"y: {tensor_y.size()}")
    print(f"conceptizator_concepts: {tensor_conceptizator_concepts.size()}")

    return (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, tensor_conceptizator_concepts,
        tensor_concepts, tensor_preds, tensor_y, tensor_concepts_bool, tensor_features, tensor_attrs,
        full_output_path
    )


def get_model(pkl, args, device, checkpoint):
    moIE = Gated_Logic_Net(
        pkl.input_size_pi,
        pkl.concept_names,
        pkl.labels,
        pkl.hidden_nodes,
        pkl.conceptizator,
        pkl.temperature_lens,
    ).to(device)
    moIE.load_state_dict(torch.load(checkpoint))

    moIE.eval()
    return moIE


def intervene_and_pred_alt_results(args, test_tensor_y, test_tensor_preds, tensor_alpha_norm, test_tensor_concepts):
    checkpoint, test_config = checkpoint_paths(args)
    pkl = pickle.load(open(test_config, "rb"))
    device = utils.get_device()
    print(f"=======>>  Checkpoint_g:  {checkpoint}")
    moIE = get_model(pkl, args, device, checkpoint)
    labels = pkl.labels

    y_pred_in = torch.FloatTensor()
    device = utils.get_device()
    with torch.no_grad():
        for idx in range(test_tensor_y.size(0)):
            target_class = test_tensor_y[idx].to(torch.int32)
            y_hat = test_tensor_preds[idx].argmax(dim=0)
            top_concepts = torch.topk(tensor_alpha_norm[y_hat], args.topK)[1]
            concepts = test_tensor_concepts[idx]
            concepts[top_concepts] = 0
            y_pred_ex, _, _ = moIE(concepts.unsqueeze(0).to(device))
            pred_in = y_pred_ex.argmax(dim=1).item()
            y_pred_in = torch.cat((y_pred_in, y_pred_ex.cpu()), dim=0)
            print(
                f"id: {idx}, "
                f"Ground Truth: {labels[target_class]} ({target_class}), "
                f"Predicted(g) : {labels[y_hat]} ({y_hat}), "
                f"Predicted(alt) : {labels[pred_in]} ({pred_in}), "
            )

        return y_pred_in


def get_concept_masks_top_k(
        tensor_preds,
        alpha_norm,
        tensor_concepts,
        topK
):
    mask_alpha = torch.BoolTensor()
    for idx in range(tensor_concepts.size(0)):
        y_hat = tensor_preds[idx].argmax(dim=0)
        mask_alpha_norm = torch.zeros(tensor_concepts.size(1))
        top_concepts = torch.topk(alpha_norm[y_hat], topK)[1]
        mask_alpha_norm[top_concepts] = True
        mask_alpha_norm = mask_alpha_norm.reshape(1, mask_alpha_norm.size(0))
        mask_alpha = torch.cat((mask_alpha, mask_alpha_norm), dim=0)

    return mask_alpha


def tti(args, test_tensor_y, test_tensor_preds, tensor_alpha_norm, test_tensor_concepts, tensor_attrs):
    checkpoint, test_config = checkpoint_paths(args)
    pkl = pickle.load(open(test_config, "rb"))
    device = utils.get_device()
    print(f"=======>>  Checkpoint_g:  {checkpoint}")
    moIE = get_model(pkl, args, device, checkpoint)
    labels = pkl.labels
    y_pred_in = torch.FloatTensor()
    device = utils.get_device()
    with torch.no_grad():
        for idx in range(test_tensor_y.size(0)):
            target_class = test_tensor_y[idx].to(torch.int32)
            y_hat = test_tensor_preds[idx].argmax(dim=0)
            top_concepts = torch.topk(tensor_alpha_norm[y_hat], args.topK)[1]
            concept_s = test_tensor_concepts[idx]
            print(concept_s[top_concepts].dtype)
            print(tensor_attrs[idx][top_concepts].dtype)
            concept_s[top_concepts] = tensor_attrs[idx][top_concepts]
            y_pred_ex, _, _ = moIE(concept_s.unsqueeze(0).to(device))
            pred_in = y_pred_ex.argmax(dim=1).item()
            y_pred_in = torch.cat((y_pred_in, y_pred_ex.cpu()), dim=0)
            print(
                f"id: {idx}, "
                f"Ground Truth: {labels[target_class]} ({target_class}), "
                f"Predicted(g) : {labels[y_hat]} ({y_hat}), "
                f"Predicted(alt) : {labels[pred_in]} ({pred_in}), "
            )

    return y_pred_in