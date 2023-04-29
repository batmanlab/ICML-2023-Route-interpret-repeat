import os
import sys

import cv2
import torch.nn.functional as F

from BB.models.VIT import VisionTransformer, CONFIGS
from BB.models.t import Logistic_Regression_t
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)

import numpy as np
from torchvision import transforms
import torch

import utils
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")


def show_results(
        idx,
        feature_names,
        test_tensor_preds,
        test_tensor_preds_bb,
        test_tensor_y,
        dataset_path,
        test_tensor_concepts_bool,
        tensor_alpha_norm,
        percentile_selection,
        concept_names,
        labels,
        model,
        test_tensor_concepts,
        device,
        args,
):
    print("---------------------------------------------------------------------")
    print(f"{idx}: ==================================================>")
    print("---------------------------------------------------------------------")
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    y_hat_bb = test_tensor_preds_bb[idx].argmax(dim=0)
    img = torch.load(
        os.path.join(dataset_path, "test_raw_images", f"raw_img_{idx}.pth.tar")
    ).squeeze(dim=0)

    print(f"Ground Truth class_label: {args.labels[target_class]} ({target_class})")
    print(f"Predicted(g) class_label: {args.labels[y_hat]} ({y_hat})")
    print(f"Predicted(BB) class_label: {args.labels[y_hat_bb]} ({y_hat_bb})")
    print(f"Check wrt BB: {y_hat_bb == y_hat}")
    print(f"Check wrt GT: {target_class == y_hat}")

    im = img.permute(1, 2, 0).numpy()
    print("----------------------------")
    print(test_tensor_concepts_bool[idx])
    print("-------------------------------")
    print(tensor_alpha_norm[y_hat])

    t = 0
    ps = 100
    while True:
        # print(percentile_selection)
        if percentile_selection == 0:
            percentile_selection = 80
            ps = 0
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        # print(percentile_val)
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = test_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = test_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        # print("------------------------------")
        # print(y_pred_ex.item())
        # print(y_hat.item())
        # print("------------------------------")
        if ps == 0:
            break
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    print(mask_indxs)
    print(imp_concepts)

    explanations = ""
    for m_idx in mask_indxs.tolist():
        if explanations:
            explanations += " & "

        if test_tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif test_tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = replace_names(explanations, concept_names)

    print("Raw Explanations: =======================>>>>")
    print(explanations)
    print("Concept Explanations: =======================>>>>")
    print(explanation_complete)
    print(
        f"Concepts in the extracted FOL: \
        {len(explanation_complete.split(' & '))} / {test_tensor_concepts_bool.size(1)}"
    )

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

    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        "concept_dict_key": y_hat.item(),
        "concept_dict_val": explanation_complete,
        "im": im,
        "test_tensor_concepts": test_tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
    }


def get_cub_models(args, iteration):
    explainer_init = "none"
    use_concepts_as_pi_input = True if args.use_concepts_as_pi_input == "y" else False
    device = utils.get_device()
    root = (
        f"lr_{args.lr[0]}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov[0]}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )
    chk_pt_explainer = os.path.join(
        args.checkpoints, args.dataset, "explainer", args.arch, root
    )
    g_chk_pt_path = os.path.join(
        chk_pt_explainer, f"cov_0.2_lr_0.01", f"{iteration}", "explainer"
    )
    glt_chk_pt = os.path.join(g_chk_pt_path, args.checkpoint_model[-1])
    print(f"=======>> G for iteration {iteration} is loaded from {glt_chk_pt}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input,
    ).to(device)
    model.load_state_dict(torch.load(glt_chk_pt))
    model.eval()

    config = CONFIGS[args.arch]
    bb = VisionTransformer(
        config,
        args.img_size,
        zero_head=True,
        num_classes=len(args.labels),
        smoothing_value=args.smoothing_value,
        grad_cam=True,
    ).to(device)
    bb.load_state_dict(
        torch.load(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/BB/lr_0.03_epochs_95/ViT-B_16/VIT_CUBS_8000_checkpoint.bin"
        )["model"]
    )
    bb.eval()

    t_model = Logistic_Regression_t(
        ip_size=768, op_size=len(args.concept_names), flattening_type="vit_flatten"
    ).to(device)
    t_model.load_state_dict(
        torch.load(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/t/lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE/g_best_model_epoch_54.pth.tar"
        )
    )
    t_model.eval()

    return model, bb, t_model


def show_results_cub_vit(
        idx, args, feature_names, test_images, test_tensor_y, test_tensor_preds, test_tensor_preds_bb,
        tensor_alpha_norm, test_tensor_concepts_bool, test_tensor_concepts, model, bb, t_model,
):
    print(f"{idx}: ==================================================>")
    device = utils.get_device()
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    y_hat_bb = test_tensor_preds_bb[idx].argmax(dim=0)
    print("----------------------------")
    print(f"Ground Truth class_label: {args.labels[target_class]} ({target_class})")
    print(f"Predicted(g) class_label: {args.labels[y_hat]} ({y_hat})")
    print(f"Predicted(BB) class_label: {args.labels[y_hat_bb]} ({y_hat_bb})")
    print(f"Check wrt BB: {y_hat_bb == y_hat}")
    print(f"Check wrt GT: {target_class == y_hat}")

    percentile_selection = 99
    ps = 100
    while True:
        # print(percentile_selection)
        if percentile_selection == 0:
            percentile_selection = 80
            ps = 0
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        # print(percentile_val)
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = test_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = test_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        # print("------------------------------")
        # print(y_pred_ex.item())
        # print(y_hat.item())
        # print("------------------------------")
        if ps == 0:
            break
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    print(mask_indxs)
    print(imp_concepts)

    dict_sample_concept = {}
    for concept in args.concept_names:
        dict_sample_concept[concept] = 0

    dict_sample_concept["y_GT"] = 0
    dict_sample_concept["y_BB"] = 0
    dict_sample_concept["y_G"] = 0
    device = utils.get_device()

    attn_weight = tensor_alpha_norm[y_hat][mask_alpha_norm]
    concepts = []
    for indx in mask_indxs:
        dict_sample_concept[args.concept_names[indx]] = 1
        concepts.append(args.concept_names[indx])
    dict_sample_concept["y_GT"] = target_class.item()
    dict_sample_concept["y_BB"] = y_hat_bb.item()
    dict_sample_concept["y_G"] = y_hat.item()
    dict_sample_concept["correctly_predicted_wrt_GT"] = (target_class.item() == y_hat.item())
    print(concepts)
    print(len(concepts))
    # concept_label.append(dict_sample_concept)

    test_transforms = transforms.Compose([
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    image = test_transforms(test_images[idx])

    logits_y, _part_tokens, att_mat, embedding_output = bb(image.unsqueeze(0).to(device))
    logit_concepts = t_model(_part_tokens[:, 0])
    gt = logits_y.argmax(dim=1)
    logits_y = logits_y.squeeze(0).cpu()
    logit_concepts = logit_concepts.squeeze(0).cpu()

    im = test_images[idx].permute(1, 2, 0).numpy()
    grad_acc = torch.zeros((embedding_output.size(0), embedding_output.size(1), embedding_output.size(2)))
    for indx, concept_val in enumerate(logit_concepts[mask_indxs]):
        grad_wrt_concept = torch.autograd.grad(concept_val, embedding_output, retain_graph=True)[0]
        grad_acc += attn_weight[indx] * grad_wrt_concept.cpu()

    result_grad = grad_acc[:, 1:, :].reshape(embedding_output.size(0),
                                             28, 28, embedding_output.size(2))

    result_grad = result_grad.transpose(2, 3).transpose(1, 2)
    pooled_grads = result_grad.mean((0, 2, 3))

    result = embedding_output[:, 1:, :].reshape(embedding_output.size(0),
                                                28, 28, embedding_output.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    result = result.squeeze()
    result = F.relu(result)
    for i in range(len(pooled_grads)):
        result[i, :, :] *= pooled_grads[i]

    heatmap = result.mean(dim=0).squeeze()
    heatmap = heatmap / torch.max(heatmap)
    mask1 = heatmap.cpu().detach().numpy()
    cmap = mpl.cm.get_cmap("jet", 256)
    heatmap_j2 = cmap(cv2.resize(mask1 / mask1.max(), (im.shape[0], im.shape[1])), alpha=0.5)

    grad_wrt_y = torch.autograd.grad(logits_y[gt], embedding_output, retain_graph=True)[0]
    y_grad = grad_wrt_y[:, 1:, :].reshape(embedding_output.size(0),
                                          28, 28, embedding_output.size(2))
    y_grad = y_grad.transpose(2, 3).transpose(1, 2)
    pooled_y_grads = y_grad.mean((0, 2, 3))
    result1 = embedding_output[:, 1:, :].reshape(embedding_output.size(0),
                                                 28, 28, embedding_output.size(2))
    result1 = result1.transpose(2, 3).transpose(1, 2)
    result1 = result1.squeeze()
    result1 = F.relu(result1)
    for i in range(len(pooled_y_grads)):
        result1[i, :, :] *= pooled_y_grads[i]

    heatmap_y = result1.mean(dim=0).squeeze()
    heatmap_y = heatmap_y / torch.max(heatmap_y)

    mask2 = heatmap_y.cpu().detach().numpy()
    cmap1 = mpl.cm.get_cmap("jet", 256)
    heatmap_i2 = cmap1(cv2.resize(mask2 / mask2.max(), (im.shape[0], im.shape[1])), alpha=0.5)

    explanations = ""
    for m_idx in mask_indxs.tolist():
        if explanations:
            explanations += " & "

        if test_tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif test_tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = replace_names(explanations, args.concept_names)
    # num_concepts.append(len(concepts))
    print("Concept Explanations: =======================>>>>")
    print(f"{args.labels[y_hat]} ({y_hat}) <=> {explanation_complete}")

    # concept_dict[int(target_class.item())].append(explanation_complete)
    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        "concept_dict_key": int(y_hat.item()) if (target_class.item() == y_hat.item()) else -1,
        "concept_dict_val": explanation_complete,
        "im": im,
        "heatmap_j2": heatmap_j2,
        "heatmap_i2": heatmap_i2,
        "test_tensor_concepts": test_tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
        "raw_explanations": explanations
    }


from utils import replace_names

percentile_selection = 50


def show_results_HAM10K(
        args, idx, val_images, val_tensor_y, val_tensor_concepts_bool, val_tensor_concepts, val_tensor_preds,
        feature_names, tensor_alpha_norm, concept_names, glt
):
    print(f"{idx}: ==================================================>")
    im = val_images[idx].permute(1, 2, 0).numpy()
    #     plt.imshow(val_images[idx].permute(1, 2, 0))
    #     plt.show()
    device = utils.get_device()
    target_class = val_tensor_y[idx].to(torch.int32)
    y_hat = val_tensor_preds[idx].argmax(dim=0)
    print("----------------------------")
    print(f"y_gt: {target_class}")
    print(f"y_hat: {y_hat}")
    print(f"class_label: {args.labels[target_class]}")
    print("----------------------------")
    print(val_tensor_concepts_bool[idx])
    print("-------------------------------")
    print(tensor_alpha_norm[y_hat])
    percentile_selection = 99
    while True:
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        print(percentile_val)
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = val_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = val_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = glt(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        print("------------------------------")
        print(y_pred_ex.item())
        print(y_hat.item())
        print("------------------------------")
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    print(mask_indxs)
    print(imp_concepts)
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

    print("Raw Explanations: =======================>>>>")
    print(explanations)
    print("Concept Explanations: =======================>>>>")
    print(explanation_complete)
    print(
        f"Concepts in the extracted FOL: \
        {len(explanation_complete.split(' & '))} / {val_tensor_concepts_bool.size(1)}"
    )

    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        "concept_dict_key": int(y_hat.item())
        if (target_class.item() == y_hat.item())
        else -1,
        "concept_dict_val": explanation_complete,
        "im": im,
        "test_tensor_concepts": val_tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
        "raw_explanations": explanations
    }


#     prediction_result_from_explanations = test_explanation(
#         explanations,
#         val_tensor_concepts_bool[idx, :],
#         val_tensor_y_1h[idx],
#         target_class,
#     )

#     print(f"prediction_result_from_explanations: {prediction_result_from_explanations}")


def show_results_cub_CNN(
        idx,
        args,
        feature_names,
        test_images,
        test_tensor_y,
        test_tensor_preds,
        test_tensor_preds_bb,
        tensor_alpha_norm,
        test_tensor_concepts_bool,
        test_tensor_concepts,
        model,
        bb,
        t_model,
        layer,
):
    print(f"{idx}: ==================================================>")
    device = utils.get_device()
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    y_hat_bb = test_tensor_preds_bb[idx].argmax(dim=0)
    print("----------------------------")
    print(f"Ground Truth class_label: {args.labels[target_class]} ({target_class})")
    print(f"Predicted(g) class_label: {args.labels[y_hat]} ({y_hat})")
    print(f"Predicted(BB) class_label: {args.labels[y_hat_bb]} ({y_hat_bb})")
    print(f"Check wrt BB: {y_hat_bb == y_hat}")
    print(f"Check wrt GT: {target_class == y_hat}")

    t = 0
    percentile_selection = 99
    while True:
        if percentile_selection == 0:
            t = -1
            percentile_selection = 90
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        print(percentile_val)
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = test_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = test_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        print("------------------------------")
        print(y_pred_ex.item())
        print(y_hat.item())
        print("------------------------------")
        if y_pred_ex.item() == y_hat.item() or t == -1:
            break
        else:
            percentile_selection = percentile_selection - 1

    print(mask_indxs)
    print(imp_concepts)

    dict_sample_concept = {}
    for concept in args.concept_names:
        dict_sample_concept[concept] = 0

    dict_sample_concept["y_GT"] = 0
    dict_sample_concept["y_BB"] = 0
    dict_sample_concept["y_G"] = 0
    device = utils.get_device()

    attn_weight = tensor_alpha_norm[y_hat][mask_alpha_norm]
    concepts = []
    for indx in mask_indxs:
        dict_sample_concept[args.concept_names[indx]] = 1
        concepts.append(args.concept_names[indx])
    dict_sample_concept["y_GT"] = target_class.item()
    dict_sample_concept["y_BB"] = y_hat_bb.item()
    dict_sample_concept["y_G"] = y_hat.item()
    dict_sample_concept["correctly_predicted_wrt_GT"] = (
            target_class.item() == y_hat.item()
    )
    print(concepts)
    print(len(concepts))
    # concept_label.append(dict_sample_concept)

    test_transforms = transforms.Compose(
        [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )
    image = test_transforms(test_images[idx])

    logits_y = bb(image.unsqueeze(0).to(device))
    embedding_output = bb.feature_store[layer]
    # print(embedding_output.size())
    logit_concepts = t_model(embedding_output)
    # print(logit_concepts)
    gt = logits_y.argmax(dim=1)
    logits_y = logits_y.squeeze(0).cpu()
    logit_concepts = logit_concepts.squeeze(0).cpu()

    im = test_images[idx].permute(1, 2, 0).numpy()
    grad_acc = torch.zeros(
        (
            embedding_output.size(0),
            embedding_output.size(1),
            embedding_output.size(2),
            embedding_output.size(3),
        )
    )
    for indx, concept_val in enumerate(logit_concepts[mask_indxs]):
        grad_wrt_concept = torch.autograd.grad(
            concept_val, embedding_output, retain_graph=True
        )[0]
        grad_acc += attn_weight[indx] * grad_wrt_concept.cpu()

    #     result_grad = grad_acc[:, 1:, :].reshape(
    #         embedding_output.size(0), 28, 28, embedding_output.size(2)
    #     )

    result_grad = grad_acc

    #     result_grad = result_grad.transpose(2, 3).transpose(1, 2)
    pooled_grads = result_grad.mean((0, 2, 3))

    #     result = embedding_output[:, 1:, :].reshape(
    #         embedding_output.size(0), 28, 28, embedding_output.size(2)
    #     )
    result = embedding_output
    result = result.transpose(2, 3)
    result = result.squeeze()
    result = F.relu(result)
    for i in range(len(pooled_grads)):
        result[i, :, :] *= pooled_grads[i]

    heatmap = result.mean(dim=0).squeeze()
    heatmap = heatmap / torch.max(heatmap)
    mask1 = heatmap.cpu().detach().numpy()
    cmap = mpl.cm.get_cmap("jet", 256)
    heatmap_j2 = cmap(
        cv2.resize(mask1 / mask1.max(), (im.shape[0], im.shape[1])), alpha=0.5
    )

    grad_wrt_y = torch.autograd.grad(logits_y[gt], embedding_output, retain_graph=True)[
        0
    ]
    #     y_grad = grad_wrt_y[:, 1:, :].reshape(
    #         embedding_output.size(0), 28, 28, embedding_output.size(2)
    #     )
    y_grad = grad_wrt_y
    y_grad = y_grad.transpose(2, 3)
    pooled_y_grads = y_grad.mean((0, 2, 3))
    #     result1 = embedding_output[:, 1:, :].reshape(
    #         embedding_output.size(0), 28, 28, embedding_output.size(2)
    #     )

    result1 = embedding_output
    result1 = result1.transpose(2, 3)
    result1 = result1.squeeze()
    result1 = F.relu(result1)
    for i in range(len(pooled_y_grads)):
        result1[i, :, :] *= pooled_y_grads[i]

    heatmap_y = result1.mean(dim=0).squeeze()
    heatmap_y = heatmap_y / torch.max(heatmap_y)

    mask2 = heatmap_y.cpu().detach().numpy()
    cmap1 = mpl.cm.get_cmap("jet", 256)
    heatmap_i2 = cmap1(
        cv2.resize(mask2 / mask2.max(), (im.shape[0], im.shape[1])), alpha=0.5
    )

    explanations = ""
    for m_idx in mask_indxs.tolist():
        if explanations:
            explanations += " & "

        if test_tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif test_tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = replace_names(explanations, args.concept_names)
    # num_concepts.append(len(concepts))
    print("Concept Explanations: =======================>>>>")
    print(f"{args.labels[y_hat]} ({y_hat}) <=> {explanation_complete}")

    # concept_dict[int(target_class.item())].append(explanation_complete)
    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        "concept_dict_key": int(y_hat.item())
        if (target_class.item() == y_hat.item())
        else -1,
        "concept_dict_val": explanation_complete,
        "im": im,
        "heatmap_j2": heatmap_j2,
        "heatmap_i2": heatmap_i2,
        "test_tensor_concepts": test_tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
    }
