import ast
import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable

import utils
from BB.models.BB_DenseNet121 import DenseNet121
from BB.models.t import Logistic_Regression_t
from MIMIC_CXR.bbox_utils import apply_mask, deprocess_image, BoundingBoxGenerator


def merge_csv_from_experts(iterations, json_file, disease, fol_path_root, dataset_path, mode="test"):
    df_list = []
    for iteration in range(iterations):
        iteration += 1
        print(f"############################### iteration: {iteration} start ###############################")
        with open(json_file) as _file:
            paths = json.load(_file)
        root = paths[disease]["MoIE_paths"][f"iter{iteration}"]["base_path"]
        fol_path = f"{fol_path_root}/{root}/iter{iteration}/g/selected/auroc/FOLs"
        df = pd.read_csv(os.path.join(fol_path, f"{mode}_results_expert_{iteration}.csv"))
        df_list.append(df)
        print(f"############################### iteration: {iteration} End ###############################")

    df_master = pd.concat(df_list, ignore_index=True)
    df_master.to_csv(os.path.join(dataset_path, "Grad_CAM_BBOX", f"{mode}_master_FOL_results.csv"))
    print(f"Master csv is saved at: {dataset_path}")

    return df_master


def get_cam(pred_logits_tensor, target, feature_map_tensor, img, path_to_save_grad_cam, img_idx, ground_truth):
    grad_wrt_y = torch.autograd.grad(pred_logits_tensor[:, target], feature_map_tensor, retain_graph=True)[0]
    feature_map_np = feature_map_tensor.cpu().data.numpy()[0, :]
    grad_wrt_y_np = grad_wrt_y.cpu().data.numpy()[0, :]
    weights = np.mean(grad_wrt_y_np, axis=(1, 2))
    cam = np.zeros(feature_map_np.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_map_np[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, img.shape[-2:][::-1])
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    image_cam = apply_mask(deprocess_image(img), cam)

    return image_cam, cam


def get_bbox_for_detector():
    pass


def get_initial_setup(output, disease):
    path = f"{output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{disease}"
    device = utils.get_device()
    print(f"Device: {device}")
    pickle_in = open(os.path.join(path, "MIMIC_test_configs.pkl"), "rb", )
    args = pickle.load(pickle_in)
    concepts_master = args.landmark_names_spec + args.abnorm_obs_concepts
    print(args.lr)
    print(args.checkpoint_t)
    len(concepts_master)

    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss1}_flattening_type_{args.flattening_type}_layer_{args.layer}"
    chk_pt_path_t = os.path.join(args.checkpoints, args.dataset, "t", root, args.arch, args.selected_obs[0])
    model_chk_pt_t = torch.load(os.path.join(chk_pt_path_t, args.checkpoint_t))
    t_model = Logistic_Regression_t(
        ip_size=1024 * 16 * 16, op_size=len(concepts_master), flattening_type=args.flattening_type
    )
    t_model.load_state_dict(model_chk_pt_t["state_dict"])
    t_model.eval()

    bb = DenseNet121(args, layer=args.layer)
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", args.bb_chkpt_folder, args.arch, disease)
    model_chk_pt = torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
    bb.load_state_dict(model_chk_pt["state_dict"])
    bb.eval()
    classifier_bb = list(bb.children())[-1]
    return t_model, classifier_bb, concepts_master


def get_grad_cam_maps_and_bbox(dataset_path, mode, classifier_bb, t_model, df, concepts_master):
    avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    path_to_save_grad_cam_for_target_disease = os.path.join(
        dataset_path, "Grad_CAM_BBOX", f"{mode}_Grad_CAM_for_target_disease"
    )
    path_to_save_grad_cam_for_concepts = os.path.join(dataset_path, "Grad_CAM_BBOX", f"{mode}_Grad_CAM_for_concepts")
    path_to_save_grad_cam_for_concepts_w_bbox = os.path.join(
        dataset_path, "Grad_CAM_BBOX", f"{mode}_Grad_CAM_with_bbox_for_concepts"
    )
    os.makedirs(path_to_save_grad_cam_for_target_disease, exist_ok=True)
    os.makedirs(path_to_save_grad_cam_for_concepts, exist_ok=True)
    os.makedirs(path_to_save_grad_cam_for_concepts_w_bbox, exist_ok=True)
    bbox_annot = {}
    for row in range(len(df)):
        idx = df.loc[row, "idx"]
        bbox_annot[f"img_{idx}"] = []
        concept_ids = df.loc[row, "concept_ids_in_explanations"]
        ground_truth = df.loc[row, "ground_truth"]
        bb_pred = df.loc[row, "bb_pred"]
        concept_names_in_explanations = ast.literal_eval(df.loc[row, "concept_names_in_explanations"])
        concept_proba = ast.literal_eval(df.loc[row, "all_concept_proba"])

        img = torch.load(os.path.join(dataset_path, f"{mode}_transformed_images", f"transformed_img_{idx}.pth.tar"))
        raw_img = torch.load(os.path.join(dataset_path, f"{mode}_raw_images", f"raw_img_{idx}.pth.tar"))
        feature_map = torch.load(os.path.join(dataset_path, f"{mode}_features", f"features_{idx}.pth.tar"))
        feature_map_tensor = Variable(feature_map, requires_grad=True)

        concept_logits = t_model(feature_map_tensor)
        pred_logits = classifier_bb(avg_pool(feature_map_tensor).reshape(-1, 1024 * 1 * 1))

        print(f"==================>>>>>> {idx} <<<<<<================== ")
        print(concept_names_in_explanations)
        image_cam, _ = get_cam(
            pred_logits, bb_pred, feature_map_tensor, img, path_to_save_grad_cam_for_target_disease,
            img_idx=idx, ground_truth=ground_truth
        )
        im = Image.fromarray(image_cam)
        img_name = os.path.join(
            path_to_save_grad_cam_for_target_disease, f"img_idx_{idx}_ground_truth_{ground_truth}.png"
        )
        im.save(img_name)
        print(f"Grad_CAM for target is saved at: {img_name}")
        for concept_name in concept_names_in_explanations:
            print(f"================= {concept_name} =================")
            concept_id_master = concepts_master.index(concept_name)
            bbox_arr = get_cam_and_bbox_per_concept(
                concept_id_master, path_to_save_grad_cam_for_concepts, path_to_save_grad_cam_for_concepts_w_bbox,
                concept_logits, img, feature_map_tensor, idx, ground_truth, concept_name
            )
            bbox_annot[f"img_{idx}"].extend(bbox_arr)

    with open(os.path.join(dataset_path, "Grad_CAM_BBOX", f"{mode}_bbox_annot.json"), "w") as outfile:
        json.dump(bbox_annot, outfile)

    pickle.dump(bbox_annot, open(os.path.join(dataset_path, "Grad_CAM_BBOX", f"{mode}_bbox_annot.pkl"), "wb"))
    return bbox_annot


def get_cam_and_bbox_per_concept(
        concept_id_master, path_to_save_grad_cam_for_concepts, path_to_save_grad_cam_for_concepts_w_bbox,
        concept_logits, img, feature_map_tensor, img_idx, ground_truth, concept_name
):
    image_cam, cam = get_cam(
        pred_logits_tensor=concept_logits, target=concept_id_master, feature_map_tensor=feature_map_tensor, img=img,
        path_to_save_grad_cam=path_to_save_grad_cam_for_concepts, img_idx=img_idx, ground_truth=ground_truth)
    im = Image.fromarray(image_cam)
    img_name_gc_concept = os.path.join(
        path_to_save_grad_cam_for_concepts, f"img-idx_{img_idx}_ground-truth_{ground_truth}_{concept_name}.png"
    )
    im.save(img_name_gc_concept)

    BBG = BoundingBoxGenerator(cam, percentile=0.95)
    bboxs = BBG.get_bbox_pct()
    img_cv2 = cv2.cvtColor(image_cam.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bbox_arr = []
    for box in bboxs:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        bbox_arr.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': concept_id_master})
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 1, )
        cv2.putText(img_cv2, concept_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    img_name_gc_concept_w_bbox = os.path.join(
        path_to_save_grad_cam_for_concepts_w_bbox, f"img-idx_{img_idx}_ground-truth_{ground_truth}_{concept_name}.png"
    )
    cv2.imwrite(img_name_gc_concept_w_bbox, img_cv2)

    print(f"Grad_CAM for {concept_name} is saved at: {img_name_gc_concept}")
    print(f"Grad_CAM with bbox for {concept_name} is saved at: {img_name_gc_concept_w_bbox}")
    return bbox_arr
