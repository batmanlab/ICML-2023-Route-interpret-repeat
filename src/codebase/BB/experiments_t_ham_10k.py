import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from torchvision import transforms
from tqdm import tqdm

from BB.models.BB_Inception_V3 import get_model
from dataset.dataset_ham10k import Derm7Concepts_Dataset


def train_for_concepts(args):
    output_path = os.path.join(
        args.output,
        args.dataset,
        "t",
        args.arch
    )
    os.makedirs(output_path, exist_ok=True)

    DERM7_META = os.path.join(args.derm7_folder, "meta", args.derm7_meta)
    TRAIN_IDX = os.path.join(args.derm7_folder, "meta", args.derm7_train_idx)
    VAL_IDX = os.path.join(args.derm7_folder, "meta", args.derm7_val_idx)

    df = pd.read_csv(DERM7_META)
    train_indexes = list(pd.read_csv(TRAIN_IDX)['indexes'])
    val_indexes = list(pd.read_csv(VAL_IDX)['indexes'])
    print(df.columns)
    # df["Sex"] = df.apply(
    #     lambda row: {"male": 0, "female": 1}[row["sex"]], axis=1)
    df["TypicalPigmentNetwork"] = df.apply(
        lambda row: {"absent": 0, "typical": 1, "atypical": -1}[row["pigment_network"]], axis=1)
    df["AtypicalPigmentNetwork"] = df.apply(
        lambda row: {"absent": 0, "typical": -1, "atypical": 1}[row["pigment_network"]], axis=1)

    df["RegularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["streaks"]], axis=1)
    df["IrregularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["streaks"]], axis=1)

    df["RegressionStructures"] = df.apply(lambda row: (1 - int(row["regression_structures"] == "absent")), axis=1)

    df["RegularDG"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["dots_and_globules"]],
                               axis=1)
    df["IrregularDG"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["dots_and_globules"]],
                                 axis=1)

    df["BWV"] = df.apply(lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]], axis=1)

    df = df.iloc[train_indexes + val_indexes]

    concepts = args.concepts
    concept_libs = {C: {} for C in args.C}

    model, model_bottom, _ = get_model(args.bb_dir, args.model_name)
    np.random.seed(args.seed)

    for c_name in concepts:
        print("\t Learning: ", c_name)
        pos_df = df[df[c_name] == 1]
        neg_df = df[df[c_name] == 0]
        base_dir = os.path.join(args.derm7_folder, "images")

        print(pos_df.shape, neg_df.shape)

        if (pos_df.shape[0] < 2 * args.n_samples) or (neg_df.shape[0] < 2 * args.n_samples):
            print("\t Not enough samples! Sampling with replacement")
            pos_df = pos_df.sample(2 * args.n_samples, replace=True)
            neg_df = neg_df.sample(2 * args.n_samples, replace=True)
        else:
            pos_df = pos_df.sample(2 * args.n_samples)
            neg_df = neg_df.sample(2 * args.n_samples)

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        pos_ds = Derm7Concepts_Dataset(pos_df, base_dir=base_dir, image_key="derm", transform=transform)
        neg_ds = Derm7Concepts_Dataset(neg_df, base_dir=base_dir, image_key="derm", transform=transform)
        pos_loader = torch.utils.data.DataLoader(pos_ds,
                                                 batch_size=2 * args.n_samples,
                                                 shuffle=True,
                                                 num_workers=args.num_workers)

        neg_loader = torch.utils.data.DataLoader(neg_ds,
                                                 batch_size=2 * args.n_samples,
                                                 shuffle=True,
                                                 num_workers=args.num_workers)

        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concepts(pos_loader, neg_loader, model_bottom, args.n_samples, args.C, device="cuda")

        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_libs[C][c_name] = cav_info[C]
            print(c_name, C, cav_info[C][1], cav_info[C][2])


        for C in concept_libs.keys():
            lib_path = os.path.join(output_path, f"derma_{args.model_name}_{C}_{args.n_samples}.pkl")
            with open(lib_path, "wb") as f:
                pickle.dump(concept_libs[C], f)
            print(f"Saved to: {lib_path}")


def learn_concepts(pos_loader, neg_loader, model_bottom, n_samples, C, train_ratio=0.5, device="cuda"):
    """Learning CAVs and related margin stats.
    Args:
        pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
        neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
        model_bottom (nn.Module): Mode
        n_samples (int): Number of positive samples to use while learning the concept.
        C (float): Regularization parameter for the SVM. Possibly multiple options.
        device (str, optional): Device to use while extracting activations. Defaults to "cuda".

    Returns:
        dict: Concept information, including the CAV and margin stats.
    """
    print("Extracting Embeddings: ")
    pos_act = get_embedding(pos_loader, model_bottom, n_samples=2 * n_samples, device=device)
    neg_act = get_embedding(neg_loader, model_bottom, n_samples=2 * n_samples, device=device)
    train_idx = int(train_ratio * pos_act.shape[0])
    X_train = np.concatenate([pos_act[:train_idx], neg_act[:train_idx]], axis=0)
    X_val = np.concatenate([pos_act[train_idx:], neg_act[train_idx:]], axis=0)
    y_train = np.concatenate([np.ones(train_idx), np.zeros(train_idx)], axis=0)
    y_val = np.concatenate([np.ones(X_val.shape[0] // 2), np.zeros(X_val.shape[0] // 2)], axis=0)
    print("\t Learning CAVS")
    concept_info = {}
    for c in C:
        concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c)
        print(f"\t Reg-C: {c}, Training Acc: {concept_info[c][1]}, Validation Acc: {concept_info[c][2]}")
    return concept_info


def get_cavs(X_train, y_train, X_val, y_val, C):
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: NumPy arrays to learn the concepts with.
        C: Regularizer for the SVM.
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean": np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": X_train.shape[0] // 2,
                   "neg_count": X_train.shape[0] // 2,
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info


@torch.no_grad()
def get_embedding(loader, model, n_samples=100, device="cuda"):
    """
    Args:
        loader ([type]): Data loader returning only the images
        model ([type]): Backbone
        n_samples (int, optional): Number of samples to extract the activations
        device (str, optional): Device to use. Defaults to "cpu".

    Returns:
        [type]: Activations as a numpy array.
    """
    activations = None
    with torch.no_grad():
        for image in tqdm(loader):
            image = image.to(device)
            try:
                batch_act = model.encode_image(image).squeeze().detach().cpu().numpy()
            except:
                batch_act = model(image).squeeze().detach().cpu().numpy()
            if activations is None:
                activations = batch_act
            else:
                activations = np.concatenate([activations, batch_act], axis=0)
            if activations.shape[0] >= n_samples:
                return activations[:n_samples]
        raise ValueError(f"Insufficient number of samples: {activations.shape}. Desired: {n_samples}")
