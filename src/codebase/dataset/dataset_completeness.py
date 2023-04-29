import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset_completeness(Dataset):
    def __init__(
            self, dataset_path, transform=None, mode=None
    ):
        self.transform = transform
        self.concept_mask = torch.load(os.path.join(dataset_path, f"{mode}_mask_alpha.pt"))
        self.raw_data = torch.load(os.path.join(dataset_path, f"{mode}_tensor_images.pt"))
        self.y = torch.load(os.path.join(dataset_path, f"{mode}_tensor_y.pt"))
        print(f"{mode}_size: {self.raw_data.size()}")
        print(f"{mode}_size: {self.concept_mask.size()}")
        print(f"{mode}_size: {self.y.size()}")

    def __getitem__(self, item):
        image = self.raw_data[item]
        if self.transform:
            image = self.transform(image)
        return image, self.y[item], self.concept_mask[item]

    def __len__(self):
        return self.y.size(0)


class Dataset_completeness_features(Dataset):
    def __init__(self, dataset_path, transform=None, mode=None):
        self.transform = transform
        self.concept_mask = torch.load(os.path.join(dataset_path, f"{mode}_mask_alpha.pt"))
        self.features = torch.load(os.path.join(dataset_path, f"{mode}_tensor_features.pt"))
        self.y = torch.load(os.path.join(dataset_path, f"{mode}_tensor_y.pt"))
        print(f"{mode}_size: {self.features.size()}")
        print(f"{mode}_size: {self.concept_mask.size()}")
        print(f"{mode}_size: {self.y.size()}")

    def __getitem__(self, item):
        return self.features[item], self.y[item], self.concept_mask[item]

    def __len__(self):
        return self.y.size(0)


class Dataset_completeness_mimic_cxr(Dataset):
    def __init__(self, dataset_path, disease, seed, top_k, model_type, transform=None, mode=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.disease = disease
        self.mode = mode
        self.master_csv = pd.read_csv(
            os.path.join(
                dataset_path, "completeness", f"seed_{seed}", "dataset", model_type, disease,
                f"{mode}_master_FOL_results.csv"
            ))
        self.concept_mask = torch.load(
            os.path.join(
                dataset_path, "completeness", f"seed_{seed}", "dataset", model_type, disease,
                f"concepts_topK_{top_k}", f"{mode}_all_mask_alpha.pt")
        )
        print(f"{mode} concept_mask_size: {self.concept_mask.size()}")
        print(f"{mode} master csv size: {self.master_csv.shape}")

    def __getitem__(self, item):
        idx = self.master_csv.loc[item, "idx"]
        gt_labels = self.master_csv.loc[item, "ground_truth"]
        # bb_preds = self.master_csv.loc[item, "bb_pred"]
        features = torch.load(
            os.path.join(
                self.dataset_path,
                "t", "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4", "densenet121",
                self.disease,
                "dataset_g",
                f"{self.mode}_features",
                f"features_{idx}.pth.tar"
            )
        )

        features = torch.squeeze(features)
        return features, gt_labels, self.concept_mask[item]

    def __len__(self):
        return self.master_csv.shape[0]
