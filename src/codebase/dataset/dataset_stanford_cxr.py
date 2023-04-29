import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class Stanford_CXR(Dataset):
    def __init__(self, args, csv_file_name, transform=None):
        self.transform = transform
        self.channels = args.channels
        self.parent_dir = args.image_dir_path
        self.uncertain = args.uncertain
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.dataset = args.dataset
        self.disease = args.disease
        csv_file_path = os.path.join(args.image_dir_path, args.image_source_dir, csv_file_name)

        self.df = pd.read_csv(csv_file_path)
        self.df = self.df.fillna(0)
        self.df = self.df.set_index(args.image_col_header)

        print(f"Dataset path: {csv_file_path}")
        print(f"The size of the dataset: {self.df.shape}")

        if args.dataset == "stanford_cxr":
            self.labels = [label.replace(" ", "_") for label in list(self.df.columns.values)[4:]]
            self.df.columns = list(self.df.columns.values)[0:4] + self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.parent_dir, self.df.index[idx]))
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = np.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            label[i] = self.df[self.labels[i].strip()].iloc[idx].astype('int')
            if label[i] == -1:
                label[i] = self.uncertain

        return image, torch.FloatTensor(label), self.df.index[idx]


class Stanford_CXR_T_SSL(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
        self.channels = args.channels
        self.parent_dir = args.image_dir_path
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.dataset = args.dataset
        self.disease = args.disease
        csv_file_path = os.path.join(
            args.output, args.dataset, "BB", "lr_0.01_epochs_10_loss_CE", args.arch, args.disease, args.csv_file
        )
        self.df = pd.read_csv(csv_file_path)

        print(f"Dataset path: {csv_file_path}")
        print(f"The size of the dataset: {self.df.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_names = self.df.loc[idx, "image_names"]
        disease_label = self.df.loc[idx, "target"]
        image = Image.open(os.path.join(self.parent_dir, image_names))
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, disease_label, image_names


class Stanford_CXR_Domain_transfer(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
        self.channels = args.channels
        self.parent_dir = args.image_dir_path
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.dataset = args.dataset
        self.disease = args.disease
        csv_file = f"master_tot_{args.tot_samples}.csv"
        csv_file_path = os.path.join(
            args.output, args.dataset, "BB", "lr_0.01_epochs_10_loss_CE", args.arch, args.disease, csv_file
        )
        self.df = pd.read_csv(csv_file_path)

        print(f"Dataset path: {csv_file_path}")
        print(f"The size of the dataset: {self.df.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_names = self.df.loc[idx, "image_names"]
        disease_label = self.df.loc[idx, "GT"]
        image = Image.open(os.path.join(self.parent_dir, image_names))
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, disease_label, image_names


class Stanford_CXR_MoIE_train(Dataset):
    def __init__(self, args, dataset_path, mode, transform=None):
        self.transform = transform
        self.channels = args.channels
        self.parent_dir = args.image_dir_path
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.dataset = args.dataset
        self.disease = args.disease
        csv_file_path = os.path.join(
            args.output, args.dataset, "BB", "lr_0.01_epochs_10_loss_CE", args.arch, args.disease, args.csv_file
        )

        self.df = pd.read_csv(csv_file_path)
        self.attributes_gt = torch.load(
            os.path.join(dataset_path, f"{mode}_sample_size_{args.tot_samples}_sub_select_proba_concepts.pt")
        )
        self.proba_concept_x = torch.load(
            os.path.join(dataset_path, f"{mode}_sample_size_{args.tot_samples}_sub_select_attributes.pt")
        )
        print(f"=======>> Dataset path: {csv_file_path}")
        print(f"=======>> The size of the {mode} dataset: {self.df.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_names = self.df.loc[idx, "image_names"]
        disease_label = self.df.loc[idx, "GT"]
        image = Image.open(os.path.join(self.parent_dir, image_names))
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, disease_label, self.proba_concept_x[idx], self.attributes_gt[idx], image_names


class Stanford_CXR_MoIE_val(Dataset):
    def __init__(self, args, dataset_path, mode, transform=None):
        self.transform = transform
        self.channels = args.channels
        self.parent_dir = args.image_dir_path
        self.uncertain = args.uncertain
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.dataset = args.dataset
        self.disease = args.disease

        csv_file_path = os.path.join(args.image_dir_path, args.image_source_dir, "valid.csv")

        self.df = pd.read_csv(csv_file_path)
        self.df = self.df.fillna(0)
        self.df = self.df.set_index(args.image_col_header)

        print(f"=======>> Dataset path: {csv_file_path}")
        print(f"=======>> The size of the val dataset: {self.df.shape}")

        self.attributes_gt = torch.load(
            os.path.join(dataset_path, f"{mode}_sample_size_{args.tot_samples}_sub_select_proba_concepts.pt")
        )
        self.proba_concept_x = torch.load(
            os.path.join(dataset_path, f"{mode}_sample_size_{args.tot_samples}_sub_select_attributes.pt")
        )

        if args.dataset == "stanford_cxr":
            self.labels = [label.replace(" ", "_") for label in list(self.df.columns.values)[4:]]
            self.df.columns = list(self.df.columns.values)[0:4] + self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.parent_dir, self.df.index[idx]))
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = np.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            label[i] = self.df[self.labels[i].strip()].iloc[idx].astype('int')
            if label[i] == -1:
                label[i] = self.uncertain

        return image, torch.FloatTensor(label), self.proba_concept_x[idx], self.attributes_gt[idx], self.df.index[idx]
