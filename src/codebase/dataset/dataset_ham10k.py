import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

DERM7_FOLDER = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/Derm7pt"


class DermDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['y'].iloc[index]))
        if self.mode == "train":
            if self.transform:
                X = self.transform(X)
            return X, y
        elif self.mode == "save":
            raw_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor()
            ])
            raw_image = raw_transform(X)
            if self.transform:
                X = self.transform(X)
            return X, raw_image, y


class Derm7Concepts_Dataset():
    def __init__(self, images, base_dir=os.path.join(DERM7_FOLDER, "images"), transform=None,
                 image_key="derm"):
        self.images = images
        self.transform = transform
        self.base_dir = base_dir
        self.image_key = image_key

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.images.iloc[idx]
        img_path = os.path.join(self.base_dir, row[self.image_key])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = os.path.abspath(self.imgs[index][0])
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def load_ham_data(args, transform, class_to_idx, mode="train"):
    print(args.dataset)
    if args.dataset == "HAM10k":
        return load_HAM10k_data(args, transform, class_to_idx, mode)
    elif args.dataset == "SIIM-ISIC":
        return load_isic(args, transform, mode)


def load_isic(args, preprocess, mode="train", n_train=2000, n_val=500):
    df = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    df['path'] = df['image_name'].map(lambda name: os.path.join(args.data_root, "train", name + '.jpg'))
    df['y'] = df['target']

    files = os.listdir(os.path.join(args.data_root, "train"))
    files = [os.path.join(args.data_root, "train", f) for f in files]
    df = df[df.path.isin(files)]

    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]

    _, df_val_pos = train_test_split(df_pos, test_size=0.20, random_state=args.seed)
    _, df_val_neg = train_test_split(df_neg, test_size=0.20, random_state=args.seed)
    df_train_pos = df_pos[~df_pos.path.isin(df_val_pos.path)]
    df_train_neg = df_neg[~df_neg.path.isin(df_val_neg.path)]

    df_train_pos = df_train_pos.sample(n_train // 5, random_state=args.seed)
    df_train_neg = df_train_neg.sample(4 * n_train // 5, random_state=args.seed)

    df_val_pos = df_val_pos.sample(n_val // 5, random_state=args.seed)
    df_val_neg = df_val_neg.sample(4 * n_val // 5, random_state=args.seed)

    df_train = pd.concat([df_train_pos, df_train_neg])
    df_val = pd.concat([df_val_pos, df_val_neg])

    trainset = DermDataset(df_train, preprocess, mode)
    valset = DermDataset(df_val, preprocess, mode)

    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                               shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.bs,
                                             shuffle=False, num_workers=args.num_workers)

    idx_to_class = {v: k for k, v in args.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


def load_HAM10k_data(args, transform, class_to_idx, mode):
    np.random.seed(args.seed)
    id_to_lesion = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'}

    benign_malignant = {
        'nv': 'benign',
        'mel': 'malignant',
        'bkl': 'benign',
        'bcc': 'malignant',
        'akiec': 'benign',
        'vasc': 'benign',
        'df': 'benign'}

    df = pd.read_csv(os.path.join(args.data_root, 'HAM10000_metadata.csv'))
    all_image_paths = glob(os.path.join(args.data_root, '*', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id]
        else:
            return "-1"

    df['path'] = df['image_id'].map(path_getter)
    df = df[df.path != "-1"]
    df['dx_name'] = df['dx'].map(lambda id: id_to_lesion[id])
    df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])

    df['y'] = df["benign_or_malignant"].map(lambda id: class_to_idx[id])

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    _, df_val = train_test_split(df, test_size=0.20, random_state=args.seed, stratify=df["dx"])
    df_train = df[~df.image_id.isin(df_val.image_id)]
    trainset = DermDataset(df_train, transform, mode)
    valset = DermDataset(df_val, transform, mode)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                               shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.bs,
                                             shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, idx_to_class
