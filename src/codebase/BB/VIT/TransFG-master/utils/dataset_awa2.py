import argparse
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

import utils


class AnimalDataset(data.dataset.Dataset):
    def __init__(self, args):
        self.predicate_binary_mat = np.array(
            np.genfromtxt(
                os.path.join(args.data_root, "predicate-matrix-binary.txt"), dtype='int')
        )

        # self.transform = transform

        class_to_index = dict()
        # Build dictionary of indices to classes
        img_names = []
        target_index = []
        index = 0
        for class_name in args.labels:
            class_to_index[class_name] = index
            FOLDER_DIR = os.path.join(args.data_root, 'JPEGImages', class_name)
            file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
            files = glob(file_descriptor)
            class_index = class_to_index[class_name]
            for file_name in files:
                img_names.append(file_name)
                target_index.append(class_index)
            index += 1
        self.class_to_index = class_to_index

        self.img_names = img_names
        self.target_index = target_index

    def __getitem__(self, index):
        target = self.target_index[index]
        attribute = self.predicate_binary_mat[target, :]
        # return im, attribute, self.img_names[index], target
        return attribute, self.img_names[index], target

    def __len__(self):
        return len(self.img_names)


class Awa2(data.dataset.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

        # self.img_arr = []
        self.img_name_arr = []
        self.attribute_arr = []
        self.target_arr = []

        for attribute, img_names, target in self.dataset:
            self.attribute_arr.append(attribute)
            self.img_name_arr.append(img_names)
            self.target_arr.append(target)

    def __getitem__(self, idx):
        im = Image.open(self.img_name_arr[idx])
        if im.getbands()[0] == 'L':
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        # if im.shape != (3, 224, 224):
        #     print(self.img_name_arr[idx])
        return im, self.attribute_arr[idx], self.img_name_arr[idx], self.target_arr[idx]

    def __len__(self):
        return len(self.dataset)


parser = argparse.ArgumentParser(description='CUB Training')
parser.add_argument('--data-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/awa2',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default="awa2", help='dataset name')
parser.add_argument('--img-size', type=int, default=224, help='image\'s size for transforms')
parser.add_argument('--arch', type=str, default="ResNet50", required=False, help='BB architecture')
parser.add_argument('--labels', nargs='+',
                    default=[
                        "antelope", "grizzly+bear", "killer+whale", "beaver", "dalmatian", "persian+cat", "horse",
                        "german+shepherd", "blue+whale", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus",
                        "leopard", "moose", "spider+monkey", "humpback+whale", "elephant", "gorilla", "ox",
                        "fox", "sheep", "seal", "chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat",
                        "giraffe", "wolf", "chihuahua", "rat", "weasel", "otter", "buffalo", "zebra", "giant+panda",
                        "deer", "bobcat", "pig", "lion", "mouse", "polar+bear", "collie",
                        "walrus", "raccoon", "cow", "dolphin"
                    ])

if __name__ == '__main__':
    args = parser.parse_args()
    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    # print(train_transform)

    dataset = AnimalDataset(args)
    print(len(dataset))
    train_indices, val_indices = train_test_split(
        list(range(len(dataset.target_index))), test_size=0.2, stratify=dataset.target_index
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    print(val_dataset)
    print(len(train_dataset))
    print(len(val_dataset))
    train_ds_awa2 = Awa2(train_dataset, train_transform)
    val_ds_awa2 = Awa2(val_dataset, train_transform)

    train_loader = DataLoader(train_ds_awa2, batch_size=3, shuffle=True, num_workers=4, pin_memory=True)
    a, b, c, d = next(iter(train_loader))
    print(a.size())
    print(b)
    print(c)
    print(d)
    print("---------------------"*10)

    val_loader = DataLoader(val_ds_awa2, batch_size=3, shuffle=False, num_workers=4, pin_memory=True)
    a, b, c, d = next(iter(val_loader))
    print(a.size())
    print(b)
    print(c)
    print(d)

