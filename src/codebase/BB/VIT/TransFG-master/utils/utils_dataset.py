import json
import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from .dataset_cubs import Dataset_cub, Dataset_cub_waterbird_landbird


def get_dataset_with_image_and_attributes(
        data_root,
        json_root,
        dataset_name,
        mode,
        attribute_file
):
    print(data_root)
    print("------------------")
    data_json = os.path.join(
        json_root,
        f"{mode}_samples_{dataset_name}.json"
    )
    print(data_json)

    if os.path.isfile(data_json):
        with open(os.path.join(data_json), "r") as f:
            json_file = json.load(f)
            data_samples = json_file["samples"]

    print(f"Length of the [{mode}] dataset: {len(data_samples)}")
    img_set = ImageFolder(data_root)
    img_dataset = [img_set[index] for index in data_samples]
    attributes = np.load(os.path.join(data_root, attribute_file))[data_samples]
    print(f"Attribute size: {attributes.shape}")
    return img_dataset, attributes

def get_dataset_with_image_and_attributes_waterbird_landbird(
        data_root,
        json_root,
        dataset_name,
        mode,
        attribute_file
):
    print(f"Attribute_file: {attribute_file}")
    data_json = os.path.join(
        json_root,
        f"{mode}_samples_{dataset_name}.json"
    )

    if os.path.isfile(data_json):
        with open(os.path.join(data_json), "r") as f:
            json_file = json.load(f)
            data_samples = json_file["samples"]

    print(f"Length of the [{mode}] dataset: {len(data_samples)}")
    img_set = ImageFolder(data_root)
    img_dataset = [img_set[index] for index in data_samples]
    y = pd.read_csv(os.path.join(data_root, "metadata.csv"), usecols=['y']).to_numpy().flatten()

    attributes = np.load(os.path.join(data_root, attribute_file))[data_samples]
    return img_dataset, attributes, y


def get_dataloader_spurious_waterbird_landbird(
        data_root, json_root, dataset_name, batch_size, train_transform, val_transform, train_shuffle=True,
        attribute_file="attributes.npy"):
    print("Loading dataloader for waterbird-landbird")
    train_set, train_attributes, y = get_dataset_with_image_and_attributes_waterbird_landbird(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train",
        attribute_file=attribute_file
    )
    print(y)
    val_set, val_attributes, y = get_dataset_with_image_and_attributes_waterbird_landbird(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file=attribute_file
    )

    train_dataset = Dataset_cub_waterbird_landbird(train_set, train_attributes, y, train_transform)
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=train_shuffle,
        pin_memory=True
    )

    val_dataset = Dataset_cub_waterbird_landbird(val_set, val_attributes, y, val_transform)
    val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader

def get_dataloader(data_root, json_root, dataset_name, batch_size, train_transform, val_transform, train_shuffle=True,
                   attribute_file="attributes.npy"):
    if dataset_name == "cub":
        train_set, train_attributes = get_dataset_with_image_and_attributes(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="train",
            attribute_file=attribute_file
        )

        val_set, val_attributes = get_dataset_with_image_and_attributes(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="val",
            attribute_file=attribute_file
        )

        train_dataset = Dataset_cub(train_set, train_attributes, train_transform)
        train_loader = DataLoader(
            train_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=train_shuffle,
            pin_memory=True
        )

        val_dataset = Dataset_cub(val_set, val_attributes, val_transform)
        val_loader = DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader


def get_test_dataloader(
        data_root, json_root, dataset_name, batch_size, test_transform, attribute_file="attributes.npy"
):
    if dataset_name == "cub":
        test_set, test_attributes = get_dataset_with_image_and_attributes(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="test",
            attribute_file=attribute_file
        )

        test_dataset = Dataset_cub(test_set, test_attributes, test_transform)
        test_loader = DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        return test_loader
