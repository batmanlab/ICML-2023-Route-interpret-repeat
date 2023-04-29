import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import utils
from dataset.dataset_awa2 import AnimalDataset
from dataset.dataset_cubs import Dataset_cub, Waterbird_LandBird_Final_Dataset, DRODatasetFinal, \
    Dataset_cub_for_explainer, Dataset_cub_waterbird_landbird
from utils import get_train_val_transforms


def get_dataset_with_image_and_attributes(
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
    attributes = np.load(os.path.join(data_root, attribute_file))[data_samples]
    print(attributes.shape)
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


def get_transforms_cub_water_bird_land_bird(args):
    train_transform = None
    test_transform = None
    if args.img_size == 224:
        scale = 256.0 / 224.0
        train_transform = transforms.Compose([
            transforms.Resize((int(args.img_size * scale), int(args.img_size * scale))),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.img_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif args.img_size == 448:
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transform, test_transform


def get_explainer_dataloader_spurious_waterbird_landbird(args, dataset_path):
    if args.spurious_waterbird_landbird:
        print("Loading dataloader for waterbird-landbird")
        if args.projected == "n":
            train_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt",
                "train_image_tensor.pt",
            )
            val_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "val_proba_concepts.pt", "val_class_labels.pt", "val_attributes.pt",
                "val_image_tensor.pt",
            )

            test_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt",
                "test_image_tensor.pt",
            )
        elif args.projected == "y":
            train_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "train_select_proba_auroc_concepts.pt", "train_class_labels.pt",
                "train_attributes.pt", "train_image_tensor.pt",
            )
            val_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "val_select_proba_auroc_concepts.pt", "val_class_labels.pt",
                "val_attributes.pt", "val_image_tensor.pt",
            )

            test_dataset = Dataset_cub_waterbird_landbird(
                dataset_path, "test_select_proba_auroc_concepts.pt", "test_class_labels.pt",
                "test_attributes.pt", "test_image_tensor.pt",
            )

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader

    else:
        train_data, train_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="train",
            attribute_file=args.attribute_file_name
        )

        val_data, val_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="val",
            attribute_file=args.attribute_file_name
        )

        test_data, test_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="test",
            attribute_file=args.attribute_file_name
        )
        transforms = get_train_val_transforms(args.dataset, args.img_size, args.arch)
        train_transform = transforms["train_transform"]
        val_transform = transforms["val_transform"]

        train_dataset = Dataset_cub_for_explainer(
            dataset_path, "train_proba_concepts.pt", "train_class_labels.pt", "train_attributes.pt",
            train_data, train_transform
        )
        val_dataset = Dataset_cub_for_explainer(
            dataset_path, "val_proba_concepts.pt", "val_class_labels.pt", "val_attributes.pt", val_data,
            val_transform
        )

        test_dataset = Dataset_cub_for_explainer(
            dataset_path, "test_proba_concepts.pt", "test_class_labels.pt", "test_attributes.pt", test_data,
            val_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader


def get_dataloader_spurious_waterbird_landbird(args):
    print("Loading dataloader for waterbird-landbird")
    train_transform, test_transform = get_transforms_cub_water_bird_land_bird(args)

    full_dataset = Waterbird_LandBird_Final_Dataset(
        root_dir=args.data_root, concepts_list=args.concept_names, train_transform=train_transform,
        eval_transform=test_transform
    )
    splits = ['train', 'val', 'test']
    subsets = full_dataset.get_splits(splits, train_frac=1.0)
    dro_subsets = [DRODatasetFinal(
        subsets[split], process_item_fn=None, n_classes=full_dataset.n_classes
    ) for split in splits]
    train_data = dro_subsets[0]
    val_data = dro_subsets[1]
    test_data = dro_subsets[2]

    train_loader = DataLoader(
        train_data, batch_size=args.bs, num_workers=4, drop_last=True, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.bs, num_workers=4,
        pin_memory=True
    ) if val_data is not None else None
    test_loader = DataLoader(
        test_data, batch_size=args.bs, num_workers=4,
        pin_memory=True
    ) if test_data is not None else None

    # args.concept_names.append("Labels")
    #
    # full_dataset = Waterbird_LandBird_Dataset(root_dir=args.data_root, concepts_list=args.concept_names)
    # splits = ['train', 'val', 'test']
    # subsets = full_dataset.get_splits(splits, train_frac=1.0)
    # dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
    #                           n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
    #                for split in splits]
    # train_data = dro_subsets[0]
    # val_data = dro_subsets[1]
    # test_data = dro_subsets[2]
    # loader_kwargs = {'batch_size': args.bs, 'num_workers': 4, 'pin_memory': True}
    # train_loader = train_data.get_loader(train=True, reweight_groups=True, **loader_kwargs)
    # val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    # test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    # print(f"Train loader: {len(train_loader.dataset)}")
    # print(f"Val loader: {len(val_loader.dataset)}")
    # print(f"test loader: {len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader


def get_dataloader(
        data_root, json_root, dataset_name, batch_size, train_transform, val_transform, train_shuffle=True,
        attribute_file="attributes.npy"
):
    print(f"Attribute_file: {attribute_file}")
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

    elif dataset_name == "awa2":
        transforms = utils.get_train_val_transforms(dataset_name, args.img_size, args.arch)
        train_transform = transforms["train_transform"]
        val_transform = transforms["val_transform"]

        dataset = AnimalDataset(args)
        train_indices, val_indices = train_test_split(
            list(range(len(dataset.target_index))), test_size=0.2, stratify=dataset.target_index
        )


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
