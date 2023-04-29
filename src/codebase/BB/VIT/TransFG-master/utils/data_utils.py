import logging
import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms

from .autoaugment import AutoAugImageNetPolicy
from .dataset_awa2 import AnimalDataset, Awa2
# from .dataset import CarsDataset, NABirds, dogs, INat2017
from .dataset_cubs import Dataset_cub, Waterbird_LandBird_Final_Dataset, DRODatasetFinal
from .dataset_mimic_cxr import MIMICCXRDataset
from .utils_dataset import get_dataset_with_image_and_attributes

logger = logging.getLogger(__name__)


def get_loader(args, mode="train", waterbird_landbird=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.dataset == 'cub' and mode == "train" and waterbird_landbird:
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
            train_data, batch_size=args.train_batch_size, num_workers=4, drop_last=True, pin_memory=True, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=args.eval_batch_size, num_workers=4,
            pin_memory=True
        ) if test_data is not None else None

        return train_loader, val_loader
    elif args.dataset == 'cub' and mode == "test" and waterbird_landbird:
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
        full_dataset = Waterbird_LandBird_Final_Dataset(
            root_dir=args.data_root, concepts_list=args.concept_names, train_transform=train_transform,
            eval_transform=test_transform
        )
        splits = ['train', 'val', 'test']
        subsets = full_dataset.get_splits(splits, train_frac=1.0)
        dro_subsets = [DRODatasetFinal(
            subsets[split], process_item_fn=None, n_classes=full_dataset.n_classes
        ) for split in splits]
        test_data = dro_subsets[2]

        test_loader = DataLoader(
            test_data, batch_size=args.eval_batch_size, num_workers=4,
            pin_memory=True
        ) if test_data is not None else None
        return test_loader

    elif args.dataset == 'cub' and mode == "train":
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_set, train_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="train",
            attribute_file=args.attribute_file_name
        )

        val_set, val_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="test",
            attribute_file=args.attribute_file_name
        )
        trainset = Dataset_cub(train_set, train_attributes, train_transform)
        testset = Dataset_cub(val_set, val_attributes, test_transform)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader
        # print("Loading dataloader for waterbird-landbird")
        # full_dataset = Waterbird_LandBird_Dataset(root_dir=args.data_root, concepts_list=args.concept_names)
        # splits = ['train', 'val', 'test']
        # subsets = full_dataset.get_splits(splits, train_frac=1.0)
        # dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
        #                           n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
        #                for split in splits]
        # train_data = dro_subsets[0]
        # val_data = dro_subsets[1]
        # test_data = dro_subsets[2]
        # loader_kwargs = {'batch_size':16, 'num_workers':4, 'pin_memory':True}
        # train_loader = train_data.get_loader(train=True, reweight_groups=True, **loader_kwargs)
        # val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
        # test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
        # return train_loader, test_loader

    elif args.dataset == 'cub' and mode == "test":
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_set, test_attributes = get_dataset_with_image_and_attributes(
            data_root=args.data_root,
            json_root=args.json_root,
            dataset_name=args.dataset,
            mode="test",
            attribute_file=args.attribute_file_name
        )
        testset = Dataset_cub(test_set, test_attributes, test_transform)
        if args.local_rank == 0:
            torch.distributed.barrier()

        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return test_loader

    elif args.dataset == 'awa2' and mode == "train":
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Prepare dataset
        dataset = AnimalDataset(args)
        train_indices, val_indices = train_test_split(
            list(range(len(dataset.target_index))), test_size=0.2, stratify=dataset.target_index
        )
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_ds_awa2 = Awa2(train_dataset, train_transform)
        val_ds_awa2 = Awa2(val_dataset, test_transform)

        train_loader = DataLoader(
            train_ds_awa2, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            val_ds_awa2, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        return train_loader, test_loader


    elif args.dataset == 'mimic_cxr' and mode == "test":
        print("--- MIMIC-CXR Test ---")
        arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
        arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ])
        dataset_path = os.path.join(
            args.output,
            args.dataset,
            "t",
            args.dataset_folder_concepts,
            "densenet121",
            args.disease_folder,
            "dataset_g",
        )
        print(dataset_path)
        testset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode=mode,
            transform=test_transform,
            feature_path=dataset_path,
            network_type="VIT"
        )

        if args.local_rank == 0:
            torch.distributed.barrier()

        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        test_loader = DataLoader(testset,
                                 shuffle=args.shuffle,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return test_loader

    elif args.dataset == 'mimic_cxr' and mode == "train":
        print("--- MIMIC-CXR Train ---")
        arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
        arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_transform = transform = transforms.Compose([
            transforms.Resize(args.resize),
            # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
            transforms.CenterCrop(args.resize),
            # transforms.RandomRotation(args.degree),
            # transforms.RandomCrop(args.crop),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ])
        dataset_path = os.path.join(
            args.output,
            args.dataset,
            "t",
            args.dataset_folder_concepts,
            "densenet121",
            args.disease_folder,
            "dataset_g",
        )
        print(dataset_path)
        trainset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode='train',
            transform=train_transform,
            feature_path=dataset_path,
            network_type="VIT"
        )

        testset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode='valid',
            transform=test_transform,
            feature_path=dataset_path,
            network_type="VIT"
        )

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  shuffle=args.shuffle,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 shuffle=args.shuffle,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader

    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               # cleaned=os.path.join(data_dir,'cleaned.dat'),
                               transform=transforms.Compose([
                                   transforms.Resize((600, 600), Image.BILINEAR),
                                   transforms.RandomCrop((448, 448)),
                                   transforms.RandomHorizontalFlip(),
                                   AutoAugImageNetPolicy(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                               )
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                              transform=transforms.Compose([
                                  transforms.Resize((600, 600), Image.BILINEAR),
                                  transforms.CenterCrop((448, 448)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader
    elif args.dataset == 'dog':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False
                        )
        testset = dogs(root=args.data_root,
                       train=False,
                       cropped=False,
                       transform=test_transform,
                       download=False
                       )
        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader
    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader
    elif args.dataset == 'INat2017':
        train_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                              transforms.RandomCrop((304, 304)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                             transforms.CenterCrop((304, 304)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader
