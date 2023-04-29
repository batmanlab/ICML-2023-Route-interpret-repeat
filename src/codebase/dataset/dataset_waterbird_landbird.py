import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset

from dataset.ConfounderDataset import ConfounderDataset

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}


class Waterbird_LandBird_Dataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir, concepts_list):
        self.root_dir = root_dir
        print(self.root_dir)
        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, 'metadata.csv'))
        print(self.metadata_df.shape)

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.n_attrs = 110
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')
        self.attr_array = self.metadata_df.loc[:, concepts_list].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.features_mat = None
        scale = 256.0 / 224.0
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((int(224 * scale), int(224 * scale))),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        for x, y, attr, g in self:
            group_array.append(g)
            y_array.append(y)
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g in self:
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train:  # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups:  # Training but not reweighting
            shuffle = True
            sampler = None
        else:  # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self) / self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
