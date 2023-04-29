import os.path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, Subset


class Dataset_cub(Dataset):
    def __init__(self, dataset, attributes, transform=None, show_image=False):
        self.dataset = dataset
        self.show_image = show_image
        self.transform = transform
        self.attributes = attributes

    def __getitem__(self, item):
        image = self.dataset[item][0]
        label = self.dataset[item][1]
        attributes = self.attributes[item]

        if self.transform:
            image = self.transform(image)
        return image, label, attributes

    def __len__(self):
        return len(self.dataset)


class Dataset_cub_waterbird_landbird(Dataset):
    def __init__(self, dataset_path, file_name_concept, file_name_y, attribute_file_name, image_file_name):
        self.image = torch.load(os.path.join(dataset_path, image_file_name))
        self.concepts = torch.load(os.path.join(dataset_path, file_name_concept))
        self.attributes = torch.load(os.path.join(dataset_path, attribute_file_name))
        self.y = torch.load(os.path.join(dataset_path, file_name_y))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)

        print(self.image.size())
        print(self.concepts.size())
        print(self.attributes.size())
        print(self.y.size())

    def __getitem__(self, item):
        return self.image[item], self.concepts[item], self.attributes[item], self.y[item], self.y_one_hot[item]

    # def __getitem__(self, item):
    #     # image = self.raw_data[item][0]
    #     # if self.transform:
    #     #     image = self.transform(image)
    #
    #     return self.concepts[item], self.attributes[item], self.y[item], self.y_one_hot[item]

    def __len__(self):
        return self.y.size(0)


class Dataset_cub_for_explainer(Dataset):
    def __init__(
            self, dataset_path, file_name_concept, file_name_y, attribute_file_name, raw_data, transform=None
    ):
        self.raw_data = raw_data
        self.transform = transform
        self.concepts = torch.load(os.path.join(dataset_path, file_name_concept))
        self.attributes = torch.load(os.path.join(dataset_path, attribute_file_name))
        self.y = torch.load(os.path.join(dataset_path, file_name_y))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)
        print(self.concepts.size())
        print(self.attributes.size())
        print(self.y.size())

    def __getitem__(self, item):
        image = self.raw_data[item][0]
        if self.transform:
            image = self.transform(image)

        return image, self.concepts[item], self.attributes[item], self.y[item], self.y_one_hot[item]

    def __len__(self):
        return self.y.size(0)

class ConfounderDataset(Dataset):
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        attr = self.attr_array[idx]
        img_filename = os.path.join(self.root_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)
        # Flatten if needed
        x = img

        return x, y, attr

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), split + ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets


class Waterbird_LandBird_Final_Dataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir, concepts_list, train_transform, eval_transform, confounder_names=None):
        self.root_dir = root_dir
        print(self.root_dir)
        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, 'metadata.csv'))
        print(self.metadata_df.shape)

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.n_attrs = 112
        self.attr_array = self.metadata_df.loc[:, concepts_list].values
        print(f"Attr size: {self.attr_array.shape}")
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.features_mat = None
        self.train_transform = train_transform
        self.eval_transform = eval_transform


class DRODatasetFinal(Dataset):
    def __init__(self, dataset, process_item_fn, n_classes):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_classes = n_classes
        y_array = []

        for x, y, attr in self:
            y_array.append(y)
        self._y_array = torch.LongTensor(y_array)
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y in self:
            return x.size()
