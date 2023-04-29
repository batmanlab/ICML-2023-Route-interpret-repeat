import os
import pickle
import torch

import pandas as pd
import numpy as np
from PIL import Image


def get_concept_loaders(dataset_name, n_samples=50, batch_size=100, num_workers=4, seed=1):
    if dataset_name == "cub":
       return cub_concept_loaders(n_samples, batch_size, num_workers, seed)

    # elif dataset_name == "derm7pt":
    #     return derm7pt_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    #
    # elif dataset_name == "broden":
    #     return broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")