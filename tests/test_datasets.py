import logging
import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from floods.datasets.flood import FloodDataset

LOG = logging.getLogger(__name__)

VV25 = -29.25275421142578
VV75 = -15.67404842376709
VH25 = -44.751529693603516
VH75 = -30.273120880126953
DEM25 = 19.0
DEM75 = 187.0


# Running mean and variance using Welford's algorithm
class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


def test_dataset_item(dataset_path: Path):
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    assert len(dataset.categories()) == 2
    image, mask = dataset.__getitem__(0)
    assert image.shape == (3, 512, 512), f'invalid image shape: {image.shape}'
    assert mask.shape == (512, 512), f'invalid mask shape: {mask.shape}'
    assert image.dtype == torch.float32, f'invalid image type:  {image.dtype}'
    assert mask.dtype == torch.uint8, f'invalid mask type:  {mask.dtype}'
    assert mask.min() >= 0 and mask.max() <= 255


def sum_sumsq_len_for_channel(data: np.array, perc25: float, perc75: float) -> Union[float, float, float]:
    valid = (data[0] > perc25) & (data[0] < perc75)
    data = data[valid]
    return data.sum(), np.square(data).sum(), len(data)


def test_dataset_iter(dataset_path: Path):

    ####### WORK IN PROGRESS ################

    # transform = eval_transforms(mean=FloodDataset.mean(), std=FloodDataset.std())
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    mean = torch.zeros(3)
    min_val = torch.ones(3) * np.finfo(np.float16).max
    max_val = torch.ones(3) * np.finfo(np.float16).min

    for image, label in tqdm(loader):
        valid = label.squeeze(0) != 255
        data = image.squeeze(0)[:, valid]
        min_val = np.minimum(min_val, data.min(axis=-1))
        max_val = np.maximum(max_val, data.max(axis=-1))
        mean += data.sum(axis=-1) / torch.count_nonzero(valid)

    mean /= len(loader)
    LOG.info(f"mean: {mean}")
    LOG.info(f"min: {min_val}")
    LOG.info(f"max: {max_val}")
