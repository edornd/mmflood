import logging
from pathlib import Path

import numpy as np
from plotille import histogram
from typing import Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from floods.datasets.flood import FloodDataset
from floods.prepare import eval_transforms
from floods.config.preproc import StatsConfig

LOG = logging.getLogger(__name__)

VV25 = -29.25275421142578
VV75 = -15.67404842376709
VH25 = -44.751529693603516
VH75 = -30.273120880126953
DEM25 = 19.0
DEM75 = 187.0

import math

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

    vv_rs = RunningStats()
    vh_rs = RunningStats()
    dem_rs = RunningStats()

    # mean = rs.mean()
    # variance = rs.variance()
    # stdev = rs.standard_deviation()
    sum_val = torch.zeros(3)
    sum_sq_val = torch.zeros(3)
    len_val = torch.zeros(3)

    hist_samples = np.zeros(shape=(3,1))

    for i, (image, label) in enumerate(tqdm(loader)):
        # Exclude values which are not valid in the label
        valid = label.squeeze(0) != 255
        data = image.squeeze(0)[:, valid]
        
        for val in data[0].numpy():
            if((val < VV75) and (val > VV25)):
                vv_rs.push(val)
        # # VV channel
        # sum_v, sum_sq_v,  len_v = sum_sumsq_len_for_channel(data[0].numpy(), VV25, VV75)
        # sum_sq_val[0] += sum_sq_v
        # sum_val[0] += sum_v
        # len_val[0] += len_v

        # # VH channel
        # sum_v, sum_sq_v,  len_v = sum_sumsq_len_for_channel(data[1].numpy(), VH25, VH75)
        # sum_sq_val[1] += sum_sq_v
        # sum_val[1] += sum_v
        # len_val[1] += len_v

        # # DEM channel
        # sum_v, sum_sq_v, len_v = sum_sumsq_len_for_channel(data[2].numpy(), DEM25, DEM75)
        # sum_sq_val[2] += sum_sq_v
        # sum_val[2] += sum_v
        # len_val[2] += len_v

        # To improve memory efficiency, we save only 10% of the values of the images
        if(i%100 == 0):
            hist_samples = np.concatenate((hist_samples, data.numpy()), axis=1)

    # LOG.info(f'{hist_samples.shape}')
    LOG.info(f"mean: {vv_rs.mean()}")
    LOG.info(f"std: {np.sqrt((sum_sq_val / len_val) - np.square(sum_val / len_val))}")

    LOG.info(histogram(hist_samples[0,:], bins=50))
    LOG.info(histogram(hist_samples[1,:], bins=50))
    LOG.info(histogram(hist_samples[2,:], bins=50))
    return

def test_dataset(config: StatsConfig):
    dataset_path = Path(config.data_root)
    test_dataset_item(dataset_path)
    LOG.info(f"Triplet of images correct - test passed")
    test_dataset_iter(dataset_path)
    LOG.info(f"Iteration of images correct - test passed")