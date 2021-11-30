import logging
import numpy as np
import torch
import time


from pathlib import Path
from plotille import histogram
from torch.utils.data import DataLoader
from datetime import datetime

from floods.datasets.flood import FloodDataset
from floods.config.preproc import StatsConfig

LOG = logging.getLogger(__name__)

# Filter values for VV, VH, DEM to calculate statistics
# Low and high filters are the 25 and 75 percentiles for each channel
LOW_FILTER = np.array([[0.5226023197174072], [0.11324059963226318], [1434.2500]])
HIGH_FILTER = np.array([[1.8945084810256958], [0.4730878472328186], [1676.5000]])

# Plot title and histogram on the console of a given array


def plot_histogram(data: np.ndarray, title: str, low: float = None, high: float = None):
    print(f'Histogram of {title}')
    print(histogram(data,
                    bins=100,
                    X_label='Value',
                    Y_label='Frequency',
                    x_min=low,
                    x_max=high))
    return

# Test a single item from the dataset to make sure it respects given standards


def test_dataset_item(dataset_path: Path):
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    assert len(dataset.categories()) == 2
    image, mask = dataset.__getitem__(0)
    assert image.shape == (3, 512, 512), f'invalid image shape: {image.shape}'
    assert mask.shape == (512, 512), f'invalid mask shape: {mask.shape}'
    assert image.dtype == torch.float32, f'invalid image type:  {image.dtype}'
    assert mask.dtype == torch.uint8, f'invalid mask type:  {mask.dtype}'
    assert mask.min() >= 0 and mask.max() <= 255


# Compute the mean and the standard deviation of a given dataset
def compute_mean_std(dataset_path: Path):

    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    loader = DataLoader(dataset, batch_size=dataset.__len__(), num_workers=4, shuffle=True)

    images, labels = next(iter(loader))
    images = np.moveaxis(images.numpy(), 1, 0)

    valid = labels != 255
    data = images[:, valid]
    idx = (data >= LOW_FILTER) & (data <= HIGH_FILTER)
    data[idx] = np.nan
    with open('outputs/stats.txt', 'a') as file:
        file.write(f'{str(datetime.now())}\n')
        file.write(f'Mean: {np.nanmean(data, axis=1)}\n')
        file.write(f'Std: {np.nanstd(data, axis=1)}\n')

    return

# Compute the mean and the standard deviation of a given dataset


def histograms_for_sanity_check(dataset_path: Path):
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

    images, labels = next(iter(loader))
    images = np.moveaxis(images.numpy(), 1, 0)

    valid = labels != 255
    data = images[:, valid]
    plot_histogram(data[0], 'VV', LOW_FILTER[0][0], HIGH_FILTER[0][0])
    plot_histogram(data[1], 'VH', LOW_FILTER[1][0], HIGH_FILTER[1][0])
    plot_histogram(data[2], 'DEM', LOW_FILTER[2][0], HIGH_FILTER[2][0])

    return


def test_dataset(config: StatsConfig):
    dataset_path = Path(config.data_root)
    test_dataset_item(dataset_path)
    LOG.info(f"Triplet of images correct - test passed")
    histograms_for_sanity_check(dataset_path)
    LOG.info(f"Histograms plotted")
    compute_mean_std(dataset_path)
    LOG.info(f"Iteration of images correct - test passed")
    return
