import logging
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from plotille import histogram
from tqdm.contrib.logging import logging_redirect_tqdm

from floods.datasets.flood import FloodDataset
from floods.utils.tiling.functional import weights_from_body_ratio

LOG = logging.getLogger(__name__)


# Plot title and histogram on the console of a given array
def plot_histogram(data: np.ndarray, title: str, low: float = None, high: float = None):
    print(f'Histogram of {title}')
    print(histogram(data, bins=100, X_label='Value', Y_label='Frequency', x_min=low, x_max=high))
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


def test_weighted_sampler(dataset_path: Path):
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform_base=None)
    with logging_redirect_tqdm():
        weights0 = weights_from_body_ratio(dataset.label_files, smoothing=0)
        weights1 = weights_from_body_ratio(dataset.label_files, smoothing=0.2)
        weights2 = weights_from_body_ratio(dataset.label_files, smoothing=0.9)
        LOG.info("min: %.2f - max: %.2f", weights0.min(), weights0.max())
        LOG.info("min: %.2f - max: %.2f", weights1.min(), weights1.max())
        LOG.info("min: %.2f - max: %.2f", weights2.min(), weights2.max())
        plt.xlim(0, 1)
        sns.histplot(weights0, bins=100, color="r")
        sns.histplot(weights1, bins=100, color="g")
        sns.histplot(weights2, bins=100, color="b")
        plt.savefig("result.png")
