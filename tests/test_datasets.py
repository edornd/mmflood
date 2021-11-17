import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from floods.datasets.flood import FloodDataset
from floods.prepare import eval_transforms

LOG = logging.getLogger(__name__)


def test_dataset_item(dataset_path: Path):
    dataset = FloodDataset(dataset_path, subset="train", ignore_index=255, include_dem=True)
    assert len(dataset.categories()) == 2
    image, mask = dataset.__getitem__(0)
    assert image.shape == (512, 512, 3)
    assert mask.shape == (512, 512)
    assert image.dtype == np.float32
    assert mask.dtype == np.uint8
    assert mask.min() >= 0 and mask.max() <= 255


def test_dataset_iter(dataset_path: Path):
    # transform = eval_transforms(mean=FloodDataset.mean(), std=FloodDataset.std())
    dataset = FloodDataset(dataset_path, subset="train", include_dem=True, transform=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
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
    return

    for image, _ in tqdm(loader):
        valid = label.squeeze(0) != 255
        data = image.squeeze(0)[:, valid]
        std += ((data - mean)**2).sum(axis=-1) / torch.count_nonzero(valid)

    std /= len(loader)
    std = std.sqrt()
    LOG.info(f"std: {std}")
