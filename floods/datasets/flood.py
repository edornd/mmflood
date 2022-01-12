from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from torch import Tensor

from floods.datasets.base import DatasetBase
from floods.utils.gis import imread


class FloodDataset(DatasetBase):

    _name = "flood"
    _categories = {0: "background", 1: "flood"}
    _palette = {0: (0, 0, 0), 1: (255, 255, 255), 255: (255, 0, 255)}
    _ignore_index = 255
    # actually using the median, more stable given the input data
    _mean = (4.9329374e-02, 1.1776519e-02, 1.4241237e+02)
    _std = (3.91287043e-02, 1.03687926e-02, 8.11010422e+01)

    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 include_dem: bool = False,
                 transform_base: Callable = None,
                 transform_sar: Callable = None,
                 transform_dem: Callable = None,
                 normalization: Callable = None) -> None:
        super().__init__()
        self._include_dem = include_dem
        self._name = "flood"
        self._subset = subset
        self.transform_base = transform_base
        self.transform_sar = transform_sar
        self.transform_dem = transform_dem
        self.normalization = normalization
        # gather files to build the list of available pairs
        path = path / subset
        self.image_files = sorted(glob(str(path / "sar" / "*.tif")))
        self.label_files = sorted(glob(str(path / "mask" / "*.tif")))
        assert len(self.image_files) > 0, f"No images found, is the given path correct? ({str(path)})"
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching tiles, just in case
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = Path(image).stem
            label_tile = Path(mask).stem
            assert image_tile == label_tile, f"image: {image_tile} != mask: {label_tile}"
        # add the optional digital elevation map (DEM)
        if self._include_dem:
            self.dem_files = sorted(glob(str(path / "dem" / "*.tif")))
            assert len(self.image_files) == len(self.dem_files), "Length mismatch between tiles and DEMs"
            for image, dem in zip(self.image_files, self.dem_files):
                image_tile = Path(image).stem
                dem_tile = Path(dem).stem
                assert image_tile == dem_tile, f"image: {image_tile} != dem: {dem_tile}"

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def categories(cls) -> Dict[int, str]:
        return cls._categories

    @classmethod
    def palette(cls) -> Dict[int, tuple]:
        return cls._palette

    @classmethod
    def ignore_index(cls) -> int:
        return cls._ignore_index

    @classmethod
    def mean(cls) -> Tuple[float, ...]:
        return cls._mean

    @classmethod
    def std(cls) -> Tuple[float, ...]:
        return cls._std

    def stage(self) -> str:
        return self._subset

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        assert len(mask) == len(self.image_files), \
            f"Mask is the wrong size! Expected {len(self.image_files)}, got {len(mask)}"
        self.image_files = [x for include, x in zip(mask, self.image_files) if include]
        self.label_files = [x for include, x in zip(mask, self.label_files) if include]
        if self._include_dem:
            self.dem_files = [x for include, x in zip(mask, self.dem_files) if include]
        if stage:
            self._subset = stage

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get the image/label pair, with optional augmentations and preprocessing steps.
        Augmentations should be provided for a training dataset, while preprocessing should contain
        the transforms required in both cases (normalizations, ToTensor, ...)
        :param index:   integer pointing to the tile
        :type index:    int
        :return:        image, mask tuple
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # read SAR image and corresponding label, augment with SAR-specific processing
        image = imread(self.image_files[index], channels_first=False).astype(np.float32)
        label = imread(self.label_files[index]).squeeze(0).astype(np.uint8)
        if self.transform_sar is not None:
            pair = self.transform_sar(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        # if requested, add digital elevation map as extra channel to the image
        # also transform with DEM-specific augmentations, if any
        if self._include_dem:
            dem = imread(self.dem_files[index], channels_first=False).astype(np.float32)
            if self.transform_dem is not None:
                pair = self.transform_dem(image=dem, mask=label)
                dem = pair.get("image")
                label = pair.get("mask")
            image = np.dstack((image, dem))
        # last, apply shared transforms (affine transformations) and standardize
        if self.transform_base is not None:
            pair = self.transform_base(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        if self.normalization:
            pair = self.normalization(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)


class RGBFloodDataset(FloodDataset):
    # R, G, B, DEM
    _mean = (0.485, 0.456, 0.406, 1.4241237e+02)
    _std = (0.229, 0.224, 0.225, 8.11010422e+01)


class WeightedFloodDataset(FloodDataset):
    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 include_dem: bool = False,
                 transform_base: Callable = None,
                 transform_sar: Callable = None,
                 transform_dem: Callable = None,
                 normalization: Callable = None,
                 class_weights: Tuple[float, float, float] = (1.0, 0.5, 5.0)) -> None:
        super().__init__(path,
                         subset=subset,
                         include_dem=include_dem,
                         transform_base=transform_base,
                         transform_sar=transform_sar,
                         transform_dem=transform_dem,
                         normalization=normalization)
        # we need 256 positions to account for 255 indices (ignore index)
        weights_array = np.zeros(256, dtype=np.float32)
        weights_array[:len(class_weights)] = np.array(class_weights)
        self.class_weights = weights_array
        self.weight_files = sorted(glob(str(path / subset / "weight" / "*.tif")))
        assert len(self.image_files) == len(self.weight_files), \
            f"Length mismatch between tiles and weights: {len(self.image_files)} != {len(self.weight_files)}"

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image, label = super().__getitem__(index)
        # read the weight map from file
        # 0 = background, 1 = thresholded water U ground truth 2 = threshold ∩ ground truth
        # based on this, we produce a pixel-wise weight map, where we aim at giving more weight
        # to areas where it's flooded and the threshold agrees, less where it's confused
        weight_indices = imread(self.weight_files[index]).squeeze(0).astype(np.uint8)
        weight = self.class_weights[weight_indices]
        return image, label, weight
