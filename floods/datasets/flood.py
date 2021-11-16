from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from torch import Tensor

from floods.datasets.base import DatasetBase
from floods.utils.functional import imread


class FloodDataset(DatasetBase):

    _categories = {0: "background", 1: "flood"}
    _palette = {0: (0, 0, 0), 1: (255, 255, 255), 255: (255, 0, 255)}

    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 ignore_index: int = 255,
                 include_dem: bool = False,
                 transform: Callable = None) -> None:
        super().__init__()
        self._ignore_index = ignore_index
        self._include_dem = include_dem
        self._name = "flood"
        self._subset = subset
        self.transform = transform
        # gather files to build the list of available pairs
        path = path / subset
        self.image_files = sorted(glob(str(path / "*_sar.tif")))
        self.label_files = sorted(glob(str(path / "*_mask.tif")))
        assert len(self.image_files) > 0, f"No images found, is the given path correct? ({str(path)})"
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching tiles, just in case
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = Path(image).stem.replace("_sar", "")
            label_tile = Path(mask).stem.replace("_mask", "")
            assert image_tile == label_tile, f"image: {image_tile} != mask: {label_tile}"
        # add the optional digital elevation map (DEM)
        if self._include_dem:
            self.dem_files = sorted(glob(str(path / "*_dem.tif")))
            assert len(self.image_files) == len(self.dem_files), "Length mismatch between tiles and DEMs"
            for image, dem in zip(self.image_files, self.dem_files):
                image_tile = Path(image).stem.replace("_sar", "")
                dem_tile = Path(dem).stem.replace("_dem", "")
                assert image_tile == dem_tile, f"image: {image_tile} != dem: {dem_tile}"

    def name(self) -> str:
        return self._name

    def stage(self) -> str:
        return self._subset

    def categories(self) -> Dict[int, str]:
        return self._categories

    def palette(self) -> Dict[int, tuple]:
        return self._palette

    def ignore_index(self) -> int:
        return self._ignore_index

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        assert len(mask) == len(self.image_files), \
            f"Mask is the wrong size! Expected {len(self.image_files)}, got {len(mask)}"
        self.image_files = [x for include, x in zip(mask, self.image_files) if include]
        self.label_files = [x for include, x in zip(mask, self.label_files) if include]
        if self._include_dsm:
            self.dsm_files = [x for include, x in zip(mask, self.dsm_files) if include]
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
        image = imread(self.image_files[index], channels_first=False).astype(np.float32)
        label = imread(self.label_files[index], channels_first=False).astype(np.uint8)
        # add digital elevation map as extra channel to the image
        if self._include_dem:
            dem = imread(self.dem_files[index]).astype(np.float32)
            image = np.dstack((image, dem))
        # preprocess if required, cast mask to Long for torch losses
        if self.transform is not None:
            pair = self.transform(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)
