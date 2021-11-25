from abc import ABC, abstractmethod
from typing import Callable, Generator, Union

import numpy as np
import torch

from floods.utils.tiling.functional import tile_fixed_overlap, tile_overlapped
from floods.utils.tiling.smooth import predict_smooth_windowing


class Tiler(ABC):
    """Generic tiling operator
    """
    def __init__(self, tile_size: int, channels_first: bool) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.channels_first = channels_first

    @abstractmethod
    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        return NotImplementedError("Implement in subclass")


class SingleImageTiler(Tiler):
    """'Fake' tiling operator that returns the coordinates for the full image.
    Used to generate the test set and avoid overlapping pixels (tiling will be done at test time).
    """
    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        if len(image.shape) == 2:
            axis = 0 if self.channels_first else -1
            image = np.expand_dims(image, axis=axis)
        if self.channels_first:
            image = np.moveaxis(image, 0, -1)
        height, width, _ = image.shape
        yield (0, 0), (0, 0, height, width)


class DynamicOverlapTiler(Tiler):
    """Tiling operator that provides dynamic overlap based on image size.
    """
    def __init__(self, tile_size: Union[tuple, int], overlap_threshold: int, channels_first: bool = False) -> None:
        super().__init__(tile_size=tile_size, channels_first=channels_first)
        self.overlap_threshold = overlap_threshold

    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        return tile_overlapped(image,
                               tile_size=self.tile_size,
                               overlap_threshold=self.overlap_threshold,
                               channels_first=self.channels_first)


class FixedOverlaptiler(Tiler):
    """Same as dynamic, but using a fixed overlap instead of computing it on the fly.
    """
    def __init__(self, tile_size: int, overlap: int, channels_first: bool = False) -> None:
        super().__init__(tile_size=tile_size, channels_first=channels_first)
        self.overlap = overlap

    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        return tile_fixed_overlap(image,
                                  tile_size=self.tile_size,
                                  overlap=self.overlap,
                                  channels_first=self.channels_first)


class SmoothTiler(Tiler):
    def __init__(self,
                 tile_size: int,
                 num_classes: int,
                 channels_first: bool = False,
                 subdivisions: int = 2,
                 batch_size: int = 4,
                 mirrored: bool = True) -> None:
        super().__init__(tile_size, channels_first)
        self.num_classes = num_classes
        self.subdivisions = subdivisions
        self.batch_size = batch_size
        self.mirrored = mirrored

    def __call__(self, image: torch.Tensor, callback: Callable) -> torch.Tensor:
        return predict_smooth_windowing(image=image,
                                        tile_size=self.tile_size,
                                        subdivisions=self.subdivisions,
                                        num_classes=self.num_classes,
                                        prediction_fn=callback,
                                        batch_size=self.batch_size,
                                        channels_first=self.channels_first,
                                        mirrored=self.mirrored)
