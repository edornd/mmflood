from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.windows import Window


def imread(path: Path, channels_first: bool = True) -> np.ndarray:
    """Wraps rasterio open functionality to read the numpy array and exit the context.

    Args:
        path (Path): path to the geoTIFF image
        channels_first (bool, optional): whether to return it channels first or not. Defaults to True.

    Returns:
        np.ndarray: image array
    """
    with rasterio.open(str(path), mode="r", driver="GTiff") as src:
        image = src.read()
    return image if channels_first else image.transpose(1, 2, 0)


def mask_raster(path: Path, mask: np.ndarray, mask_value: int = 0) -> None:
    """Masks the given raster (a 3D image with channels first) with the given mask (2D).

    Args:
        path (Path): path to the image to be masked
        mask (np.ndarray): mask containing the pixels to be set to the given value
        mask_value (int, optional): Value for the invalid pixels. Defaults to 0.
    """
    with rasterio.open(str(path), mode="r", driver="GTiff") as src:
        current = src.read()
        profile = src.profile
    with rasterio.open(str(path), mode="w", **profile) as dst:
        assert current.shape[1:] == mask.shape, \
            f"Mask shape ({mask.shape}) doesn't match target shape ({current.shape})"
        current[:, mask] = mask_value
        dst.write(current)


def write_window(window: Window, source: DatasetReader, path: Path, transform: Affine = None) -> None:
    kwargs = source.meta.copy()
    transform = transform or source.transform
    transform = rasterio.windows.transform(window, transform)
    kwargs.update(dict(height=window.height, width=window.width, transform=transform))

    with rasterio.open(str(path), "w", **kwargs) as dst:
        dst.write(source.read(window=window))


def rgb_ratio(sar_image: np.ndarray,
              channels_first: bool = False,
              weights: tuple = (1.0, 1.0, 0.2),
              scale: float = 255,
              dtype: np.dtype = np.uint8) -> np.ndarray:
    assert sar_image.ndim == 3, "batch not supported"
    if channels_first:
        sar_image = sar_image.transpose(1, 2, 0)
    vv = sar_image[:, :, 0]
    vh = sar_image[:, :, 1]
    a, b, c = weights
    rgb = np.stack((a * vv, b * vh, c * (vv / vh)), axis=-1)
    image = np.clip(rgb, 0, 1)
    image = (image * scale).astype(dtype)
    return image
