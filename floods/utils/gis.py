from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.windows import Window


def imread(path: Path, channels_first: bool = True, return_metadata: bool = False) -> np.ndarray:
    """Wraps rasterio open functionality to read the numpy array and exit the context.

    Args:
        path (Path): path to the geoTIFF image
        channels_first (bool, optional): whether to return it channels first or not. Defaults to True.

    Returns:
        np.ndarray: image array
    """
    with rasterio.open(str(path), mode="r", driver="GTiff") as src:
        image = src.read()
        metadata = src.profile.copy()
    image = image if channels_first else image.transpose(1, 2, 0)
    if return_metadata:
        return image, metadata
    return image


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
    """Stores the data inside the given window, cutting it from the source dataset.
    The image is stored in the given path. When the transform is None, the source transform is used instead.

    Args:
        window (Window): rasterio Window to delimit the target image
        source (DatasetReader): source TIFF to be cut
        path (Path): path to the target file to be created
        transform (Affine, optional): Optional alternative transform. Defaults to None.
    """
    kwargs = source.meta.copy()
    transform = transform or source.transform
    transform = rasterio.windows.transform(window, transform)
    kwargs.update(dict(height=window.height, width=window.width, transform=transform))

    with rasterio.open(str(path), "w", **kwargs) as dst:
        dst.write(source.read(window=window))


def rgb_ratio(sar_image: np.ndarray,
              channels_first: bool = False,
              weights: tuple = (2.0, 5.0, 0.1),
              scale: float = 255,
              dtype: np.dtype = np.uint8) -> np.ndarray:
    assert sar_image.ndim == 3, "batch not supported"
    if channels_first:
        sar_image = sar_image.transpose(1, 2, 0)
    vv = sar_image[:, :, 0]
    vh = sar_image[:, :, 1]
    a, b, c = weights
    rgb = np.stack((a * vv, b * vh, c * (vv / vh)), axis=-1)
    return as_image(rgb, scale=scale, dtype=dtype)


def as_image(array: np.ndarray, scale: float = 255, dtype: np.dtype = np.uint8) -> np.ndarray:
    image = np.clip(array, 0, 1)
    return (image * scale).astype(dtype)
