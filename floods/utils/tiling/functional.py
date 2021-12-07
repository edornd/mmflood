from glob import glob
from typing import Generator, Union

import numpy as np
from tqdm import tqdm

from floods.utils.gis import imread


def tile_overlapped(image: np.ndarray,
                    tile_size: Union[tuple, int],
                    overlap_threshold: int,
                    channels_first: bool = False) -> Generator[tuple, None, None]:
    """Generates a set of tiles with dynamically computed overlap, so that every tile is contained inside the image
    bounds.

    Args:
        image (np.ndarray): input image to be tiled.
        tile_size (Union[tuple, int], optional): size of the tile in pixels, assuming a square tile. Defaults to 256.
        overlap_threshold (int): if it overlaps for more tha X pixels, discard the second one
        channels_first (bool, optional): whether the image has CxHxW format or HxWxC. Defaults to False.

    Raises:
        ValueError: when the image is smaller than a single tile

    Returns:
        Generator[int, int, int, int]: x, y coordinates with x and y offsets to crop windows
    """
    if len(image.shape) == 2:
        axis = 0 if channels_first else -1
        image = np.expand_dims(image, axis=axis)
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    # assume height, width, channels from now on
    height, width, channels = image.shape
    tile_h, tile_w = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    # if the image is too short, pad with ignored
    if height <= tile_h or width <= tile_w:
        height = max(height, tile_h)
        width = max(width, tile_w)
    # number of expected tiles
    tile_count_h = int(np.ceil(height / tile_h))
    tile_count_w = int(np.ceil(width / tile_w))
    # compute total remainder for the expanded window
    remainder_h = (tile_count_h * tile_h) - height
    remainder_w = (tile_count_w * tile_w) - width
    # divide remainders among tiles as overlap (floor to keep overlap to the minimum)
    overlap_h = int(np.floor(remainder_h / float(tile_count_h - 1))) if tile_count_h > 1 else 0
    overlap_w = int(np.floor(remainder_w / float(tile_count_w - 1))) if tile_count_w > 1 else 0
    # special case: tiles (in hor. or ver. direction), with almost complete overlap
    # if the overlapped region is over 'overlap_thres' pixels discard the overlap and cut the
    # final piece out.
    offset_h, offset_w = 0, 0
    if overlap_h >= overlap_threshold:
        tile_count_h -= 1
        overlap_h = 0
        offset_h = (height - (tile_count_h * tile_h)) // 2
    if overlap_w >= overlap_threshold:
        tile_count_w -= 1
        overlap_w = 0
        offset_w = (width - (tile_count_w * tile_h)) // 2

    # iterate rows and columns to compute the effective window positions with offsets
    for row in range(tile_count_h):
        for col in range(tile_count_w):
            # get the starting indices, accounting from initial positions
            x = max(row * tile_h - overlap_h, 0) + offset_h
            y = max(col * tile_w - overlap_w, 0) + offset_w
            # if it exceeds horizontally or vertically in the last rows or cols, increase overlap to fit
            if (x + tile_h) >= height:
                x -= abs(x + tile_h - height)
            if (y + tile_w) >= width:
                y -= abs(y + tile_w - width)
            # yield coordinates
            yield (row, col), (x, y, x + tile_h, y + tile_h)


def tile_fixed_overlap(image: np.ndarray,
                       tile_size: Union[tuple, int],
                       overlap: int,
                       channels_first: bool = False) -> Generator[tuple, None, None]:
    if len(image.shape) == 2:
        axis = 0 if channels_first else -1
        image = np.expand_dims(image, axis=axis)
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    # transform in tuple
    tile_dims = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    # assume channels-last with 3 dims from now on
    height, width, _ = image.shape
    strides = [t - overlap for t in tile_dims]
    tiles_x, tiles_y = [int(np.ceil(dim / float(s))) for dim, s in zip((height, width), strides)]

    for tile_x in range(tiles_x):
        for tile_y in range(tiles_y):
            x = tile_x * strides[0]
            y = tile_y * strides[1]
            yield (tile_x, tile_y), (x, x + tile_dims[0], y, y + tile_dims[1])


def tile_body_water_ratio(image: np.ndarray) -> float:
    """
    Computes the body water ratio from the given image.
    """
    assert len(image.shape) == 3, "Expected 3D image"
    assert image.shape[0] == 1, "Expected single channel image"

    # flat the image into 1D array, then remove nan values
    nan_filtered = image.flat
    nan_filtered = nan_filtered[~np.isnan(nan_filtered)]

    # get counter of 1s
    _, u_cnt = np.unique(nan_filtered, return_counts=True)
    # return the ratio
    if (len(u_cnt) != 2):  # if there is no body water, return the ratio as empty
        return 0

    return u_cnt[1] / len(nan_filtered)


def mask_body_ratio_from_threshold(gt_list: list, ratio_threshold: float) -> np.ndarray:
    """
    Returns a binary mask with the images having a body water ratio above the threshold.
    """

    # for each mask in the path read the image array and get the tile body water ratio
    assert len(gt_list) > 0, "No masks found in the path"

    # check if the mask has been cached already
    if (glob(f'data/cache/_mask_t{str(ratio_threshold)[-2:]}.npy')):

        mask = np.load(f'./data/cache/_mask_t{ratio_threshold}.npy')
        _, counts = np.unique(mask, return_counts=True)

        return mask, counts

    # create the list for filtering the elements, set to zero by default
    mask = np.zeros(len(gt_list))

    for i, gt in enumerate(tqdm(gt_list)):
        # 1. read the image
        # 2. get the body water ratio
        # 3. if the ratio is above the threshold, set the mask to 1, meaning use the image
        image = imread(gt)
        ratio = tile_body_water_ratio(image)
        if ratio >= ratio_threshold:
            mask[i] = 1

    _, counts = np.unique(mask, return_counts=True)

    # save a cache file of the mask for future usage
    np.save(f'./data/cache/_mask_t{str(ratio_threshold)[-2:]}.npy', mask)

    return mask, counts
