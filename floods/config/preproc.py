from enum import Enum
from pathlib import Path
from typing import List, Set

from pydantic import Field

from floods.config.base import EnvConfig


class ImageType(Enum):
    SAR = ("sar", 0)
    DEM = ("dem", 0)
    MASK = ("mask", 255)


class StatsConfig(EnvConfig):
    data_root: Path = Field(description="Path where the processed tiles are stored (train set most likely)")
    subset: str = Field("train", description="Which subset to use for statistics, usually  training set")


class PreparationConfig(EnvConfig):
    data_source: Path = Field(description="Path where to find sar/dem/mask folders")
    data_processed: Path = Field(description="Path where to store results")
    subset: Set[str] = Field(["train", "val", "test"],
                             description="Select which subset to preprocess (requires an existing split)")
    summary_file: str = Field(description="JSON file containing all th required information on the dataset")
    tiling: bool = Field(True, description="whether to skip the tiling or not (also skips mask preprocessing)")
    scale: List[int] = Field([1], description="Scaling multipliers for each tile (before resizing to tile_size).")
    tile_size: int = Field(512, description="base dimension of the squared tile")
    tile_max_overlap: int = Field(400, description="how much the tiles can overlap before skipping the next one")
    make_context: bool = Field(False, description="Whether to generate the context-based variant of the set")
    decibel: bool = Field(True, description="Apply a log10 transformation to the SAR signal")
    clip_dem: bool = Field(True, description="Whether to apply min-max normalization to the DEM")
    morphology: bool = Field(True, description="whether to use morphological operators or not")
    morph_kernel: int = Field(5, description="Kernel size for mask preprocessing using opening/closing")
    nan_threshold: float = Field(0.75, description="Percentage of invalid pixels before discaring the tile")
    vv_multiplier: float = Field(5.0, description="Fixed multiplier for threshold-based pseudolabeling (1st channel)")
    vh_multiplier: float = Field(10.0, description="Fixed multiplier for threshold-based pseudolabeling (2nd channel)")

    def subset_exists(cls, v):
        allowed = {"train", "test", "val"}
        if not v:
            raise ValueError("Specify a subset before running")
        if not set(v) <= allowed:
            raise ValueError(f"subsets must belong to: {allowed}")
        return set(v)
