import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

F32_EPS = np.finfo(np.float32).eps
F16_EPS = np.finfo(np.float16).eps


def identity(args: Any) -> Any:
    return args


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def progressbar(dataloder: DataLoader, epoch: int = 0, stage: str = "train", disable: bool = False):
    pbar = tqdm(dataloder, file=sys.stdout, unit="batch", postfix={"loss": "--"}, disable=disable)
    pbar.set_description(f"Epoch {epoch:<3d} - {stage}")
    return pbar


def get_rank() -> int:
    if not distributed.is_initialized():
        return 0
    return distributed.get_rank()


def only_rank(rank: int = 0):
    def decorator(wrapped_fn: Callable):
        def wrapper(*args, **kwargs):
            if not distributed.is_initialized() or distributed.get_rank() == rank:
                return wrapped_fn(*args, **kwargs)

        return wrapper

    return decorator


def find_best_checkpoint(folder: Path, model_name: str = "*.pth", divider: str = "_") -> Path:
    wildcard_path = folder / model_name
    models = list(glob(str(wildcard_path)))
    assert len(models) > 0, f"No models found for pattern '{wildcard_path}'"
    current_best = None
    current_best_metric = None

    for model_path in models:
        model_name = Path(model_path).stem
        # expecting model_loss-value_metric-value, so take only the last
        metric_str = model_name.rsplit(divider, maxsplit=1)[-1]
        model_metric = float(metric_str.split("-")[-1])
        if not current_best_metric or current_best_metric < model_metric:
            current_best_metric = model_metric
            current_best = model_path

    return Path(current_best)


def load_class_weights(weights_path: Path, device: torch.device, normalize: bool = False) -> torch.Tensor:
    # load class weights, if any
    if weights_path is None or not weights_path.exists() or not weights_path.is_file():
        raise ValueError(f"Path '{str(weights_path)}' does not exist or it's not a numpy array")

    weights = np.load(weights_path).astype(np.float32)
    if normalize:
        weights /= weights.max()
    return torch.from_numpy(weights).to(device)


def compute_class_weights(data: Dict[Any, int], smoothing: float = 0.15, clip: float = 10.0):
    assert smoothing >= 0 and smoothing <= 1, "Smoothing factor out of range"
    if smoothing > 0:
        # the larger the smooth factor, the bigger the quantities to sum to the remaining counts (additive smoothing)
        smoothed_maxval = max(list(data.values())) * smoothing
        for k in data.keys():
            data[k] += smoothed_maxval
    # retrieve the (new) max value, divide by counts, round to 2 digits and clip to the given value
    # max / value allows to keep the majority class' weights to 1, while the others will be >= 1 and <= clip
    majority = max(data.values())
    return {k: np.clip(round(float(majority / v), ndigits=2), 0, clip) for k, v in data.items()}


def initialize_weights(*models: nn.Sequential) -> None:
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(module, f):
    child = module if isinstance(module, (list, tuple)) else list(module.children())
    if isinstance(module, nn.Module):
        f(module)
    if len(child) > 0:
        for layer in child:
            apply_leaf(layer, f)


def set_trainable(layers: list, value: bool) -> None:
    apply_leaf(layers, lambda m: set_trainable_attr(m, value))


def entropy(label: np.ndarray, ignore: int = 255) -> np.ndarray:
    valid = label.copy()
    valid[valid == ignore] = 0
    marg = np.histogramdd(valid.ravel(), bins=2)[0] / label.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    return -np.sum(np.multiply(marg, np.log2(marg)))
