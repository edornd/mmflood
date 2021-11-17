import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

F32_EPS = np.finfo(np.float32).eps
F16_EPS = np.finfo(np.float16).eps


def identity(*args: Any) -> Any:
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


def one_hot(target: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """source: https://github.com/PhoenixDL/rising. Computes one-hot encoding of input tensor.
    Args:
        target (torch.Tensor): tensor to be converted
        num_classes (Optional[int], optional): number of classes. If None, the maximum value of target is used.
    Returns:
        torch.Tensor: one-hot encoded tensor of the target
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def find_best_checkpoint(folder: Path, model_name: str = "*.pth", divider: str = "_") -> Path:
    wildcard_path = folder / model_name
    models = list(glob(str(wildcard_path)))
    assert len(models) > 0, f"No models found for pattern '{wildcard_path}'"
    current_best = None
    current_best_metric = None

    for model_path in models:
        model_name = Path(model_path).stem
        mtype, _, metric_str = model_name.split(divider)
        assert mtype == "classifier" or mtype == "segmenter", f"Unknown model type '{mtype}'"
        model_metric = float(metric_str.split("-")[-1])
        if not current_best_metric or current_best_metric < model_metric:
            current_best_metric = model_metric
            current_best = model_path

    return current_best


def load_class_weights(weights_path: Path, device: torch.device, normalize: bool = False) -> torch.Tensor:
    # load class weights, if any
    if weights_path is None or not weights_path.exists() or not weights_path.is_file():
        raise ValueError(f"Path '{str(weights_path)}' does not exist or it's not a numpy array")

    weights = np.load(weights_path).astype(np.float32)
    if normalize:
        weights /= weights.max()
    return torch.from_numpy(weights).to(device)
