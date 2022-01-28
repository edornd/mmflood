from typing import Sequence, Union

import numpy as np
import torch
from albumentations import Normalize
from torch import Tensor


class Denormalize:

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        single_image = tensor.ndim == 3
        tensor = tensor.unsqueeze(0) if single_image else tensor
        channels = tensor.size(1)
        # slice to support a lower number of channels
        means = self.mean[:channels].view(1, -1, 1, 1).to(tensor.device)
        stds = self.std[:channels].view(1, -1, 1, 1).to(tensor.device)
        tensor = tensor * stds + means
        # swap from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


class ClipNormalize(Normalize):

    def __init__(self,
                 mean: tuple,
                 std: tuple,
                 clip_min: Union[float, tuple],
                 clip_max: Union[float, tuple],
                 max_pixel_value: float = 1.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        result = super().apply(image=image, **params)
        return np.clip(result, self.clip_min, self.clip_max)

    def get_transform_init_args_names(self):
        parent = list(super().get_transform_init_args_names())
        return tuple(parent + ["clip_min", "clip_max"])
