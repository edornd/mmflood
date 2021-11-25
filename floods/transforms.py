from typing import Any, Dict, Sequence

import numpy as np
from albumentations import Normalize
from torch import Tensor


class Denormalize:

    def __init__(self, mean: Sequence[float] = (0.485, 0.456, 0.406), std: Sequence[float] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

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
        means = self.mean[:channels]
        stds = self.std[:channels]
        for t, mean, std in zip(tensor, means, stds):
            t[:3].mul_(std - 1e-6).add_(mean)
        # swap from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


class ClipNormalize(Normalize):

    def __init__(self,
                 mean: tuple,
                 std: tuple,
                 clip_min: tuple,
                 clip_max: tuple,
                 max_pixel_value: float = 1.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)
        self.min = clip_min
        self.max = clip_max

    def __call__(self, image, **params) -> Dict[str, Any]:
        image = np.clip(image, self.min, self.max)
        return super().__call__(image=image, **params)
