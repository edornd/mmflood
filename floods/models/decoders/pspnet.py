from typing import List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as fn

from floods.models import Decoder


class PSPBlock(nn.Module):
    """Single Pyramid Pooling block, where the input is shrunk into a smaller spatial dimension,
    then processed by a standard 1x1 convolution + BN + activation.
    """
    def __init__(self, in_channels: int, out_channels: int, pool_size: int, norm_layer: Type[nn.Module],
                 act_layer: Type[nn.Module]):
        super().__init__()
        if pool_size == 1:
            norm_layer = nn.Identity  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                  norm_layer(out_channels), act_layer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = self.pool(x)
        x = fn.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    """Ensemble of parallel PSP blocks. Each block has a different pooling size, so that we gather different
    contexts every time. Results are then concatenated and sent to the lightweight decoder.
    """
    def __init__(self,
                 in_channels: int,
                 norm_layer: Type[nn.Module],
                 act_layer: Type[nn.Module],
                 pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, out_channels, pool_size=size, norm_layer=norm_layer, act_layer=act_layer)
            for size in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPNet(Decoder):
    """PSPNet implementation, which consists of a Pyramid Scene Parsing (PSP) layer, followed by a simple convolutional
    decoder. The PSP module resembles the ASPP from DeepLab, its purpose is to extract multi-scale features from the
    last layer of the encoder. Since it doesn't use features from lower levels, PSPNet works well in general and on large
    image, but it doesn't segment details very well.
    """
    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 out_channels: int = 512,
                 dropout: float = 0.2):
        assert len(feature_channels) == len(feature_reductions) == 1, "PSPNet only requires the last layer"
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        last_channels = feature_channels[-1]
        self.reduction = feature_reductions[-1]
        self.channels = out_channels
        self.psp = PSPModule(in_channels=last_channels, norm_layer=norm_layer, act_layer=act_layer)
        self.conv = nn.Sequential(nn.Conv2d(last_channels * 2, out_channels, kernel_size=1, bias=False),
                                  norm_layer(out_channels), act_layer())
        self.dropout = nn.Dropout2d(p=dropout)

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return (3, )
        return (4, )

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        x = self.psp(*x)
        x = self.conv(x)
        x = self.dropout(x)
        return x
