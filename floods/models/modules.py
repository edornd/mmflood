from typing import Type

import torch
from torch import nn

from floods.models.base import Head


class SegmentationHead(Head):
    """Binary segmentation head that simply provides a final 1x1 convolution from an arbitrary
    amount of channels to 1 output channel, for binary segmentation of an image.
    """
    def __init__(self, in_channels: int, upscale: int = None, num_classes: int = 1):
        super().__init__(in_channels=in_channels)
        upscale = upscale or 1
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upscale) if upscale > 1 else nn.Identity()
        self.reduce = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        return self.upsample(x).squeeze(dim=1)


class MultimodalAdapter(nn.Module):
    """Self-Supervised Multi-Modal Adaptation block, from https://arxiv.org/abs/1808.03833
    Also called SSMA, it merges together features coming from parallel layers, using Squeeze and Excitation
    blocks, then rescaled the input using channel (sigmoid-based) attention.
    """
    def __init__(self,
                 sar_channels: int,
                 dem_channels: int,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 bottleneck_factor: int = 4):
        super().__init__()
        # compute input channels and bottleneck channels
        total_chs = sar_channels + dem_channels
        bottleneck_chs = total_chs // bottleneck_factor
        self.bottleneck = nn.Sequential(nn.Conv2d(total_chs, bottleneck_chs, kernel_size=3, padding=1, bias=False),
                                        norm_layer(bottleneck_chs), act_layer(),
                                        nn.Conv2d(bottleneck_chs, total_chs, kernel_size=3, padding=1, bias=False),
                                        norm_layer(total_chs), nn.Sigmoid())
        # the output is given by the RGB network, which is supposed to be bigger
        # also, this allows for easier integration with decoders
        self.out_bn = norm_layer(total_chs)
        self.out_conv = nn.Conv2d(total_chs, sar_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, sar: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        x1 = torch.cat((sar, dem), dim=1)
        x = self.bottleneck(x1)
        recalibrated = x1 * x
        x = self.out_bn(recalibrated)
        return self.out_conv(x)
