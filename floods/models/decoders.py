from functools import partial
from typing import Iterable, List, Type

import torch
from torch import nn

from floods.models.base import Decoder
from floods.models.modules import UNetDecodeBlock, UNetHead


class DeepLabV3(Decoder):
    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        last = 3 if encoder.startswith("tresnet") else 4
        return (last, )


class DeepLabV3Plus(Decoder):
    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        last = 3 if encoder.startswith("tresnet") else 4
        return (1, last)


class UNet(Decoder):
    def __init__(self,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 bilinear: bool = True,
                 num_classes: int = None,
                 drop_channels: bool = False):
        super().__init__()
        # invert sequences to decode
        channels = feature_channels[::-1]
        reductions = feature_reductions[::-1] + [1]
        scaling_factors = [int(reductions[i] // reductions[i + 1]) for i in range(len(reductions) - 1)]
        # dinamically create the decoder blocks (some encoders do not have all 5 layers)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                UNetDecodeBlock(in_channels=channels[i],
                                mid_channels=channels[i] // 2,
                                skip_channels=channels[i + 1],
                                out_channels=channels[i + 1],
                                act_layer=act_layer,
                                norm_layer=norm_layer,
                                scale_factor=scaling_factors[i],
                                bilinear=bilinear))
        self.out_channels = channels[-1]
        self.out = UNetHead(in_channels=self.out_channels,
                            num_classes=num_classes,
                            scale_factor=scaling_factors[-1],
                            drop_channels=drop_channels)

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return None
        return [i for i in range(5)]

    def output(self) -> int:
        return self.out_channels

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        # features = x1, x2, x3, x4, x5
        x, skips = features[-1], features[:-1]
        # we now start from the bottom and combine x with x4, x3, x2 ...
        for module, feature in zip(self.blocks, reversed(skips)):
            x = module(x, feature)
        return self.out(x)


available_decoders = {
    "unet": partial(UNet, bilinear=True),
    "deeplabv3": partial(DeepLabV3),
    "deeplabv3p": partial(DeepLabV3Plus)
}
