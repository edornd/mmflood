from typing import Iterable, List, Type

import torch
from torch import nn

from floods.models import Decoder


class UNetDecodeBlock(nn.Module):
    """UNet basic block, providing an upscale from the lower features and a skip connection
    from the encoder. This specific version adopts a residual skip similar to ResNets.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 scale_factor: int = 2,
                 bilinear: bool = True):
        """Creates a new UNet block, with residual skips.

        Args:
            in_channels (int): number of input channels
            skip_channels (int): number of channels coming from the skip connection (usually 2 * input)
            out_channels (int): number of desired channels in output
            scale_factor (int, optional): How much should the input be scaled. Defaults to 2.
            bilinear (bool, optional): Upscale with bilinear and conv1x1 or transpose conv. Defaults to True.
            norm_layer (Type[nn.Module]: normalization layer.
            act_layer (Type[nn.Module]): activation layer.
        """
        super().__init__()
        self.upsample = self._upsampling(in_channels, mid_channels, factor=scale_factor, bilinear=bilinear)
        self.conv = self._upconv(mid_channels + skip_channels, out_channels, act_layer=act_layer, norm_layer=norm_layer)
        self.adapter = nn.Conv2d(mid_channels, out_channels, 1) if mid_channels != out_channels else nn.Identity()

    def _upsampling(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = True):
        """Create the upsampling block, bilinear + conv1x2 or convTranspose2D. The former typically yields
        better results, avoiding checkerboards artifacts.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            factor (int): upscaling factor
            bilinear (bool, optional): Use bilinear or upconvolutions. Defaults to True.

        Returns:
            nn.Module: upsampling block
        """
        if bilinear:
            return nn.Sequential(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)

    def _upconv(self, in_channels: int, out_channels: int, act_layer: Type[nn.Module],
                norm_layer: Type[nn.Module]) -> nn.Sequential:
        """Creates a decoder block in the UNet standard architecture.
        Two conv3x3 with batch norms and activations.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            act_layer (Type[nn.Module]): activation layer
            norm_layer (Type[nn.Module]): normalization layer

        Returns:
            nn.Sequential: UNet basic decoder block.
        """
        mid_channels = out_channels
        return nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(mid_channels), act_layer(),
                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(out_channels), act_layer())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x2 = self.conv(torch.cat((x, skip), dim=1))
        x1 = self.adapter(x)
        return x1 + x2


class UNet(Decoder):
    """UNet architecture with dynamic adaptation to the encoder.
    This UNet variant also has residual skips in the decoder, just to be fancy, other than that is faithful.
    """
    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 bilinear: bool = True,
                 num_classes: int = None,
                 drop_channels: bool = False,
                 dropout_prob: int = 0.5):
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
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
        drop_class = nn.Dropout2d if drop_channels else nn.Dropout
        self.dropout = drop_class(p=dropout_prob)
        self.channels = channels[-1]
        self.reduction = scaling_factors[-1]

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return None
        return [i for i in range(5)]

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        # features = x1, x2, x3, x4, x5
        x, skips = features[-1], features[:-1]
        # we now start from the bottom and combine x with x4, x3, x2 ...
        for module, feature in zip(self.blocks, reversed(skips)):
            x = module(x, feature)
        return self.dropout(x)
