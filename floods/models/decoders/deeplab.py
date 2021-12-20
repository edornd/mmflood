from typing import List, Type

import torch
from torch import nn

from floods.models import Decoder


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module: this block is responsible for the multi-scale feature extraction,
    using multiple parallel convolutional blocks (conv, bn, relu) with different dilations.
    The four feature groups are then recombined into a single tensor together with an upscaled average pooling
    (that contrasts information loss), then again processed by a 1x1 convolution + dropout
    """
    def __init__(self,
                 in_size: int,
                 in_channels: int,
                 output_stride: int,
                 out_channels: int,
                 norm_layer: Type[nn.Module],
                 act_layer: Type[nn.Module],
                 aspp_channels: int = 256):
        """Creates a new Atrous spatial Pyramid Pooling block. This module is responsible
        for the extraction of features at different scales from the input tensor (which is
        an encoder version of the image with high depth and low height/width).
        The module combines these multi-scale features into a single tensor via 1x convolutions
        Args:
            in_size (int, optional): Size of the input tensor, defaults to 32 for the last layer of ResNet50/101.
            in_channels (int, optional): Channels of the input tensor, defaults to 2048 for ResNet50/101.
            dilations (Tuple[int, int, int, int], optional): dilations, depending on stride. Defaults to (1, 6, 12, 18).
            out_channels (int, optional): Number of output channels. Defaults to 256.
        """
        super().__init__()
        assert output_stride in (8, 16), f"Cannot handle output stride = {output_stride}"
        dil_factor = int(16 // output_stride)  # equals 1 or 2 if os = 8
        dilations = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, aspp_channels, 1, 0, dilations[0], norm_layer, act_layer)
        self.aspp2 = self.aspp_block(in_channels, aspp_channels, 3, dilations[1], dilations[1], norm_layer, act_layer)
        self.aspp3 = self.aspp_block(in_channels, aspp_channels, 3, dilations[2], dilations[2], norm_layer, act_layer)
        self.aspp4 = self.aspp_block(in_channels, aspp_channels, 3, dilations[3], dilations[3], norm_layer, act_layer)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
                                     norm_layer(aspp_channels), act_layer(),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        aspp_output = aspp_channels * 5
        self.merge = self.aspp_block(aspp_output, out_channels, 1, 0, 1, norm_layer, act_layer)
        self.dropout = nn.Dropout(p=0.5)

    def aspp_block(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   batch_norm: Type[nn.Module], activation: Type[nn.Module]) -> nn.Sequential:
        """Creates a basic ASPP block, a sequential module with convolution, batch normalization and relu activation.
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels (usually fixed to 256)
        :type out_channels: int
        :param kernel: kernel size for the convolution (usually 3)
        :type kernel: int
        :param padding: convolution padding, usually equal to the dilation, unless no dilation is applied
        :type padding: int
        :param dilation: dilation for the atrous convolution, depends on ASPPVariant
        :type dilation: int
        :param batch_norm: batch normalization class yet to be instantiated
        :type batch_norm: Type[nn.Module]
        :return: sequential block representing an ASPP component
        :rtype: nn.Sequential
        """
        module = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False), batch_norm(out_channels), activation())
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the ASPP module.
        The same input is processed five times with different dilations. Output sizes are the same,
        except for the pooled layer, which requires an upscaling.
        :param batch: input tensor with dimensions [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: output tensor with dimensions [batch, 256, height, width]
        :rtype: torch.Tensor
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.avgpool(x)
        x5 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.merge(x5)
        return self.dropout(x)


class DecoderV3(nn.Sequential):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for performance, but, if memory is a problem, this can save a little space.
    """
    def __init__(self,
                 in_channels: int = 256,
                 out_channels: int = 256,
                 dropout: float = 0.1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU):
        """Decoder output for the simpler DeepLabV3: this module simply processes the ASPP output
        and upscales it to the input size.The 3x3 convolution and the dropout do not appear in the paper,
        but they are implemented in the official release.
        :param dropout: dropout probability before the final convolution, defaults to 0.1
        :type dropout: float, optional
        :param batch_norm: batch normalization class, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        # yapf: disable
        super(DecoderV3, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        norm_layer(out_channels),
                                        act_layer(),
                                        nn.Dropout(p=dropout))
        # yapf: enable


class DecoderV3Plus(nn.Module):
    """DeepLabV3+ decoder branch, with a skip branch embedding low level
    features (higher resolution) into the highly dimensional output. This typically
    produces much better results than a naive 16x upsampling.
    Original paper: https://arxiv.org/abs/1802.02611
    """
    def __init__(self,
                 skip_channels: int,
                 lower_upscale: int,
                 aspp_channels: int = 256,
                 hidd_channels: int = 256,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU):
        """Returns a new Decoder for DeepLabV3+.
        The upsampling is divided into two parts: a fixed 4x from 128 to 512, and a 2x or 4x
        from 32 or 64 (when input=512x512) to 128, depending on the output stride.
        :param low_level_channels: how many channels on the lo-level skip branch
        :type low_level_channels: int
        :param output_stride: downscaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super().__init__()
        self.low_level = nn.Sequential(nn.Conv2d(skip_channels, 48, 1, bias=False), norm_layer(48), act_layer())
        self.upsample = nn.Upsample(scale_factor=lower_upscale, mode="bilinear", align_corners=True)

        # Table 2, best performance with two 3x3 convs, yapf: disable
        self.output = nn.Sequential(nn.Conv2d(48 + aspp_channels, hidd_channels, 3, stride=1, padding=1, bias=False),
                                    norm_layer(hidd_channels),
                                    act_layer(),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(hidd_channels, hidd_channels, 3, stride=1, padding=1, bias=False),
                                    norm_layer(hidd_channels),
                                    act_layer(),
                                    nn.Dropout(0.1))
        # yapf: enable

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass on the decoder. Low-level features 'skip' are processed and merged
        with the upsampled high-level features 'x'. The output then restores the tensor
        to the original height and width.
        :param x: high-level features, [batch, 2048, X, X], where X = input size / output stride
        :type x: torch.Tensor
        :param skip: low-level features, [batch, Y, 128, 128] where Y = 256 for ResNet, 128 for Xception
        :type skip: torch.Tensor
        :return: tensor with the final output, [batch, classes, input height, input width]
        :rtype: torch.Tensor
        """
        skip = self.low_level(skip)
        x = self.upsample(x)
        return self.output(torch.cat((skip, x), dim=1))


class DeepLabV3(Decoder):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for segmentation performance but, if memory is a problem, this can save a little space.
    """
    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 aspp_channels: int = 256):
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        assert len(feature_channels) == len(feature_reductions) == 1, \
            "DeepLabV3 onlt requires the final encoder output"
        channels = feature_channels[0]
        reduction = feature_reductions[0]
        self.channels = aspp_channels
        self.reduction = reduction
        self.aspp = ASPPModule(in_size=int(input_size / reduction),
                               in_channels=channels,
                               output_stride=reduction,
                               out_channels=aspp_channels,
                               norm_layer=norm_layer,
                               act_layer=act_layer)
        self.decoder = DecoderV3(in_channels=aspp_channels,
                                 out_channels=aspp_channels,
                                 norm_layer=norm_layer,
                                 act_layer=act_layer)

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return (3, )
        return (4, )

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.aspp(*features)
        return self.decoder(x)


class DeepLabV3Plus(Decoder):
    """Decoder for DeepLabV3+, consisting of the classical ASPP with some convolutions, plus a lightweight decoder.
    Instead of a single 16X upscale, the decoder divides it into two separate 4X.
    """
    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 aspp_channels: int = 256) -> None:
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        assert len(feature_channels) == len(feature_reductions) == 2, \
            "DeepLabV3+ requires the last encoder output and one skip connection"
        skip_channels, out_channels = feature_channels
        skip_reduction, out_reduction = feature_reductions
        self.channels = aspp_channels
        self.reduction = skip_reduction
        self.aspp = ASPPModule(in_size=int(input_size / out_reduction),
                               in_channels=out_channels,
                               output_stride=out_reduction,
                               out_channels=aspp_channels,
                               norm_layer=norm_layer,
                               act_layer=act_layer)
        self.decoder = DecoderV3Plus(skip_channels=skip_channels,
                                     lower_upscale=(out_reduction // skip_reduction),
                                     aspp_channels=aspp_channels,
                                     hidd_channels=aspp_channels,
                                     norm_layer=norm_layer,
                                     act_layer=act_layer)

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return (1, 3)
        return (1, 4)

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        skip, out = features
        out = self.aspp(out)
        return self.decoder(out, skip)
