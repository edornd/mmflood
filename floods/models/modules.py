from typing import Type

import torch
from torch import nn

from floods.models.base import Head


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module: this block is responsible for the multi-scale feature extraction,
    using multiple parallel convolutional blocks (conv, bn, relu) with different dilations.
    The four feature groups are then recombined into a single tensor together with an upscaled average pooling
    (that contrasts information loss), then again processed by a 1x1 convolution + dropout
    """
    def __init__(self,
                 in_size: int = 32,
                 in_channels: int = 2048,
                 output_stride: int = 16,
                 out_channels: int = 256,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Creates a new Atrous spatial Pyramid Pooling block. This module is responsible
        for the extraction of features at different scales from the input tensor (which is
        an encoder version of the image with high depth and low height/width).
        The module combines these multi-scale features into a single tensor via 1x convolutions
        Args:
            in_size (int, optional): Size of the input tensor, defaults to 32 for the last layer of ResNet50/101.
            in_channels (int, optional): Channels of the input tensor, defaults to 2048 for ResNet50/101.
            dilations (Tuple[int, int, int, int], optional): dilations, depending on stride. Defaults to (1, 6, 12, 18).
            out_channels (int, optional): Number of output channels. Defaults to 256.
            batch_norm (Type[nn.Module], optional): batch normalization layer. Defaults to nn.BatchNorm2d.
        """
        super().__init__()
        dil_factor = int(output_stride // 16)  # equals 1 or 2 if os = 8
        dilations = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, 256, 1, 0, dilations[0], batch_norm=batch_norm)
        self.aspp2 = self.aspp_block(in_channels, 256, 3, dilations[1], dilations[1], batch_norm=batch_norm)
        self.aspp3 = self.aspp_block(in_channels, 256, 3, dilations[2], dilations[2], batch_norm=batch_norm)
        self.aspp4 = self.aspp_block(in_channels, 256, 3, dilations[3], dilations[3], batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale, yapf: disable
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                                     batch_norm(256),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        self.merge = self.aspp_block(256 * 5, out_channels, kernel=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)
        # yapf: enable

    def aspp_block(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   batch_norm: Type[nn.Module]) -> nn.Sequential:
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
                      bias=False), batch_norm(out_channels), nn.ReLU(inplace=True))
        return module

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the ASPP module.
        The same input is processed five times with different dilations. Output sizes are the same,
        except for the pooled layer, which requires an upscaling.
        :param batch: input tensor with dimensions [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: output tensor with dimensions [batch, 256, height, width]
        :rtype: torch.Tensor
        """
        x1 = self.aspp1(batch)
        x2 = self.aspp2(batch)
        x3 = self.aspp3(batch)
        x4 = self.aspp4(batch)
        x5 = self.avgpool(batch)
        x5 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.merge(x5)
        return self.dropout(x)


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
        self.conv = self._upconv(mid_channels + skip_channels,
                                 out_channels,
                                 act_layer=act_layer,
                                 norm_layer=norm_layer)
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


class UNetHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 scale_factor: int = 2,
                 dropout_prob: float = 0.5,
                 drop_channels: bool = False):
        super().__init__()
        drop_class = nn.Dropout2d if drop_channels else nn.Dropout
        self.dropout = drop_class(p=dropout_prob, inplace=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.out = nn.Conv2d(in_channels, num_classes, kernel_size=1) if num_classes else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.upsample(x)
        return self.out(x)


class SegmentationHead(Head):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        # this contains the number of classes in the current step
        # e.g. with steps 0 1 | 4 5 | 3 6 7, num classes will be 2 | 2 | 3
        self.num_classes = num_classes
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)
