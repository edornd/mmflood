from functools import partial

import torch
from torch import nn

from floods.models import create_decoder, create_encoder
from floods.models.base import Segmenter
from floods.models.modules import SegmentationHead

NORM_LAYER = nn.BatchNorm2d
ACT_LAYER = partial(nn.ReLU, inplace=True)


def create_model(input_size: int, encoder: str, decoder: str):
    encoder = create_encoder(name=encoder,
                             decoder=decoder,
                             pretrained=False,
                             freeze=False,
                             output_stride=8,
                             act_layer=ACT_LAYER,
                             norm_layer=NORM_LAYER,
                             channels=3)
    decoder = create_decoder(name=decoder,
                             input_size=input_size,
                             feature_info=encoder.feature_info,
                             act_layer=ACT_LAYER,
                             norm_layer=NORM_LAYER)
    # create final segmentation head and build model
    head = SegmentationHead(in_channels=decoder.out_channels(), upscale=decoder.out_reduction())

    return Segmenter(encoder, decoder, head, return_features=False)


def test_resnet34_unet():
    model = create_model(input_size=512, encoder="resnet34", decoder="unet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet50_unet():
    model = create_model(input_size=512, encoder="resnet50", decoder="unet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_efficientnet_unet():
    model = create_model(input_size=512, encoder="efficientnet_b3", decoder="unet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_tresnet_unet():
    model = create_model(input_size=512, encoder="tresnet_m", decoder="unet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet50_deeplabv3():
    model = create_model(input_size=512, encoder="resnet50", decoder="deeplabv3")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet101_deeplabv3():
    model = create_model(input_size=512, encoder="resnet101", decoder="deeplabv3")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_efficientnet_deeplabv3():
    model = create_model(input_size=512, encoder="efficientnet_b3", decoder="deeplabv3")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_tresnet_deeplabv3():
    model = create_model(input_size=512, encoder="tresnet_m", decoder="deeplabv3")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet50_deeplabv3plus():
    model = create_model(input_size=512, encoder="resnet50", decoder="deeplabv3p")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet101_deeplabv3plus():
    model = create_model(input_size=512, encoder="resnet101", decoder="deeplabv3p")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_efficientnet_deeplabv3plus():
    model = create_model(input_size=512, encoder="efficientnet_b3", decoder="deeplabv3p")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_tresnet_deeplabv3plus():
    model = create_model(input_size=512, encoder="tresnet_m", decoder="deeplabv3p")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_resnet50_pspnet():
    model = create_model(input_size=512, encoder="resnet50", decoder="pspnet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_efficientnet_pspnet():
    model = create_model(input_size=512, encoder="efficientnet_b3", decoder="pspnet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)


def test_tresnet_pspnet():
    model = create_model(input_size=512, encoder="tresnet_m", decoder="pspnet")
    x = torch.rand((4, 3, 512, 512))
    out = model(x)
    assert out.shape == (4, 512, 512)
