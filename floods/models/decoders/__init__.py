from functools import partial

from floods.models.decoders.deeplab import DeepLabV3, DeepLabV3Plus
from floods.models.decoders.pspnet import PSPNet
from floods.models.decoders.unet import UNet

__all__ = ["DeepLabV3", "DeepLabV3Plus", "UNet", "PSPNet"]

available_decoders = {
    "unet": partial(UNet, bilinear=True),
    "pspnet": partial(PSPNet),
    "deeplabv3": partial(DeepLabV3),
    "deeplabv3p": partial(DeepLabV3Plus)
}
