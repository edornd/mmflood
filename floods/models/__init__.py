from typing import Type

import timm
from timm.models.features import FeatureInfo
from torch import nn

from floods.models.base import Decoder, Encoder
from floods.models.decoders import available_decoders
from floods.models.encoders import available_encoders


def filter_encoder_args(encoder: str, pretrained: bool, **kwargs: dict) -> dict:
    """Some encoders have different structures and/or different parameters.
    Especially efficient ones (TResNets, EfficientNets) have such a custom architectures that some
    parameters don't event exist, thus filter them out.
    """
    exclude = set()
    if encoder.startswith("tresnet"):
        exclude = exclude.union(["norm_layer", "act_layer", "output_stride"])
    if encoder.startswith("efficientnet") or pretrained:
        exclude = exclude.union(["norm_layer", "act_layer"])
    for arg in exclude:
        kwargs.pop(arg, None)
    return kwargs


def create_encoder(name: str,
                   decoder: str,
                   pretrained: bool,
                   freeze: bool,
                   output_stride: int,
                   act_layer: Type[nn.Module],
                   norm_layer: Type[nn.Module],
                   channels: int = 3,
                   **kwargs) -> Encoder:
    # assert that the encoder exists or is among custom ones
    assert name in available_encoders, f"Encoder '{name}' not supported"
    assert decoder in available_decoders, f"Decoder '{name}' not supported"
    # build a dictionary of additional arguments, not every model has them
    additional_args = kwargs or {}
    additional_args.update(act_layer=act_layer, norm_layer=norm_layer, output_stride=output_stride)
    additional_args = filter_encoder_args(encoder=name, pretrained=pretrained, **additional_args)
    additional_args.update(in_chans=channels)
    # create the encoder, selecting indices to be returned
    # this depends on the decoder first of all, and then on the encoder structure
    # - TResNets have reductions:   4, 8, 16, 32
    # - ResNets have reductions: 2, 4, 8, 16, 32
    indices = available_decoders[decoder].func.required_indices(encoder=name)
    model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=indices, **additional_args)
    # if channels > 3:
    #     model = expand_input(model, num_copies=(channels - 3))
    # freeze layers in the encoder if required
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    # return the encoder
    return model


def create_decoder(name: str, feature_info: FeatureInfo, act_layer: Type[nn.Module], norm_layer: Type[nn.Module],
                   **kwargs: dict) -> Decoder:
    # sanity check to keep going with no worries
    assert name in available_decoders, f"Decoder '{name}' not implemented"
    # retrieve the partial object and instantiate with the common params
    decoder_class = available_decoders.get(name)
    decoder = decoder_class(feature_channels=feature_info.channels(),
                            feature_reductions=feature_info.reduction(),
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            **kwargs)
    return decoder
