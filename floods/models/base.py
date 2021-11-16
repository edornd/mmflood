from abc import abstractclassmethod, abstractmethod
from typing import List, Tuple

import torch
from timm.models.features import FeatureInfo
from torch import nn


class Encoder(nn.Module):
    @property
    @abstractmethod
    def feature_info(self) -> FeatureInfo:
        ...


class Decoder(nn.Module):
    @abstractclassmethod
    def required_indices(cls, encoder: str) -> List[int]:
        ...

    # @abstractmethod
    # def output_channels(self) -> List[int]:
    #     ...

    @abstractmethod
    def output(self) -> int:
        ...


class Head(nn.Module):
    @abstractmethod
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes


class Segmenter(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, head: Head, return_features: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.return_features = return_features

    def forward_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
        head_out = self.head(decoder_out)
        return head_out, decoder_out

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        out, features = self.forward_features(inputs)
        features = features if self.return_features else None
        return out, features

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
