from abc import abstractclassmethod, abstractmethod
from itertools import chain
from typing import Iterator, List, Tuple, Type

import torch
from timm.models.features import FeatureInfo
from torch import nn


class Encoder(nn.Module):
    @property
    @abstractmethod
    def feature_info(self) -> FeatureInfo:
        ...


class Decoder(nn.Module):
    @abstractmethod
    def __init__(self, input_size: int, feature_channels: List[int], feature_reductions: List[int],
                 act_layer: Type[nn.Module], norm_layer: Type[nn.Module]):
        super().__init__()

    @abstractclassmethod
    def required_indices(cls, encoder: str) -> List[int]:
        ...

    @abstractmethod
    def out_channels(self) -> int:
        ...

    @abstractmethod
    def out_reduction(self) -> int:
        ...


class Head(nn.Module):
    @abstractmethod
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels


class Segmenter(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, head: Head, return_features: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.return_features = return_features

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x = self.encoder(x)
        features = self.decoder(x)
        out = self.head(features)
        return (out, features) if self.return_features else out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def encoder_params(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_params(self) -> Iterator[nn.Parameter]:
        return chain(self.decoder.parameters(), self.head.parameters())


class MultiBranchSegmenter(Segmenter):
    def __init__(self, encoder: Encoder, decoder: Decoder, head: Head, auxiliary: Head, return_features: bool = False):
        super().__init__(encoder, decoder, head, return_features=return_features)
        self.auxiliary = auxiliary

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x: Tuple[torch.Tensor] = self.encoder(x)
        features = self.decoder(x)
        aux = self.auxiliary(x[-1])
        out = self.head(features)
        if self.return_features:
            return (out, aux), features
        else:
            return out, aux
