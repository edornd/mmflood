from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from torch import nn


class BaseLogger(ABC):
    @abstractmethod
    def step(self, iteration: int = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_model(self, model: nn.Module, input_size: tuple, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_table(self, name: str, table: Dict[str, str], **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_results(self, name: str, headers: List[str], results: Dict[str, List[float]]) -> None:
        raise NotImplementedError()
