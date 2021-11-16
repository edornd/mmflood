from typing import Dict, List

import numpy as np
from torch import nn

from floods.logging import BaseLogger


class EmptyLogger(BaseLogger):
    """Empty logger that does literally nothing, it just avoids endless null checks in the trainer.
    It ain't much but it's honest work.
    """
    def step(self, iteration: int = None) -> None:
        pass

    def log_model(self, model: nn.Module, input_size: tuple, **kwargs) -> None:
        pass

    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        pass

    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        pass

    def log_table(self, name: str, table: Dict[str, str], **kwargs) -> None:
        pass

    def log_results(self, name: str, headers: List[str], results: Dict[str, List[float]]) -> None:
        pass
