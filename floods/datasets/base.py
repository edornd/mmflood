from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict, Tuple

from torch.utils.data import Dataset


class DatasetBase(Dataset, ABC):
    @abstractclassmethod
    def name(cls) -> str:
        """Returns the name of the curent dataset.
        """
        ...

    @abstractclassmethod
    def categories(cls) -> Dict[int, str]:
        """Returns a dictionary of <index, category name>, representing the classes
        available for the current dataset.
        """
        ...

    @abstractclassmethod
    def palette(cls) -> Dict[int, tuple]:
        """Returns a dictionary of <index, color tuple>, representing the color associated with the given
        category index.
        """
        ...

    @abstractclassmethod
    def ignore_index(cls) -> int:
        """Returns the index to be ignored in case of losses and such, usually 255.
        """
        ...

    def mean(cls) -> Tuple[float, ...]:
        """Returns an array of channel-wise means.
        """
        ...

    def std(cls) -> Tuple[float, ...]:
        """Returns an array of channel-wise stds.
        """
        ...

    @abstractmethod
    def stage(self) -> str:
        ...
