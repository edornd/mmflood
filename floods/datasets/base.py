from abc import ABC, abstractmethod
from typing import Dict, List

from torch.utils.data import Dataset


class DatasetBase(Dataset, ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def stage(self) -> str:
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def categories(self) -> Dict[int, str]:
        """Returns a dictionary of <index, category name>, representing the classes
        available for the current dataset.

        Raises:
            NotImplementedError: abstract method to be implemented in a subclass.

        Returns:
            Dict[int, str]: dictionary of categories with their index as key.
        """
        raise NotImplementedError("Implement in a subclass")

    @abstractmethod
    def palette(self) -> Dict[int, tuple]:
        """Returns a dictionary of <index, color tuple>, representing the color associated with the given
        category index.

        Raises:
            NotImplementedError: abstract method to be implemented in a subclass

        Returns:
            Dict[int, tuple]: dictionary of class indices and associated color tuple
        """
        raise NotImplementedError("Implement in a subclass")

    @abstractmethod
    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        """Applies a boolean mask to the current dataset, effectively eliminating those items
        marked with False.

        Args:
            mask (List[bool]): list of true/false values, one per item.

        Raises:
            NotImplementedError: abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Implement in a subclass")

    @abstractmethod
    def ignore_index(self) -> int:
        """Returns the index to be ignored in case of losses and such, usually 255.

        Raises:
            NotImplementedError: implement in subclass

        Returns:
            int: index to be ignored in labels
        """
        raise NotImplementedError("Implement in a subclass")
