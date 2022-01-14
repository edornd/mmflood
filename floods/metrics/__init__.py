from abc import abstractmethod
from typing import Any, Callable, Iterable, Optional

import torch

from floods.metrics import functional as func
from floods.utils.ml import identity


def lenient_argmax(*args: Iterable[torch.Tensor], ndims: int = 3) -> Iterable[torch.Tensor]:
    result = list()
    for tensor in args:
        tensor = tensor.argmax(dim=1) if tensor.ndim > ndims else tensor
        result.append(tensor)
    return result


def lenient_sigmoid(*args: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
    result = list()
    for tensor in args:
        tensor = (torch.sigmoid(tensor) > 0.5).int()
        result.append(tensor)
    return result


class Metric:
    def __init__(self, transform: Callable, device: str) -> None:
        self.transform = transform or identity
        self.device = torch.device(device)

    @abstractmethod
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        raise NotImplementedError("Override in subclass")

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError("Override in subclass")

    @abstractmethod
    def reset(self) -> None:
        """ Overridden by subclasses """
        raise NotImplementedError("Override in subclass")

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Any:
        y_true, y_pred = self.transform(y_true, y_pred)
        self.update(y_true, y_pred)


class ConfusionMatrix(Metric):
    """Computes the confusion matrix over the classes on the given predictions.
    The process ignores the given ignore_index to exclude those positions from the computation (optional).
    """
    def __init__(self,
                 ignore_index: Optional[int] = 255,
                 transform: Callable = lenient_sigmoid,
                 device: str = "cpu") -> None:
        super().__init__(transform=transform, device=device)
        self.ignore_index = ignore_index
        self.reset()

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Accumulate data, to be averaged by the compute pass.
        :param pred: prediction tensor, as logits [N, C, ...] or [N, ...] with class indices
        :type pred: torch.Tensor
        :param target: target tensor, already provided in index format [N, ...]
        :type target: torch.Tensor
        """
        # assume 0=batch size, 1=classes, 2, 3 = dims
        partial = func.binary_confusion_matrix(y_true, y_pred, ignore_index=self.ignore_index)
        self.confusion_matrix += partial

    def compute(self) -> torch.Tensor:
        """Returns the current confusion matrix, as a tensor with size [C, C], where C = # of classes.

        Returns:
            torch.Tensor: confusion matrix CxC
        """
        return self.confusion_matrix

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros((2, 2), device=self.device)


class GeneralStatistics(Metric):
    """Computes a set of standard generic statistics, mostly used in combination in other metrics (e.g. F1).
    Specifically, the final output is a set of five values, consisting of TP, FP, TN, FN and support.
    Micro VS macro: https://datascience.stackexchange.com/questions/15989
    """
    def __init__(self,
                 ignore_index: Optional[int] = 255,
                 reduction: Optional[str] = "micro",
                 transform: Callable = lenient_sigmoid,
                 device: str = "cpu",
                 background: bool = False) -> None:
        super().__init__(transform=transform, device=device)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.reduce_first = reduction == "micro"
        self.should_reduce = reduction is not None
        self.background = background
        self.reset()

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Updates the statistics by including the provided predictions and targets.

        Args:
            y_true (torch.Tensor): true targets, provided as indices, size [B, H, W]
            y_pred (torch.Tensor): prediction batch, yet to be tansformed into indices, with size [B, C, H, W]
        """
        if self.ignore_index is not None:
            flat_true, flat_pred = func.valid_samples(self.ignore_index, y_true=y_true, y_pred=y_pred)
        else:
            flat_true, flat_pred = y_true.view(-1), y_pred.view(-1)

        onehot_true = (flat_true > 0).float()  # one_hot(flat_true, num_classes=self.num_classes)
        onehot_pred = flat_pred  # one_hot(flat_pred, num_classes=self.num_classes)
        tp, fp, tn, fn = func.statistics_from_one_hot(onehot_true, onehot_pred, reduce=self.reduce_first)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """Returns a tensor with shape (5,) for micro average or (C, 5) for macro average,
        consisting of TP, FP, TN, FN and support (obtained from TP + FN).

        Returns:
            torch.Tensor: tensor with size (5,), containing (TP, FP, TN, FN, support)
        """
        outputs = [
            self.tp.unsqueeze(-1),
            self.fp.unsqueeze(-1),
            self.tn.unsqueeze(-1),
            self.fn.unsqueeze(-1),
        ]
        # add support
        outputs.append(sum(outputs))
        return torch.cat(outputs, dim=-1)

    def reset(self) -> None:
        shape = ()  # if self.reduce_first else (self.num_classes, )
        self.tp = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.fp = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.tn = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.fn = torch.zeros(shape, dtype=torch.long, device=self.device)


class Precision(GeneralStatistics):
    """Precision metric, computed as ratio of true positives and predicted positives (tp + fp).
    The precision indicates how well the model assign the right class: precision=1.0 for class C
    means that every sample classified as C belongs to C, however some samples with true label = C
    maybe be ended up with different predicted labels. The recall takes care of this.
    """
    def compute(self) -> torch.Tensor:
        """Computes the final precision score, synching among devices.
        The tensor is averaged only for micro-avg, in the other cases it is computed over classes.
        In case of macro-avg result, the score is reduced _after_, in case of no reduction is kept as tensor.
        The return value is then a tensor with dimension () in the first cases, or (C,) in the latter case.

        Returns:
            torch.Tensor: tensor with empty size when reduced, or (C,) where C in the number of classes
        """
        score = func.precision_score(tp=self.tp, fp=self.fp, reduce=self.reduce_first)
        return score.mean() if self.should_reduce else score


class Recall(GeneralStatistics):
    """Computes the recall metric, computed as ratio between predicted class positives and actual positives (tp + fn).
    Recall indicates how well the model 'covers' a given class, regardless of how many false positives it generates.
    """
    def compute(self) -> torch.Tensor:
        """Computes the final recall score over every device.
        The tensor is averaged only for micro-avg, in the other cases it is computed over classes.
        In case of macro-avg result, the score is reduced _after_, in case of no reduction is kept as tensor.
        The return value is then a tensor with dimension () in the first cases, or (C,) in the latter case.

        Returns:
            torch.Tensor: tensor with empty size when reduced, or (C,) where C in the number of classes
        """
        score = func.recall_score(tp=self.tp, fn=self.fn, reduce=self.reduce_first)
        return score.mean() if self.should_reduce else score


class F1Score(GeneralStatistics):
    """F1 score is defined as harmonic mean between precision and recall: 2* (P * R) / (P + R).
    Combining the two metrics gives the best overall idea on how well the model covers the data and how precise it is.
    """
    def compute(self) -> torch.Tensor:
        """Computes the F1 score over every device, using the accumulated statistics.
        Same micro and macro-average considerations hold for this metric as well.

        Returns:
            torch.Tensor: tensor with empty size when reduced, or (C,) where C in the number of classes
        """
        if self.background:
            score = func.f1_score(tp=self.tn, fp=self.fn, fn=self.fp, reduce=self.reduce_first)
        else:
            score = func.f1_score(tp=self.tp, fp=self.fp, fn=self.fn, reduce=self.reduce_first)
        return score.mean() if self.should_reduce else score


class IoU(GeneralStatistics):
    """Computes the Intersection over Union metric, taking into account the number of classes
    and the ignore index to exclude pixels from the computation (optional).
    """
    def compute(self) -> torch.Tensor:
        """Computes the IoU metric using the internal statistics.

        Returns:
            torch.Tensor: IoU vector, if divided by class, or a mean value (macro/micro averaged)
        """
        if self.background:
            score = func.iou_score(tp=self.tn, fp=self.fn, fn=self.fp, reduce=self.reduce_first)
        else:
            score = func.iou_score(tp=self.tp, fp=self.fp, fn=self.fn, reduce=self.reduce_first)
        return score.mean() if self.should_reduce else score
