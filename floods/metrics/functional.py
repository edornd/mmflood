from typing import Optional, Tuple, Union

import torch

from floods.utils.ml import F32_EPS


def valid_samples(ignore_index: int,
                  y_true: torch.Tensor,
                  y_pred: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Receives one or two tensors (when the prediction is included) to exclude ignored indices.

    Args:
        ignore_index (int): index value to be discarded, usually something like 255
        target (torch.Tensor): target input, expected as 2D indexed ground truth or 3D batch of indices
        pred (Optional[torch.Tensor], optional): optional tensor of 'argmaxed' predictions. Defaults to None.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: tensor with values different from `ignore_index`,
        or a tuple of matching tensors when `y_pred` is also provided.
    """
    valid_indices = y_true != ignore_index
    valid_target = y_true[valid_indices]
    if y_pred is not None:
        assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_true.shape} </> {y_pred.shape}"
        valid_pred = y_pred[valid_indices]
        return valid_target.flatten().long(), valid_pred.flatten().long()
    return valid_target.flatten().long()


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
    """Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Args:
        y_true (torch.Tensor): ground truth label indices, same format as the predictions.
        y_pred (torch.Tensor): estimated targets, in 'argmax' format (indices of the predicted classes).
        num_classes (Optional[int], optional): number of classes if possible. Defaults to None.
        ignore_index (Optional[int], optional): index to be ignored, if any. Defaults to None.

    Returns:
        torch.Tensor: confusion matric C [num_classes, num_classes]
    """
    flat_target = y_true.view(-1)
    flat_pred = y_pred.view(-1)
    # exclude indices belonging to the ignore_index
    if ignore_index is not None:
        flat_target, flat_pred = valid_samples(ignore_index, y_true=y_true, y_pred=y_pred)
    # use bins to compute the CM
    unique_labels = flat_target + flat_pred
    bins = torch.bincount(unique_labels, minlength=4)
    cm = bins.reshape(2, 2).squeeze().int()
    return cm


def binary_confusion_matrix(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            ignore_index: Optional[int] = None) -> torch.Tensor:
    """Computes the binary confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Args:
        y_true (torch.Tensor): ground truth label indices, same format as the predictions.
        y_pred (torch.Tensor): estimated targets, in 'argmax' format (indices of the predicted classes).
        ignore_index (Optional[int], optional): index to be ignored, if any. Defaults to None.

    Returns:
        torch.Tensor: confusion matric C [num_classes, num_classes]
    """
    flat_target = y_true.view(-1)
    flat_pred = y_pred.view(-1)
    # exclude indices belonging to the ignore_index
    if ignore_index is not None:
        flat_target, flat_pred = valid_samples(ignore_index, y_true=y_true, y_pred=y_pred)
    # use bins to compute the CM
    tp = (flat_target * flat_pred).sum()
    tn = ((1 - flat_target) * (1 - flat_pred)).sum()
    fp = ((1 - flat_target) * flat_pred).sum()
    fn = (flat_target * (1 - flat_pred)).sum()
    return torch.tensor([[tn, fp], [fn, tp]], device=y_pred.device, dtype=torch.int64)


def statistics_from_one_hot(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            reduce: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the number of true/false positives, true/false negatives.
    Source https://github.com/PyTorchLightning/metrics
    :param pred: A ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
    :type pred: torch.Tensor
    :param target: A ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
    :type target: torch.Tensor
    :param reduce: One of ``'micro'``, ``'macro'``
    :type target: str
    :return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depnds on the shape of the inputs
        and the ``reduce`` parameter:
        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
    """

    true_pred = y_true == y_pred
    false_pred = y_true != y_pred
    pos_pred = y_pred == 1
    neg_pred = y_pred == 0
    tp = (true_pred * pos_pred).sum()
    fp = (false_pred * pos_pred).sum()
    tn = (true_pred * neg_pred).sum()
    fn = (false_pred * neg_pred).sum()
    return tp.long(), fp.long(), tn.long(), fn.long()


def iou_score(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the IoU from the provided statistical measures. The result is a tensor, with size (C,)
    when reduce is false, or empty size in all other cases.
    :param tp: true positives, with dimension (C,) if not reduced, or ()
    :type tp: torch.Tensor
    :param fp: false positives, with dims (C,) if not reduced, or ()
    :type fp: torch.Tensor
    :param fn: false negatives, same criteria as previous ones
    :type fn: torch.Tensor
    :param reduce: whether to reduce to mean or not, defaults to True
    :type reduce: bool, optional
    :return: tensor representing the intersection over union for each class (C,), or a mean ()
    :rtype: torch.Tensor
    """
    score = tp / (tp + fp + fn + F32_EPS)
    return score.mean() if reduce else score


def precision_score(tp: torch.Tensor, fp: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the precision using true positives and false positives.
    :param tp: true positives, dims (C,) or ()
    :type tp: torch.Tensor
    :param fp: false positives, dims (C,) or ()
    :type fp: torch.Tensor
    :param reduce: whether to compute a mean precision or a class precision, defaults to True
    :type reduce: bool, optional
    :return: tensor representing the class precision (C,) or a mean precision ()
    :rtype: torch.Tensor
    """
    score = tp / (tp + fp + F32_EPS)
    return score.mean() if reduce else score


def recall_score(tp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the recall using true positives and false negatives.
    :param tp: true positives, dims (C,) or () when micro-avg is applied
    :type tp: torch.Tensor
    :param fn: false negatives, dims (C,) or () when micro-avg is applied
    :type fn: torch.Tensor
    :param reduce: whether to reduce to mean or not, defaults to True
    :type reduce: bool, optional
    :return: recall score
    :rtype: torch.Tensor
    """
    score = tp / (tp + fn + F32_EPS)
    return score.mean() if reduce else score


def f1_score(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the F1 score using TP, FP and FN, in turn required for precision and recall.
    :param tp: true positives, (C,) or () when micro averaging
    :type tp: torch.Tensor
    :param fp: false positives, (C,) or () when micro averaging
    :type fp: torch.Tensor
    :param fn: false negatives, (C,) or () when micro averaging
    :type fn: torch.Tensor
    :param reduce: whether to compute a mean result or not, defaults to True
    :type reduce: bool, optional
    :return: (micro/macro) averaged F1 score, or class F1 score
    :rtype: torch.Tensor
    """
    # do not reduce sub-metrics, otherwise when the F1 score reduce param is True it never computes the macro,
    # since it also collapses the precision and recall.
    precision = precision_score(tp=tp, fp=fp, reduce=False)
    recall = recall_score(tp=tp, fn=fn, reduce=False)
    f1 = 2 * (precision * recall) / (precision + recall + F32_EPS)
    return f1.mean() if reduce else f1
