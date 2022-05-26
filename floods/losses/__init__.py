from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as func

from floods.losses.functional import lovasz_hinge


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, reduction: str = "mean", ignore_index: int = 255, weight: torch.Tensor = None, **kwargs: dict):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        mask = targets != self.ignore_index

        targets = targets[mask]
        preds = preds[mask]

        return func.binary_cross_entropy_with_logits(preds, targets.float(), reduction=self.reduction)


class CombinedLoss(nn.Module):
    """Simply combines two losses into a single one, with weights.
    """

    def __init__(self,
                 criterion_a: Callable,
                 criterion_b: Callable,
                 weight_a: float = 1.0,
                 weight_b: float = 1.0,
                 **kwargs: dict):
        super().__init__()
        self.criterion_a = criterion_a(**kwargs)
        self.criterion_b = criterion_b(**kwargs)
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_a = self.criterion_a(preds, targets)
        loss_b = self.criterion_b(preds, targets)
        return self.weight_a * loss_a + self.weight_b * loss_b


class FocalLoss(nn.Module):
    """Simple implementation of focal loss.
    The focal loss can be seen as a generalization of the cross entropy, where more effort is put on
    hard examples, thanks to its gamma parameter.
    """

    def __init__(self,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 weight: torch.Tensor = None,
                 **kwargs: dict):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = torch.mean if reduction == "mean" else torch.sum
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        targets = targets[mask]
        preds = preds[mask]

        ce_loss = func.binary_cross_entropy_with_logits(preds, targets.float(), reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return self.reduction(focal_loss)


class FocalTverskyLoss(nn.Module):
    """Custom implementation of a generalized Dice loss (called Tversky loss) with focal components.
    """

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        # normalize weights so that they sum to 1
        if isinstance(weight, torch.Tensor):
            weight /= weight.sum()
        self.weight = weight if weight is not None else 1.0

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        targets = targets[mask]
        preds = preds[mask]

        probs = torch.sigmoid(preds)

        # sum over batch, height width, leave classes (dim 1)
        tp = (targets * probs).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        index = self.weight * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


class LovaszSoftmax(nn.Module):

    def __init__(self, classes='present', per_image=True, ignore_index=255, weight=None):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # weightning of the loss not implemented yet
        loss = lovasz_hinge(preds, targets, ignore=self.ignore_index)
        return loss
