from functools import partial
from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as func

from floods.losses.functional import lovasz_softmax, one_hot_batch, smooth_weights, tanimoto_loss


class CombinedLoss(nn.Module):
    """Simply combines two losses into a single one, with weights.
    """
    def __init__(self, criterion_a: Callable, criterion_b: Callable, alpha: float = 0.5, **kwargs: dict):
        super().__init__()
        self.criterion_a = criterion_a(**kwargs)
        self.criterion_b = criterion_b(**kwargs)
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_a = self.criterion_a(preds, targets)
        loss_b = self.criterion_b(preds, targets)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b


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
        ce_loss = func.cross_entropy(preds,
                                     targets,
                                     reduction='none',
                                     ignore_index=self.ignore_index,
                                     weight=self.weight)
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
        num_classes = preds.size(1)
        onehot = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot = onehot.float().to(preds.device)
        probs = func.softmax(preds, dim=1)

        # sum over batch, height width, leave classes (dim 1)
        dims = (0, 2, 3)
        tp = (onehot * probs).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)

        index = self.weight * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


class TanimotoLoss(nn.Module):
    """Computes the Dice loss using the Tanimoto formulation of the Jaccard index
    (https://en.wikipedia.org/wiki/Jaccard_index), as described in https://arxiv.org/abs/1904.00592
    """
    def __init__(self, ignore_index: int = 255, gamma: float = 2.0, eps: float = 1e-6, **kwargs: dict):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.eps = eps
        self.softmax = partial(func.softmax, dim=1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Tanimoto loss.

        Args:
            preds (torch.Tensor): prediction tensor, in the form [batch, classes, h, w]
            targets (torch.Tensor): target tensor with class indices, with shape [batch, h, w]
        Returns:
            torch.Tensor: singleton tensor
        """
        num_classes = preds.size(1)
        targets_onehot = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        # mean class volume per batch: sum over H and W, average over batch (dim 1 = one hot labels)
        # final tensor shape: (classes,)
        class_volume = targets_onehot.sum(dim=(2, 3)).mean(dim=0)
        vol_weights = smooth_weights(class_volume, normalize=True)
        # compute softmax probabilities
        dims = (0, 2, 3)
        probs = self.softmax(preds)
        tp = (targets_onehot * probs).sum(dim=dims)
        l2 = (targets_onehot * targets_onehot).sum(dim=dims)
        p2 = (probs * probs).sum(dim=dims)
        denominator = l2 + p2 - tp
        # compute weighted dot(p,t) / dot(t,t) + dot(p,p) - dot(p,t)
        index = ((vol_weights * tp) + self.eps) / ((vol_weights * denominator) + self.eps)
        return ((1 - index).mean())**self.gamma


class DualTanimotoLoss(nn.Module):
    """Computes the Dice loss using the Tanimoto formulation of the Jaccard index
    (https://en.wikipedia.org/wiki/Jaccard_index). Also computes the dual formulation,
    then averages intersection and non-intersections, as described in https://arxiv.org/abs/1904.00592.
    """
    def __init__(self, ignore_index: int = 255, alpha: float = 0.5, gamma: float = 1.0, **kwargs: dict):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = partial(func.softmax, dim=1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the dual formulation of the Tanimoto loss

        Args:
            preds (torch.Tensor): prediction tensor, in the form [batch, classes, h, w]
            targets (torch.Tensor): target tensor with class indices, with shape [batch, h, w]
        Returns:
            torch.Tensor: dual Tanimoto loss.
        """
        num_classes = preds.size(1)
        onehot_pos = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot_neg = 1 - onehot_pos
        probs_pos = self.softmax(preds)
        probs_neg = 1 - probs_pos
        # mean class volume per batch: sum over H and W, average over batch (dim 1 = one hot labels)
        # final tensor shape: (classes,)
        weights_pos = smooth_weights(onehot_pos.sum(dim=(2, 3)).mean(dim=0), normalize=True)
        weights_neg = smooth_weights(onehot_neg.sum(dim=(2, 3)).mean(dim=0), normalize=True)
        # compute die/tanimoto and dual dice
        dims = (0, 2, 3)
        loss = tanimoto_loss(probs_pos, onehot_pos, weights_pos, dims=dims, gamma=self.gamma)
        dual = tanimoto_loss(probs_neg, onehot_neg, weights_neg, dims=dims, gamma=self.gamma)
        return self.alpha * loss + (1 - self.alpha) * dual


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255, weight=None):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, output, target):
        logits = func.softmax(output, dim=1)
        # weightning of the loss not implemented yet
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
