import torch
from torch.autograd import Variable

from floods.utils.ml import F16_EPS

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def unbiased_softmax(
    inputs: torch.Tensor,
    old_index: int,
) -> torch.Tensor:
    """Computes the (log) softmax of the given input tensor (provided as logits), using the unbiased setting
    from Modeling the Background for incremental-class learning in semantic segmentation.
    The gist is to compute the softmax for current classes as usual, while the background is replaced with
    p(0) = p(0) + p(old classes)

    Args:
        inputs (torch.Tensor): logits input tensor
        old_index (int): index splitting old and new classes

    Returns:
        torch.Tensor: softmax-ed tensor unbiased against old classes.
    """
    # contruct a zero-initialized tensor with same dims as input [batch, (bkgr + old + new), height, width]
    outputs = torch.zeros_like(inputs)
    # build log sum(exp(inputs))  [batch, height, width], denominator for the softmax
    denominator = torch.logsumexp(inputs, dim=1)
    # compute the softmax for background (based on old classes) and new classes (minus operator because of logs)
    outputs[:, 0] = torch.logsumexp(inputs[:, :old_index], dim=1) - denominator  # [batch, h, w] p(O)
    outputs[:, old_index:] = inputs[:, old_index:] - denominator.unsqueeze(dim=1)  # [batch, new, h, w] p(new_i)
    return outputs


def one_hot_batch(batch: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Generates a one-hot encoded version on the input batch tensor.
    Batch is expected as tensor of indices with shape [batch, height, width].

    Args:
        batch (torch.Tensor): input tensor, [b, h, w]
        num_classes (int): how many classes are there, a.k.a. the one-hot dimension.
        ignore_index (int, optional): index to be exluded from computation. Defaults to 255.

    Returns:
        torch.Tensor: one-hot batch, with shape [b, classes, h, w]
    """
    mask = batch == ignore_index
    target = batch.clone()
    target[mask] = num_classes
    onehot_target = torch.eye(num_classes + 1)[target]
    return onehot_target[:, :, :, :-1].permute(0, 3, 1, 2).to(batch.device)


def smooth_weights(class_freqs: torch.Tensor,
                   smoothing: float = 0.15,
                   clip: float = 10.0,
                   normalize: bool = True) -> torch.Tensor:
    """Compute smoothed weights starting from class frequencies (pixel counts).

    Args:
        class_freqs (torch.Tensor): tensor with shape (num_classes,)
        smoothing (float, optional): smoothing factor. Defaults to 0.15.
        clip (float, optional): maximum value before clipping. Defaults to 10.0.
        normalize (bool, optional): whether to map them to range [0, 1]. Defaults to True.

    Returns:
        torch.Tensor: weights inversely proportial to frequencies, normalized if required
    """
    # the larger the smooth factor, the bigger the quantities to sum to the remaining counts (additive smoothing)
    freqs = class_freqs.float() + class_freqs.max() * smoothing
    # retrieve the (new) max value, divide by counts, clip to max. value
    weights = torch.clamp(freqs.max() / class_freqs, min=1.0, max=clip)
    if normalize:
        weights /= weights.max()
    return weights


def isnan(x):
    return x != x


def mean(inputs, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    inputs = iter(inputs)
    if ignore_nan:
        inputs = ifilterfalse(isnan, inputs)
    try:
        n = 1
        acc = next(inputs)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(inputs, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def tanimoto_loss(predictions: torch.Tensor,
                  targets: torch.Tensor,
                  weights: torch.Tensor,
                  dims: tuple = (0, 2, 3),
                  gamma: float = 1.0) -> torch.Tensor:
    tp = (targets * predictions).sum(dim=dims)
    l2 = (targets * targets).sum(dim=dims)
    p2 = (predictions * predictions).sum(dim=dims)
    denominator = l2 + p2 - tp
    # compute weighted dot(p,t) / dot(t,t) + dot(p,p) - dot(p,t)
    index = ((weights * tp) + F16_EPS) / ((weights * denominator) + F16_EPS)
    return ((1 - index).mean())**gamma


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
