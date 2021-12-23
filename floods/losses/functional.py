import torch
import torch.nn.functional as F
from torch.autograd import Variable

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


# --------------------------- BINARY LOVASZ TAKEN FROM https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py ---------------------------


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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -infty and +infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
