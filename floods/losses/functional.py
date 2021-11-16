import torch

from floods.utils.ml import F16_EPS


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
