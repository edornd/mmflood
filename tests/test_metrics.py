import logging
from typing import Tuple

import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from floods.metrics import ConfusionMatrix, F1Score, GeneralStatistics, IoU, Precision, Recall, lenient_sigmoid

LOG = logging.getLogger(__name__)
EPS = 1e-6
PYTEST_PARAMETERS = ("seed", "reduction")
PYTEST_VALUES = [(42, "micro"), (42, "macro"), (42, None), (1377, "micro"), (1377, "macro"), (1377, None)]


def random_gt(shape: Tuple[int, ...]) -> torch.Tensor:
    """Generate random ground-truth tensor.
    """
    return (torch.rand(shape) > 0.5).type(torch.int)


def random_pred(rand_gt: torch.Tensor, error_rate: float = 0.9) -> torch.Tensor:
    """Generate random prediction tensor.
    """
    return torch.abs(rand_gt - (torch.rand(rand_gt.shape) > error_rate).type(torch.int))


def int_zeros(*dims: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(dims, dtype=torch.int)


def int_ones(*dims: Tuple[int, ...]) -> torch.Tensor:
    return torch.ones(dims, dtype=torch.int)


@pytest.mark.parametrize(("inputs", "preds", "expected"), [(int_zeros(2, 8, 8), int_zeros(2, 8, 8), [0, 0, 128, 0]),
                                                           (int_zeros(2, 8, 8), int_ones(2, 8, 8), [0, 128, 0, 0]),
                                                           (int_ones(2, 8, 8), int_zeros(2, 8, 8), [0, 0, 0, 128]),
                                                           (int_ones(2, 8, 8), int_ones(2, 8, 8), [128, 0, 0, 0])])
def test_generic_stats(inputs: torch.Tensor, preds: torch.Tensor, expected: list):
    tp, fp, tn, fn = expected
    support = sum(expected)
    metric = GeneralStatistics()
    # transform and update
    metric(inputs, preds)
    assert metric.tp == tp
    assert metric.fp == fp
    assert metric.tn == tn
    assert metric.fn == fn
    # now compute the final output (a tensor)
    result = metric.compute()
    LOG.debug(result)
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor(expected + [support])
    assert torch.all(expected == result)
    # also verify that reset works
    metric.reset()
    assert metric.tp == metric.fp == metric.tn == metric.fn == 0


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_random_continuous_matrix_stats_2D(seed: int, reduction: str):
    torch.manual_seed(seed)

    rand_gt = (torch.rand((512, 512)) > 0.5).type(torch.int)
    rand_pred = (torch.rand(rand_gt.shape) - torch.rand(rand_gt.shape)) * 5
    rand_pred_discrete = lenient_sigmoid(rand_pred)[0]

    metric = GeneralStatistics(reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()
    tp, fp, tn, fn = result[:4]

    sk_tn, sk_fp, sk_fn, sk_tp = skm.confusion_matrix(rand_gt.numpy().flatten(),
                                                      rand_pred_discrete.numpy().flatten(),
                                                      labels=[0, 1]).flatten()

    assert tp.numpy() == sk_tp
    assert tn.numpy() == sk_tn
    assert fp.numpy() == sk_fp
    assert fn.numpy() == sk_fn


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_random_matrix_stats_2D(seed: int, reduction: str):
    torch.manual_seed(seed)

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    assert torch.max(rand_gt) == 1 and torch.min(rand_gt) == 0 and torch.unique(rand_gt).shape[0] == 2
    assert torch.max(rand_pred) == 1 and torch.min(rand_pred) == 0 and torch.unique(rand_pred).shape[0] == 2

    metric = GeneralStatistics(reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()
    tp, fp, tn, fn = result[:4]

    sk_tn, sk_fp, sk_fn, sk_tp = skm.confusion_matrix(rand_gt.flatten(), rand_pred.flatten(), labels=[0, 1]).flatten()

    assert tp == sk_tp
    assert tn == sk_tn
    assert fp == sk_fp
    assert fn == sk_fn


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_precision(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = Precision(ignore_index=255, reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    skm_result = skm.precision_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_recall(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = Recall(ignore_index=255, reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    skm_result = skm.recall_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_F1_score(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = F1Score(ignore_index=255, reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    skm_result = skm.f1_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_F1_score_background(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = F1Score(ignore_index=255, reduction=reduction, background=True)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    rand_gt = np.abs(1 - rand_gt)
    rand_pred = np.abs(1 - rand_pred)
    skm_result = skm.f1_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_IoU(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = IoU(ignore_index=255, reduction=reduction)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    skm_result = skm.jaccard_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize(PYTEST_PARAMETERS, PYTEST_VALUES)
def test_IoU_background(seed: int, reduction: str):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = IoU(ignore_index=255, reduction=reduction, background=True)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    rand_gt = np.abs(1 - rand_gt)
    rand_pred = np.abs(1 - rand_pred)
    skm_result = skm.jaccard_score(rand_gt.numpy().flatten(), rand_pred.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)


@pytest.mark.parametrize("seed", [42, 1337])
def test_confusion_matrix(seed: int):

    # rand_gt is a random 2D matrix
    # rand_pred is a pertubation of rand_gt, where only about 10% of the elements are different
    torch.manual_seed(seed)
    rand_gt = random_gt((512, 512))
    rand_pred = random_pred(rand_gt)

    metric = ConfusionMatrix(ignore_index=255)
    metric(rand_gt, rand_pred)
    result = metric.compute()

    skm_result = skm.confusion_matrix(rand_gt.numpy().flatten(), rand_pred.numpy().flatten())

    LOG.debug(result)
    LOG.debug(skm_result)

    assert (result.numpy() == skm_result).all()
