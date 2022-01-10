import logging
from typing import Tuple

import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from floods.metrics import GeneralStatistics, Precision

LOG = logging.getLogger(__name__)
EPS = 1e-6


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


@pytest.mark.parametrize(("inputs", "preds", "expected"), [(int_zeros(2, 8, 8), int_zeros(2, 8, 8), 0.0),
                                                           (int_zeros(2, 8, 8), int_ones(2, 8, 8), 0.0),
                                                           (int_ones(2, 8, 8), int_zeros(2, 8, 8), 0.0),
                                                           (int_ones(2, 8, 8), int_ones(2, 8, 8), 1.0)])
def test_precision(inputs: torch.Tensor, preds: torch.Tensor, expected: float):
    metric = Precision(ignore_index=255, reduction=None)
    metric(inputs.type(torch.int), preds.type(torch.int))
    result = metric.compute()
    skm_result = skm.precision_score(inputs.numpy().flatten(), preds.numpy().flatten(), average="binary")
    LOG.debug("%.4f - %.4f", result, skm_result)
    np.testing.assert_allclose(result, skm_result, atol=EPS)
