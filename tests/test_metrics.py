import torch
from flood.metrics import GeneralStatistics
from sklearn import metrics


def test_generic_stats():
    num_classes = 5
    y_true = torch.randint(0, num_classes, size=(4, 256, 256))
    y_pred = torch.randint_like(y_true, num_classes)
    metric = GeneralStatistics()
    metric(y_true, y_pred)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
