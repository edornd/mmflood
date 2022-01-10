import logging

import torch

from floods.losses import FocalTverskyLoss

LOG = logging.getLogger(__name__)


def test_tversky_loss():
    # emulate a logits output
    torch.manual_seed(42)
    y_pred = torch.rand((2, 5, 256, 256)) * 5
    y_true = torch.randint(0, 4, (2, 256, 256))
    # compute loss
    criterion = FocalTverskyLoss(ignore_index=255)
    loss = criterion(y_pred, y_true)
    LOG.info(loss)
    assert loss >= 0 and loss <= 1
