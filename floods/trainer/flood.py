from pathlib import Path
from typing import Any, Callable, Dict

import torch
from torch import nn
from torch.nn import functional as func
from torch.optim import Optimizer

from accelerate import Accelerator
from floods.datasets.flood import FloodDataset
from floods.logging import BaseLogger
from floods.logging.functional import save_grid
from floods.metrics import Metric
from floods.trainer import Trainer, TrainerStage


class FloodTrainer(Trainer):
    def __init__(self,
                 accelerator: Accelerator,
                 model: nn.Module,
                 criterion: nn.Module,
                 categories: Dict[int, str],
                 optimizer: Optimizer = None,
                 scheduler: Any = None,
                 tiler: Callable = None,
                 train_metrics: Dict[str, Metric] = None,
                 val_metrics: Dict[str, Metric] = None,
                 logger: BaseLogger = None,
                 sample_batches: int = None,
                 stage: str = "train",
                 debug: bool = False) -> None:
        super().__init__(accelerator,
                         model,
                         optimizer,
                         scheduler,
                         criterion,
                         categories,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         logger=logger,
                         sample_batches=sample_batches,
                         stage=stage,
                         debug=debug)
        self.tiler = tiler

    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x, y = batch

        # forward and loss on segmentation task
        with self.accelerator.autocast():
            out = self.model(x)
            loss = self.criterion(out, y.long())

        # gather and update metrics
        # we group only the 'standard' images, not the rotated ones
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        # debug if active
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        return loss, {}

    def validation_batch(self, batch: Any, batch_index: int):
        # init losses and retrieve x, y
        x, y = batch
        # forward and loss on main task, using AMP
        with self.accelerator.autocast():
            out = self.model(x)
            loss = self.criterion(out, y.long())
        # gather stuff from DDP
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        # store samples for visualization, if present. Requires a plot callback.
        # Better to unpack now, so that we don't have to deal with the batch size later
        # also, we take just the first one, a lil bit hardcoded i know
        # TODO: better sampling from batches
        if self.sample_batches is not None and batch_index in self.sample_batches:
            y_pred = (torch.sigmoid(y_pred) > 0.5).int()
            images = self.accelerator.gather(x)
            self._store_samples(images[:1], y_pred[:1], y_true[:1].int())
        # update metrics and return losses
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        return loss, {}

    def test_batch(self, batch: Any, batch_index: int, output_path: Path = None):
        # x and y are full-size images, with same height and width
        x, y = batch
        assert x.shape[0] == 1, "Batch images not allowed"
        x = x.to(device=self.accelerator.device)
        y = y.to(device=self.accelerator.device)

        # define a callback with forward
        def callback(patches: torch.Tensor) -> torch.Tensor:
            patch_preds = self.model(patches)
            return patch_preds

        y_pred = self.tiler(x[0], callback)
        y_pred = y_pred.unsqueeze(0)  # .permute(2, 0, 1)
        loss = self.criterion(y_pred, y.long())
        # cannot gather if every image has different dimensions
        # store samples for visualization, if required.
        if output_path:
            if self.sample_batches is None or batch_index in self.sample_batches:
                save_grid(x[0],
                          y[0].int(), (torch.sigmoid(y_pred[0]) > 0.5).int(),
                          filepath=output_path,
                          filename=f"{batch_index:06d}",
                          palette=FloodDataset.palette())
                # store only some selected samples
        # update metrics and return losses
        self._update_metrics(y_true=y, y_pred=y_pred, stage=TrainerStage.test)
        # result_data = {"inputs": x.cpu(), "targets": y.cpu(), "preds": torch.argmax(y_pred, dim=1).cpu()}
        return loss, {}


class MultiBranchTrainer(FloodTrainer):
    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x, y = batch

        # forward and loss on segmentation task
        with self.accelerator.autocast():
            out, aux = self.model(x)
            loss = self.criterion(out, y.long())
            loss += self.criterion(aux, y.long()) * 0.4

        # gather and update metrics
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        # debug if active
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        return loss, {}

    def validation_batch(self, batch: Any, batch_index: int):
        # init losses and retrieve x, y
        x, y = batch
        # forward and loss on main task, using AMP
        with self.accelerator.autocast():
            out, _ = self.model(x)
            loss = self.criterion(out, y.long())
        # gather stuff from DDP
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        # store samples for visualization, if present. Requires a plot callback.
        # Better to unpack now, so that we don't have to deal with the batch size later
        # also, we take just the first one, a lil bit hardcoded i know
        # TODO: better sampling from batches
        if self.sample_batches is not None and batch_index in self.sample_batches:
            images = self.accelerator.gather(x)
            self._store_samples(images[:1], y_pred[:1], y_true[:1])
        # update metrics and return losses
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        return loss, {}
