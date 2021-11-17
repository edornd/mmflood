from typing import Any

import torch

from floods.trainer import Trainer, TrainerStage


class FloodTrainer(Trainer):
    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x, y = batch
        # forward and loss on segmentation task
        with self.accelerator.autocast():
            out, _ = self.model(x)
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
