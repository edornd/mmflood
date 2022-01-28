from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
from matplotlib import pyplot as plt

from floods.logging.functional import make_grid, mask_to_rgb
from floods.utils.common import get_logger, prepare_folder

if TYPE_CHECKING:
    from floods.trainer.base import Trainer

LOG = get_logger(__name__)


class BaseCallback:
    def __init__(self, call_every: int = 1, call_once: int = None) -> None:
        assert call_every is not None or call_once is not None, "Specify at least one between call_every and call_once"
        if call_every is not None:
            assert call_every > 0, "call_every should be >= 1"
        if call_once is not None:
            assert call_once >= 0, "call_once should be >= 0"
        self.call_every = call_every
        self.call_once = call_once
        self.expired = False

    def __call__(self, trainer: "Trainer", *args: Any, **kwds: Any) -> Any:
        # early exit for one-time callbacks
        if self.expired:
            return
        if self.call_once is not None and self.call_once == trainer.current_epoch:
            data = self.call(trainer, *args, **kwds)
            self.expired = True
            return data
        if self.call_every is not None:
            if (trainer.current_epoch % self.call_every) == 0:
                return self.call(trainer, *args, **kwds)

    def setup(self, trainer: "Trainer"):
        pass

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Callback not implemented!")

    def dispose(self, trainer: "Trainer"):
        pass


class EarlyStoppingCriterion(Enum):
    minimum = torch.lt
    maximum = torch.gt


class EarlyStopping(BaseCallback):

    criteria = {"min": torch.lt, "max": torch.gt}

    def __init__(self,
                 call_every: int,
                 metric: str,
                 criterion: EarlyStoppingCriterion.minimum,
                 patience: int = 10) -> None:
        super().__init__(call_every=call_every)
        self.metric = metric
        self.criterion = criterion.value
        self.patience = patience
        self.patience_counter = None

    def setup(self, trainer: "Trainer"):
        metrics = trainer.metrics["val"]
        if self.metric not in metrics:
            raise ValueError(f"Monitored metric '{self.metric}' not in validation metrics: {list(metrics.keys())}")
        self.patience_counter = 0

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        # first, a quick check for nan losses, since it happens often in AMP
        if trainer.current_loss is not None and torch.isnan(trainer.current_loss):
            LOG.info("[Epoch %d] NaN loss detected, interrupting the training loop", trainer.current_epoch)
            raise KeyboardInterrupt
        # wait for everyone should not be needed, since scores are already computed
        # after we have waited every process, but better to be safe
        current_score = trainer.current_scores["val"][self.metric]
        if trainer.best_score is None or self.criterion(current_score, trainer.best_score):
            self.patience_counter = 0
            trainer.accelerator.wait_for_everyone()
            trainer.best_score = current_score
            trainer.best_epoch = trainer.current_epoch
            # we do not unwrap, so that later we can load without re-unwrapping
            trainer.best_state_dict = trainer.model.state_dict()
        else:
            self.patience_counter += 1
            LOG.info("[Epoch %2d] Early stopping patience increased to: %d/%d", trainer.current_epoch,
                     self.patience_counter, self.patience)
            if self.patience_counter == self.patience:
                LOG.info("[Epoch %2d] Early stopping triggered", trainer.current_epoch)
                trainer.model.load_state_dict(trainer.best_state_dict)
                # stop iterating with an exceptionm it will be caught by the training loop
                raise KeyboardInterrupt

    def dispose(self, trainer: "Trainer"):
        self.patience_counter = 0


class Checkpoint(BaseCallback):
    def __init__(self,
                 call_every: int,
                 model_folder: Path,
                 monitor: str,
                 name_suffix: str = "",
                 save_every: int = None,
                 save_best: bool = True,
                 verbose: bool = True) -> None:
        super().__init__(call_every=call_every)
        model_folder = prepare_folder(model_folder)
        assert model_folder.exists() and model_folder.is_dir(), f"Invalid path '{str(model_folder)}'"
        assert save_every or save_best, "Specify one between save_every or save_best"
        self.model_folder = model_folder
        self.name_format = name_suffix
        self.save_every = save_every
        self.save_best = save_best
        self.verbose = verbose
        self.monitor = monitor
        self.best_epoch = None

    def _should_save(self, trainer: "Trainer") -> bool:
        # if save every iteration, check if we are on the right epoch
        if self.save_every and (self.save_every % trainer.current_epoch == 0):
            return True
        # if save only best iterations, check that the current one is better
        if self.save_best and (not self.best_epoch or self.best_epoch < trainer.best_epoch):
            self.best_epoch = trainer.best_epoch
            return True
        return False

    def setup(self, trainer: "Trainer"):
        self.best_epoch = None

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        if self._should_save(trainer):
            # build the name of the current model
            score = trainer.current_scores["val"][self.monitor]
            epoch = trainer.current_epoch
            model_name = f"model-{epoch:02d}_loss-{trainer.current_loss:.2f}_{self.monitor}-{score:.2f}"
            filename = self.model_folder / f"{model_name}.pth"
            # wait, then store from process 0
            trainer.accelerator.wait_for_everyone()
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            trainer.accelerator.save(unwrapped_model.state_dict(), filename)
            if self.verbose:
                LOG.info("[Epoch %2d] Checkpoint saved: %s", trainer.current_epoch, str(filename))
        else:
            if self.verbose:
                LOG.info("[Epoch %2d] No checkpoint saved", trainer.current_epoch)

    def dispose(self, trainer: "Trainer"):
        self.best_epoch = None


class DisplaySamples(BaseCallback):
    def __init__(self,
                 inverse_transform: Callable,
                 mask_palette: Dict[int, tuple],
                 image_transform: Optional[Callable] = None,
                 slice_at: int = -1,
                 call_every: int = 1,
                 stage: str = "val") -> None:
        super().__init__(call_every=call_every)
        self.inverse_transform = inverse_transform
        self.image_transform = image_transform
        self.color_palette = mask_palette
        self.slice_at = slice_at
        self.stage = stage

    def setup(self, trainer: "Trainer"):
        if trainer.sample_batches is None or len(trainer.sample_batches) == 0:
            LOG.warn("An ImagePlotter callback is active, but no samples have been found, have you set them?")

    def call(self, trainer: "Trainer", *args: Any, filepath: str = None, filename: str = None, **kwargs: Any) -> Any:
        if not trainer.sample_content:
            LOG.warn("No content to be displayed")
        for i, (image, y_true, y_pred) in enumerate(trainer.sample_content):
            image = self.inverse_transform(image)
            if self.image_transform is not None:
                image = self.image_transform(image[:, :, :self.slice_at])
            true_masks = mask_to_rgb(y_true.numpy(), palette=self.color_palette)
            pred_masks = mask_to_rgb(y_pred.numpy(), palette=self.color_palette)
            grid = make_grid(image, true_masks, pred_masks)
            if filepath:
                plt.imsave(filepath / f"{filename}.png", grid)
            else:
                trainer.logger.log_image(f"{self.stage}/sample-{i}", image=grid, step=trainer.current_epoch)

    def dispose(self, trainer: "Trainer"):
        trainer.sample_content.clear()
