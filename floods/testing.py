from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from floods.config.testing import TestConfig
from floods.config.training import TrainConfig
from floods.datasets.flood import FloodDataset, RGBFloodDataset
from floods.logging.functional import plot_confusion_matrix
from floods.logging.tensorboard import TensorBoardLogger
from floods.prepare import eval_transforms, inverse_transform, prepare_model, prepare_test_metrics
from floods.trainer.callbacks import DisplaySamples
from floods.trainer.flood import FloodTrainer
from floods.utils.common import check_or_make_dir, get_logger, init_experiment, load_config, print_config
from floods.utils.gis import as_image, rgb_ratio
from floods.utils.ml import find_best_checkpoint, load_class_weights, seed_everything, seed_worker
from floods.utils.tiling import SmoothTiler

LOG = get_logger(__name__)


def test(test_config: TestConfig):
    # assertions before starting
    assert test_config.name is not None, "Specify the experiment name to test!"

    # prepare the test log
    log_name = "output-test.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=test_config, log_name=log_name)
    config_path = out_folder / "config.yaml"
    config: TrainConfig = load_config(path=config_path, config_class=TrainConfig)
    print_config(LOG, config)

    # # prepare accelerator
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)
    accelerator.wait_for_everyone()

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare evaluation transforms
    LOG.info("Loading test dataset...")
    num_classes = 1
    # RGB is needed with either 4 - 1, or 3 - 0
    use_rgb = (config.data.in_channels - int(config.data.include_dem)) == 3
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset
    mean = dataset_cls.mean()[:config.data.in_channels]
    std = dataset_cls.std()[:config.data.in_channels]
    test_transform = eval_transforms(mean=mean,
                                     std=std,
                                     clip_max=30,
                                     clip_min=-30)

    LOG.debug("Eval. transforms: %s", str(test_transform))
    # create the test dataset
    test_dataset = dataset_cls(path=Path(config.data.path),
                               subset="test",
                               include_dem=config.data.include_dem,
                               normalization=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,  # fixed at 1 because in test we have full-size images
                             shuffle=False,
                             num_workers=test_config.trainer.num_workers,
                             worker_init_fn=seed_worker)
    # prepare model for inference, set pretrained to False to avoid loading additional weights
    LOG.info("Preparing model...")
    config.model.pretrained = False
    model = prepare_model(config=config, num_classes=num_classes, stage="test")
    # load the best checkpoint available
    if test_config.checkpoint_path is not None:
        ckpt_path = Path(test_config.checkpoint_path)
    else:
        ckpt_path = find_best_checkpoint(model_folder)
    # load the checkpoint found: make sure to load without strict in case of aux losses
    assert ckpt_path.exists(), f"Checkpoint '{str(ckpt_path)}' not found"
    strict_load = not config.model.multibranch
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"), strict=strict_load)
    LOG.info("Model restored from: %s", str(ckpt_path))

    # prepare losses
    weights = None
    if config.data.class_weights:
        weights = load_class_weights(Path(config.data.class_weights), device=accelerator.device, normalize=False)
        LOG.info("Using class weights: %s", str(weights))
    loss = config.loss.instantiate(ignore_index=255, weight=weights)
    # prepare metrics and logger
    logger = TensorBoardLogger(log_folder=logs_folder, comment=config.comment, filename_suffix="-test")

    # randomly choose images to store (too heavy to visualize)
    LOG.info("Storing predicted images: %s", str(test_config.store_predictions).lower())
    num_samples = int(test_config.store_predictions) * test_config.prediction_count
    LOG.info("Storing batches: %s", str(num_samples))

    # prepare tiler to produce tiled batches and trainer
    tiler = SmoothTiler(tile_size=test_config.image_size,
                        channels_first=True,
                        batch_size=test_config.trainer.batch_size,
                        mirrored=False)
    trainer = FloodTrainer(accelerator=accelerator,
                           model=model,
                           criterion=loss,
                           tiler=tiler,
                           categories=test_dataset.categories(),
                           logger=logger,
                           sample_batches=num_samples,
                           stage="test",
                           debug=test_config.debug)
    image_trf = as_image if use_rgb else rgb_ratio
    slice_at = config.data.in_channels - int(config.data.include_dem)
    trainer.add_callback(DisplaySamples(inverse_transform=inverse_transform(test_dataset.mean(), test_dataset.std()),
                                        mask_palette=test_dataset.palette(),
                                        image_transform=image_trf,
                                        slice_at=slice_at,
                                        stage="test"))
    # prepare testing metrics, same as validation with the addition of a confusion matrix
    eval_metrics = prepare_test_metrics(config=test_config, device=accelerator.device)

    predictions_path = check_or_make_dir(out_folder / "images")
    losses, _ = trainer.predict(test_dataloader=test_loader,
                                metrics=eval_metrics,
                                logger_exclude=["conf_mat"],
                                output_path=predictions_path)
    scores = trainer.current_scores["test"]
    # logging stuff to file and storing images if required
    LOG.info("Testing completed, average loss: %.4f", np.mean(losses))

    LOG.info("Average results:")
    classwise = dict()
    for i, (name, score) in enumerate(scores.items()):
        # only printing reduced metrics
        if score.ndim == 0:
            LOG.info(f"{name:<20s}: {score.item():.4f}")
        elif name != "conf_mat":
            classwise[name] = score

    # LOG.info("Class-wise results:")
    # header = f"{'score':<20s}  " + "|".join([f"{label:<15s}" for label in trainer.categories.values()])
    # LOG.info(header)
    # for i, (name, score) in enumerate(classwise.items()):
    #     scores_str = [f"{v:.4f}" for v in score]
    #     scores_str = "|".join(f"{s:<15s}" for s in scores_str)
    #     LOG.info(f"{name:<20s}: {scores_str}")

    LOG.info("Plotting confusion matrix...")
    cm_name = f"cm_{Path(ckpt_path).stem}"
    plot_folder = check_or_make_dir(out_folder / "plots")
    plot_confusion_matrix(scores["conf_mat"].cpu().numpy(),
                          destination=plot_folder / f"{cm_name}.png",
                          labels=trainer.categories.values(),
                          title=cm_name,
                          normalize=False)
    LOG.info("Testing done!")
