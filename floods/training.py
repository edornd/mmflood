from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from floods.config import TrainConfig
from floods.logging.tensorboard import TensorBoardLogger
from floods.prepare import inverse_transform, prepare_datasets, prepare_metrics, prepare_model
from floods.trainer.callbacks import Checkpoint, DisplaySamples, EarlyStopping, EarlyStoppingCriterion
from floods.trainer.flood import FloodTrainer
from floods.utils.common import flatten_config, get_logger, git_revision_hash, init_experiment, store_config
from floods.utils.ml import load_class_weights, seed_everything, seed_worker

LOG = get_logger(__name__)


def train(config: TrainConfig):
    # assertions before starting
    assert config.trainer.amp and torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."
    # prepare accelerator ASAP
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)

    # Create the directory tree:
    # outputs
    #  |-- exp_name
    #      |-- models
    #      |-- logs
    accelerator.wait_for_everyone()
    log_name = "output.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=config, log_name=log_name)
    config_path = out_folder / "config.yaml"
    LOG.info("Run started")
    LOG.info("Experiment ID: %s", exp_id)
    LOG.info("Output folder: %s", out_folder)
    LOG.info("Models folder: %s", model_folder)
    LOG.info("Logs folder:   %s", logs_folder)
    LOG.info("Configuration: %s", config_path)

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed, deterministic=True)
    # prepare datasets
    LOG.info("Loading datasets...")
    train_set, valid_set = prepare_datasets(config=config)
    num_classes = len(train_set.categories())
    LOG.info("Full sets - train set: %d samples, validation set: %d samples", len(train_set), len(valid_set))
    # construct data loaders
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=True,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=False,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker)
    # prepare models
    LOG.info("Preparing model...")
    new_model = prepare_model(config=config, num_classes=num_classes).to(accelerator.device)

    # prepare optimizer and scheduler
    optimizer = config.optimizer.instantiate(new_model.parameters())
    scheduler = config.scheduler.instantiate(optimizer)
    # prepare losses
    weights = None
    if config.class_weights:
        weights = load_class_weights(Path(config.class_weights), device=accelerator.device, normalize=False)
        LOG.info("Using class weights: %s", str(weights))
    loss = config.loss.instantiate(ignore_index=255, weight=weights)

    # prepare metrics and logger
    monitored = config.trainer.monitor.name
    train_metrics, valid_metrics = prepare_metrics(config, device=accelerator.device, num_classes=num_classes)
    logger = TensorBoardLogger(log_folder=logs_folder, comment=config.comment)
    # logging configuration to tensorboard
    LOG.debug("Logging flattened configuration to TensorBoard")
    logger.log_table("config", flatten_config(config.dict()))

    # prepare trainer
    LOG.info("Visualize: %s, num. batches for visualization: %s", str(config.visualize), str(config.num_samples))
    num_samples = int(config.visualize) * config.num_samples
    # choose trainer class depending on task or regularization
    trainer = FloodTrainer(accelerator=accelerator,
                           model=new_model,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           criterion=loss,
                           categories=train_set.categories(),
                           train_metrics=train_metrics,
                           val_metrics=valid_metrics,
                           logger=logger,
                           sample_batches=num_samples,
                           debug=config.debug)
    trainer.add_callback(EarlyStopping(call_every=1,
                                       metric=monitored,
                                       criterion=EarlyStoppingCriterion.maximum,
                                       patience=config.trainer.patience)) \
           .add_callback(Checkpoint(call_every=1,
                                    monitor=monitored,
                                    model_folder=model_folder,
                                    save_best=True)) \
           .add_callback(DisplaySamples(inverse_transform=inverse_transform(mean=train_set.mean(), std=train_set.std()),
                                        color_palette=train_set.palette()))
    # storing config and starting training
    config.version = git_revision_hash()
    store_config(config, path=config_path)
    trainer.fit(train_dataloader=train_loader, val_dataloader=valid_loader, max_epochs=config.trainer.max_epochs)
    LOG.info(f"Training completed at epoch {trainer.current_epoch:<2d} "
             f"(best {monitored}: {trainer.best_score:.4f})")
    LOG.info("Experiment %s completed!", exp_id)
