from pathlib import Path
from typing import Dict, Tuple

import albumentations as alb
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data.sampler import WeightedRandomSampler

from floods.config import TestConfig, TrainConfig
from floods.datasets.base import DatasetBase
from floods.datasets.flood import FloodDataset, RGBFloodDataset
from floods.metrics import ConfusionMatrix, F1Score, IoU, Metric, Precision, Recall
from floods.models import create_decoder, create_encoder, create_multi_encoder
from floods.models.base import MultiBranchSegmenter, Segmenter
from floods.models.modules import SegmentationHead
from floods.transforms import ClipNormalize, Denormalize
from floods.utils.common import get_logger
from floods.utils.tiling.functional import entropy_weights, mask_body_ratio_from_threshold

LOG = get_logger(__name__)


def train_transforms_base(image_size: int):
    min_crop = image_size // 2
    max_crop = image_size
    transforms = [
        alb.RandomSizedCrop(min_max_height=(min_crop, max_crop), height=image_size, width=image_size, p=0.8),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        alb.GridDistortion(p=0.5)
    ]
    # if input channels are 4 and mean and std are for RGB only, copy red for IR
    return alb.Compose(transforms)


def train_transforms_sar():
    transforms = [
        alb.OneOf([
            alb.GaussianBlur(blur_limit=(3, 13), p=0.5),
            alb.MultiplicativeNoise(multiplier=(0.7, 1.3), elementwise=True, per_channel=True, p=0.5),
        ], p=0.6)
    ]
    return alb.Compose(transforms)


def train_transforms_dem(channel_dropout: float = 0.0):
    transforms = []
    if channel_dropout > 0:
        transforms.append(alb.ChannelDropout(p=channel_dropout))
    return alb.Compose(transforms)


def eval_transforms(mean: tuple,
                    std: tuple,
                    clip_min: tuple,
                    clip_max: tuple) -> alb.Compose:
    return alb.Compose([ClipNormalize(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max),
                        ToTensorV2()])


def inverse_transform(mean: tuple, std: tuple):
    return Denormalize(mean=mean, std=std)


def prepare_datasets(config: TrainConfig, use_rgb: bool = False) -> Tuple[DatasetBase, DatasetBase]:
    # sanity check: least channels = 2, (vv, vh only), max. channels = 4 (rgb ratio + DEM)
    assert config.data.in_channels >= 2 and config.data.in_channels <= 4, \
        f"Declared channels: {config.data.in_channels} not supported"
    # pick which dataset to use, depending on configuration
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset

    # instantiate transforms for training and evaluation
    # adapt hardcoded tensors to the current number of channels
    data_root = Path(config.data.path)
    mean = dataset_cls.mean()[:config.data.in_channels]
    std = dataset_cls.std()[:config.data.in_channels]
    # 3 different blocks required:
    # - base is applied to everything (affine transforms mainly)
    # - sar, dem are only applied to the namesake components
    base_trf = train_transforms_base(image_size=config.image_size)
    sar_trf = train_transforms_sar()
    dem_trf = train_transforms_dem(channel_dropout=0)
    # store here just for config logging purposes
    config.model.transforms = str(base_trf) + str(sar_trf) + str(dem_trf)
    normalize = eval_transforms(mean=mean,
                                std=std,
                                clip_min=-30.0,
                                clip_max=30.0,)
    # also print them, just in case
    LOG.info("Train transforms: %s", config.model.transforms)
    LOG.info("Eval. transforms: %s", str(normalize))
    # create train and validation sets
    train_dataset = dataset_cls(path=data_root,
                                subset="train",
                                include_dem=config.data.include_dem,
                                transform_base=base_trf,
                                transform_sar=sar_trf,
                                transform_dem=dem_trf,
                                normalization=normalize)
    valid_dataset = dataset_cls(path=data_root,
                                subset="val",
                                include_dem=config.data.include_dem,
                                normalization=normalize)
    # create a temporary dataset to generate a mask useful to filter all the images
    # for which the amout of segmentation is lower than a given percentage
    if(config.data.mask_body_ratio is not None and config.data.mask_body_ratio > 0):
        # get and apply mask to the training set
        # we are directly iterating filenames to avoid transforms
        train_imgs_mask, train_counts = mask_body_ratio_from_threshold(labels=train_dataset.label_files,
                                                                       ratio_threshold=config.data.mask_body_ratio,
                                                                       label="train",
                                                                       cache_hash=config.data.cache_hash)
        train_dataset.add_mask(train_imgs_mask)
        LOG.info("Filtering training set with %d images", len(train_imgs_mask))
        LOG.info(f"Number of elements kept: {train_counts[1]}")
        LOG.info(f"Ratio: {train_counts[1]/len(train_imgs_mask):.2f}%")
        # get and apply mask to the validation set
        val_imgs_mask, val_counts = mask_body_ratio_from_threshold(labels=valid_dataset.label_files,
                                                                   ratio_threshold=config.data.mask_body_ratio,
                                                                   label="val",
                                                                   cache_hash=config.data.cache_hash)
        valid_dataset.add_mask(val_imgs_mask)
        LOG.info("Filtering validation set with %d images", len(val_imgs_mask))
        LOG.info(f"Number of elements kept: {val_counts[1]}")
        LOG.info(f"Ratio: {val_counts[1]/len(val_imgs_mask):.2f}%")

    return train_dataset, valid_dataset


def prepare_sampler(dataset: FloodDataset, cache_hash: str, smoothing: float = 0.8) -> WeightedRandomSampler:
    target_file = Path("data/cache") / f"sample-weights_{cache_hash}.npy"
    if target_file.exists() and target_file.is_file():
        LOG.info("Found an existing array of sample weights")
        weights = np.load(str(target_file))
    else:
        LOG.info("Computing weights for weighted random sampling")
        weights = entropy_weights(dataset.label_files, smoothing=smoothing)
        np.save(str(target_file), weights)
    # completely arbitrary, this is just here to maximize the amount of images we look at
    num_samples = len(dataset) * 2
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def prepare_model(config: TrainConfig, num_classes: int, stage: str = "train") -> nn.Module:
    cfg = config.model

    # instead of creating a new var, encoder is exploited for different purposes
    # we expect a single encoder name, or a comma-separated list of names, one for each modality
    # e.g. valid examples: 'tresnet_m' - 'resnet34,resnet34'
    enc_names = cfg.encoder.split(",")
    if len(enc_names) == 1:
        encoder = create_encoder(name=enc_names[0],
                                 decoder=cfg.decoder,
                                 pretrained=cfg.pretrained,
                                 freeze=cfg.freeze,
                                 output_stride=cfg.output_stride,
                                 act_layer=cfg.act,
                                 norm_layer=cfg.norm,
                                 channels=config.data.in_channels)
    else:
        # we only support at most two encoders, one for SAR, one for DEM
        # channels are inferred at runtime: we always expect DEM, so SAr channels = total - 1
        assert len(enc_names) == 2, f"Multimodal encoders not supported: {cfg.encoder}"
        assert config.data.in_channels >= 3, "Multimodal approach only works with 3+ channels (SAR + DEM)"
        LOG.info("Creating a multimodal encoder (%s, %s)", enc_names[0], enc_names[1])
        encoder = create_multi_encoder(sar_name=enc_names[0],
                                       dem_name=enc_names[1],
                                       channels=config.data.in_channels,
                                       config=cfg,
                                       return_features=False)
    # create decoder: always uses the main encoder as reference
    additional_args = dict()

    # Additional dropout executed by channel, not implemented in all decoders, so commented for now
    # if hasattr(config.model, "dropout2d"):
    #     LOG.info("Adding 2D dropout to the decoder's head: %s", str(config.model.dropout2d))
    #     additional_args.update(drop_channels=config.model.dropout2d)
    decoder = create_decoder(name=cfg.decoder,
                             input_size=config.image_size,
                             feature_info=encoder.feature_info,
                             act_layer=cfg.act,
                             norm_layer=cfg.norm,
                             **additional_args)
    # extract intermediate features when encoder KD is required
    # TODO: remove extract features if not used down the line
    extract_features = False
    LOG.info("Returning intermediate features: %s", str(extract_features))
    # create final segmentation head and build model
    head = SegmentationHead(in_channels=decoder.out_channels(),
                            num_classes=num_classes,
                            upscale=decoder.out_reduction())
    if cfg.multibranch and stage != "test":
        auxiliary = SegmentationHead(in_channels=encoder.feature_info.channels()[-1],
                                     num_classes=num_classes,
                                     upscale=encoder.feature_info.reduction()[-1])
        return MultiBranchSegmenter(encoder, decoder, head, auxiliary=auxiliary, return_features=extract_features)
    return Segmenter(encoder, decoder, head, return_features=extract_features)


def prepare_metrics(config: TrainConfig, device: torch.device) -> Tuple[dict, dict]:
    # prepare metrics
    t_metrics = config.trainer.train_metrics
    v_metrics = config.trainer.val_metrics
    train_metrics = {e.name: e.value(device=device) for e in t_metrics}
    valid_metrics = {e.name: e.value(device=device) for e in v_metrics}
    valid_metrics.update(dict(class_iou=IoU(reduction=None, device=device),
                              class_f1=F1Score(reduction=None, device=device)))
    LOG.debug("Train metrics: %s", str(list(train_metrics.keys())))
    LOG.debug("Eval. metrics: %s", str(list(valid_metrics.keys())))
    return train_metrics, valid_metrics


def prepare_test_metrics(config: TestConfig, device: torch.device) -> Dict[str, Metric]:
    test_metrics = {e.name: e.value(device=device) for e in config.test_metrics}
    # include class-wise metrics
    test_metrics.update(dict(precision=Precision(reduction=None, device=device),
                             recall=Recall(reduction=None, device=device),
                             fg_iou=IoU(reduction=None, device=device),
                             fg_f1=F1Score(reduction=None, device=device),
                             bg_iou=IoU(reduction=None, device=device, background=True),
                             bg_f1=F1Score(reduction=None, device=device, background=True)))
    # include a confusion matrix
    test_metrics.update(dict(conf_mat=ConfusionMatrix(device=device)))
    return test_metrics
