from pathlib import Path
from typing import Dict, Tuple

import albumentations as alb
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn

from floods.config import TrainConfig
from floods.config.testing import TestConfig
from floods.datasets.base import DatasetBase
from floods.datasets.flood import FloodDataset
from floods.metrics import ConfusionMatrix, F1Score, IoU, Metric
from floods.models import create_decoder, create_encoder
from floods.models.base import Segmenter
from floods.models.modules import SegmentationHead
from floods.transforms import ClipNormalize, Denormalize
from floods.utils.common import get_logger
from floods.utils.tiling.functional import mask_body_ratio_from_threshold

LOG = get_logger(__name__)


def train_transforms_base(image_size: int):
    min_crop = image_size // 2
    max_crop = image_size
    transforms = [
        alb.RandomSizedCrop(min_max_height=(min_crop, max_crop), height=image_size, width=image_size, p=0.8),
        alb.ElasticTransform(alpha=1, sigma=30, alpha_affine=30),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
    ]
    # if input channels are 4 and mean and std are for RGB only, copy red for IR
    return alb.Compose(transforms)


def train_transforms_sar():
    transforms = [
        alb.GaussianBlur(blur_limit=(3, 11), p=0.5)
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


def prepare_datasets(config: TrainConfig) -> Tuple[DatasetBase, DatasetBase]:
    # a bit dirty, but at least check that in_channels allows for DEM if present
    required_channels = 3 if config.include_dem else 2
    assert config.in_channels == required_channels, \
        f"Declared channels: {required_channels}, required: {config.in_channels}"

    # instantiate transforms for training and evaluation
    data_root = Path(config.data_root)
    mean = FloodDataset.mean()[:config.in_channels]
    std = FloodDataset.std()[:config.in_channels]
    clip_min = FloodDataset.clip_min()[:config.in_channels]
    clip_max = FloodDataset.clip_max()[:config.in_channels]

    base_trf = train_transforms_base(image_size=config.image_size)
    sar_trf = train_transforms_sar()
    dem_trf = train_transforms_dem(config.channel_drop)
    # store here just for config logging purposes
    config.model.transforms = str(base_trf) + str(sar_trf) + str(dem_trf)
    normalize = eval_transforms(mean=mean,
                                std=std,
                                clip_min=clip_min,
                                clip_max=clip_max,)
    # also print them, just in case
    LOG.info("Train transforms: %s", config.model.transforms)
    LOG.info("Eval. transforms: %s", str(normalize))
    # create train and validation sets
    train_dataset = FloodDataset(path=data_root,
                                 subset="train",
                                 include_dem=config.include_dem,
                                 transform_base=base_trf,
                                 transform_sar=sar_trf,
                                 transform_dem=dem_trf,
                                 normalization=normalize)
    valid_dataset = FloodDataset(path=data_root,
                                 subset="val",
                                 include_dem=config.include_dem,
                                 normalization=normalize)
    # create a temporary dataset to generate a mask useful to filter all the images
    # for which the amout of segmentation is lower than a given percentage
    if(config.segmentation_threshold is not None and config.segmentation_threshold > 0):
        complete_dataset = FloodDataset(path=data_root,
                                        subset="train",
                                        include_dem=config.include_dem)

        # get and apply mask to the training set
        imgs_mask, counts = mask_body_ratio_from_threshold(complete_dataset.label_files, config.segmentation_threshold)
        train_dataset.add_mask(imgs_mask)

        LOG.info(f"Number of elements kept: {counts[1]}")
        LOG.info(f"Number of elements removed: {counts[0]}")
        LOG.info(f"Ratio: {round(counts[1]/len(imgs_mask), 2)}")

    return train_dataset, valid_dataset


def prepare_model(config: TrainConfig, num_classes: int) -> nn.Module:
    cfg = config.model

    # if cfg.decoder in ('deeplab', 'pspnet'):
    #     #...
    #     return # model

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
                                 channels=config.in_channels)
    else:
        NotImplementedError(f"Multimodal encoders not supported: {cfg.encoder}")
    # create decoder: always uses the main encoder as reference
    additional_args = dict()
    if hasattr(config.model, "dropout2d"):
        LOG.info("Adding 2D dropout to the decoder's head: %s", str(config.model.dropout2d))
        additional_args.update(drop_channels=config.model.dropout2d)
    decoder = create_decoder(name=cfg.decoder,
                             feature_info=encoder.feature_info,
                             act_layer=cfg.act,
                             norm_layer=cfg.norm,
                             **additional_args)
    # extract intermediate features when encoder KD is required
    # TODO: remove extract features if not used down the line
    extract_features = False
    LOG.info("Returning intermediate features: %s", str(extract_features))
    # create final segmentation head and build model
    head = SegmentationHead(in_channels=decoder.output(), num_classes=num_classes)
    model = Segmenter(encoder, decoder, head, return_features=extract_features)
    return model


def prepare_metrics(config: TrainConfig, device: torch.device, num_classes: int) -> Tuple[dict, dict]:
    # prepare metrics
    t_metrics = config.trainer.train_metrics
    v_metrics = config.trainer.val_metrics
    train_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in t_metrics}
    valid_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in v_metrics}
    valid_metrics.update(dict(class_iou=IoU(num_classes=num_classes, reduction=None, device=device),
                              class_f1=F1Score(num_classes=num_classes, reduction=None, device=device)))
    LOG.debug("Train metrics: %s", str(list(train_metrics.keys())))
    LOG.debug("Eval. metrics: %s", str(list(valid_metrics.keys())))
    return train_metrics, valid_metrics


def prepare_test_metrics(config: TestConfig, device: torch.device, num_classes: int) -> Dict[str, Metric]:
    test_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in config.test_metrics}
    # include class-wise metrics
    test_metrics.update(dict(class_iou=IoU(num_classes=num_classes, reduction=None, device=device),
                             class_f1=F1Score(num_classes=num_classes, reduction=None, device=device)))
    # include a confusion matrix
    test_metrics.update(dict(conf_mat=ConfusionMatrix(num_classes=num_classes, device=device)))
    return test_metrics
