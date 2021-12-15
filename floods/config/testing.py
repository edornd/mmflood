from typing import List

from pydantic import Field

from floods.config.training import Metrics, TrainConfig


class TestConfig(TrainConfig):
    data_root: str = Field(None, description="Path to the dataset")
    checkpoint_path: str = Field(None, description="Path to the checkpoint to be tested")
    store_predictions: bool = Field(True, description="Whether to store predicted images or not")
    prediction_count: int = Field(16, description="How many predictions to store (chosen randomly)")
    test_metrics: List[Metrics] = Field([Metrics.f1, Metrics.iou, Metrics.precision, Metrics.recall],
                                        description="Which validation metrics to use")
