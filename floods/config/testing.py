from pydantic import Field

from floods.config.training import TrainConfig


class TestConfig(TrainConfig):
    data_root: str = Field(None, description="Path to the dataset")
    store_predictions: bool = Field(False, description="Whether to store predicted images or not")
    pred_count: int = Field(100, description="How many predictions to store (chosen randomly)")
