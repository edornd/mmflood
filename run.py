from clidantic import Parser
from tqdm.contrib.logging import logging_redirect_tqdm

from floods import preproc, testing, training
from floods.config import PreparationConfig, StatsConfig, TrainConfig
from floods.config.testing import TestConfig
from floods.utils.common import prepare_logging
from tests.test_datasets import test_dataset
cli = Parser()


@cli.command()
def preprocess(config: PreparationConfig):
    return preproc.preprocess_data(config=config)


@cli.command()
def stats(config: StatsConfig):
    return test_dataset(config=config)


@cli.command()
def train(config: TrainConfig):
    training.train(config=config)


@cli.command()
def test(config: TestConfig):
    testing.test(config)


if __name__ == '__main__':
    prepare_logging()
    with logging_redirect_tqdm():
        cli()
