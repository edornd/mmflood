from clidantic import Parser
from tqdm.contrib.logging import logging_redirect_tqdm

from floods import preproc, testing, training
from floods.config import PreparationConfig, StatsConfig, TrainConfig
from floods.config.testing import TestConfig
from floods.utils.common import prepare_logging

cli = Parser()


@cli.command()
def preprocess(config: PreparationConfig):
    return preproc.preprocess_data(config=config)


@cli.command()
def stats(config: StatsConfig):
    return preproc.compute_statistics(config=config)


@cli.command()
def train(config: TrainConfig):
    training.train(config=config)


def test(config: TestConfig):
    testing.test(config=config)


if __name__ == '__main__':
    prepare_logging()
    with logging_redirect_tqdm():
        cli()
