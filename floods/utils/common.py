import collections
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union
from uuid import uuid4

import yaml
from pydantic import BaseSettings

from floods.logging.console import DistributedLogger


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")


def git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def generate_id() -> str:
    return str(uuid4())


def prepare_logging():
    """Initializes logging to print infos and with standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)-24s: %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )


def prepare_file_logging(experiment_folder: Path, filename: str = "output.log") -> None:
    logger = logging.getLogger()
    handler = logging.FileHandler(experiment_folder / filename)
    handler.setLevel(logging.INFO)
    # get the handler from the base handler
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return DistributedLogger(logging.getLogger(name))


def check_or_make_dir(path: Union[str, Path]) -> Path:
    """Utility that checks whether the folder already exists or not,
    creates it and always return a pathlib instance.

    Args:
        path (Union[str, Path]): string or path object

    Returns:
        Path: path instance to existing folder
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_config(logger: logging.Logger, config: BaseSettings):
    """Well, it prints the config, one param per line.

    Args:
        config (PreparationConfig): input configuration
    """
    for k, v in config.dict().items():
        logger.info(f"{k:<20s}: {v}")


def prepare_folder(root_folder: Path, experiment_id: str = ""):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    full_path = root_folder / experiment_id
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def store_config(config: BaseSettings, path: Path) -> None:
    with open(str(path), "w") as file:
        yaml.dump(config.dict(), file)


def load_config(path: Path, config_class: Callable) -> BaseSettings:
    assert path.exists(), f"Missing training configuration for path: {path.parent}"
    # load the training configuration
    with open(str(path), "r", encoding="utf-8") as file:
        train_params = yaml.load(file, Loader=yaml.Loader)
    return config_class(**train_params)


def flatten_config(config: dict, parent_key: str = "", separator: str = "/") -> Dict[str, Any]:
    items = []
    for k, v in config.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_config(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_experiment(config: BaseSettings, log_name: str = "output.log"):
    # initialize experiment
    experiment_id = config.name or current_timestamp()
    out_folder = Path(config.output_folder)
    # prepare folders and log outputs
    output_folder = prepare_folder(out_folder, experiment_id=experiment_id)
    prepare_file_logging(output_folder, filename=log_name)

    # prepare experiment directories
    model_folder = prepare_folder(output_folder / "models")
    logs_folder = prepare_folder(output_folder / "logs")
    if config.name is not None:
        assert model_folder.exists() and logs_folder.exists()
    return experiment_id, output_folder, model_folder, logs_folder
