import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()


def get_env(name: str) -> str:
    if (result := os.getenv(name)) is None:
        raise ValueError(f"Missing env variable '{name}'")
    return result


@pytest.fixture(scope="session")
def dataset_path():
    return Path(get_env("DATA_PROCESSED"))
