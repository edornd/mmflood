from enum import Enum
from functools import partial
from typing import Any

from pydantic import BaseSettings, Extra


class EnvConfig(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = Extra.ignore


class CallableEnum(Enum):
    """Enum that allows to be called directly, useful for enums that allow
    classes and such as arguments.
    """
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.value(*args, **kwds)


class InstantiableSettings(EnvConfig):
    """Settings that present an instantiate method to create
    their own object in place.
    """
    def instantiate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement in subclass")


class Initializer(partial):
    """Picklable partial version, thanks to the eq implementation.
    """
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, partial):
            return False
        return self.func == o.func and self.args == o.args and self.keywords == o.keywords
