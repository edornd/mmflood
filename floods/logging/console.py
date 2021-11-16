from logging import Logger

from floods.utils.ml import only_rank


class DistributedLogger:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    @only_rank()
    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    @only_rank()
    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    @only_rank()
    def warn(self, *args, **kwargs):
        self.logger.warn(*args, **kwargs)
