from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from floods.logging import BaseLogger
from floods.utils.ml import only_rank


class TensorBoardLogger(BaseLogger):
    def __init__(self,
                 log_folder: Path = Path("logs"),
                 filename_suffix: str = "",
                 current_step: int = 0,
                 comment: str = "") -> None:
        super().__init__()
        self.log = SummaryWriter(log_dir=log_folder, filename_suffix=filename_suffix, comment=comment)
        self.current_step = current_step

    def step(self, iteration: int = None) -> None:
        if not iteration:
            self.current_step += 1
        else:
            self.current_step = iteration

    def get_step(self, kwargs: dict) -> int:
        return kwargs.pop("step", self.current_step)

    @only_rank(0)
    def log_model(self, model: nn.Module, input_size: tuple = (1, 4, 256, 256), device: str = "cpu") -> None:
        sample_input = torch.rand(input_size, device=device)
        self.log.add_graph(model, input_to_model=sample_input)

    @only_rank(0)
    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        self.log.add_scalar(name, value, global_step=self.get_step(kwargs), **kwargs)

    @only_rank(0)
    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        self.log.add_image(name, image, global_step=self.get_step(kwargs), dataformats="HWC", **kwargs)

    @only_rank(0)
    def log_figure(self, name: str, figure: Figure, **kwargs) -> None:
        self.log.add_figure(name, figure, global_step=self.get_step(kwargs), **kwargs)

    @only_rank(0)
    def log_table(self, name: str, table: Dict[str, str], **kwargs: dict):
        table_html = "<table width=\"100%\"> "
        table_html += "<tr><th>Key</th><th>Value</th></tr>"
        # iterate dictionary rows
        for k, v in table.items():
            table_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        table_html += "</table>"
        # log the table as html text
        self.log.add_text(name, table_html, global_step=self.get_step(kwargs))

    @only_rank(0)
    def log_results(self, name: str, headers: List[str], results: Dict[str, List[float]], **kwargs: dict):
        header_html = "".join([f"<th>{h}</th>" for h in headers])
        table_html = f"<table width=\"100%\"><tr><th>metric/class</th>{header_html}</tr>"
        # iterate results and write them into a table
        for score_name, scores in results.items():
            row_html = "".join([f"<td>{x:.4f}</td>" for x in scores])
            table_html += f"<tr><td>{score_name}</td>{row_html}</tr>"
        table_html += "</table>"
        self.log.add_text(name, table_html, global_step=self.get_step(kwargs))
