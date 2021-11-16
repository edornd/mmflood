from pathlib import Path

from torch.utils.data import Dataset


class RawFloodDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
    ) -> None:
        super().__init__()
        self.data_root = data_root
