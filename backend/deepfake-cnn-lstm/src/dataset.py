from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, index_csv: str | Path):
        self.df = pd.read_csv(index_csv)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["sequence_path"])
        x = torch.from_numpy(x).float()
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y
