"""LSTM model and dataset for spot price forecasting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Windowed dataset for LSTM training.

    Each sample: input = data[idx : idx+seq_len], target = data[idx+seq_len+horizon-1, target_idx].
    This predicts `target_idx` column `horizon` steps ahead of the window end.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        horizon: int,
        target_idx: int,
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len + self.horizon - 1, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    """Single-output LSTM regressor for time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
