from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

TARGET_COLUMN = "target"
DATE_COLUMN = "date"
DATASET_NAME = "S12-hw-dataset.csv"
BASELINE_FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_24",
    "lag_168",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_24",
    "rolling_std_24",
    "day_of_week",
    "hour",
    "is_weekend",
]


@dataclass(frozen=True)
class TemporalSplit:
    train_end: int
    val_end: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def val_start_timestamp(self) -> pd.Timestamp:
        return self.val.iloc[0][DATE_COLUMN]

    @property
    def test_start_timestamp(self) -> pd.Timestamp:
        return self.test.iloc[0][DATE_COLUMN]

    @property
    def summary(self) -> str:
        train_range = f"{self.train.iloc[0][DATE_COLUMN]}..{self.train.iloc[-1][DATE_COLUMN]}"
        val_range = f"{self.val.iloc[0][DATE_COLUMN]}..{self.val.iloc[-1][DATE_COLUMN]}"
        test_range = f"{self.test.iloc[0][DATE_COLUMN]}..{self.test.iloc[-1][DATE_COLUMN]}"
        return (
            f"train={len(self.train)}[{train_range}]; "
            f"val={len(self.val)}[{val_range}]; "
            f"test={len(self.test)}[{test_range}]"
        )


class SequenceForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        scaled_series: np.ndarray,
        sample_indices: Sequence[int],
        window_size: int,
        horizon: int = 1,
    ) -> None:
        self.scaled_series = np.asarray(scaled_series, dtype=np.float32)
        self.sample_indices = list(sample_indices)
        self.window_size = int(window_size)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_idx = self.sample_indices[idx]
        start_idx = target_idx - self.window_size
        end_idx = target_idx
        window = self.scaled_series[start_idx:end_idx]
        target = self.scaled_series[target_idx + self.horizon - 1]
        x = torch.tensor(window[:, None], dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y


class GRUForecastModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_series_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    return frame


def summarize_frame(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "n_rows": int(len(frame)),
        "date_min": frame[DATE_COLUMN].min(),
        "date_max": frame[DATE_COLUMN].max(),
        "missing_by_column": frame.isna().sum().to_dict(),
        "frequency": pd.infer_freq(frame[DATE_COLUMN]),
    }


def temporal_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> TemporalSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train = frame.iloc[:train_end].copy()
    val = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:].copy()

    return TemporalSplit(
        train_end=train_end,
        val_end=val_end,
        train=train,
        val=val,
        test=test,
    )


def add_baseline_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = frame.copy()
    for lag in (1, 7, 14, 24, 168):
        featured[f"lag_{lag}"] = featured[TARGET_COLUMN].shift(lag)

    for window in (7, 24):
        shifted = featured[TARGET_COLUMN].shift(1)
        featured[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        featured[f"rolling_std_{window}"] = shifted.rolling(window).std()

    featured["day_of_week"] = featured[DATE_COLUMN].dt.dayofweek
    featured["hour"] = featured[DATE_COLUMN].dt.hour
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    return featured


def build_baseline_frame(frame: pd.DataFrame, split: TemporalSplit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    featured = add_baseline_features(frame)
    featured = featured.dropna().reset_index(drop=True)

    train = featured[featured[DATE_COLUMN] < split.val_start_timestamp].copy()
    val = featured[
        (featured[DATE_COLUMN] >= split.val_start_timestamp)
        & (featured[DATE_COLUMN] < split.test_start_timestamp)
    ].copy()
    test = featured[featured[DATE_COLUMN] >= split.test_start_timestamp].copy()
    return train, val, test


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    epsilon = 1e-8
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / np.clip(np.abs(true), epsilon, None))) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape}


def naive_last_predictions(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET_COLUMN].shift(1)


def moving_average_predictions(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame[TARGET_COLUMN].shift(1).rolling(window).mean()


def fit_ridge_baseline(train_frame: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_frame[BASELINE_FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    return model


def evaluate_ridge(
    model: Pipeline,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions = model.predict(frame[BASELINE_FEATURE_COLUMNS])
    metrics = regression_metrics(frame[TARGET_COLUMN], predictions)
    return predictions, metrics


def build_sequence_indices(
    series_length: int,
    split: TemporalSplit,
    window_size: int,
    horizon: int = 1,
) -> dict[str, list[int]]:
    indices = {"train": [], "val": [], "test": []}
    for target_idx in range(window_size, series_length - horizon + 1):
        if target_idx < split.train_end:
            indices["train"].append(target_idx)
        elif target_idx < split.val_end:
            indices["val"].append(target_idx)
        else:
            indices["test"].append(target_idx)
    return indices


def make_sequence_dataloaders(
    frame: pd.DataFrame,
    split: TemporalSplit,
    window_size: int,
    batch_size: int,
    horizon: int = 1,
) -> tuple[dict[str, DataLoader], StandardScaler, dict[str, list[int]]]:
    scaler = StandardScaler()
    scaler.fit(frame.iloc[: split.train_end][[TARGET_COLUMN]])

    scaled_series = scaler.transform(frame[[TARGET_COLUMN]]).astype(np.float32).reshape(-1)
    indices = build_sequence_indices(
        series_length=len(frame),
        split=split,
        window_size=window_size,
        horizon=horizon,
    )

    loaders: dict[str, DataLoader] = {}
    for split_name, sample_indices in indices.items():
        dataset = SequenceForecastDataset(
            scaled_series=scaled_series,
            sample_indices=sample_indices,
            window_size=window_size,
            horizon=horizon,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
        )
    return loaders, scaler, indices


def inverse_scale_array(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def predict_gru(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).detach().cpu().numpy()
            predictions.append(outputs)
    if not predictions:
        return np.array([], dtype=float)
    scaled_predictions = np.concatenate(predictions)
    return inverse_scale_array(scaled_predictions, scaler)


def evaluate_gru(
    model: nn.Module,
    loader: DataLoader,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    target_indices: Sequence[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    predictions = predict_gru(model, loader, scaler, device)
    targets = frame.iloc[list(target_indices)][TARGET_COLUMN].to_numpy()
    metrics = regression_metrics(targets, predictions)
    return targets, predictions, metrics


def train_gru_model(
    loaders: dict[str, DataLoader],
    frame: pd.DataFrame,
    scaler: StandardScaler,
    split_indices: dict[str, list[int]],
    device: torch.device,
    *,
    window_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float], int]:
    model = GRUForecastModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_val_mae = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        _, _, val_metrics = evaluate_gru(
            model=model,
            loader=loaders["val"],
            frame=frame,
            scaler=scaler,
            target_indices=split_indices["val"],
            device=device,
        )
        train_loss = float(np.mean(batch_losses)) if batch_losses else math.nan
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mape": val_metrics["mape"],
            }
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("GRU training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model, history, best_metrics, best_epoch


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def rows_to_frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))
