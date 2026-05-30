"""Training script for the LSTM spot-price forecasting model.

Usage:
    cd project
    python -m src.train
    python -m src.train --data data/g5_2xlarge_6h_after_2024-07_with_features.parquet \\
                        --config configs/training.yaml \\
                        --artifacts artifacts/
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.features import FEATURE_COLUMNS
from src.model import LSTMModel, TimeSeriesDataset

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "g5_2xlarge_6h_after_2024-07_with_features.parquet"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data_path: Path = DEFAULT_DATA,
    config_path: Path = DEFAULT_CONFIG,
    artifact_dir: Path = DEFAULT_ARTIFACTS,
    device: torch.device | None = None,
) -> tuple[LSTMModel, StandardScaler]:
    _set_seed(42)
    if device is None:
        device = _get_device()
    print(f"Device: {device}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    seq_len: int = train_cfg["seq_len"]
    horizon: int = train_cfg["horizon"]
    batch_size: int = train_cfg["batch_size"]
    epochs: int = train_cfg["epochs"]
    lr: float = train_cfg["learning_rate"]

    df = pl.read_parquet(data_path)
    data = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    target_idx = FEATURE_COLUMNS.index("cost")
    input_size = len(FEATURE_COLUMNS)

    split_idx = int(len(data) * 0.8)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(data[:split_idx])
    test_data = scaler.transform(data[split_idx:])

    train_ds = TimeSeriesDataset(train_data, seq_len, horizon, target_idx)
    test_ds = TimeSeriesDataset(test_data, seq_len, horizon, target_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=input_size,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training LSTM: input_size={input_size}, hidden={model_cfg['hidden_size']}, "
          f"layers={model_cfg['num_layers']}, seq_len={seq_len}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={epoch_loss / len(train_loader):.5f}")

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy().reshape(-1)
            y_pred_list.extend(preds)
            y_true_list.extend(y_batch.numpy().reshape(-1))

    cost_std = scaler.scale_[target_idx]
    cost_mean = scaler.mean_[target_idx]
    y_true = np.array(y_true_list) * cost_std + cost_mean
    y_pred = np.array(y_pred_list) * cost_std + cost_mean

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    print(f"Test → MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "lstm.pt")
    with open(artifact_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2)}
    import json
    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {artifact_dir}/")
    return model, scaler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM spot-price forecaster")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(data_path=args.data, config_path=args.config, artifact_dir=args.artifacts)
