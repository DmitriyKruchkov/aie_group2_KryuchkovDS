"""FastAPI service for AWS Spot price forecasting.

Start:
    cd project
    uvicorn src.service:app --reload --port 8000

Or via Docker (see src/Dockerfile).

Endpoints:
    GET  /health   — liveness check
    POST /predict  — predict price 24 h ahead and check against budget
"""
from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_COLUMNS, compute_new_row_features
from src.model import LSTMModel

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("spot_forecaster")

PROJECT_ROOT = Path(__file__).parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "g5_2xlarge_6h_after_2024-07_with_features.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"

# ---- global state ----------------------------------------------------------
_model: LSTMModel | None = None
_scaler = None
_seq_len: int = 14
_horizon: int = 4
_feature_window: np.ndarray | None = None  # shape (N, 22) — last rows from parquet
_raw_costs: list[float] = []               # last 50 cost values for lag computation


def _load_artifacts() -> bool:
    global _model, _scaler, _seq_len, _horizon, _feature_window, _raw_costs

    if not (ARTIFACT_DIR / "lstm.pt").exists():
        log.warning(
            "Model artifact not found at %s. "
            "Run `python -m src.train` first, then restart the service.",
            ARTIFACT_DIR / "lstm.pt",
        )
        return False

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    _seq_len = cfg["training"]["seq_len"]
    _horizon = cfg["training"]["horizon"]

    with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)

    mc = cfg["model"]
    _model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    )
    _model.load_state_dict(
        torch.load(ARTIFACT_DIR / "lstm.pt", map_location="cpu", weights_only=True)
    )
    _model.eval()

    df = pl.read_parquet(DATA_PATH)
    _feature_window = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    _raw_costs = df["cost"].tail(50).to_list()

    log.info(
        "Model loaded: seq_len=%d, horizon=%d (%d h), history_rows=%d",
        _seq_len, _horizon, _horizon * 6, len(_feature_window),
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="AWS Spot Price Forecaster",
    description=(
        "Predicts g5.2xlarge spot price 24 h ahead and warns if the "
        "predicted price will exceed your max_price budget."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---- schemas ----------------------------------------------------------------

class PredictRequest(BaseModel):
    current_price: float = Field(..., gt=0, description="Current spot price (USD/hr)")
    max_price: float = Field(..., gt=0, description="Your max_price budget (USD/hr)")


class PredictResponse(BaseModel):
    current_price: float
    predicted_price: float
    max_price: float
    horizon_hours: int
    is_safe: bool
    message: str


# ---- endpoints --------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    loaded = _model is not None
    log.debug("Health check: model_loaded=%s", loaded)
    return {
        "status": "ok",
        "model_loaded": loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    log.info(
        "Predict request: current_price=%.4f, max_price=%.4f",
        request.current_price, request.max_price,
    )

    if _model is None or _scaler is None or _feature_window is None:
        log.error("Predict called but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first.",
        )

    history_df = pl.DataFrame({"cost": _raw_costs})
    new_row = compute_new_row_features(
        history=history_df,
        current_price=request.current_price,
        dt=datetime.now(),
    )

    # Build input window: last (seq_len - 1) historical rows + new row
    hist_rows = _feature_window[-(_seq_len - 1):]  # (seq_len-1, 22)
    window = np.vstack([hist_rows, new_row[np.newaxis, :]])  # (seq_len, 22)

    scaled_window = _scaler.transform(window)

    x = torch.tensor(scaled_window[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = float(_model(x).item())

    target_idx = FEATURE_COLUMNS.index("cost")
    cost_mean = float(_scaler.mean_[target_idx])
    cost_std = float(_scaler.scale_[target_idx])
    predicted_price = round(pred_scaled * cost_std + cost_mean, 4)

    is_safe = predicted_price <= request.max_price
    horizon_hours = _horizon * 6

    if is_safe:
        msg = (
            f"Safe to launch: predicted price ${predicted_price:.4f}/hr "
            f"is within your budget of ${request.max_price:.4f}/hr."
        )
    else:
        msg = (
            f"Danger: predicted price ${predicted_price:.4f}/hr "
            f"exceeds your max_price of ${request.max_price:.4f}/hr. "
            "Consider raising max_price or waiting."
        )

    log.info(
        "Predict result: predicted=%.4f, is_safe=%s",
        predicted_price, is_safe,
    )
    return PredictResponse(
        current_price=request.current_price,
        predicted_price=predicted_price,
        max_price=request.max_price,
        horizon_hours=horizon_hours,
        is_safe=is_safe,
        message=msg,
    )


# ---- CLI entry-point --------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.service:app", host="0.0.0.0", port=8000, reload=False)
