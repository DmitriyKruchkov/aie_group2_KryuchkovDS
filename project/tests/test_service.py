"""Tests for the FastAPI service (src/service.py).

Uses FastAPI's TestClient with mocked ML dependencies so no trained
model is required to run the test suite.
"""
from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# TestClient requires httpx; skip gracefully if not installed
pytest.importorskip("httpx", reason="httpx required for TestClient")
from fastapi.testclient import TestClient

from src.features import FEATURE_COLUMNS
from src.model import LSTMModel
from src.service import app


# ------------------------------------------------------------------ fixtures

@pytest.fixture()
def mock_model():
    """Minimal LSTM that always predicts 0.95 (scaled)."""
    model = LSTMModel(input_size=len(FEATURE_COLUMNS), hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        # Zero-out weights so output is near 0 in scaled space (will denorm to ~mean)
        for p in model.parameters():
            p.zero_()
    return model


@pytest.fixture()
def mock_scaler():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    n_features = len(FEATURE_COLUMNS)
    dummy = np.ones((10, n_features), dtype=np.float32)
    dummy[:, 0] = np.linspace(0.9, 1.1, 10)  # cost column variation
    scaler.fit(dummy)
    return scaler


@pytest.fixture()
def mock_feature_window():
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.default_rng(42)
    return rng.random((50, n_features)).astype(np.float32)


@pytest.fixture()
def client(mock_model, mock_scaler, mock_feature_window):
    """TestClient with mocked globals."""
    import src.service as svc

    svc._model = mock_model
    svc._scaler = mock_scaler
    svc._seq_len = 14
    svc._horizon = 4
    svc._feature_window = mock_feature_window
    svc._raw_costs = [0.95 + 0.001 * i for i in range(50)]

    with TestClient(app) as c:
        yield c

    # cleanup
    svc._model = None
    svc._scaler = None
    svc._feature_window = None
    svc._raw_costs = []


# ------------------------------------------------------------------ /health

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded_true(self, client):
        data = resp = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_without_model(self):
        with patch("src.service._load_artifacts", return_value=False):
            with TestClient(app) as c:
                resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


# ------------------------------------------------------------------ /predict

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={"current_price": 1.0, "max_price": 1.2})
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json={"current_price": 0.95, "max_price": 1.1})
        data = resp.json()
        required = {
            "current_price", "predicted_price", "max_price",
            "horizon_hours", "is_safe", "message",
        }
        assert required.issubset(data.keys())

    def test_predict_horizon_hours(self, client):
        resp = client.post("/predict", json={"current_price": 0.95, "max_price": 1.1})
        # horizon=4 steps × 6 h = 24 h
        assert resp.json()["horizon_hours"] == 24

    def test_predict_echoes_inputs(self, client):
        resp = client.post("/predict", json={"current_price": 0.95, "max_price": 1.05})
        data = resp.json()
        assert abs(data["current_price"] - 0.95) < 1e-4
        assert abs(data["max_price"] - 1.05) < 1e-4

    def test_predict_safe_when_below_max(self, client, mock_scaler):
        import src.service as svc

        # Make model return a scaled value that denorms to 0.95
        target_idx = FEATURE_COLUMNS.index("cost")
        cost_mean = mock_scaler.mean_[target_idx]
        cost_std = mock_scaler.scale_[target_idx]
        desired_price = 0.95
        scaled_val = (desired_price - cost_mean) / cost_std

        with patch.object(svc._model, "forward", return_value=torch.tensor([scaled_val])):
            resp = client.post("/predict", json={"current_price": 0.95, "max_price": 2.0})

        data = resp.json()
        assert data["is_safe"] is True
        assert "Safe" in data["message"]

    def test_predict_unsafe_when_above_max(self, client, mock_scaler):
        import src.service as svc

        target_idx = FEATURE_COLUMNS.index("cost")
        cost_mean = mock_scaler.mean_[target_idx]
        cost_std = mock_scaler.scale_[target_idx]
        desired_price = 1.5
        scaled_val = (desired_price - cost_mean) / cost_std

        with patch.object(svc._model, "forward", return_value=torch.tensor([scaled_val])):
            resp = client.post("/predict", json={"current_price": 1.5, "max_price": 1.0})

        data = resp.json()
        assert data["is_safe"] is False
        assert "Danger" in data["message"] or "exceeds" in data["message"]

    def test_predict_invalid_price_rejected(self, client):
        resp = client.post("/predict", json={"current_price": -1.0, "max_price": 1.0})
        assert resp.status_code == 422

    def test_predict_zero_price_rejected(self, client):
        resp = client.post("/predict", json={"current_price": 0.0, "max_price": 1.0})
        assert resp.status_code == 422

    def test_predict_503_when_no_model(self):
        with patch("src.service._load_artifacts", return_value=False):
            with TestClient(app) as c:
                resp = c.post("/predict", json={"current_price": 1.0, "max_price": 1.2})
        assert resp.status_code == 503
