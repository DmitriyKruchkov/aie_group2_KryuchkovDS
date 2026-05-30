"""Tests for LSTM model and dataset (src/model.py)."""
import numpy as np
import pytest
import torch

from src.model import LSTMModel, TimeSeriesDataset


def _random_data(n: int = 100, n_features: int = 22) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, n_features)).astype(np.float32)


# ------------------------------------------------------------------ TimeSeriesDataset

class TestTimeSeriesDataset:
    def test_length(self):
        data = _random_data(100)
        ds = TimeSeriesDataset(data, seq_len=14, horizon=4, target_idx=0)
        expected = 100 - 14 - 4 + 1
        assert len(ds) == expected

    def test_item_shapes(self):
        data = _random_data(100, 22)
        ds = TimeSeriesDataset(data, seq_len=14, horizon=4, target_idx=0)
        x, y = ds[0]
        assert x.shape == (14, 22)
        assert y.shape == ()  # scalar

    def test_target_is_horizon_steps_ahead(self):
        n, seq_len, horizon = 50, 10, 4
        data = np.arange(n * 1, dtype=np.float32).reshape(n, 1)
        ds = TimeSeriesDataset(data, seq_len=seq_len, horizon=horizon, target_idx=0)
        x, y = ds[0]
        # window covers [0..9], target at [0+10+4-1] = [13]
        expected_y = float(data[seq_len + horizon - 1, 0])
        assert abs(float(y) - expected_y) < 1e-6

    def test_tensors_are_float32(self):
        data = _random_data()
        ds = TimeSeriesDataset(data, 14, 4, 0)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_different_seq_lengths(self):
        data = _random_data(200)
        for seq_len in (14, 28, 56):
            ds = TimeSeriesDataset(data, seq_len=seq_len, horizon=4, target_idx=0)
            assert len(ds) == 200 - seq_len - 4 + 1
            x, y = ds[0]
            assert x.shape == (seq_len, 22)


# ------------------------------------------------------------------ LSTMModel

class TestLSTMModel:
    def test_output_shape_batch(self):
        model = LSTMModel(input_size=22, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(8, 14, 22)
        out = model(x)
        assert out.shape == (8,)

    def test_output_shape_single(self):
        model = LSTMModel(input_size=22, hidden_size=64)
        x = torch.randn(1, 14, 22)
        out = model(x)
        assert out.shape == (1,)

    def test_different_hyperparams(self):
        for hidden, layers in [(32, 1), (64, 2), (128, 3)]:
            model = LSTMModel(input_size=22, hidden_size=hidden, num_layers=layers, dropout=0.1)
            x = torch.randn(4, 14, 22)
            out = model(x)
            assert out.shape == (4,), f"Failed for hidden={hidden}, layers={layers}"

    def test_forward_is_deterministic(self):
        model = LSTMModel(input_size=22, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()
        x = torch.randn(2, 14, 22)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flows(self):
        model = LSTMModel(input_size=22, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(4, 14, 22)
        y = torch.randn(4)
        loss = torch.nn.MSELoss()(model(x), y)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(g > 0 for g in grad_norms)

    def test_dropout_disabled_when_eval(self):
        model = LSTMModel(input_size=22, hidden_size=64, num_layers=2, dropout=0.5)
        model.eval()
        x = torch.randn(2, 14, 22)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        torch.testing.assert_close(o1, o2)

    def test_state_dict_save_load(self, tmp_path):
        model = LSTMModel(input_size=22, hidden_size=32, num_layers=1, dropout=0.0)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = LSTMModel(input_size=22, hidden_size=32, num_layers=1, dropout=0.0)
        model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model2.eval()
        model.eval()

        x = torch.randn(1, 14, 22)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model2(x))
