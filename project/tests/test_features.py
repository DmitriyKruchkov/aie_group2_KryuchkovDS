"""Tests for feature engineering (src/features.py)."""
import math
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from src.features import (
    FEATURE_COLUMNS,
    ON_DEMAND_PRICE,
    build_features,
    compute_new_row_features,
)


def _make_raw_df(n: int = 60) -> pl.DataFrame:
    """Synthetic 6-h spaced datetime + cost series."""
    start = datetime(2025, 1, 1)
    datetimes = [
        datetime(2025, 1, 1, (i * 6) % 24, 0) for i in range(n)
    ]
    costs = [0.9 + 0.01 * (i % 10) for i in range(n)]
    return pl.DataFrame({"datetime": datetimes, "cost": costs}).with_columns(
        pl.col("datetime").cast(pl.Datetime)
    )


# ------------------------------------------------------------------ build_features

class TestBuildFeatures:
    def test_output_columns(self):
        df = _make_raw_df(60)
        result = build_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nulls_after_build(self):
        df = _make_raw_df(60)
        result = build_features(df)
        assert result.null_count().sum_horizontal()[0] == 0

    def test_drops_first_rows(self):
        df = _make_raw_df(60)
        result = build_features(df)
        # lag_1w = shift(28) → first 28 rows become null → dropped
        assert len(result) < len(df)
        assert len(result) <= len(df) - 28

    def test_lag_6h_is_previous_cost(self):
        df = _make_raw_df(60)
        result = build_features(df)
        # After drop_nulls, first valid row is row 28 (0-indexed)
        # cost_lag_6h should equal cost of the preceding row in raw df
        first_cost = result["cost"][0]
        first_lag = result["cost_lag_6h"][0]
        # lag should be the cost 1 step earlier in raw df
        raw_costs = df["cost"].to_list()
        row_in_raw = df["cost"].to_list().index(first_cost)
        assert abs(first_lag - raw_costs[row_in_raw - 1]) < 1e-5

    def test_discount_ratio(self):
        df = _make_raw_df(60)
        result = build_features(df)
        ratio = result["discount_ratio"].to_numpy()
        cost = result["cost"].to_numpy()
        np.testing.assert_allclose(ratio, cost / ON_DEMAND_PRICE, rtol=1e-4)

    def test_hour_sin_cos_bounds(self):
        df = _make_raw_df(60)
        result = build_features(df)
        for col in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"):
            vals = result[col].to_numpy()
            assert vals.min() >= -1.0 - 1e-6
            assert vals.max() <= 1.0 + 1e-6

    def test_is_weekend_binary(self):
        df = _make_raw_df(60)
        result = build_features(df)
        assert set(result["is_weekend"].to_list()).issubset({0, 1})

    def test_is_holiday_binary(self):
        df = _make_raw_df(60)
        result = build_features(df)
        assert set(result["is_holiday"].to_list()).issubset({0, 1})


# ------------------------------------------------------------------ compute_new_row_features

class TestComputeNewRowFeatures:
    def _history(self, n: int = 35) -> pl.DataFrame:
        return pl.DataFrame({"cost": [0.95 + 0.001 * i for i in range(n)]})

    def test_returns_correct_length(self):
        row = compute_new_row_features(self._history(), current_price=1.0)
        assert len(row) == len(FEATURE_COLUMNS)

    def test_returns_float32(self):
        row = compute_new_row_features(self._history(), current_price=1.0)
        assert row.dtype == np.float32

    def test_cost_is_current_price(self):
        row = compute_new_row_features(self._history(), current_price=1.05)
        cost_idx = FEATURE_COLUMNS.index("cost")
        assert abs(row[cost_idx] - 1.05) < 1e-5

    def test_discount_ratio(self):
        price = 1.0
        row = compute_new_row_features(self._history(), current_price=price)
        idx = FEATURE_COLUMNS.index("discount_ratio")
        assert abs(row[idx] - price / ON_DEMAND_PRICE) < 1e-5

    def test_lag_6h_is_last_history_cost(self):
        h = self._history(35)
        price = 1.0
        row = compute_new_row_features(h, current_price=price)
        idx = FEATURE_COLUMNS.index("cost_lag_6h")
        assert abs(row[idx] - h["cost"][-1]) < 1e-5

    def test_hour_sin_cos_consistent(self):
        dt = datetime(2025, 6, 15, 12, 0)
        row = compute_new_row_features(self._history(), current_price=1.0, dt=dt)
        sin_idx = FEATURE_COLUMNS.index("hour_sin")
        cos_idx = FEATURE_COLUMNS.index("hour_cos")
        expected_sin = math.sin(2 * math.pi * 12 / 24)
        expected_cos = math.cos(2 * math.pi * 12 / 24)
        assert abs(row[sin_idx] - expected_sin) < 1e-5
        assert abs(row[cos_idx] - expected_cos) < 1e-5

    def test_weekend_detection(self):
        # 2025-01-04 is Saturday
        sat = datetime(2025, 1, 4, 6, 0)
        row = compute_new_row_features(self._history(), current_price=1.0, dt=sat)
        idx = FEATURE_COLUMNS.index("is_weekend")
        assert row[idx] == 1.0

        # 2025-01-06 is Monday
        mon = datetime(2025, 1, 6, 6, 0)
        row2 = compute_new_row_features(self._history(), current_price=1.0, dt=mon)
        assert row2[idx] == 0.0

    def test_short_history_no_crash(self):
        short = pl.DataFrame({"cost": [0.9, 0.91, 0.92]})
        row = compute_new_row_features(short, current_price=0.93)
        assert len(row) == len(FEATURE_COLUMNS)

    def test_feature_columns_order(self):
        row = compute_new_row_features(self._history(), current_price=1.0)
        for i, col in enumerate(FEATURE_COLUMNS):
            assert np.isfinite(row[i]) or col in ("cost_lag_1w",), (
                f"Unexpected non-finite value at col={col} idx={i}"
            )
