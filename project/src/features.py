"""Feature engineering for AWS Spot price time series."""
from __future__ import annotations

import math
from datetime import datetime

import holidays
import numpy as np
import polars as pl

ON_DEMAND_PRICE = 1.212
_US_HOLIDAYS = holidays.US(years=range(2020, 2031))

FEATURE_COLUMNS: list[str] = [
    "cost",
    "cost_lag_6h",
    "cost_lag_12h",
    "cost_lag_18h",
    "cost_lag_1d",
    "cost_lag_2d",
    "cost_lag_1w",
    "hour",
    "weekday",
    "month",
    "day",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "cost_diff",
    "cost_pct_change",
    "cost_vs_mean",
    "discount_ratio",
]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all feature columns to a DataFrame with 'datetime' and 'cost' columns.

    Drops rows with nulls (first 28 rows needed for lag_1w and rolling stats).
    """
    df = df.with_columns([
        pl.col("cost").shift(1).alias("cost_lag_6h"),
        pl.col("cost").shift(2).alias("cost_lag_12h"),
        pl.col("cost").shift(3).alias("cost_lag_18h"),
        pl.col("cost").shift(4).alias("cost_lag_1d"),
        pl.col("cost").shift(8).alias("cost_lag_2d"),
        pl.col("cost").shift(28).alias("cost_lag_1w"),
    ])

    df = df.with_columns([
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.day().alias("day"),
        pl.col("datetime").dt.week().alias("week_of_year"),
        (pl.col("datetime").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        pl.col("datetime").map_elements(
            lambda x: int(x.date() in _US_HOLIDAYS), return_dtype=pl.Int8
        ).alias("is_holiday"),
    ])

    df = df.with_columns([
        (2 * math.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
        (2 * math.pi * pl.col("weekday") / 7).sin().alias("weekday_sin"),
        (2 * math.pi * pl.col("weekday") / 7).cos().alias("weekday_cos"),
    ])

    df = df.with_columns([
        (pl.col("cost") - pl.col("cost").shift(1)).alias("cost_diff"),
        (
            (pl.col("cost") - pl.col("cost").shift(1)) / pl.col("cost").shift(1)
        ).alias("cost_pct_change"),
        (pl.col("cost") / pl.col("cost").rolling_mean(28)).alias("cost_vs_mean"),
        (pl.col("cost") / ON_DEMAND_PRICE).alias("discount_ratio"),
    ])

    return df.drop_nulls()


def compute_new_row_features(
    history: pl.DataFrame,
    current_price: float,
    dt: datetime | None = None,
) -> np.ndarray:
    """Compute feature vector for a new row given historical data.

    Args:
        history: DataFrame with at least 'cost' column; last rows provide lags.
                 Needs ≥28 rows for all features to be non-null.
        current_price: Current spot price (USD/hr).
        dt: Timestamp for the new row; defaults to now.

    Returns:
        1-D float32 array of length len(FEATURE_COLUMNS), in canonical order.
    """
    if dt is None:
        dt = datetime.now()

    costs = history["cost"].to_list()

    def _lag(n: int) -> float:
        return float(costs[-n]) if len(costs) >= n else float("nan")

    hour = dt.hour
    weekday = dt.weekday()  # 0=Mon, 6=Sun (same as polars)
    rolling_window = costs[-28:] if len(costs) >= 28 else costs
    rolling_mean = float(np.mean(rolling_window)) if rolling_window else current_price
    prev_cost = float(costs[-1]) if costs else current_price

    values = {
        "cost": current_price,
        "cost_lag_6h": _lag(1),
        "cost_lag_12h": _lag(2),
        "cost_lag_18h": _lag(3),
        "cost_lag_1d": _lag(4),
        "cost_lag_2d": _lag(8),
        "cost_lag_1w": _lag(28),
        "hour": float(hour),
        "weekday": float(weekday),
        "month": float(dt.month),
        "day": float(dt.day),
        "week_of_year": float(dt.isocalendar()[1]),
        "is_weekend": float(weekday >= 5),
        "is_holiday": float(dt.date() in _US_HOLIDAYS),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "weekday_cos": math.cos(2 * math.pi * weekday / 7),
        "cost_diff": current_price - prev_cost,
        "cost_pct_change": (current_price - prev_cost) / prev_cost if prev_cost else 0.0,
        "cost_vs_mean": current_price / rolling_mean if rolling_mean else 1.0,
        "discount_ratio": current_price / ON_DEMAND_PRICE,
    }

    return np.array([values[col] for col in FEATURE_COLUMNS], dtype=np.float32)
