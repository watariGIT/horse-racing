"""Shared pytest fixtures for the horse racing test suite.

Provides sample DataFrames, mock models, and helper factories
used across unit and integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Load the full sample_data.json fixture."""
    with open(FIXTURES_DIR / "sample_data.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Race DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_races_df(sample_data: dict[str, Any]) -> pl.DataFrame:
    """DataFrame of race metadata (one row per race)."""
    records: list[dict[str, Any]] = []
    for race in sample_data["races"]:
        records.append(
            {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "race_name": race["race_name"],
                "race_number": race["race_number"],
                "course": race["course"],
                "distance": race["distance"],
                "track_type": race["track_type"],
                "track_condition": race["track_condition"],
                "weather": race["weather"],
                "grade": race["grade"],
                "num_entries": race["num_entries"],
            }
        )
    return pl.DataFrame(records)


@pytest.fixture
def sample_entries_df(sample_data: dict[str, Any]) -> pl.DataFrame:
    """DataFrame of race entries (one row per horse per race)."""
    records: list[dict[str, Any]] = []
    for race in sample_data["races"]:
        for entry in race["entries"]:
            records.append(
                {
                    "race_id": race["race_id"],
                    "race_date": race["race_date"],
                    "course": race["course"],
                    "distance": race["distance"],
                    "track_type": race["track_type"],
                    "track_condition": race["track_condition"],
                    "weather": race["weather"],
                    "grade": race["grade"],
                    "num_entries": race["num_entries"],
                    "horse_id": entry["horse_id"],
                    "horse_name": entry["horse_name"],
                    "jockey_id": entry["jockey_id"],
                    "odds": entry["odds"],
                    "finish_position": entry["finish_position"],
                    "actual_position": entry["finish_position"],
                    "weight": entry["weight"],
                }
            )
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Prediction DataFrames (for evaluator tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prediction_df() -> pl.DataFrame:
    """DataFrame mimicking model prediction output for 3 races."""
    np.random.seed(42)
    records: list[dict[str, Any]] = []
    for race_idx in range(3):
        race_id = f"R{race_idx:03d}"
        n_horses = 5
        for h in range(n_horses):
            finish = h + 1
            prob = max(0.01, 1.0 / (finish + 0.5) + np.random.normal(0, 0.05))
            records.append(
                {
                    "race_id": race_id,
                    "horse_id": f"H{race_idx * 10 + h:03d}",
                    "actual_position": finish,
                    "predicted_prob": prob,
                    "is_win": 1 if finish == 1 else 0,
                    "predicted_win": 1 if prob >= 0.3 else 0,
                    "odds": float(finish * 3 + 1),
                    "bet_amount": 100.0,
                    "payout": float(finish * 3 + 1) * 100.0 if finish == 1 else 0.0,
                }
            )
    df = pl.DataFrame(records)

    # Compute predicted_rank per race
    df = df.with_columns(
        pl.col("predicted_prob")
        .rank(method="ordinal", descending=True)
        .over("race_id")
        .alias("predicted_rank")
    )
    return df


# ---------------------------------------------------------------------------
# Feature DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features_df(sample_entries_df: pl.DataFrame) -> pl.DataFrame:
    """DataFrame with dummy features, target, and metadata for backtest."""
    np.random.seed(42)
    n = len(sample_entries_df)
    return sample_entries_df.with_columns(
        pl.Series("feat_distance_norm", np.random.randn(n)),
        pl.Series("feat_weight_norm", np.random.randn(n)),
        pl.Series("feat_odds_log", np.log1p(sample_entries_df["odds"].to_numpy())),
        pl.Series("feat_jockey_score", np.random.rand(n)),
        (pl.col("finish_position") == 1).cast(pl.Int64).alias("is_win"),
    )


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockModel:
    """Simple mock model that returns random predictions.

    Implements the PredictionModel protocol expected by BacktestEngine.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._fitted = False

    def fit(self, X: Any, y: Any, **kwargs: Any) -> MockModel:
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X)
        return (self._rng.rand(n) >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        probs = self._rng.rand(n)
        return np.column_stack([1 - probs, probs])


@pytest.fixture
def mock_model() -> MockModel:
    """Provide a MockModel instance."""
    return MockModel()
