"""End-to-end integration test for the full prediction pipeline.

Exercises: data collection (mocked) -> feature engineering -> training
(mock model) -> prediction -> evaluation.  No real API or GCP calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from src.data_collector.collectors.race_collector import RaceCollector
from src.data_collector.jra_client import JRAClient
from src.data_collector.validators.data_validator import DataValidator
from src.evaluator.backtest_engine import BacktestEngine, BacktestResult
from src.evaluator.metrics import RacingMetrics
from src.evaluator.monitor import PerformanceMonitor
from src.evaluator.reporter import Reporter
from src.feature_engineering.extractors.horse_features import HorseFeatureExtractor
from src.feature_engineering.extractors.race_features import RaceFeatureExtractor
from src.feature_engineering.pipeline import FeaturePipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_race_data() -> list[dict[str, Any]]:
    """Create realistic mock race data for several dates."""
    races = []
    base_horses = [
        {"horse_id": "H001", "horse_name": "Thunder Bolt", "jockey_id": "J001"},
        {"horse_id": "H002", "horse_name": "Wind Runner", "jockey_id": "J002"},
        {"horse_id": "H003", "horse_name": "Star Light", "jockey_id": "J003"},
        {"horse_id": "H004", "horse_name": "Dark Shadow", "jockey_id": "J004"},
        {"horse_id": "H005", "horse_name": "Gold Rush", "jockey_id": "J005"},
    ]

    dates = [
        "2024-01-01",
        "2024-01-07",
        "2024-01-14",
        "2024-01-21",
        "2024-01-28",
        "2024-02-04",
        "2024-02-11",
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]

    for i, date_str in enumerate(dates):
        race_id = f"R{i:04d}"
        entries = []
        for j, horse in enumerate(base_horses):
            entries.append(
                {
                    **horse,
                    "odds": float(3 + j * 2 + np.random.rand()),
                    "finish_position": j + 1,
                    "weight": 470 + j * 5,
                }
            )
        races.append(
            {
                "race_id": race_id,
                "race_date": date_str,
                "race_name": f"Race {i + 1}",
                "race_number": 1,
                "course": "Tokyo",
                "distance": 1600 + (i % 3) * 200,
                "track_type": "\u829d",
                "track_condition": "\u826f",
                "weather": "\u6674",
                "grade": "G1" if i % 3 == 0 else "",
                "num_entries": len(entries),
                "entries": entries,
            }
        )
    return races


class SimpleMockModel:
    """A deterministic mock model for pipeline testing."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: Any, y: Any, **kwargs: Any) -> SimpleMockModel:
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.zeros(n, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        # Give first row highest probability, linearly decreasing
        probs = np.linspace(0.8, 0.1, n)
        return np.column_stack([1 - probs, probs])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """Full pipeline integration test."""

    def test_data_to_evaluation(self):
        """Test the entire flow from raw data to evaluation metrics."""
        # 1. Simulate data collection (mock API, create DataFrame directly)
        raw_races = _build_mock_race_data()

        # Build entries-level DataFrame (as a collector + normalizer would)
        records: list[dict[str, Any]] = []
        for race in raw_races:
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
                        "jockey_id": entry["jockey_id"],
                        "odds": entry["odds"],
                        "finish_position": entry["finish_position"],
                        "actual_position": entry["finish_position"],
                        "weight": float(entry["weight"]),
                        # Pre-aggregated horse stats (simulate)
                        "avg_finish": float(entry["finish_position"]),
                        "win_rate": 1.0 / (entry["finish_position"] + 0.5),
                        "top3_rate": 1.0 if entry["finish_position"] <= 3 else 0.0,
                        "days_since_last_race": 14,
                        "horse_age": 4,
                        "num_past_races": 10,
                    }
                )
        entries_df = pl.DataFrame(records)
        assert not entries_df.is_empty()

        # 2. Data validation
        validator = DataValidator(
            required_columns=["race_id", "horse_id", "finish_position"]
        )
        errors = validator.validate(entries_df)
        assert len(errors) == 0

        # 3. Feature engineering
        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor(), HorseFeatureExtractor()],
        )
        features_df = pipeline.fit_transform(entries_df)

        # Verify features were created
        feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
        assert len(feature_cols) > 0

        # Add target column
        features_df = features_df.with_columns(
            (pl.col("finish_position") == 1).cast(pl.Int64).alias("is_win")
        )

        # 4. Model training and prediction (mock)
        model = SimpleMockModel()
        X_train = features_df.select(feature_cols).to_pandas()
        y_train = features_df["is_win"].to_pandas()
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_train)
        win_proba = proba[:, 1]

        # 5. Build prediction results
        pred_df = features_df.select(
            ["race_id", "horse_id", "actual_position", "is_win"]
        ).with_columns(
            pl.Series("predicted_prob", win_proba),
            pl.Series("predicted_win", (win_proba >= 0.3).astype(int)),
        )

        pred_df = pred_df.with_columns(
            pl.col("predicted_prob")
            .rank(method="ordinal", descending=True)
            .over("race_id")
            .alias("predicted_rank")
        )

        # 6. Evaluate
        metrics = RacingMetrics.compute_all(pred_df, has_betting=False)
        assert "win_accuracy" in metrics.values
        assert "auc_roc" in metrics.values
        assert "ndcg" in metrics.values
        assert 0.0 <= metrics["win_accuracy"] <= 1.0
        assert 0.0 <= metrics["ndcg"] <= 1.0

    def test_backtest_pipeline(self):
        """Test backtest engine with feature pipeline output."""
        raw_races = _build_mock_race_data()

        records: list[dict[str, Any]] = []
        for race in raw_races:
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
                        "finish_position": entry["finish_position"],
                        "actual_position": entry["finish_position"],
                        "weight": float(entry["weight"]),
                        "avg_finish": float(entry["finish_position"]),
                        "win_rate": 1.0 / (entry["finish_position"] + 0.5),
                        "top3_rate": 1.0 if entry["finish_position"] <= 3 else 0.0,
                        "days_since_last_race": 14,
                        "horse_age": 4,
                        "num_past_races": 10,
                    }
                )

        entries_df = pl.DataFrame(records)

        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor(), HorseFeatureExtractor()],
        )
        features_df = pipeline.fit_transform(entries_df)
        features_df = features_df.with_columns(
            (pl.col("finish_position") == 1).cast(pl.Int64).alias("is_win")
        )

        model = SimpleMockModel()
        engine = BacktestEngine(
            train_window_days=21,
            test_window_days=7,
            step_days=7,
        )
        result = engine.run(features_df, model)

        assert isinstance(result, BacktestResult)
        # With 10 weeks of data and 21-day training window, we should get periods
        assert len(result.periods) >= 1

        # Overall metrics should be populated
        assert "win_accuracy" in result.overall_metrics.values

    def test_evaluation_to_report(self):
        """Test that evaluation results can be rendered as a report."""
        raw_races = _build_mock_race_data()

        records: list[dict[str, Any]] = []
        for race in raw_races:
            for entry in race["entries"]:
                records.append(
                    {
                        "race_id": race["race_id"],
                        "race_date": race["race_date"],
                        "horse_id": entry["horse_id"],
                        "actual_position": entry["finish_position"],
                        "is_win": 1 if entry["finish_position"] == 1 else 0,
                        "predicted_prob": 1.0 / (entry["finish_position"] + 0.5),
                        "predicted_win": 1 if entry["finish_position"] <= 2 else 0,
                    }
                )

        df = pl.DataFrame(records)
        df = df.with_columns(
            pl.col("predicted_prob")
            .rank(method="ordinal", descending=True)
            .over("race_id")
            .alias("predicted_rank")
        )

        metrics = RacingMetrics.compute_all(df, has_betting=False)

        reporter = Reporter(model_name="integration_test")
        report = reporter.generate_metrics_report(metrics)

        assert "# Horse Racing Model Evaluation Report" in report
        assert "integration_test" in report
        assert "Win Accuracy" in report

    def test_monitoring_integration(self):
        """Test that monitoring works with real evaluation output."""
        monitor = PerformanceMonitor(thresholds={"win_accuracy": 0.05, "auc_roc": 0.50})

        raw_races = _build_mock_race_data()

        for i, race in enumerate(raw_races):
            # Create prediction for this race
            records = []
            for entry in race["entries"]:
                records.append(
                    {
                        "race_id": race["race_id"],
                        "horse_id": entry["horse_id"],
                        "actual_position": entry["finish_position"],
                        "is_win": 1 if entry["finish_position"] == 1 else 0,
                        "predicted_prob": 1.0 / (entry["finish_position"] + 0.5),
                        "predicted_win": 1 if entry["finish_position"] == 1 else 0,
                    }
                )

            df = pl.DataFrame(records)
            df = df.with_columns(
                pl.col("predicted_prob")
                .rank(method="ordinal", descending=True)
                .over("race_id")
                .alias("predicted_rank")
            )

            metrics = RacingMetrics.compute_all(df, has_betting=False)
            monitor.record(metrics, label=race["race_date"])

        report = monitor.check()
        assert isinstance(report.is_healthy, bool)
        assert len(monitor.history) == len(raw_races)

    def test_race_collector_to_features(self):
        """Test that RaceCollector output can be fed to FeaturePipeline."""
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_races.return_value = [
            {"race_id": "R001"},
            {"race_id": "R002"},
        ]
        mock_client.get_race_detail.side_effect = [
            {
                "race_id": "R001",
                "race_date": "2024-01-01",
                "race_name": "Test Race",
                "race_number": 1,
                "course": "Tokyo",
                "distance": 1600,
                "track_type": "\u829d",
                "track_condition": "\u826f",
                "weather": "\u6674",
                "grade": "G1",
                "entries": [{"horse_id": "H001"}, {"horse_id": "H002"}],
            },
            {
                "race_id": "R002",
                "race_date": "2024-01-01",
                "race_name": "Test Race 2",
                "race_number": 2,
                "course": "Tokyo",
                "distance": 2000,
                "track_type": "\u30c0\u30fc\u30c8",
                "track_condition": "\u91cd",
                "weather": "\u96e8",
                "grade": "",
                "entries": [],
            },
        ]

        from datetime import date

        collector = RaceCollector(client=mock_client)
        race_df = collector.collect_by_date(date(2024, 1, 1))

        assert not race_df.is_empty()

        pipeline = FeaturePipeline(extractors=[RaceFeatureExtractor()])
        features_df = pipeline.fit_transform(race_df)

        assert "feat_distance" in features_df.columns
        assert "feat_track_condition" in features_df.columns
        assert features_df["feat_distance"][0] == 1600
