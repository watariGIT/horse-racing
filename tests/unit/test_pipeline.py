"""Unit tests for the pipeline orchestrator and data preparer."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest

from src.pipeline.data_preparer import DataPreparer
from src.pipeline.orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_race_df() -> pl.DataFrame:
    """Synthetic race result DataFrame spanning multiple dates and horses."""
    records: list[dict[str, Any]] = []
    base_date = date(2020, 1, 1)
    horses = ["H001", "H002", "H003", "H004", "H005", "H006"]

    for day_offset in range(0, 210, 7):  # ~30 race days
        race_date = base_date + timedelta(days=day_offset)
        race_id = f"R{day_offset:03d}"
        for idx, horse_id in enumerate(horses):
            records.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "horse_id": horse_id,
                    "jockey_id": f"J{idx:03d}",
                    "finish_position": ((day_offset // 7 + idx) % 6) + 1,
                    "distance": 2000,
                    "track_condition": "良",
                    "weather": "晴",
                    "track_type": "芝",
                    "num_entries": 12,
                    "grade": "" if idx > 0 else "G1",
                    "course": "東京",
                    "weight": 480.0 + idx * 5,
                    "age": 3 + idx,
                }
            )
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# DataPreparer tests
# ---------------------------------------------------------------------------


class TestDataPreparer:
    """Tests for DataPreparer history aggregation."""

    def test_no_future_data_leakage(self, raw_race_df: pl.DataFrame) -> None:
        """History columns must only use data from before each row's date."""
        preparer = DataPreparer(n_past_races=5)
        result = preparer.prepare_for_training(raw_race_df)

        # First race of each horse should have null/0 history
        first_races = result.sort("race_date").group_by("horse_id").head(1)
        for row in first_races.iter_rows(named=True):
            assert (
                row["num_past_races"] == 0
            ), f"Horse {row['horse_id']} should have 0 past races on first appearance"

    def test_history_columns_added(self, raw_race_df: pl.DataFrame) -> None:
        """All expected history columns should be present."""
        preparer = DataPreparer(n_past_races=5)
        result = preparer.prepare_for_training(raw_race_df)

        expected = [
            "avg_finish",
            "win_rate",
            "top3_rate",
            "days_since_last_race",
            "num_past_races",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_target_columns_created(self, raw_race_df: pl.DataFrame) -> None:
        """is_win and actual_position targets should be added."""
        preparer = DataPreparer(n_past_races=5)
        result = preparer.prepare_for_training(raw_race_df)

        assert "is_win" in result.columns
        assert "actual_position" in result.columns

        # is_win should be 1 only when finish_position == 1
        wins = result.filter(pl.col("is_win") == 1)
        assert (wins["finish_position"] == 1).all()

    def test_null_finish_positions_filtered(self) -> None:
        """Rows with null finish_position should be excluded."""
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R2"],
                "race_date": [date(2020, 1, 1)] * 2 + [date(2020, 1, 8)],
                "horse_id": ["H1", "H2", "H1"],
                "finish_position": [1, None, 2],
                "distance": [2000, 2000, 2000],
                "track_condition": ["良", "良", "良"],
                "weather": ["晴", "晴", "晴"],
                "track_type": ["芝", "芝", "芝"],
                "num_entries": [10, 10, 10],
                "grade": ["", "", ""],
                "weight": [480.0, 480.0, 480.0],
            }
        )
        preparer = DataPreparer()
        result = preparer.prepare_for_training(df)
        assert len(result) == 2  # H2's null row excluded

    def test_rolling_stats_accumulate(self, raw_race_df: pl.DataFrame) -> None:
        """num_past_races should increase with each race for a horse."""
        preparer = DataPreparer(n_past_races=5)
        result = preparer.prepare_for_training(raw_race_df)

        for horse_id in ["H001", "H002", "H003"]:
            horse_data = result.filter(pl.col("horse_id") == horse_id).sort("race_date")
            past_races = horse_data["num_past_races"].to_list()
            # Should be monotonically increasing
            for i in range(1, len(past_races)):
                assert (
                    past_races[i] >= past_races[i - 1]
                ), f"num_past_races not increasing for {horse_id}"


# ---------------------------------------------------------------------------
# PipelineOrchestrator tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator stage methods."""

    def test_import_data_csv(self, raw_race_df: pl.DataFrame) -> None:
        """import_data should load from CSV when source is csv."""
        orchestrator = PipelineOrchestrator(data_source="csv")

        with patch.object(orchestrator, "_load_from_csv", return_value=raw_race_df):
            result = orchestrator.import_data()
            assert len(result) == len(raw_race_df)

    def test_import_data_bigquery(self, raw_race_df: pl.DataFrame) -> None:
        """import_data should load from BigQuery when source is bigquery."""
        orchestrator = PipelineOrchestrator(data_source="bigquery")

        with patch.object(
            orchestrator, "_load_from_bigquery", return_value=raw_race_df
        ):
            result = orchestrator.import_data()
            assert len(result) == len(raw_race_df)

    def test_prepare_features_requires_import(self) -> None:
        """prepare_features should raise if import_data not called."""
        orchestrator = PipelineOrchestrator()
        with pytest.raises(RuntimeError, match="import_data"):
            orchestrator.prepare_features()

    def test_train_model_requires_features(self) -> None:
        """train_model should raise if prepare_features not called."""
        orchestrator = PipelineOrchestrator()
        with pytest.raises(RuntimeError, match="prepare_features"):
            orchestrator.train_model()

    def test_evaluate_requires_train(self) -> None:
        """evaluate_model should raise if train_model not called."""
        orchestrator = PipelineOrchestrator()
        with pytest.raises(RuntimeError, match="train_model"):
            orchestrator.evaluate_model()

    def test_prepare_features_adds_feat_columns(
        self, raw_race_df: pl.DataFrame
    ) -> None:
        """prepare_features should add feat_ prefixed columns."""
        orchestrator = PipelineOrchestrator(data_source="csv")
        orchestrator._raw_df = raw_race_df

        result = orchestrator.prepare_features()
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) > 0, "No feature columns found"

    def test_full_pipeline_integration(self, raw_race_df: pl.DataFrame) -> None:
        """Full pipeline should complete without errors (mocked GCP)."""
        orchestrator = PipelineOrchestrator(data_source="csv")

        with (
            patch.object(orchestrator, "_load_from_csv", return_value=raw_race_df),
            patch.object(orchestrator, "_save_to_gcp", return_value={}),
        ):
            result = orchestrator.run_full()
            assert "overall_metrics" in result
            assert "n_periods" in result


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestPipelineCLI:
    """Tests for __main__.py argument parsing."""

    def test_parse_default_args(self) -> None:
        """Default args should set stage=full."""
        from src.pipeline.__main__ import _parse_args

        args = _parse_args([])
        assert args.stage == "full"
        assert args.train_window == 365
        assert args.test_window == 30
        assert args.model_name == "win_classifier"

    def test_parse_custom_args(self) -> None:
        """Custom args should be parsed correctly."""
        from src.pipeline.__main__ import _parse_args

        args = _parse_args(
            [
                "--stage",
                "train",
                "--date-from",
                "2020-01-01",
                "--date-to",
                "2021-06-30",
                "--train-window",
                "180",
                "--test-window",
                "14",
                "--model-name",
                "my_model",
            ]
        )
        assert args.stage == "train"
        assert args.date_from == "2020-01-01"
        assert args.date_to == "2021-06-30"
        assert args.train_window == 180
        assert args.test_window == 14
        assert args.model_name == "my_model"
