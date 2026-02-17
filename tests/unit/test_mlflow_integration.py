"""Unit tests for MLflow experiment management integration.

Tests for MLflowConfig, enhanced ExperimentTracker features,
compare_experiments CLI, and orchestrator MLflow integration.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.common.config import MLflowConfig
from src.model_training.experiment_tracker import ExperimentTracker

# ---------------------------------------------------------------------------
# MLflowConfig tests
# ---------------------------------------------------------------------------


class TestMLflowConfig:
    """Tests for MLflowConfig Pydantic model."""

    def test_defaults(self) -> None:
        config = MLflowConfig()
        assert config.tracking_uri == "file:./mlruns"
        assert config.experiment_name == "horse-racing-prediction"
        assert config.enabled is True

    def test_custom_values(self) -> None:
        config = MLflowConfig(
            tracking_uri="gs://bucket/mlruns",
            experiment_name="test-exp",
            enabled=False,
        )
        assert config.tracking_uri == "gs://bucket/mlruns"
        assert config.experiment_name == "test-exp"
        assert config.enabled is False

    @patch("src.common.config._load_yaml_config")
    def test_settings_includes_mlflow(self, mock_yaml: MagicMock) -> None:
        mock_yaml.return_value = {
            "mlflow": {
                "experiment_name": "from-yaml",
                "enabled": False,
            }
        }
        from src.common.config import AppSettings, Environment

        settings = AppSettings(environment=Environment.DEV, **mock_yaml.return_value)
        assert settings.mlflow.experiment_name == "from-yaml"
        assert settings.mlflow.enabled is False


# ---------------------------------------------------------------------------
# Enhanced ExperimentTracker tests
# ---------------------------------------------------------------------------


class TestExperimentTrackerEnhanced:
    """Tests for new ExperimentTracker features."""

    def test_from_config(self) -> None:
        config = MLflowConfig(
            tracking_uri="file:./test_mlruns",
            experiment_name="config-test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config.tracking_uri = f"file:{tmpdir}/mlruns"
            tracker = ExperimentTracker.from_config(config)
            assert tracker._experiment_name == "config-test"
            assert tracker._tracking_uri == config.tracking_uri

    def test_generate_run_name_format(self) -> None:
        name = ExperimentTracker.generate_run_name("lgbm_classifier")
        assert name.startswith("lgbm_classifier_")
        # Format: lgbm_classifier_YYYYMMDD_HHMMSS
        parts = name.split("_", 2)
        assert len(parts) == 3
        assert len(parts[2]) == 15  # YYYYMMDD_HHMMSS

    def test_set_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="tag-test",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            tracker.start_run(run_name="tag-run")
            # Should not raise
            tracker.set_tags({"env": "dev", "model": "lgbm"})
            tracker.end_run()

    def test_set_tags_without_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="tag-noop",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            # Should not raise when no active run
            tracker.set_tags({"key": "value"})

    def test_log_dict_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="dict-art",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            tracker.start_run(run_name="dict-run")
            tracker.log_dict_artifact(
                {"feature_a": 0.5, "feature_b": 0.3},
                "test_artifact.json",
            )
            tracker.end_run()

    def test_log_dict_artifact_without_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="dict-noop",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            # Should not raise
            tracker.log_dict_artifact({"key": "value"}, "test.json")

    def test_log_figure(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="fig-test",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            tracker.start_run(run_name="fig-run")
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            tracker.log_figure(fig, "test_plot.png")
            plt.close(fig)
            tracker.end_run()


# ---------------------------------------------------------------------------
# Compare experiments tests
# ---------------------------------------------------------------------------


class TestCompareExperiments:
    """Tests for compare_experiments CLI."""

    def test_fetch_runs_no_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(f"file:{tmpdir}/mlruns")
            from src.model_training.compare_experiments import fetch_runs

            result = fetch_runs("nonexistent-experiment")
            assert result.empty

    def test_format_comparison_table_empty(self) -> None:
        from src.model_training.compare_experiments import format_comparison_table

        result = format_comparison_table(pd.DataFrame())
        assert result.empty

    def test_format_comparison_table_with_data(self) -> None:
        from src.model_training.compare_experiments import format_comparison_table

        runs = pd.DataFrame(
            {
                "run_name": ["run1", "run2"],
                "start_time": ["2026-01-01", "2026-01-02"],
                "params.model_type": ["lgbm", "lgbm"],
                "metrics.val_accuracy": [0.85, 0.90],
            }
        )
        table = format_comparison_table(runs, sort_metric="val_accuracy")
        assert len(table) == 2
        assert "val_accuracy" in table.columns

    @patch("src.model_training.compare_experiments.get_settings")
    @patch("src.model_training.compare_experiments.mlflow")
    def test_main_no_runs(
        self, mock_mlflow: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.return_value = MagicMock(
            mlflow=MLflowConfig(tracking_uri="file:./test")
        )
        mock_mlflow.get_experiment_by_name.return_value = None

        from src.model_training.compare_experiments import main

        main(["--last", "3"])  # Should not raise


# ---------------------------------------------------------------------------
# Orchestrator MLflow integration tests
# ---------------------------------------------------------------------------


class TestOrchestratorMLflowIntegration:
    """Tests for MLflow integration in PipelineOrchestrator."""

    @patch("src.pipeline.orchestrator.get_settings")
    def test_tracker_created_when_enabled(self, mock_settings: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = MagicMock(
                mlflow=MLflowConfig(
                    tracking_uri=f"file:{tmpdir}/mlruns",
                    enabled=True,
                ),
                environment=MagicMock(value="dev"),
                model=MagicMock(feature_version="v1"),
                gcs=MagicMock(bucket_processed="", bucket_models=""),
                bigquery=MagicMock(dataset="test"),
                gcp=MagicMock(project_id="test"),
            )
            from src.pipeline.orchestrator import PipelineOrchestrator

            orch = PipelineOrchestrator(data_source="csv")
            assert orch._tracker is not None

    @patch("src.pipeline.orchestrator.get_settings")
    def test_tracker_none_when_disabled(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value = MagicMock(
            mlflow=MLflowConfig(enabled=False),
            environment=MagicMock(value="dev"),
            model=MagicMock(feature_version="v1"),
            gcs=MagicMock(bucket_processed="", bucket_models=""),
            bigquery=MagicMock(dataset="test"),
            gcp=MagicMock(project_id="test"),
        )
        from src.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(data_source="csv")
        assert orch._tracker is None
