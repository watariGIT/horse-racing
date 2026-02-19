"""Unit tests for the model training module.

Tests model implementations, trainer, experiment tracker,
tuner, and model registry with mock data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.model_training.experiment_tracker import ExperimentTracker
from src.model_training.models import MODEL_REGISTRY, create_model
from src.model_training.models.base import BaseModel
from src.model_training.models.ensemble import EnsembleModel
from src.model_training.models.lgbm_classifier import (
    LGBMClassifierModel,
    find_optimal_threshold,
)
from src.model_training.models.lgbm_ranker import LGBMRankerModel
from src.model_training.trainer import ModelTrainer

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_classification_data(
    n_samples: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic binary classification data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.rand(n_samples) > 0.85).astype(int), name="target")
    return X, y


def _make_ranking_data(
    n_races: int = 20,
    horses_per_race: int = 10,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic ranking data with race groups."""
    rng = np.random.RandomState(seed)
    n_samples = n_races * horses_per_race
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    race_ids = pd.Series(
        [f"R{i:03d}" for i in range(n_races) for _ in range(horses_per_race)],
        name="race_id",
    )
    # Relevance labels: higher is better
    positions = []
    for _ in range(n_races):
        positions.extend(range(horses_per_race, 0, -1))
    y = pd.Series(positions, name="relevance")
    return X, y, race_ids


# ---------------------------------------------------------------------------
# BaseModel tests
# ---------------------------------------------------------------------------


class TestBaseModel:
    """Tests for BaseModel ABC."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_save_and_load(self):
        X, y = _make_classification_data(n_samples=50)
        model = LGBMClassifierModel()
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)
            assert path.exists()

            loaded = BaseModel.load(path)
            assert isinstance(loaded, LGBMClassifierModel)
            assert loaded.is_fitted

            preds_original = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_load_invalid_type_raises(self):
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pkl"
            with open(path, "wb") as f:
                pickle.dump({"not": "a model"}, f)
            with pytest.raises(TypeError):
                BaseModel.load(path)

    def test_repr(self):
        model = LGBMClassifierModel()
        assert "not fitted" in repr(model)


# ---------------------------------------------------------------------------
# LGBMClassifierModel tests
# ---------------------------------------------------------------------------


class TestLGBMClassifier:
    """Tests for LGBMClassifierModel."""

    def test_fit_and_predict(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel()
        model.fit(X, y)

        assert model.is_fitted
        assert model.model_type == "lgbm_classifier"

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_feature_importances(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel()
        model.fit(X, y)

        importances = model.feature_importances
        assert len(importances) == X.shape[1]

    def test_predict_before_fit_raises(self):
        model = LGBMClassifierModel()
        X, _ = _make_classification_data(n_samples=5)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_custom_params(self):
        params = {"n_estimators": 10, "learning_rate": 0.1}
        model = LGBMClassifierModel(params=params)
        assert model.params["n_estimators"] == 10
        assert model.params["learning_rate"] == 0.1
        # Defaults should also be present
        assert "subsample" in model.params

    def test_calibrate(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 50})
        model.fit(X, y)

        # Calibrate on the same data (for test simplicity)
        model.calibrate(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_optimal_threshold(self):
        model = LGBMClassifierModel()
        assert model.optimal_threshold == 0.5

        model.optimal_threshold = 0.1
        assert model.optimal_threshold == 0.1

    def test_predict_uses_optimal_threshold(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 50})
        model.fit(X, y)

        # With default threshold 0.5
        preds_default = model.predict(X)

        # With lower threshold, should predict more positives
        model.optimal_threshold = 0.05
        preds_low = model.predict(X)

        assert preds_low.sum() >= preds_default.sum()

    def test_calibrate_before_fit_raises(self):
        model = LGBMClassifierModel()
        X, y = _make_classification_data(n_samples=10)
        with pytest.raises(RuntimeError):
            model.calibrate(X, y)


# ---------------------------------------------------------------------------
# LGBMRankerModel tests
# ---------------------------------------------------------------------------


class TestLGBMRanker:
    """Tests for LGBMRankerModel."""

    def test_fit_and_predict(self):
        X, y, race_ids = _make_ranking_data()
        groups = LGBMRankerModel.compute_group_sizes(race_ids)

        model = LGBMRankerModel()
        model.fit(X, y, group=groups)

        assert model.is_fitted
        assert model.model_type == "lgbm_ranker"

        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_fit_without_group_raises(self):
        X, y, _ = _make_ranking_data()
        model = LGBMRankerModel()
        with pytest.raises(ValueError, match="group"):
            model.fit(X, y)

    def test_compute_group_sizes(self):
        race_ids = pd.Series(["R1", "R1", "R1", "R2", "R2"])
        groups = LGBMRankerModel.compute_group_sizes(race_ids)
        assert list(groups) == [3, 2]

    def test_create_relevance_labels(self):
        positions = pd.Series([1, 2, 3, 10, 18])
        labels = LGBMRankerModel.create_relevance_labels(positions)
        assert labels.iloc[0] == 18  # 1st place gets highest
        assert labels.iloc[-1] == 1  # 18th place gets lowest

    def test_feature_importances(self):
        X, y, race_ids = _make_ranking_data()
        groups = LGBMRankerModel.compute_group_sizes(race_ids)
        model = LGBMRankerModel()
        model.fit(X, y, group=groups)

        importances = model.feature_importances
        assert len(importances) == X.shape[1]


# ---------------------------------------------------------------------------
# EnsembleModel tests
# ---------------------------------------------------------------------------


class TestEnsembleModel:
    """Tests for EnsembleModel."""

    def test_add_fitted_models(self):
        X, y = _make_classification_data()

        m1 = LGBMClassifierModel(params={"n_estimators": 10})
        m1.fit(X, y)
        m2 = LGBMClassifierModel(params={"n_estimators": 20})
        m2.fit(X, y)

        ensemble = EnsembleModel()
        ensemble.add_model(m1, weight=0.6)
        ensemble.add_model(m2, weight=0.4)

        assert ensemble.is_fitted
        preds = ensemble.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba(self):
        X, y = _make_classification_data()

        m1 = LGBMClassifierModel(params={"n_estimators": 10})
        m1.fit(X, y)

        ensemble = EnsembleModel()
        ensemble.add_model(m1)

        scores = ensemble.predict_proba(X)
        assert len(scores) == len(X)
        assert all(0 <= s <= 1 for s in scores)

    def test_feature_importances(self):
        X, y = _make_classification_data()

        m1 = LGBMClassifierModel(params={"n_estimators": 10})
        m1.fit(X, y)

        ensemble = EnsembleModel()
        ensemble.add_model(m1)

        importances = ensemble.feature_importances
        assert len(importances) == X.shape[1]

    def test_predict_before_fit_raises(self):
        ensemble = EnsembleModel()
        X, _ = _make_classification_data(n_samples=5)
        with pytest.raises(RuntimeError):
            ensemble.predict(X)


# ---------------------------------------------------------------------------
# Model Factory tests
# ---------------------------------------------------------------------------


class TestModelFactory:
    """Tests for model factory function."""

    def test_create_known_model(self):
        model = create_model("lgbm_classifier")
        assert isinstance(model, LGBMClassifierModel)

    def test_create_with_params(self):
        model = create_model("lgbm_classifier", params={"n_estimators": 10})
        assert model.params["n_estimators"] == 10

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown_model")

    def test_registry_contains_expected(self):
        assert "lgbm_classifier" in MODEL_REGISTRY
        assert "lgbm_ranker" in MODEL_REGISTRY
        assert "ensemble" in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# ModelTrainer tests
# ---------------------------------------------------------------------------


class TestModelTrainer:
    """Tests for ModelTrainer."""

    def test_train_classifier(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 10})
        trainer = ModelTrainer()

        result = trainer.train(model, X, y)

        assert "model" in result
        assert "metrics" in result
        assert result["model"].is_fitted
        assert "val_accuracy" in result["metrics"]

    def test_train_with_dates(self):
        X, y = _make_classification_data()
        dates = pd.Series(pd.date_range("2024-01-01", periods=len(X), freq="D"))

        model = LGBMClassifierModel(params={"n_estimators": 10})
        trainer = ModelTrainer()

        result = trainer.train(model, X, y, race_dates=dates)
        assert result["model"].is_fitted

    def test_train_ranker(self):
        X, y, race_ids = _make_ranking_data()
        model = LGBMRankerModel(params={"n_estimators": 10})
        trainer = ModelTrainer()

        result = trainer.train(model, X, y, race_ids=race_ids)
        assert result["model"].is_fitted

    def test_cross_validate(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 10})
        trainer = ModelTrainer()

        result = trainer.cross_validate(model, X, y, n_splits=3)

        assert "fold_metrics" in result
        assert "mean_metrics" in result
        assert "std_metrics" in result
        assert len(result["fold_metrics"]) == 3

    def test_train_calibrates_and_optimizes_threshold(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 50})
        trainer = ModelTrainer()

        result = trainer.train(model, X, y)

        # Model should have been calibrated
        assert model._calibrated_model is not None
        # Optimal threshold should be set (likely < 0.5 for imbalanced data)
        assert "optimal_threshold" in result["metrics"]
        assert 0.0 < model.optimal_threshold < 1.0

    def test_find_optimal_threshold(self):
        rng = np.random.RandomState(42)
        y_true = pd.Series((rng.rand(1000) > 0.9).astype(int))
        proba = rng.rand(1000)
        # Make proba correlated with y_true
        proba[y_true == 1] += 0.3
        proba = np.clip(proba, 0, 1)

        threshold = find_optimal_threshold(y_true, proba)
        assert 0.01 <= threshold < 0.50

    def test_train_with_tracker(self):
        X, y = _make_classification_data()
        model = LGBMClassifierModel(params={"n_estimators": 10})

        tracker = MagicMock(spec=ExperimentTracker)
        tracker.is_active = True

        trainer = ModelTrainer(tracker=tracker)
        trainer.train(model, X, y)

        tracker.log_params.assert_called()
        tracker.log_metrics.assert_called()


# ---------------------------------------------------------------------------
# ExperimentTracker tests
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_tracker_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test-exp",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )

            assert not tracker.is_active

            tracker.start_run(run_name="test-run")
            assert tracker.is_active
            assert tracker.run_id is not None

            tracker.log_param("key", "value")
            tracker.log_metric("accuracy", 0.85)
            tracker.log_metrics({"f1": 0.8, "auc": 0.9})

            tracker.end_run()
            assert not tracker.is_active

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test-ctx",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            with tracker:
                assert tracker.is_active
                tracker.log_param("test", "value")
            assert not tracker.is_active

    def test_log_without_active_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test-noop",
                tracking_uri=f"file:{tmpdir}/mlruns",
            )
            # Should not raise
            tracker.log_param("key", "value")
            tracker.log_metric("metric", 1.0)


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for ModelRegistry with mocked GCS."""

    @patch("src.model_training.model_registry.GCSClient")
    @patch("src.model_training.model_registry.get_settings")
    def test_save_model(self, mock_settings, mock_gcs_cls):
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_models="test-bucket")
        )
        mock_gcs = MagicMock()
        mock_gcs_cls.return_value = mock_gcs

        from src.model_training.model_registry import ModelRegistry

        registry = ModelRegistry()

        X, y = _make_classification_data(n_samples=50)
        model = LGBMClassifierModel(params={"n_estimators": 10})
        model.fit(X, y)

        version = registry.save_model(model, "test_model", metrics={"accuracy": 0.85})

        assert version is not None
        assert mock_gcs.upload_file.called
        assert mock_gcs.upload_json.called

    @patch("src.model_training.model_registry.GCSClient")
    @patch("src.model_training.model_registry.get_settings")
    def test_save_model_with_empty_metrics_and_metadata(
        self, mock_settings, mock_gcs_cls
    ):
        """Regression: empty dict args must not raise ambiguous truth errors."""
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_models="test-bucket")
        )
        mock_gcs = MagicMock()
        mock_gcs_cls.return_value = mock_gcs

        from src.model_training.model_registry import ModelRegistry

        registry = ModelRegistry()

        X, y = _make_classification_data(n_samples=50)
        model = LGBMClassifierModel(params={"n_estimators": 10})
        model.fit(X, y)

        # Empty dicts and None should all work without errors
        version = registry.save_model(
            model, "test_model", metrics={}, extra_metadata={}
        )
        assert version is not None

        version2 = registry.save_model(
            model, "test_model", metrics=None, extra_metadata=None
        )
        assert version2 is not None

    @patch("src.model_training.model_registry.GCSClient")
    @patch("src.model_training.model_registry.get_settings")
    def test_list_versions(self, mock_settings, mock_gcs_cls):
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_models="test-bucket")
        )
        mock_gcs = MagicMock()
        mock_gcs.list_blobs.return_value = [
            "models/my_model/20260101_abc12345/model.pkl",
            "models/my_model/20260101_abc12345/metadata.json",
            "models/my_model/20260215_def67890/model.pkl",
            "models/my_model/20260215_def67890/metadata.json",
        ]
        mock_gcs_cls.return_value = mock_gcs

        from src.model_training.model_registry import ModelRegistry

        registry = ModelRegistry()
        versions = registry.list_versions("my_model")

        assert len(versions) == 2
        assert versions[0] == "20260101_abc12345"
        assert versions[1] == "20260215_def67890"

    @patch("src.model_training.model_registry.GCSClient")
    @patch("src.model_training.model_registry.get_settings")
    def test_get_latest_version(self, mock_settings, mock_gcs_cls):
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_models="test-bucket")
        )
        mock_gcs = MagicMock()
        mock_gcs.list_blobs.return_value = [
            "models/my_model/20260101_aaa/model.pkl",
            "models/my_model/20260215_bbb/model.pkl",
        ]
        mock_gcs_cls.return_value = mock_gcs

        from src.model_training.model_registry import ModelRegistry

        registry = ModelRegistry()
        latest = registry.get_latest_version("my_model")

        assert latest == "20260215_bbb"

    @patch("src.model_training.model_registry.GCSClient")
    @patch("src.model_training.model_registry.get_settings")
    def test_no_versions_returns_none(self, mock_settings, mock_gcs_cls):
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_models="test-bucket")
        )
        mock_gcs = MagicMock()
        mock_gcs.list_blobs.return_value = []
        mock_gcs_cls.return_value = mock_gcs

        from src.model_training.model_registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.get_latest_version("nonexistent") is None
