"""Unit tests for the predictor module.

Tests model loading, prediction pipeline, and batch predictor
with mock data and mocked GCS/BigQuery services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.model_training.models.base import BaseModel
from src.model_training.models.lgbm_classifier import LGBMClassifierModel
from src.predictor.batch_predictor import BatchPredictor
from src.predictor.model_loader import ModelLoader
from src.predictor.prediction_pipeline import PredictionPipeline

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_mock_model() -> MagicMock:
    """Create a mock model that accepts any number of features."""
    model = MagicMock(spec=BaseModel)
    model.is_fitted = True
    model.model_type = "lgbm_classifier"

    def mock_predict(X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(42)
        return (rng.rand(len(X)) > 0.85).astype(int)

    def mock_predict_proba(X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(42)
        probs = rng.rand(len(X))
        return np.column_stack([1 - probs, probs])

    model.predict = mock_predict
    model.predict_proba = mock_predict_proba
    return model


def _make_real_model() -> LGBMClassifierModel:
    """Create a real fitted classifier for pipeline tests."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        rng.randn(100, 5),
        columns=[f"feat_{i}" for i in range(5)],
    )
    y = pd.Series((rng.rand(100) > 0.85).astype(int))
    model = LGBMClassifierModel(params={"n_estimators": 10, "verbose": -1})
    model.fit(X, y)
    return model


def _make_raw_race_data() -> pl.DataFrame:
    """Create mock raw race data for prediction."""
    return pl.DataFrame(
        {
            "race_id": ["R001"] * 5 + ["R002"] * 5,
            "horse_id": [f"H{i:03d}" for i in range(10)],
            "distance": [1600] * 10,
            "track_condition": ["良"] * 10,
            "weather": ["晴"] * 10,
            "track_type": ["芝"] * 10,
            "num_entries": [10] * 10,
            "grade": ["G1"] * 10,
            "avg_finish": [2.5, 3.0, 1.5, 4.0, 5.0, 2.0, 3.5, 1.0, 4.5, 6.0],
            "win_rate": [0.3, 0.2, 0.5, 0.1, 0.05, 0.35, 0.15, 0.6, 0.08, 0.02],
            "top3_rate": [0.6, 0.4, 0.8, 0.2, 0.1, 0.7, 0.3, 0.9, 0.15, 0.05],
            "days_since_last_race": [14, 21, 7, 30, 45, 10, 28, 14, 35, 60],
            "horse_age": [4, 5, 3, 6, 4, 3, 5, 4, 7, 3],
            "weight": [
                480.0,
                470.0,
                490.0,
                460.0,
                500.0,
                475.0,
                485.0,
                465.0,
                510.0,
                455.0,
            ],
            "num_past_races": [10, 15, 5, 20, 8, 12, 18, 6, 25, 3],
            "jockey_win_rate": [
                0.15,
                0.10,
                0.20,
                0.08,
                0.12,
                0.18,
                0.09,
                0.25,
                0.07,
                0.11,
            ],
            "jockey_top3_rate": [
                0.35,
                0.25,
                0.45,
                0.20,
                0.30,
                0.40,
                0.22,
                0.50,
                0.18,
                0.28,
            ],
            "jockey_course_win_rate": [
                0.20,
                0.12,
                0.25,
                0.10,
                0.15,
                0.22,
                0.11,
                0.30,
                0.08,
                0.14,
            ],
            "jockey_experience": [500, 300, 800, 200, 400, 600, 250, 1000, 150, 350],
        }
    )


# ---------------------------------------------------------------------------
# ModelLoader tests
# ---------------------------------------------------------------------------


class TestModelLoader:
    """Tests for ModelLoader."""

    def test_load_with_cache(self):
        mock_registry = MagicMock()
        mock_model = _make_mock_model()
        mock_registry.load_model.return_value = mock_model
        mock_registry.get_latest_version.return_value = "v1"

        loader = ModelLoader(registry=mock_registry, cache_size=2)

        # First load - from registry
        model1 = loader.load("test_model", "v1")
        assert mock_registry.load_model.call_count == 1

        # Second load - from cache
        model2 = loader.load("test_model", "v1")
        assert mock_registry.load_model.call_count == 1  # No new call
        assert model1 is model2

    def test_load_latest(self):
        mock_registry = MagicMock()
        mock_model = _make_mock_model()
        mock_registry.load_model.return_value = mock_model
        mock_registry.get_latest_version.return_value = "v2"

        loader = ModelLoader(registry=mock_registry)
        loader.load("test_model")

        mock_registry.get_latest_version.assert_called_once_with("test_model")
        mock_registry.load_model.assert_called_once_with("test_model", "v2")

    def test_load_no_versions_raises(self):
        mock_registry = MagicMock()
        mock_registry.get_latest_version.return_value = None

        loader = ModelLoader(registry=mock_registry)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_model")

    def test_cache_eviction(self):
        mock_registry = MagicMock()
        mock_registry.load_model.return_value = _make_mock_model()

        loader = ModelLoader(registry=mock_registry, cache_size=2)

        loader.load("model_a", "v1")
        loader.load("model_b", "v1")
        loader.load("model_c", "v1")  # Should evict model_a

        assert "model_a:v1" not in loader._cache
        assert "model_b:v1" in loader._cache
        assert "model_c:v1" in loader._cache

    def test_clear_cache(self):
        mock_registry = MagicMock()
        mock_registry.load_model.return_value = _make_mock_model()

        loader = ModelLoader(registry=mock_registry)
        loader.load("test", "v1")
        assert len(loader._cache) == 1

        loader.clear_cache()
        assert len(loader._cache) == 0


# ---------------------------------------------------------------------------
# PredictionPipeline tests
# ---------------------------------------------------------------------------


class TestPredictionPipeline:
    """Tests for PredictionPipeline."""

    def test_predict(self):
        from src.feature_engineering.extractors.horse_features import (
            HorseFeatureExtractor,
        )
        from src.feature_engineering.extractors.jockey_features import (
            JockeyFeatureExtractor,
        )
        from src.feature_engineering.extractors.race_features import (
            RaceFeatureExtractor,
        )
        from src.feature_engineering.pipeline import FeaturePipeline

        raw_data = _make_raw_race_data()

        # Build and fit feature pipeline
        feature_pipeline = FeaturePipeline(
            extractors=[
                RaceFeatureExtractor(),
                HorseFeatureExtractor(),
                JockeyFeatureExtractor(),
            ],
        )
        feature_pipeline.fit(raw_data)

        # Get feature columns for training a model
        transformed = feature_pipeline.transform(raw_data)
        feat_cols = [c for c in transformed.columns if c.startswith("feat_")]
        X_train = transformed.select(feat_cols).to_pandas()
        y_train = pd.Series([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        model = LGBMClassifierModel(params={"n_estimators": 10, "verbose": -1})
        model.fit(X_train, y_train)

        # Run prediction pipeline
        pipeline = PredictionPipeline(model=model, feature_pipeline=feature_pipeline)
        result = pipeline.predict(raw_data)

        assert "prediction" in result.columns
        assert "win_probability" in result.columns
        assert "race_id" in result.columns
        assert "horse_id" in result.columns
        assert len(result) == 10

    def test_predict_with_ranking(self):
        from src.feature_engineering.extractors.horse_features import (
            HorseFeatureExtractor,
        )
        from src.feature_engineering.extractors.jockey_features import (
            JockeyFeatureExtractor,
        )
        from src.feature_engineering.extractors.race_features import (
            RaceFeatureExtractor,
        )
        from src.feature_engineering.pipeline import FeaturePipeline

        raw_data = _make_raw_race_data()

        feature_pipeline = FeaturePipeline(
            extractors=[
                RaceFeatureExtractor(),
                HorseFeatureExtractor(),
                JockeyFeatureExtractor(),
            ],
        )
        feature_pipeline.fit(raw_data)

        transformed = feature_pipeline.transform(raw_data)
        feat_cols = [c for c in transformed.columns if c.startswith("feat_")]
        X_train = transformed.select(feat_cols).to_pandas()
        y_train = pd.Series([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        model = LGBMClassifierModel(params={"n_estimators": 10, "verbose": -1})
        model.fit(X_train, y_train)

        pipeline = PredictionPipeline(model=model, feature_pipeline=feature_pipeline)
        result = pipeline.predict_with_ranking(raw_data)

        assert "rank" in result.columns
        # Check ranks within each race
        for race_id in result["race_id"].unique():
            race_ranks = result[result["race_id"] == race_id]["rank"]
            assert race_ranks.min() == 1


# ---------------------------------------------------------------------------
# BatchPredictor tests
# ---------------------------------------------------------------------------


class TestBatchPredictor:
    """Tests for BatchPredictor."""

    def test_predict_batch(self):
        raw_data = _make_raw_race_data()
        mock_model = _make_mock_model()

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model

        predictor = BatchPredictor(
            model_name="test_model",
            feature_config={"extractors": ["race", "horse", "jockey"]},
            model_loader=mock_loader,
        )

        result = predictor.predict_batch(raw_data)

        assert "prediction" in result.columns
        assert "win_probability" in result.columns
        assert "predicted_at" in result.columns
        assert "model_name" in result.columns
        assert len(result) == 10

    @patch("src.predictor.batch_predictor.BigQueryClient")
    def test_save_to_bigquery(self, mock_bq_cls):
        raw_data = _make_raw_race_data()
        mock_model = _make_mock_model()

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model

        predictor = BatchPredictor(
            model_name="test_model",
            feature_config={"extractors": ["race", "horse", "jockey"]},
            model_loader=mock_loader,
        )

        predictor.predict_batch(raw_data, save_to_bq=True)

        mock_bq_cls.return_value.load_dataframe.assert_called_once()

    @patch("src.predictor.batch_predictor.get_settings")
    @patch("src.predictor.batch_predictor.GCSClient")
    def test_save_to_gcs(self, mock_gcs_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            gcs=MagicMock(bucket_processed="test-bucket")
        )
        raw_data = _make_raw_race_data()
        mock_model = _make_mock_model()

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model

        predictor = BatchPredictor(
            model_name="test_model",
            feature_config={"extractors": ["race", "horse", "jockey"]},
            model_loader=mock_loader,
        )

        predictor.predict_batch(raw_data, save_to_gcs=True)

        mock_gcs_cls.return_value.upload_json.assert_called_once()
