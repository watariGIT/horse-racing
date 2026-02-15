"""Prediction pipeline: data -> features -> model -> results.

Orchestrates the end-to-end prediction flow from raw input
data through feature engineering to model prediction.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from src.common.logging import get_logger
from src.feature_engineering.pipeline import FeaturePipeline
from src.model_training.models.base import BaseModel

logger = get_logger(__name__)


class PredictionPipeline:
    """End-to-end prediction pipeline.

    Chains feature engineering and model prediction into a
    single callable flow. Handles data format conversion
    between Polars (features) and Pandas (model).

    Args:
        model: Trained model for predictions.
        feature_pipeline: Fitted feature pipeline.
    """

    def __init__(
        self,
        model: BaseModel,
        feature_pipeline: FeaturePipeline,
    ) -> None:
        self._model = model
        self._feature_pipeline = feature_pipeline

    def predict(self, raw_data: pl.DataFrame) -> pd.DataFrame:
        """Run the full prediction pipeline.

        Steps:
            1. Extract features using the feature pipeline
            2. Select feature columns (feat_* prefix)
            3. Convert to pandas for model input
            4. Generate predictions and probabilities
            5. Combine with identifiers

        Args:
            raw_data: Raw input data with race/horse/jockey info.

        Returns:
            DataFrame with columns: race_id, horse_id (if present),
            prediction, win_probability.
        """
        # Extract features
        features_df = self._feature_pipeline.transform(raw_data)

        # Get feature-only columns
        feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
        X = features_df.select(feature_cols).to_pandas()

        # Predict
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)

        # Build result
        result = pd.DataFrame({"prediction": predictions})

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            result["win_probability"] = probabilities[:, 1]
        else:
            result["win_probability"] = probabilities

        # Add identifiers from raw data
        if "race_id" in raw_data.columns:
            result["race_id"] = raw_data["race_id"].to_pandas().values
        if "horse_id" in raw_data.columns:
            result["horse_id"] = raw_data["horse_id"].to_pandas().values

        # Sort by probability (highest first) within each race
        if "race_id" in result.columns:
            result = result.sort_values(
                ["race_id", "win_probability"],
                ascending=[True, False],
            ).reset_index(drop=True)

        logger.info(
            "Prediction completed",
            n_predictions=len(result),
        )

        return result

    def predict_with_ranking(self, raw_data: pl.DataFrame) -> pd.DataFrame:
        """Predict with rank position within each race.

        Args:
            raw_data: Raw input data.

        Returns:
            Prediction DataFrame with added 'rank' column.
        """
        result = self.predict(raw_data)

        if "race_id" in result.columns:
            result["rank"] = (
                result.groupby("race_id")["win_probability"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        else:
            result["rank"] = (
                result["win_probability"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

        return result
