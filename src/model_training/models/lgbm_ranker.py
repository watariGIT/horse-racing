"""LightGBM LambdaRank model for finish position prediction.

Wraps LightGBM's LGBMRanker to predict horse finish order
within each race group.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model_training.models.base import BaseModel

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "label_gain": list(range(18, 0, -1)),
}


class LGBMRankerModel(BaseModel):
    """LightGBM LambdaRank model for finish position ranking.

    Predicts the relative order of horses within a race.
    Requires group information (number of horses per race).

    Args:
        params: Override default hyperparameters.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**_DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self._model: lgb.LGBMRanker | None = None

    @property
    def model_type(self) -> str:
        return "lgbm_ranker"

    @property
    def feature_importances(self) -> np.ndarray:
        self._check_fitted()
        assert self._model is not None
        return self._model.feature_importances_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> LGBMRankerModel:
        """Train the LightGBM ranker.

        Args:
            X: Feature matrix.
            y: Relevance labels (higher = better finish position).
                Typically: max_position - finish_position + 1.
            **kwargs: Must include 'group' (array of group sizes).
                May also include eval_set, eval_group, callbacks.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If 'group' is not provided in kwargs.
        """
        if "group" not in kwargs:
            raise ValueError(
                "LGBMRankerModel.fit requires 'group' parameter "
                "(array of group sizes per race)"
            )
        self._model = lgb.LGBMRanker(**self._params)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores for ranking.

        Higher scores indicate higher predicted finish position.

        Args:
            X: Feature matrix.

        Returns:
            Array of relevance scores.
        """
        self._check_fitted()
        assert self._model is not None
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores (same as predict for ranker).

        LambdaRank does not produce calibrated probabilities.
        Returns raw scores normalized per group if needed.

        Args:
            X: Feature matrix.

        Returns:
            Array of relevance scores.
        """
        return self.predict(X)

    @staticmethod
    def compute_group_sizes(race_ids: pd.Series) -> np.ndarray:
        """Compute group sizes from race IDs.

        Args:
            race_ids: Series of race identifiers, ordered so that
                rows from the same race are contiguous.

        Returns:
            Array of group sizes (number of horses per race).
        """
        return np.asarray(race_ids.groupby(race_ids, sort=False).count().values)

    @staticmethod
    def create_relevance_labels(
        finish_positions: pd.Series,
        max_position: int = 18,
    ) -> pd.Series:
        """Convert finish positions to relevance labels for LambdaRank.

        Higher relevance = better position. 1st place gets the highest score.

        Args:
            finish_positions: Series of finish positions (1-based).
            max_position: Maximum possible position for scaling.

        Returns:
            Series of relevance labels.
        """
        return pd.Series(
            max_position - finish_positions.clip(upper=max_position) + 1,
            index=finish_positions.index,
            name="relevance",
        )
