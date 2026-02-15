"""LightGBM classifier for win prediction (binary classification).

Wraps LightGBM's LGBMClassifier to predict whether a horse finishes
first (1) or not (0).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model_training.models.base import BaseModel

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
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
}


class LGBMClassifierModel(BaseModel):
    """LightGBM binary classifier for win prediction.

    Predicts whether a horse finishes in 1st place.

    Args:
        params: Override default hyperparameters.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**_DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self._model: lgb.LGBMClassifier | None = None

    @property
    def model_type(self) -> str:
        return "lgbm_classifier"

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
    ) -> LGBMClassifierModel:
        """Train the LightGBM classifier.

        Args:
            X: Feature matrix.
            y: Binary target (1 = win, 0 = not win).
            **kwargs: Extra args passed to lgb.LGBMClassifier.fit
                (e.g. eval_set, callbacks).

        Returns:
            Self for method chaining.
        """
        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class (win / not win).

        Args:
            X: Feature matrix.

        Returns:
            Array of 0/1 predictions.
        """
        self._check_fitted()
        assert self._model is not None
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [P(not win), P(win)].
        """
        self._check_fitted()
        assert self._model is not None
        return self._model.predict_proba(X)
