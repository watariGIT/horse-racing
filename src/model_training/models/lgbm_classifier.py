"""LightGBM classifier for win prediction (binary classification).

Wraps LightGBM's LGBMClassifier to predict whether a horse finishes
first (1) or not (0). Supports probability calibration and
optimal threshold selection for imbalanced datasets.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score

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
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._optimal_threshold: float = 0.5

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
        """Predict binary class using optimal threshold.

        Args:
            X: Feature matrix.

        Returns:
            Array of 0/1 predictions.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba_pos = proba[:, 1]
        else:
            proba_pos = proba
        return (proba_pos >= self._optimal_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability.

        Uses calibrated model if available, otherwise raw LightGBM output.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [P(not win), P(win)].
        """
        self._check_fitted()
        assert self._model is not None
        if self._calibrated_model is not None:
            return self._calibrated_model.predict_proba(X)
        return self._model.predict_proba(X)

    def calibrate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "isotonic",
    ) -> LGBMClassifierModel:
        """Calibrate predicted probabilities using held-out data.

        Args:
            X: Validation feature matrix.
            y: Validation binary target.
            method: Calibration method (``"isotonic"`` or ``"sigmoid"``).

        Returns:
            Self for method chaining.
        """
        self._check_fitted()
        assert self._model is not None
        self._calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(self._model), method=method
        )
        self._calibrated_model.fit(X, y)
        return self

    @property
    def optimal_threshold(self) -> float:
        """Return the current optimal classification threshold."""
        return self._optimal_threshold

    @optimal_threshold.setter
    def optimal_threshold(self, value: float) -> None:
        """Set the optimal classification threshold."""
        self._optimal_threshold = value


def find_optimal_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    low: float = 0.01,
    high: float = 0.50,
    step: float = 0.01,
) -> float:
    """Find the threshold that maximizes F1 score.

    Args:
        y_true: True binary labels.
        proba: Predicted probabilities (shape (n,2) or (n,)).
        low: Lower bound of threshold search range.
        high: Upper bound of threshold search range.
        step: Step size for threshold search.

    Returns:
        Optimal threshold value.
    """
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba_pos = proba[:, 1]
    else:
        proba_pos = proba

    best_threshold = low
    best_f1 = 0.0

    for threshold in np.arange(low, high, step):
        preds = (proba_pos >= threshold).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold
