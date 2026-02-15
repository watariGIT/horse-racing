"""Ensemble model combining multiple base models.

Supports weighted averaging of predictions from classifier
and ranker models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.model_training.models.base import BaseModel


class EnsembleModel(BaseModel):
    """Weighted ensemble of multiple base models.

    Combines predictions from multiple models using configurable weights.
    Supports both classification probabilities and ranking scores.

    Args:
        params: Configuration dict. Supported keys:
            - weights: List of floats for model weighting.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._models: list[BaseModel] = []
        self._weights: list[float] = (params or {}).get("weights", [])

    @property
    def model_type(self) -> str:
        return "ensemble"

    @property
    def models(self) -> list[BaseModel]:
        return self._models

    @property
    def feature_importances(self) -> np.ndarray:
        self._check_fitted()
        if not self._models:
            return np.array([])
        importances = []
        for model, weight in zip(self._models, self._effective_weights):
            importances.append(model.feature_importances * weight)
        return np.mean(importances, axis=0)

    @property
    def _effective_weights(self) -> list[float]:
        """Return weights, defaulting to uniform if not specified."""
        if self._weights and len(self._weights) == len(self._models):
            total = sum(self._weights)
            return [w / total for w in self._weights]
        n = len(self._models)
        return [1.0 / n] * n if n > 0 else []

    def add_model(self, model: BaseModel, weight: float = 1.0) -> EnsembleModel:
        """Add a pre-trained model to the ensemble.

        Args:
            model: A fitted BaseModel instance.
            weight: Weight for this model's predictions.

        Returns:
            Self for method chaining.
        """
        self._models.append(model)
        self._weights.append(weight)
        if all(m.is_fitted for m in self._models):
            self._fitted = True
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> EnsembleModel:
        """Fit all sub-models on the same data.

        For models requiring group info (ranker), pass via kwargs.

        Args:
            X: Feature matrix.
            y: Target values.
            **kwargs: Additional args passed to each sub-model.

        Returns:
            Self for method chaining.
        """
        for model in self._models:
            model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted average predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average of all model predictions.
        """
        self._check_fitted()
        weights = self._effective_weights
        predictions = np.zeros(len(X))
        for model, weight in zip(self._models, weights):
            predictions += model.predict(X).astype(float) * weight
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted average probability predictions.

        For models that produce probability arrays (n_samples, n_classes),
        uses the positive class column (index 1). For ranker models,
        uses raw scores.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average probability/score array.
        """
        self._check_fitted()
        weights = self._effective_weights
        scores = np.zeros(len(X))
        for model, weight in zip(self._models, weights):
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                scores += proba[:, 1] * weight
            else:
                scores += proba * weight
        return scores
