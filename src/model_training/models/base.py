"""Abstract base model class for horse racing prediction.

Defines the interface that all model implementations must follow,
including fit/predict/save/load and feature importance access.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for prediction models.

    All model implementations must implement fit, predict, and
    predict_proba methods. Provides save/load via pickle.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = params or {}
        self._model: Any = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def params(self) -> dict[str, Any]:
        return self._params.copy()

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return a string identifier for this model type."""

    @property
    @abstractmethod
    def feature_importances(self) -> np.ndarray:
        """Return feature importance values from the trained model.

        Raises:
            RuntimeError: If the model has not been fitted.
        """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> BaseModel:
        """Train the model on the given data.

        Args:
            X: Feature matrix.
            y: Target values.
            **kwargs: Additional training arguments.

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given features.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities (shape depends on implementation).
        """

    def save(self, path: str | Path) -> Path:
        """Save the model to a file using pickle.

        Args:
            path: File path to save the model.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a model from a pickle file.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected BaseModel subclass"
            )
        return model

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction"
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status}, params={self._params})"
