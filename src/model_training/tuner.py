"""Optuna-based hyperparameter tuning.

Provides automated hyperparameter optimization for LightGBM models
using Optuna with time-series aware validation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.common.logging import get_logger
from src.model_training.models.base import BaseModel
from src.model_training.models.lgbm_ranker import LGBMRankerModel

logger = get_logger(__name__)


def _default_classifier_search_space(
    trial: optuna.Trial,
) -> dict[str, Any]:
    """Default search space for LightGBM classifier."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _default_ranker_search_space(
    trial: optuna.Trial,
) -> dict[str, Any]:
    """Default search space for LightGBM ranker."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner for horse racing models.

    Performs Bayesian optimization of model hyperparameters
    using time-series cross-validation.

    Args:
        n_trials: Number of Optuna trials to run.
        n_cv_splits: Number of TimeSeriesSplit folds.
        metric: Metric to optimize. "auc" or "logloss" for classifiers.
        direction: "maximize" or "minimize".
        search_space_fn: Custom search space function.
            If None, uses default search space for the model type.
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_cv_splits: int = 3,
        metric: str = "auc",
        direction: str = "maximize",
        search_space_fn: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    ) -> None:
        self._n_trials = n_trials
        self._n_cv_splits = n_cv_splits
        self._metric = metric
        self._direction = direction
        self._search_space_fn = search_space_fn
        self._study: optuna.Study | None = None

    @property
    def best_params(self) -> dict[str, Any]:
        """Return the best hyperparameters found.

        Raises:
            RuntimeError: If tuning has not been run yet.
        """
        if self._study is None:
            raise RuntimeError("Must call tune() before accessing best_params")
        return self._study.best_params

    @property
    def best_score(self) -> float:
        """Return the best score achieved.

        Raises:
            RuntimeError: If tuning has not been run yet.
        """
        if self._study is None:
            raise RuntimeError("Must call tune() before accessing best_score")
        return self._study.best_value

    @property
    def study(self) -> optuna.Study | None:
        return self._study

    def tune(
        self,
        model_class: type[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
        race_ids: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            model_class: Model class to tune (e.g. LGBMClassifierModel).
            X: Feature matrix.
            y: Target values.
            race_ids: Race ID series (required for ranker models).

        Returns:
            Dict with best_params and best_score.
        """
        search_fn = self._search_space_fn
        if search_fn is None:
            if issubclass(model_class, LGBMRankerModel):
                search_fn = _default_ranker_search_space
            else:
                search_fn = _default_classifier_search_space

        def objective(trial: optuna.Trial) -> float:
            params = search_fn(trial)
            return self._cv_evaluate(model_class, params, X, y, race_ids)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._study = optuna.create_study(direction=self._direction)
        self._study.optimize(objective, n_trials=self._n_trials)

        logger.info(
            "Tuning completed",
            best_score=round(self._study.best_value, 4),
            best_params=self._study.best_params,
            n_trials=self._n_trials,
        )

        return {
            "best_params": self._study.best_params,
            "best_score": self._study.best_value,
        }

    def _cv_evaluate(
        self,
        model_class: type[BaseModel],
        params: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        race_ids: pd.Series | None,
    ) -> float:
        """Evaluate params with time-series cross-validation.

        Args:
            model_class: Model class to instantiate.
            params: Hyperparameters to evaluate.
            X: Feature matrix.
            y: Target values.
            race_ids: Race ID series for ranker group computation.

        Returns:
            Mean metric value across folds.
        """
        tscv = TimeSeriesSplit(n_splits=self._n_cv_splits)
        scores: list[float] = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = model_class(params=params)
            fit_kwargs: dict[str, Any] = {}

            if isinstance(model, LGBMRankerModel) and race_ids is not None:
                train_groups = LGBMRankerModel.compute_group_sizes(
                    race_ids.iloc[train_idx]
                )
                fit_kwargs["group"] = train_groups

            model.fit(X_train, y_train, **fit_kwargs)
            score = self._compute_score(model, X_val, y_val)
            scores.append(score)

        return float(np.mean(scores))

    def _compute_score(
        self,
        model: BaseModel,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Compute the optimization metric.

        Args:
            model: Fitted model.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Score value.
        """
        if self._metric == "auc":
            proba = model.predict_proba(X_val)
            if proba.ndim == 2:
                proba = proba[:, 1]
            n_classes = len(np.unique(y_val))
            if n_classes < 2:
                return 0.5
            return float(roc_auc_score(y_val, proba))

        if self._metric == "logloss":
            proba = model.predict_proba(X_val)
            if proba.ndim == 2:
                proba = proba[:, 1]
            return float(log_loss(y_val, proba))

        # Default: accuracy
        pred = model.predict(X_val)
        return float(np.mean(pred == y_val))
