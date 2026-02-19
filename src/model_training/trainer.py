"""Model trainer with time-series aware cross-validation.

Orchestrates the training loop with MLflow integration,
ensuring no future data leakage in temporal splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.common.logging import get_logger
from src.model_training.experiment_tracker import ExperimentTracker
from src.model_training.models.base import BaseModel
from src.model_training.models.lgbm_classifier import LGBMClassifierModel
from src.model_training.models.lgbm_ranker import LGBMRankerModel

logger = get_logger(__name__)


class ModelTrainer:
    """Trains models with time-series aware validation.

    Provides train/validate splitting that respects temporal ordering
    to prevent future data leakage. Integrates with MLflow for
    experiment tracking.

    Args:
        tracker: Optional ExperimentTracker for MLflow logging.
    """

    def __init__(self, tracker: ExperimentTracker | None = None) -> None:
        self._tracker = tracker

    def train(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        race_dates: pd.Series | None = None,
        race_ids: pd.Series | None = None,
        validation_ratio: float = 0.2,
        **fit_kwargs: Any,
    ) -> dict[str, Any]:
        """Train a model with temporal train/validation split.

        Splits data by time so validation uses only future data
        relative to training. Records metrics via MLflow if a
        tracker is configured.

        Args:
            model: Model instance to train.
            X: Feature matrix.
            y: Target values.
            race_dates: Optional date series for temporal splitting.
                If None, uses the last validation_ratio of rows.
            race_ids: Race ID series (required for ranker models).
            validation_ratio: Fraction of data used for validation.
            **fit_kwargs: Extra args passed to model.fit.

        Returns:
            Dict with train/validation metrics and the fitted model.
        """
        train_idx, val_idx = self._temporal_split(X, race_dates, validation_ratio)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Handle ranker group info
        if isinstance(model, LGBMRankerModel) and race_ids is not None:
            train_groups = LGBMRankerModel.compute_group_sizes(race_ids.iloc[train_idx])
            val_groups = LGBMRankerModel.compute_group_sizes(race_ids.iloc[val_idx])
            fit_kwargs["group"] = train_groups
            fit_kwargs.setdefault("eval_set", [(X_val, y_val)])
            fit_kwargs.setdefault("eval_group", [val_groups])

        # Train
        model.fit(X_train, y_train, **fit_kwargs)

        # Calibrate and optimize threshold for classifiers
        if isinstance(model, LGBMClassifierModel):
            model.calibrate(X_val, y_val)
            optimal_thresh = self._find_optimal_threshold(
                y_val, model.predict_proba(X_val)
            )
            model.optimal_threshold = optimal_thresh

        # Evaluate
        metrics = self._evaluate(model, X_train, y_train, X_val, y_val)

        # Log to MLflow
        if self._tracker and self._tracker.is_active:
            self._tracker.log_params(model.params)
            self._tracker.log_metrics(metrics)
            self._tracker.log_param("model_type", model.model_type)
            self._tracker.log_param("train_size", len(X_train))
            self._tracker.log_param("val_size", len(X_val))

        logger.info(
            "Training completed",
            model_type=model.model_type,
            train_size=len(X_train),
            val_size=len(X_val),
            val_metrics={
                k: round(v, 4) for k, v in metrics.items() if k.startswith("val_")
            },
        )

        return {"model": model, "metrics": metrics}

    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        race_ids: pd.Series | None = None,
        n_splits: int = 5,
        **fit_kwargs: Any,
    ) -> dict[str, Any]:
        """Perform time-series cross-validation.

        Uses sklearn's TimeSeriesSplit to create temporal folds,
        ensuring each validation fold comes after the training fold.

        Args:
            model: Model instance (a fresh copy is trained per fold).
            X: Feature matrix.
            y: Target values.
            race_ids: Race ID series (required for ranker models).
            n_splits: Number of CV folds.
            **fit_kwargs: Extra args passed to model.fit.

        Returns:
            Dict with per-fold and averaged metrics.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create a fresh model with the same params
            fold_model = model.__class__(params=model.params)
            fold_fit_kwargs = dict(fit_kwargs)

            if isinstance(fold_model, LGBMRankerModel) and race_ids is not None:
                train_groups = LGBMRankerModel.compute_group_sizes(
                    race_ids.iloc[train_idx]
                )
                fold_fit_kwargs["group"] = train_groups

            fold_model.fit(X_train, y_train, **fold_fit_kwargs)
            metrics = self._evaluate(fold_model, X_train, y_train, X_val, y_val)
            fold_metrics.append(metrics)

            logger.info(
                "CV fold completed",
                fold=fold_idx + 1,
                n_splits=n_splits,
                val_metrics={
                    k: round(v, 4) for k, v in metrics.items() if k.startswith("val_")
                },
            )

        # Aggregate metrics
        avg_metrics: dict[str, float] = {}
        std_metrics: dict[str, float] = {}
        all_keys = fold_metrics[0].keys()
        for key in all_keys:
            values = [m[key] for m in fold_metrics]
            avg_metrics[f"cv_mean_{key}"] = float(np.mean(values))
            std_metrics[f"cv_std_{key}"] = float(np.std(values))

        if self._tracker and self._tracker.is_active:
            self._tracker.log_metrics(avg_metrics)
            self._tracker.log_metrics(std_metrics)

        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": avg_metrics,
            "std_metrics": std_metrics,
        }

    def _temporal_split(
        self,
        X: pd.DataFrame,
        race_dates: pd.Series | None,
        validation_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split data temporally, keeping future data for validation.

        Args:
            X: Feature matrix (for index/length reference).
            race_dates: Optional date series for sorting.
            validation_ratio: Fraction of data for validation.

        Returns:
            Tuple of (train_indices, validation_indices).
        """
        n = len(X)
        if race_dates is not None:
            sorted_idx = race_dates.argsort().values
        else:
            sorted_idx = np.arange(n)

        split_point = int(n * (1 - validation_ratio))
        train_idx = sorted_idx[:split_point]
        val_idx = sorted_idx[split_point:]

        return train_idx, val_idx

    @staticmethod
    def _find_optimal_threshold(
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

        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(low, high, step):
            preds = (proba_pos >= threshold).astype(int)
            score = float(f1_score(y_true, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        return best_threshold

    def _evaluate(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, float]:
        """Compute evaluation metrics for train and validation sets.

        For classifier models, computes accuracy, AUC, F1, precision,
        recall, and log loss. For ranker models, computes only basic
        prediction stats.

        Args:
            model: Fitted model.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Dict of metric name -> value.
        """
        metrics: dict[str, float] = {}

        if isinstance(model, LGBMRankerModel):
            val_pred = model.predict(X_val)
            metrics["val_pred_mean"] = float(np.mean(val_pred))
            metrics["val_pred_std"] = float(np.std(val_pred))
            return metrics

        # Record optimal threshold if available
        if isinstance(model, LGBMClassifierModel):
            metrics["optimal_threshold"] = model.optimal_threshold

        # Classifier metrics
        for prefix, X_set, y_set in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
        ]:
            pred = model.predict(X_set)
            proba = model.predict_proba(X_set)

            metrics[f"{prefix}_accuracy"] = float(accuracy_score(y_set, pred))
            metrics[f"{prefix}_f1"] = float(f1_score(y_set, pred, zero_division=0))
            metrics[f"{prefix}_precision"] = float(
                precision_score(y_set, pred, zero_division=0)
            )
            metrics[f"{prefix}_recall"] = float(
                recall_score(y_set, pred, zero_division=0)
            )

            if proba.ndim == 2 and proba.shape[1] == 2:
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba

            n_classes = len(np.unique(y_set))
            if n_classes == 2:
                metrics[f"{prefix}_auc"] = float(roc_auc_score(y_set, proba_pos))
                metrics[f"{prefix}_logloss"] = float(log_loss(y_set, proba_pos))

        return metrics
