"""Walk-forward backtesting engine for horse racing models.

Implements a time-series aware sliding-window approach where the model
is retrained on each expanding or rolling window and evaluated on the
subsequent period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import polars as pl

from src.common.logging import get_logger
from src.evaluator.metrics import MetricsResult, RacingMetrics

logger = get_logger(__name__)


class PredictionModel(Protocol):
    """Minimal interface expected from a prediction model."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> np.ndarray: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...


@dataclass
class BacktestPeriod:
    """Results for a single backtest evaluation window."""

    period_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    metrics: MetricsResult


@dataclass
class BacktestResult:
    """Aggregated backtest results across all periods."""

    periods: list[BacktestPeriod] = field(default_factory=list)
    overall_metrics: MetricsResult = field(default_factory=MetricsResult)

    def summary_df(self) -> pl.DataFrame:
        """Return a DataFrame summarizing per-period metrics."""
        records: list[dict[str, Any]] = []
        for p in self.periods:
            row: dict[str, Any] = {
                "period": p.period_index,
                "train_start": p.train_start,
                "train_end": p.train_end,
                "test_start": p.test_start,
                "test_end": p.test_end,
                "n_train": p.n_train,
                "n_test": p.n_test,
            }
            row.update(p.metrics.to_dict())
            records.append(row)
        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)


class BacktestEngine:
    """Time-series walk-forward backtesting.

    Splits data by date into train/test windows and evaluates model
    performance on each test window after retraining.

    Args:
        train_window_days: Number of days in the training window.
            Use ``None`` for expanding window (all data up to split).
        test_window_days: Number of days in each test window.
        step_days: Number of days to advance the window each step.
        date_col: Column name containing race dates.
        race_id_col: Column containing race identifiers.
        target_col: Column containing the target variable.
        feature_cols: List of feature column names. If None, all
            columns starting with ``feat_`` are used.
    """

    def __init__(
        self,
        train_window_days: int | None = 365,
        test_window_days: int = 30,
        step_days: int = 30,
        date_col: str = "race_date",
        race_id_col: str = "race_id",
        target_col: str = "is_win",
        feature_cols: list[str] | None = None,
    ) -> None:
        self._train_window = train_window_days
        self._test_window = test_window_days
        self._step = step_days
        self._date_col = date_col
        self._race_id_col = race_id_col
        self._target_col = target_col
        self._feature_cols = feature_cols

    def run(
        self,
        df: pl.DataFrame,
        model: PredictionModel,
        has_betting: bool = False,
    ) -> BacktestResult:
        """Execute the walk-forward backtest.

        Args:
            df: Full dataset with features, target, dates.
            model: Model instance that will be cloned/retrained each period.
            has_betting: Whether betting columns are present.

        Returns:
            BacktestResult with per-period and overall metrics.
        """
        df = df.sort(self._date_col)

        feature_cols = self._feature_cols
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith("feat_")]

        dates = df[self._date_col].cast(pl.Date)
        min_date = dates.min()
        max_date = dates.max()
        if min_date is None or max_date is None:
            logger.warning("No valid dates found in data")
            return BacktestResult()

        periods: list[BacktestPeriod] = []
        all_predictions: list[pl.DataFrame] = []

        period_idx = 0

        # Determine the first test window start
        if self._train_window is not None:
            from datetime import timedelta

            test_start = min_date + timedelta(days=self._train_window)
        else:
            # Expanding window: need at least some data, start after 90 days
            from datetime import timedelta

            test_start = min_date + timedelta(days=90)

        from datetime import timedelta

        while test_start <= max_date:
            test_end = test_start + timedelta(days=self._test_window - 1)

            # Determine training range
            if self._train_window is not None:
                train_start = test_start - timedelta(days=self._train_window)
            else:
                train_start = min_date

            # Split data
            train_df = df.filter(
                (pl.col(self._date_col).cast(pl.Date) >= train_start)
                & (pl.col(self._date_col).cast(pl.Date) < test_start)
            )
            test_df = df.filter(
                (pl.col(self._date_col).cast(pl.Date) >= test_start)
                & (pl.col(self._date_col).cast(pl.Date) <= test_end)
            )

            if train_df.is_empty() or test_df.is_empty():
                test_start += timedelta(days=self._step)
                continue

            # Train
            X_train = train_df.select(feature_cols).to_pandas()
            y_train = train_df[self._target_col].to_pandas()

            # Calibrate and find optimal threshold for calibratable models
            X_test = test_df.select(feature_cols).to_pandas()
            threshold = 0.5
            if hasattr(model, "calibrate") and hasattr(model, "optimal_threshold"):
                # Hold out last 20% of train for calibration (not seen by model)
                cal_split = int(len(X_train) * 0.8)
                X_train_fit = X_train.iloc[:cal_split]
                y_train_fit = y_train.iloc[:cal_split]
                X_cal = X_train.iloc[cal_split:]
                y_cal = y_train.iloc[cal_split:]
                model.fit(X_train_fit, y_train_fit)
                model.calibrate(X_cal, y_cal)

                from src.model_training.models.lgbm_classifier import (
                    find_optimal_threshold,
                )

                threshold = find_optimal_threshold(y_cal, model.predict_proba(X_cal))
                model.optimal_threshold = threshold
            else:
                model.fit(X_train, y_train)

            # Predict
            proba = model.predict_proba(X_test)

            # Handle shape: (n, 2) for classifiers or (n,) for rankers
            if proba.ndim == 2:
                win_proba = proba[:, 1]
            else:
                win_proba = proba

            # Build prediction DataFrame
            pred_df = test_df.select(
                [self._race_id_col, "horse_id", self._target_col]
                + (
                    [self._date_col]
                    if self._date_col
                    not in [self._race_id_col, "horse_id", self._target_col]
                    else []
                )
                + (["bet_amount", "payout", "odds"] if has_betting else [])
            ).with_columns(
                pl.Series("predicted_prob", win_proba),
                pl.Series("predicted_win", (win_proba >= threshold).astype(int)),
                pl.Series(
                    "actual_position",
                    (
                        test_df["actual_position"].to_numpy()
                        if "actual_position" in test_df.columns
                        else np.zeros(len(test_df))
                    ),
                ),
            )

            # Compute predicted_rank per race
            pred_df = pred_df.with_columns(
                pl.col("predicted_prob")
                .rank(method="ordinal", descending=True)
                .over(self._race_id_col)
                .alias("predicted_rank")
            )

            # Rename target if needed
            if self._target_col != "is_win":
                pred_df = pred_df.rename({self._target_col: "is_win"})

            # Evaluate
            metrics = RacingMetrics.compute_all(pred_df, has_betting=has_betting)

            period = BacktestPeriod(
                period_index=period_idx,
                train_start=str(train_start),
                train_end=str(test_start - timedelta(days=1)),
                test_start=str(test_start),
                test_end=str(test_end),
                n_train=len(train_df),
                n_test=len(test_df),
                metrics=metrics,
            )
            periods.append(period)
            all_predictions.append(pred_df)

            logger.info(
                "Backtest period completed",
                period=period_idx,
                test_range=f"{test_start} to {test_end}",
                n_test=len(test_df),
                win_accuracy=metrics.values.get("win_accuracy", 0),
            )

            period_idx += 1
            test_start += timedelta(days=self._step)

        # Compute overall metrics on all predictions
        overall_metrics = MetricsResult()
        if all_predictions:
            combined = pl.concat(all_predictions)
            overall_metrics = RacingMetrics.compute_all(
                combined, has_betting=has_betting
            )

        result = BacktestResult(periods=periods, overall_metrics=overall_metrics)

        logger.info(
            "Backtest completed",
            n_periods=len(periods),
            overall_win_accuracy=overall_metrics.values.get("win_accuracy", 0),
        )

        return result
