"""Feature scaling transformers.

Provides standard scaling (z-score) and min-max scaling for
numeric features in Polars DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnStats:
    """Learned statistics for a single column."""

    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0


class FeatureScaler:
    """Scales numeric feature columns.

    Supports "standard" (z-score normalization) and "minmax" strategies.
    Learned statistics are stored for consistent transform on new data.

    Args:
        columns: List of column names to scale.
        strategy: "standard" for z-score, "minmax" for 0-1 range.
    """

    def __init__(
        self,
        columns: list[str],
        strategy: str = "standard",
    ) -> None:
        self._columns = columns
        self._strategy = strategy
        self._stats: dict[str, ColumnStats] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pl.DataFrame) -> FeatureScaler:
        """Learn scaling statistics from training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self for method chaining.
        """
        for col in self._columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found during fit, skipping", col)
                continue
            series = df[col].drop_nulls().cast(pl.Float64)
            stats = ColumnStats(
                mean=series.mean() or 0.0,
                std=series.std() or 1.0,
                min=series.min() or 0.0,
                max=series.max() or 1.0,
            )
            # Guard against zero std / zero range
            if stats.std == 0.0:
                stats.std = 1.0
            if stats.max == stats.min:
                stats.max = stats.min + 1.0
            self._stats[col] = stats

        self._fitted = True
        logger.info(
            "FeatureScaler fitted",
            columns=list(self._stats.keys()),
            strategy=self._strategy,
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply scaling to a DataFrame.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with scaled columns (suffixed with _scaled).

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform")

        result = df.clone()

        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            scaled_col = f"{col}_scaled"
            if self._strategy == "standard":
                result = result.with_columns(
                    ((pl.col(col).cast(pl.Float64) - stats.mean) / stats.std).alias(
                        scaled_col
                    )
                )
            elif self._strategy == "minmax":
                result = result.with_columns(
                    (
                        (pl.col(col).cast(pl.Float64) - stats.min)
                        / (stats.max - stats.min)
                    ).alias(scaled_col)
                )

        return result

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)
