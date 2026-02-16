"""Data preparation with history aggregation for training.

Computes per-horse rolling statistics from race history, ensuring
no future data leakage by only using races prior to each row's date.
"""

from __future__ import annotations

import polars as pl

from src.common.logging import get_logger

logger = get_logger(__name__)


class DataPreparer:
    """Prepares raw race data for model training.

    Aggregates historical horse performance for each race row,
    using only data from before that race to prevent leakage.

    Args:
        n_past_races: Number of recent past races to consider
            when computing rolling statistics.
    """

    def __init__(self, n_past_races: int = 5) -> None:
        self._n_past = n_past_races

    def prepare_for_training(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Add historical aggregation columns and target to each race row.

        For each row, computes horse statistics using only races with
        ``race_date < current_race_date`` to prevent future data leakage.

        Adds columns:
            - avg_finish: Mean finish position over past N races
            - win_rate: Win rate over past N races
            - top3_rate: Top-3 rate over past N races
            - days_since_last_race: Days since horse's previous race
            - num_past_races: Total number of prior races
            - is_win: Binary target (1 if finish_position == 1)
            - actual_position: Copy of finish_position

        Args:
            raw_df: Raw race results DataFrame with at minimum
                horse_id, race_date, finish_position columns.

        Returns:
            DataFrame enriched with history features and targets.
        """
        df = raw_df.filter(pl.col("finish_position").is_not_null())
        df = df.sort("race_date")

        logger.info(
            "Preparing data for training",
            rows=len(df),
            n_past_races=self._n_past,
        )

        history_cols = self._compute_horse_history(df)
        df = df.with_columns(history_cols)

        # Add target columns
        df = df.with_columns(
            (pl.col("finish_position") == 1).cast(pl.Int64).alias("is_win"),
            pl.col("finish_position").alias("actual_position"),
        )

        logger.info("Data preparation complete", rows=len(df))
        return df

    def _compute_horse_history(self, df: pl.DataFrame) -> list[pl.Expr]:
        """Compute rolling history features using sorted group operations.

        Uses Polars shift-based approach: for each horse, rolling stats
        are computed over the previous N races (shifted by 1 to exclude
        the current row).

        Args:
            df: Sorted DataFrame.

        Returns:
            List of Polars expressions for history columns.
        """
        n = self._n_past

        # Shifted finish_position within each horse group (excludes current race)
        # We use rolling_mean over a window of n on the shifted column.
        # Polars' over() with shift ensures we only look at past data.

        return [
            # avg_finish: rolling mean of past n finish positions
            pl.col("finish_position")
            .shift(1)
            .rolling_mean(window_size=n, min_samples=1)
            .over("horse_id")
            .alias("avg_finish"),
            # win_rate: rolling mean of (finish == 1) over past n
            (pl.col("finish_position") == 1)
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=n, min_samples=1)
            .over("horse_id")
            .alias("win_rate"),
            # top3_rate: rolling mean of (finish <= 3) over past n
            (pl.col("finish_position") <= 3)
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=n, min_samples=1)
            .over("horse_id")
            .alias("top3_rate"),
            # days_since_last_race
            (pl.col("race_date") - pl.col("race_date").shift(1).over("horse_id"))
            .dt.total_days()
            .alias("days_since_last_race"),
            # num_past_races: cumulative count - 1 (exclude current)
            pl.col("finish_position")
            .cum_count()
            .over("horse_id")
            .sub(1)
            .alias("num_past_races"),
        ]
