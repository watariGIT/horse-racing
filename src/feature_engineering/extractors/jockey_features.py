"""Jockey-level feature extractor.

Extracts jockey performance features: overall win rate,
course-specific win rate, and experience.
"""

from __future__ import annotations

import polars as pl

from src.feature_engineering.extractors.base import BaseFeatureExtractor


class JockeyFeatureExtractor(BaseFeatureExtractor):
    """Extracts jockey-level features.

    Computes win rate, course-specific performance, and experience metrics.
    """

    _FEATURES = [
        "feat_jockey_win_rate",
        "feat_jockey_top3_rate",
        "feat_jockey_course_win_rate",
        "feat_jockey_experience",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract jockey features.

        Expected input columns:
            jockey_win_rate, jockey_top3_rate, jockey_course_win_rate,
            jockey_experience (pre-aggregated stats)

        Args:
            df: DataFrame with jockey statistics.

        Returns:
            DataFrame with added jockey feature columns.
        """
        result = df.clone()

        # Overall win rate
        if "jockey_win_rate" in df.columns:
            result = result.with_columns(
                pl.col("jockey_win_rate").cast(pl.Float64).alias("feat_jockey_win_rate")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_jockey_win_rate")
            )

        # Top-3 rate
        if "jockey_top3_rate" in df.columns:
            result = result.with_columns(
                pl.col("jockey_top3_rate")
                .cast(pl.Float64)
                .alias("feat_jockey_top3_rate")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_jockey_top3_rate")
            )

        # Course-specific win rate
        if "jockey_course_win_rate" in df.columns:
            result = result.with_columns(
                pl.col("jockey_course_win_rate")
                .cast(pl.Float64)
                .alias("feat_jockey_course_win_rate")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_jockey_course_win_rate")
            )

        # Experience (total races ridden)
        if "jockey_experience" in df.columns:
            result = result.with_columns(
                pl.col("jockey_experience")
                .cast(pl.Int64)
                .alias("feat_jockey_experience")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_jockey_experience")
            )

        return result

    @staticmethod
    def aggregate_history(
        history_df: pl.DataFrame,
        course: str | None = None,
    ) -> pl.DataFrame:
        """Aggregate jockey race history into summary statistics.

        Args:
            history_df: DataFrame with columns jockey_id, race_id,
                course, finish_position.
            course: If specified, also compute course-specific stats.

        Returns:
            Aggregated DataFrame with one row per jockey.
        """
        overall = history_df.group_by("jockey_id").agg(
            (pl.col("finish_position") == 1).mean().alias("jockey_win_rate"),
            (pl.col("finish_position") <= 3).mean().alias("jockey_top3_rate"),
            pl.col("race_id").n_unique().alias("jockey_experience"),
        )

        if course:
            course_df = (
                history_df.filter(pl.col("course") == course)
                .group_by("jockey_id")
                .agg(
                    (pl.col("finish_position") == 1)
                    .mean()
                    .alias("jockey_course_win_rate")
                )
            )
            overall = overall.join(course_df, on="jockey_id", how="left")
        else:
            overall = overall.with_columns(
                pl.lit(None).cast(pl.Float64).alias("jockey_course_win_rate")
            )

        return overall
