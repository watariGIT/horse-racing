"""Horse-level feature extractor.

Extracts features from horse past performance data:
average finish position, win rate, days since last race, age, weight,
market odds, gate position, and condition indicators.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.feature_engineering.extractors.base import BaseFeatureExtractor

# Sex encoding: male/colt=0, female/filly=1, gelding=2
_SEX_MAP = {"牡": 0, "牝": 1, "騸": 2}


class HorseFeatureExtractor(BaseFeatureExtractor):
    """Extracts horse-level features from past performance data.

    Computes rolling statistics over each horse's recent race history.

    Args:
        n_past_races: Number of recent races to consider for stats.
    """

    _FEATURES = [
        "feat_avg_finish",
        "feat_win_rate",
        "feat_top3_rate",
        "feat_days_since_last_race",
        "feat_horse_age",
        "feat_horse_weight",
        "feat_num_past_races",
        "feat_win_odds",
        "feat_win_favorite",
        "feat_bracket_number",
        "feat_post_position",
        "feat_carried_weight",
        "feat_sex",
        "feat_horse_weight_change",
    ]

    def __init__(self, n_past_races: int = 5) -> None:
        self._n_past = n_past_races

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract horse features.

        Expected input columns:
            horse_id, finish_position, race_date, horse_age, weight,
            past_results (optional nested or pre-joined)

        For pre-aggregated input (one row per horse with stats), the
        method uses them directly. For detailed history, it aggregates.

        Args:
            df: DataFrame with horse data.

        Returns:
            DataFrame with added horse feature columns.
        """
        result = df.clone()

        # Average finish position over past N races
        if "avg_finish" in df.columns:
            result = result.with_columns(
                pl.col("avg_finish").cast(pl.Float64).alias("feat_avg_finish")
            )
        elif "finish_position" in df.columns:
            result = result.with_columns(
                pl.col("finish_position").cast(pl.Float64).alias("feat_avg_finish")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_avg_finish")
            )

        # Win rate
        if "win_rate" in df.columns:
            result = result.with_columns(
                pl.col("win_rate").cast(pl.Float64).alias("feat_win_rate")
            )
        elif "finish_position" in df.columns:
            result = result.with_columns(
                (pl.col("finish_position") == 1).cast(pl.Float64).alias("feat_win_rate")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_win_rate")
            )

        # Top-3 rate
        if "top3_rate" in df.columns:
            result = result.with_columns(
                pl.col("top3_rate").cast(pl.Float64).alias("feat_top3_rate")
            )
        elif "finish_position" in df.columns:
            result = result.with_columns(
                (pl.col("finish_position") <= 3)
                .cast(pl.Float64)
                .alias("feat_top3_rate")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_top3_rate")
            )

        # Days since last race
        if "days_since_last_race" in df.columns:
            result = result.with_columns(
                pl.col("days_since_last_race")
                .cast(pl.Int64)
                .alias("feat_days_since_last_race")
            )
        elif "last_race_date" in df.columns:
            today = date.today()
            result = result.with_columns(
                (pl.lit(today) - pl.col("last_race_date").cast(pl.Date))
                .dt.total_days()
                .alias("feat_days_since_last_race")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_days_since_last_race")
            )

        # Horse age
        if "horse_age" in df.columns:
            result = result.with_columns(
                pl.col("horse_age").cast(pl.Int64).alias("feat_horse_age")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_horse_age")
            )

        # Horse weight
        if "weight" in df.columns:
            result = result.with_columns(
                pl.col("weight").cast(pl.Float64).alias("feat_horse_weight")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_horse_weight")
            )

        # Number of past races
        if "num_past_races" in df.columns:
            result = result.with_columns(
                pl.col("num_past_races").cast(pl.Int64).alias("feat_num_past_races")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_num_past_races")
            )

        # Win odds (market evaluation)
        if "win_odds" in df.columns:
            result = result.with_columns(
                pl.col("win_odds").cast(pl.Float64).alias("feat_win_odds")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_win_odds")
            )

        # Win favorite (popularity rank)
        if "win_favorite" in df.columns:
            result = result.with_columns(
                pl.col("win_favorite").cast(pl.Int64).alias("feat_win_favorite")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_win_favorite")
            )

        # Bracket number (gate group)
        if "bracket_number" in df.columns:
            result = result.with_columns(
                pl.col("bracket_number").cast(pl.Int64).alias("feat_bracket_number")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_bracket_number")
            )

        # Post position (gate number)
        if "post_position" in df.columns:
            result = result.with_columns(
                pl.col("post_position").cast(pl.Int64).alias("feat_post_position")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Int64).alias("feat_post_position")
            )

        # Carried weight (handicap)
        if "carried_weight" in df.columns:
            result = result.with_columns(
                pl.col("carried_weight").cast(pl.Float64).alias("feat_carried_weight")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_carried_weight")
            )

        # Sex (ordinal encoded)
        if "sex" in df.columns:
            result = result.with_columns(
                pl.col("sex")
                .replace_strict(_SEX_MAP, default=-1)
                .cast(pl.Int64)
                .alias("feat_sex")
            )
        else:
            result = result.with_columns(pl.lit(None).cast(pl.Int64).alias("feat_sex"))

        # Horse weight change
        if "horse_weight_change" in df.columns:
            result = result.with_columns(
                pl.col("horse_weight_change")
                .cast(pl.Float64)
                .alias("feat_horse_weight_change")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_horse_weight_change")
            )

        return result

    @staticmethod
    def aggregate_history(history_df: pl.DataFrame, n_past: int = 5) -> pl.DataFrame:
        """Aggregate per-race history into per-horse summary statistics.

        Args:
            history_df: DataFrame with columns horse_id, race_date,
                finish_position, weight.
            n_past: Number of most recent races to consider.

        Returns:
            Aggregated DataFrame with one row per horse.
        """
        # Sort and take most recent N races per horse
        sorted_df = history_df.sort(["horse_id", "race_date"], descending=[False, True])
        recent = sorted_df.group_by("horse_id").head(n_past)

        agg = recent.group_by("horse_id").agg(
            pl.col("finish_position").mean().alias("avg_finish"),
            (pl.col("finish_position") == 1).mean().alias("win_rate"),
            (pl.col("finish_position") <= 3).mean().alias("top3_rate"),
            pl.col("race_date").max().alias("last_race_date"),
            pl.col("weight").last().alias("weight"),
            pl.col("finish_position").count().alias("num_past_races"),
        )
        return agg
