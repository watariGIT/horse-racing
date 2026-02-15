"""Race-level feature extractor.

Extracts features from race metadata: distance, track condition,
weather, course, grade, and number of entries.
"""

from __future__ import annotations

import polars as pl

from src.feature_engineering.extractors.base import BaseFeatureExtractor

# Mappings for categorical encoding
_TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
_WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}
_TRACK_TYPE_MAP = {"芝": 0, "ダート": 1, "障害": 2}


class RaceFeatureExtractor(BaseFeatureExtractor):
    """Extracts race-level features.

    Produces numeric features from race metadata including
    distance, encoded track condition/weather/course, grade, and
    number of entries.
    """

    _FEATURES = [
        "feat_distance",
        "feat_track_condition",
        "feat_weather",
        "feat_track_type",
        "feat_num_entries",
        "feat_is_grade_race",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract race features.

        Expected input columns:
            distance, track_condition, weather, track_type,
            num_entries, grade

        Args:
            df: DataFrame with race data.

        Returns:
            DataFrame with added race feature columns.
        """
        result = df.clone()

        # Distance as-is (already numeric)
        if "distance" in df.columns:
            result = result.with_columns(
                pl.col("distance").cast(pl.Int64).alias("feat_distance")
            )
        else:
            result = result.with_columns(pl.lit(0).alias("feat_distance"))

        # Track condition encoding
        if "track_condition" in df.columns:
            result = result.with_columns(
                pl.col("track_condition")
                .replace_strict(_TRACK_CONDITION_MAP, default=-1, return_dtype=pl.Int64)
                .alias("feat_track_condition")
            )
        else:
            result = result.with_columns(pl.lit(-1).alias("feat_track_condition"))

        # Weather encoding
        if "weather" in df.columns:
            result = result.with_columns(
                pl.col("weather")
                .replace_strict(_WEATHER_MAP, default=-1, return_dtype=pl.Int64)
                .alias("feat_weather")
            )
        else:
            result = result.with_columns(pl.lit(-1).alias("feat_weather"))

        # Track type encoding
        if "track_type" in df.columns:
            result = result.with_columns(
                pl.col("track_type")
                .replace_strict(_TRACK_TYPE_MAP, default=-1, return_dtype=pl.Int64)
                .alias("feat_track_type")
            )
        else:
            result = result.with_columns(pl.lit(-1).alias("feat_track_type"))

        # Number of entries
        if "num_entries" in df.columns:
            result = result.with_columns(
                pl.col("num_entries").cast(pl.Int64).alias("feat_num_entries")
            )
        else:
            result = result.with_columns(pl.lit(0).alias("feat_num_entries"))

        # Grade race flag
        if "grade" in df.columns:
            result = result.with_columns(
                (pl.col("grade").is_not_null() & (pl.col("grade") != ""))
                .cast(pl.Int64)
                .alias("feat_is_grade_race")
            )
        else:
            result = result.with_columns(pl.lit(0).alias("feat_is_grade_race"))

        return result
