"""Running style feature extractor.

Extracts features derived from historical race running patterns:
average final corner position (running style) and average closing
speed (last_3f_time). These use pre-computed historical averages
from DataPreparer to avoid data leakage.
"""

from __future__ import annotations

import polars as pl

from src.feature_engineering.extractors.base import BaseFeatureExtractor


class RunningStyleFeatureExtractor(BaseFeatureExtractor):
    """Extracts historical running style features.

    Uses pre-aggregated historical averages computed by DataPreparer
    (avg_corner_pos_4, avg_last_3f_time) to represent a horse's
    running style and closing speed without leakage.
    """

    _FEATURES = [
        "feat_avg_corner_pos_4",
        "feat_avg_last_3f_time",
    ]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract running style features.

        Expected input columns (pre-aggregated by DataPreparer):
            avg_corner_pos_4: Historical average of final corner position.
            avg_last_3f_time: Historical average of last 3 furlong time.

        Args:
            df: DataFrame with pre-aggregated running style data.

        Returns:
            DataFrame with added running style feature columns.
        """
        result = df.clone()

        # Average final corner position (running style indicator)
        if "avg_corner_pos_4" in df.columns:
            result = result.with_columns(
                pl.col("avg_corner_pos_4")
                .cast(pl.Float64)
                .alias("feat_avg_corner_pos_4")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_avg_corner_pos_4")
            )

        # Average last 3F time (closing speed indicator)
        if "avg_last_3f_time" in df.columns:
            result = result.with_columns(
                pl.col("avg_last_3f_time")
                .cast(pl.Float64)
                .alias("feat_avg_last_3f_time")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_avg_last_3f_time")
            )

        return result
