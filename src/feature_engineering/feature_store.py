"""Feature store backed by BigQuery.

Provides read/write access to materialized feature tables
for training and serving.
"""

from __future__ import annotations

import polars as pl

from src.common.config import get_settings
from src.common.gcp_client import BigQueryClient
from src.common.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_FEATURE_TABLE = "features_v1"


class FeatureStore:
    """Manages feature storage and retrieval in BigQuery.

    Stores computed features for training and serves them
    for prediction.

    Args:
        bq_client: Optional pre-configured BigQuery client.
        table_id: Feature table name.
    """

    def __init__(
        self,
        bq_client: BigQueryClient | None = None,
        table_id: str = _DEFAULT_FEATURE_TABLE,
    ) -> None:
        self._client = bq_client or BigQueryClient()
        self._table_id = table_id
        settings = get_settings()
        self._dataset = settings.bigquery.dataset

    def save_features(
        self,
        df: pl.DataFrame,
        write_disposition: str = "WRITE_APPEND",
    ) -> None:
        """Save a feature DataFrame to BigQuery.

        Args:
            df: Feature DataFrame to persist.
            write_disposition: WRITE_APPEND or WRITE_TRUNCATE.
        """
        if df.is_empty():
            logger.debug("Skipping empty feature save")
            return

        pandas_df = df.to_pandas()
        self._client.load_dataframe(
            pandas_df,
            self._table_id,
            write_disposition=write_disposition,
        )
        logger.info(
            "Saved features",
            table=self._table_id,
            rows=len(df),
        )

    def load_features(
        self,
        race_ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load features from BigQuery.

        Args:
            race_ids: Optional list of race IDs to filter by.
            columns: Optional list of columns to select.

        Returns:
            Polars DataFrame of features.
        """
        col_clause = ", ".join(columns) if columns else "*"
        sql = f"SELECT {col_clause} FROM `{self._dataset}.{self._table_id}`"

        if race_ids:
            placeholders = ", ".join(f"'{rid}'" for rid in race_ids)
            sql += f" WHERE race_id IN ({placeholders})"

        pandas_df = self._client.query(sql)
        result = pl.from_pandas(pandas_df)
        logger.info(
            "Loaded features",
            table=self._table_id,
            rows=len(result),
        )
        return result

    def table_exists(self) -> bool:
        """Check if the feature table exists in BigQuery.

        Returns:
            True if the table exists.
        """
        return self._client.table_exists(self._table_id)
