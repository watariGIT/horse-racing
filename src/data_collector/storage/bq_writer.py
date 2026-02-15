"""BigQuery writer for structured data storage.

Uses the common BigQueryClient to persist collected data
into BigQuery tables.
"""

from __future__ import annotations

import polars as pl

from src.common.gcp_client import BigQueryClient
from src.common.logging import get_logger

logger = get_logger(__name__)


class BQWriter:
    """Writes DataFrames to BigQuery tables.

    Converts Polars DataFrames to pandas for BigQuery ingestion,
    using the existing BigQueryClient wrapper.

    Args:
        bq_client: Optional pre-configured BigQuery client.
    """

    def __init__(self, bq_client: BigQueryClient | None = None) -> None:
        self._client = bq_client or BigQueryClient()

    def write(
        self,
        df: pl.DataFrame,
        table_id: str,
        write_disposition: str = "WRITE_APPEND",
        partition_field: str | None = None,
    ) -> None:
        """Write a Polars DataFrame to a BigQuery table.

        Args:
            df: DataFrame to write.
            table_id: Target table name (without dataset prefix).
            write_disposition: WRITE_APPEND, WRITE_TRUNCATE, or WRITE_EMPTY.
            partition_field: Optional field for time partitioning.
        """
        if df.is_empty():
            logger.debug("Skipping empty DataFrame write", table_id=table_id)
            return

        # Convert Polars -> pandas for BigQuery client compatibility
        pandas_df = df.to_pandas()

        self._client.load_dataframe(
            pandas_df,
            table_id,
            write_disposition=write_disposition,
            partition_field=partition_field,
        )
        logger.info(
            "Wrote to BigQuery",
            table_id=table_id,
            rows=len(df),
        )
