"""GCS writer for raw data storage in Parquet format.

Uses the common GCSClient to persist collected data as Parquet files
organized by data type and date.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from src.common.config import get_settings
from src.common.gcp_client import GCSClient
from src.common.logging import get_logger

logger = get_logger(__name__)


class GCSWriter:
    """Writes DataFrames to GCS as Parquet files.

    Files are organized as:
        gs://{bucket}/{data_type}/{partition_key}/{data_type}.parquet

    For date-partitioned data (e.g. races), partition_key is YYYY/MM/DD.

    Args:
        gcs_client: Optional pre-configured GCS client.
        bucket_name: Override bucket name (defaults to settings).
    """

    def __init__(
        self,
        gcs_client: GCSClient | None = None,
        bucket_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._client = gcs_client or GCSClient()
        self._bucket = bucket_name or settings.gcs.bucket_raw

    def write_parquet(
        self,
        df: pl.DataFrame,
        data_type: str,
        partition_key: str,
    ) -> str:
        """Write a Polars DataFrame to GCS as a Parquet file.

        Args:
            df: DataFrame to write.
            data_type: Category of data (e.g. "races", "horses").
            partition_key: Sub-path for partitioning (e.g. "2024-01-15"
                will be stored as 2024/01/15).

        Returns:
            The GCS URI of the written file.
        """
        if df.is_empty():
            logger.debug("Skipping empty DataFrame write", data_type=data_type)
            return ""

        # Convert date-like partition keys (YYYY-MM-DD -> YYYY/MM/DD)
        path_key = partition_key.replace("-", "/")
        blob_path = f"{data_type}/{path_key}/{data_type}.parquet"

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            df.write_parquet(tmp_path)
            uri = self._client.upload_file(self._bucket, str(tmp_path), blob_path)
            logger.info(
                "Wrote parquet to GCS",
                uri=uri,
                rows=len(df),
                columns=df.columns,
            )
            return uri
        finally:
            tmp_path.unlink(missing_ok=True)
