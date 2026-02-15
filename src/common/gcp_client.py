"""GCS and BigQuery client wrappers with error handling.

Provides simplified interfaces for common GCP storage and query operations
used throughout the horse racing ML pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery, storage
from google.cloud.exceptions import GoogleCloudError

from src.common.config import get_settings

logger = logging.getLogger(__name__)


class GCSClient:
    """Wrapper around Google Cloud Storage client."""

    def __init__(self, project_id: str | None = None) -> None:
        settings = get_settings()
        self._project = project_id or settings.gcp.project_id
        self._client = storage.Client(project=self._project)

    def upload_file(
        self, bucket_name: str, source_path: str | Path, destination_blob: str
    ) -> str:
        """Upload a local file to GCS.

        Args:
            bucket_name: Target GCS bucket name.
            source_path: Path to local file.
            destination_blob: Destination path in the bucket.

        Returns:
            The GCS URI of the uploaded file.
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(str(source_path))
            uri = f"gs://{bucket_name}/{destination_blob}"
            logger.info("Uploaded file to %s", uri)
            return uri
        except GoogleCloudError as e:
            logger.error(
                "Failed to upload to gs://%s/%s: %s", bucket_name, destination_blob, e
            )
            raise

    def upload_json(
        self, bucket_name: str, data: dict[str, Any] | list[Any], destination_blob: str
    ) -> str:
        """Upload JSON data directly to GCS.

        Args:
            bucket_name: Target GCS bucket name.
            data: JSON-serializable data.
            destination_blob: Destination path in the bucket.

        Returns:
            The GCS URI of the uploaded file.
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(destination_blob)
            blob.upload_from_string(
                json.dumps(data, ensure_ascii=False, default=str),
                content_type="application/json",
            )
            uri = f"gs://{bucket_name}/{destination_blob}"
            logger.info("Uploaded JSON to %s", uri)
            return uri
        except GoogleCloudError as e:
            logger.error(
                "Failed to upload JSON to gs://%s/%s: %s",
                bucket_name,
                destination_blob,
                e,
            )
            raise

    def download_file(
        self, bucket_name: str, source_blob: str, destination_path: str | Path
    ) -> Path:
        """Download a file from GCS to local filesystem.

        Args:
            bucket_name: Source GCS bucket name.
            source_blob: Path of the blob in the bucket.
            destination_path: Local path to save the file.

        Returns:
            Path to the downloaded file.
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(source_blob)
            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            logger.info("Downloaded gs://%s/%s to %s", bucket_name, source_blob, dest)
            return dest
        except GoogleCloudError as e:
            logger.error(
                "Failed to download gs://%s/%s: %s", bucket_name, source_blob, e
            )
            raise

    def download_json(self, bucket_name: str, source_blob: str) -> Any:
        """Download and parse a JSON file from GCS.

        Args:
            bucket_name: Source GCS bucket name.
            source_blob: Path of the JSON blob in the bucket.

        Returns:
            Parsed JSON data.
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(source_blob)
            content = blob.download_as_text()
            return json.loads(content)
        except GoogleCloudError as e:
            logger.error(
                "Failed to download JSON from gs://%s/%s: %s",
                bucket_name,
                source_blob,
                e,
            )
            raise

    def list_blobs(self, bucket_name: str, prefix: str = "") -> list[str]:
        """List blobs in a GCS bucket with optional prefix filter.

        Args:
            bucket_name: GCS bucket name.
            prefix: Optional prefix to filter blobs.

        Returns:
            List of blob names.
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except GoogleCloudError as e:
            logger.error(
                "Failed to list blobs in gs://%s/%s: %s", bucket_name, prefix, e
            )
            raise


class BigQueryClient:
    """Wrapper around Google BigQuery client."""

    def __init__(self, project_id: str | None = None) -> None:
        settings = get_settings()
        self._project = project_id or settings.gcp.project_id
        self._dataset = settings.bigquery.dataset
        self._location = settings.bigquery.location
        self._client = bigquery.Client(project=self._project, location=self._location)

    def query(
        self, sql: str, params: list[bigquery.ScalarQueryParameter] | None = None
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string.
            params: Optional query parameters.

        Returns:
            Query results as a pandas DataFrame.
        """
        try:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = params
            query_job = self._client.query(sql, job_config=job_config)
            result = query_job.result()
            logger.info("Query completed: %d rows returned", result.total_rows)
            return result.to_dataframe()
        except GoogleCloudError as e:
            logger.error("Query failed: %s", e)
            raise

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_id: str,
        write_disposition: str = "WRITE_APPEND",
        partition_field: str | None = None,
    ) -> None:
        """Load a pandas DataFrame into a BigQuery table.

        Args:
            df: DataFrame to load.
            table_id: Table name (without dataset prefix).
            write_disposition: WRITE_APPEND, WRITE_TRUNCATE, or WRITE_EMPTY.
            partition_field: Optional field for time partitioning.
        """
        full_table_id = f"{self._project}.{self._dataset}.{table_id}"
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition=write_disposition,
            )
            if partition_field:
                job_config.time_partitioning = bigquery.TimePartitioning(
                    field=partition_field
                )
            load_job = self._client.load_table_from_dataframe(
                df, full_table_id, job_config=job_config
            )
            load_job.result()
            logger.info("Loaded %d rows into %s", len(df), full_table_id)
        except GoogleCloudError as e:
            logger.error("Failed to load data into %s: %s", full_table_id, e)
            raise

    def table_exists(self, table_id: str) -> bool:
        """Check if a BigQuery table exists.

        Args:
            table_id: Table name (without dataset prefix).

        Returns:
            True if the table exists.
        """
        full_table_id = f"{self._project}.{self._dataset}.{table_id}"
        try:
            self._client.get_table(full_table_id)
            return True
        except Exception:
            return False
