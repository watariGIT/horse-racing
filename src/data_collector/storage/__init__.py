"""Storage writers for GCS and BigQuery."""

from src.data_collector.storage.bq_writer import BQWriter
from src.data_collector.storage.gcs_writer import GCSWriter

__all__ = ["GCSWriter", "BQWriter"]
