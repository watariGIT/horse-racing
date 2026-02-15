"""Jockey data collector.

Fetches jockey profiles and performance statistics from the JRA API.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from src.common.logging import get_logger
from src.data_collector.jra_client import JRAClient
from src.data_collector.storage.bq_writer import BQWriter
from src.data_collector.storage.gcs_writer import GCSWriter
from src.data_collector.validators.data_validator import DataValidator

logger = get_logger(__name__)

_JOCKEY_REQUIRED_COLUMNS = ["jockey_id", "jockey_name"]


class JockeyCollector:
    """Collects jockey information from JRA API.

    Args:
        client: JRA API client instance.
        gcs_writer: Optional GCS writer for raw data storage.
        bq_writer: Optional BigQuery writer for structured storage.
    """

    def __init__(
        self,
        client: JRAClient,
        gcs_writer: GCSWriter | None = None,
        bq_writer: BQWriter | None = None,
    ) -> None:
        self._client = client
        self._gcs = gcs_writer
        self._bq = bq_writer
        self._validator = DataValidator(required_columns=_JOCKEY_REQUIRED_COLUMNS)

    def collect_jockey(self, jockey_id: str) -> dict[str, Any]:
        """Collect profile data for a single jockey.

        Args:
            jockey_id: Unique jockey identifier.

        Returns:
            Jockey profile dictionary.
        """
        logger.info("Collecting jockey", jockey_id=jockey_id)
        return self._client.get_jockey(jockey_id)

    def collect_jockey_results(
        self, jockey_id: str, year: int | None = None
    ) -> pl.DataFrame:
        """Collect race results for a jockey.

        Args:
            jockey_id: Unique jockey identifier.
            year: Optional year filter.

        Returns:
            DataFrame of jockey race results.
        """
        results = self._client.get_jockey_results(jockey_id, year=year)
        if not results:
            return pl.DataFrame()

        records: list[dict[str, Any]] = []
        for r in results:
            records.append(
                {
                    "jockey_id": jockey_id,
                    "race_id": r.get("race_id", ""),
                    "race_date": r.get("race_date", ""),
                    "course": r.get("course", ""),
                    "distance": r.get("distance", 0),
                    "horse_id": r.get("horse_id", ""),
                    "finish_position": r.get("finish_position", 0),
                }
            )
        return pl.DataFrame(records)

    def collect_jockeys_from_race(
        self,
        entries: list[dict[str, Any]],
        year: int | None = None,
        save: bool = True,
    ) -> pl.DataFrame:
        """Collect jockey data for all entries in a race.

        Args:
            entries: List of entry dicts (must contain 'jockey_id').
            year: Optional year filter for results.
            save: Whether to persist collected data.

        Returns:
            Combined DataFrame of all jockeys' results.
        """
        seen: set[str] = set()
        all_frames: list[pl.DataFrame] = []

        for entry in entries:
            jockey_id = entry.get("jockey_id", "")
            if not jockey_id or jockey_id in seen:
                continue
            seen.add(jockey_id)

            df = self.collect_jockey_results(jockey_id, year=year)
            if not df.is_empty():
                all_frames.append(df)

        if not all_frames:
            return pl.DataFrame()

        combined = pl.concat(all_frames)

        if save:
            self._save(combined)

        return combined

    def _save(self, df: pl.DataFrame) -> None:
        """Persist jockey data to storage backends.

        Args:
            df: Jockey results DataFrame.
        """
        if df.is_empty():
            return

        if self._gcs:
            self._gcs.write_parquet(df, "jockeys", "results")
            logger.info("Saved jockey results to GCS", count=len(df))

        if self._bq:
            self._bq.write(df, "jockey_results_raw")
            logger.info("Saved jockey results to BigQuery", count=len(df))
