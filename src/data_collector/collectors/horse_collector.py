"""Horse data collector.

Fetches horse profiles and past performance data from the JRA API.
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

_HORSE_REQUIRED_COLUMNS = ["horse_id", "horse_name"]


class HorseCollector:
    """Collects horse information from JRA API.

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
        self._validator = DataValidator(required_columns=_HORSE_REQUIRED_COLUMNS)

    def collect_horse(self, horse_id: str) -> dict[str, Any]:
        """Collect profile data for a single horse.

        Args:
            horse_id: Unique horse identifier.

        Returns:
            Horse profile dictionary.
        """
        logger.info("Collecting horse", horse_id=horse_id)
        return self._client.get_horse(horse_id)

    def collect_horse_results(self, horse_id: str, limit: int = 20) -> pl.DataFrame:
        """Collect past race results for a horse.

        Args:
            horse_id: Unique horse identifier.
            limit: Maximum number of results.

        Returns:
            DataFrame of past results.
        """
        results = self._client.get_horse_results(horse_id, limit=limit)
        if not results:
            return pl.DataFrame()

        records: list[dict[str, Any]] = []
        for r in results:
            records.append(
                {
                    "horse_id": horse_id,
                    "race_id": r.get("race_id", ""),
                    "race_date": r.get("race_date", ""),
                    "course": r.get("course", ""),
                    "distance": r.get("distance", 0),
                    "track_condition": r.get("track_condition", ""),
                    "finish_position": r.get("finish_position", 0),
                    "time": r.get("time", ""),
                    "weight": r.get("weight", 0),
                    "jockey_id": r.get("jockey_id", ""),
                }
            )
        return pl.DataFrame(records)

    def collect_horses_from_race(
        self,
        entries: list[dict[str, Any]],
        results_limit: int = 10,
        save: bool = True,
    ) -> pl.DataFrame:
        """Collect horse data for all entries in a race.

        Args:
            entries: List of entry dicts (must contain 'horse_id').
            results_limit: Max past results per horse.
            save: Whether to persist collected data.

        Returns:
            Combined DataFrame of all horses' past results.
        """
        all_frames: list[pl.DataFrame] = []
        for entry in entries:
            horse_id = entry.get("horse_id", "")
            if not horse_id:
                continue
            df = self.collect_horse_results(horse_id, limit=results_limit)
            if not df.is_empty():
                all_frames.append(df)

        if not all_frames:
            return pl.DataFrame()

        combined = pl.concat(all_frames)

        if save:
            self._save(combined)

        return combined

    def _save(self, df: pl.DataFrame) -> None:
        """Persist horse data to storage backends.

        Args:
            df: Horse results DataFrame.
        """
        if df.is_empty():
            return

        if self._gcs:
            self._gcs.write_parquet(df, "horses", "results")
            logger.info("Saved horse results to GCS", count=len(df))

        if self._bq:
            self._bq.write(df, "horse_results_raw")
            logger.info("Saved horse results to BigQuery", count=len(df))
