"""Race data collector.

Fetches race information from the JRA API, validates it,
and writes to configured storage backends.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from src.common.logging import get_logger
from src.data_collector.jra_client import JRAClient
from src.data_collector.storage.bq_writer import BQWriter
from src.data_collector.storage.gcs_writer import GCSWriter
from src.data_collector.validators.data_validator import DataValidator

logger = get_logger(__name__)

# Required columns for race data
_RACE_REQUIRED_COLUMNS = [
    "race_id",
    "race_date",
    "race_name",
    "course",
    "distance",
    "track_condition",
    "weather",
]


class RaceCollector:
    """Collects race information from JRA API and persists it.

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
        self._validator = DataValidator(required_columns=_RACE_REQUIRED_COLUMNS)

    def collect_by_date(self, target_date: date) -> pl.DataFrame:
        """Collect all races for a single date.

        Args:
            target_date: The date to fetch races for.

        Returns:
            DataFrame containing race data for the date.
        """
        date_str = target_date.isoformat()
        logger.info("Collecting races", date=date_str)

        raw_races = self._client.get_races(date=date_str)
        if not raw_races:
            logger.info("No races found", date=date_str)
            return pl.DataFrame()

        # Fetch detailed info for each race
        detailed_races: list[dict[str, Any]] = []
        for race_summary in raw_races:
            race_id = race_summary.get("race_id", "")
            if not race_id:
                continue
            detail = self._client.get_race_detail(race_id)
            detailed_races.append(detail)

        df = self._normalize_races(detailed_races)
        logger.info("Collected races", date=date_str, count=len(df))
        return df

    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
        save: bool = True,
    ) -> pl.DataFrame:
        """Collect races over a date range.

        Args:
            start_date: First date (inclusive).
            end_date: Last date (inclusive).
            save: Whether to persist collected data.

        Returns:
            Combined DataFrame of all races in range.
        """
        logger.info(
            "Collecting race range",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        all_frames: list[pl.DataFrame] = []
        current = start_date
        while current <= end_date:
            df = self.collect_by_date(current)
            if not df.is_empty():
                all_frames.append(df)
            current += timedelta(days=1)

        if not all_frames:
            logger.info("No races found in date range")
            return pl.DataFrame()

        combined = pl.concat(all_frames)

        if save:
            self._save(combined)

        return combined

    def _normalize_races(self, raw_races: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert raw API responses to a normalized DataFrame.

        Args:
            raw_races: List of race detail dictionaries.

        Returns:
            Normalized Polars DataFrame.
        """
        records: list[dict[str, Any]] = []
        for race in raw_races:
            entries = race.get("entries", [])
            num_entries = len(entries) if isinstance(entries, list) else 0

            record: dict[str, Any] = {
                "race_id": race.get("race_id", ""),
                "race_date": race.get("race_date", ""),
                "race_name": race.get("race_name", ""),
                "race_number": race.get("race_number", 0),
                "course": race.get("course", ""),
                "distance": race.get("distance", 0),
                "track_type": race.get("track_type", ""),
                "track_condition": race.get("track_condition", ""),
                "weather": race.get("weather", ""),
                "grade": race.get("grade", ""),
                "num_entries": num_entries,
            }
            records.append(record)

        if not records:
            return pl.DataFrame()

        df = pl.DataFrame(records)

        # Validate
        errors = self._validator.validate(df)
        if errors:
            logger.warning("Validation issues in race data", errors=errors)

        return df

    def _save(self, df: pl.DataFrame) -> None:
        """Persist race DataFrame to configured storage backends.

        Args:
            df: Race DataFrame to save.
        """
        if df.is_empty():
            return

        if self._gcs:
            dates = df["race_date"].unique().to_list()
            for d in dates:
                day_df = df.filter(pl.col("race_date") == d)
                self._gcs.write_parquet(day_df, "races", str(d))
            logger.info("Saved races to GCS", count=len(df))

        if self._bq:
            self._bq.write(df, "races_raw", partition_field="race_date")
            logger.info("Saved races to BigQuery", count=len(df))
