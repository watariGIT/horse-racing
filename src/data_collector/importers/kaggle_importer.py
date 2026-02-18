"""Kaggle dataset importer for batch ingestion into GCS and BigQuery.

Orchestrates the full import pipeline: load CSV → extract views
(races, horse_results, jockey_results) → validate → persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from src.common.logging import get_logger
from src.data_collector.kaggle_loader import KaggleDataLoader
from src.data_collector.schemas import (
    EXTENDED_HORSE_RESULT_SCHEMA,
    JOCKEY_RESULT_SCHEMA,
    RACE_SCHEMA,
)
from src.data_collector.storage.bq_writer import BQWriter
from src.data_collector.storage.gcs_writer import GCSWriter
from src.data_collector.validators.data_validator import DataValidator

logger = get_logger(__name__)


@dataclass
class ImportResult:
    """Summary of a Kaggle import run.

    Attributes:
        races_count: Number of unique races imported.
        horse_results_count: Number of horse result rows imported.
        jockey_results_count: Number of jockey result rows imported.
        validation_errors: List of validation error messages.
    """

    races_count: int = 0
    horse_results_count: int = 0
    jockey_results_count: int = 0
    validation_errors: list[str] = field(default_factory=list)


class KaggleImporter:
    """Imports Kaggle JRA dataset into GCS and BigQuery.

    Extracts three normalized views from the denormalized CSV:
    - races: one row per race (deduplicated by race_id)
    - horse_results: one row per horse per race
    - jockey_results: one row per jockey per race

    Args:
        loader: KaggleDataLoader instance.
        gcs_writer: Optional GCSWriter for Parquet persistence.
        bq_writer: Optional BQWriter for BigQuery persistence.
    """

    def __init__(
        self,
        loader: KaggleDataLoader,
        gcs_writer: GCSWriter | None = None,
        bq_writer: BQWriter | None = None,
    ) -> None:
        self._loader = loader
        self._gcs_writer = gcs_writer
        self._bq_writer = bq_writer

    def _extract_races(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract unique race records from the denormalized data.

        Groups by race_id to deduplicate and counts entries per race.

        Args:
            df: Full denormalized DataFrame.

        Returns:
            DataFrame with one row per race.
        """
        race_cols = [c for c in RACE_SCHEMA if c != "num_entries" and c in df.columns]

        if "race_id" not in df.columns:
            logger.warning("race_id column missing, returning empty races")
            return pl.DataFrame(schema=RACE_SCHEMA)

        # Count entries per race
        entry_counts = df.group_by("race_id").agg(pl.len().alias("num_entries"))

        # Get unique race records
        races = df.select(race_cols).unique(subset=["race_id"])

        # Join entry counts
        races = races.join(entry_counts, on="race_id", how="left")

        return races

    def _extract_horse_results(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract horse result records.

        Each row in the original CSV is already one horse per race.

        Args:
            df: Full denormalized DataFrame.

        Returns:
            DataFrame with horse result columns.
        """
        hr_cols = [c for c in EXTENDED_HORSE_RESULT_SCHEMA if c in df.columns]
        if not hr_cols:
            return pl.DataFrame(schema=EXTENDED_HORSE_RESULT_SCHEMA)
        return df.select(hr_cols)

    def _extract_jockey_results(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract jockey result records.

        Args:
            df: Full denormalized DataFrame.

        Returns:
            DataFrame with jockey result columns.
        """
        jr_cols = [c for c in JOCKEY_RESULT_SCHEMA if c in df.columns]
        if not jr_cols:
            return pl.DataFrame(schema=JOCKEY_RESULT_SCHEMA)
        return df.select(jr_cols)

    def _validate(
        self,
        df: pl.DataFrame,
        name: str,
        required_columns: list[str],
    ) -> list[str]:
        """Validate a DataFrame and return error messages.

        Args:
            df: DataFrame to validate.
            name: Human-readable name for logging.
            required_columns: Columns that must be present.

        Returns:
            List of validation error message strings.
        """
        validator = DataValidator(
            required_columns=required_columns,
            max_null_rate=0.5,
        )
        errors = validator.validate(df)
        if errors:
            for err in errors:
                logger.warning(
                    "Validation error",
                    dataset=name,
                    column=err.column,
                    error_type=err.error_type,
                    message=err.message,
                )
        return [f"[{name}] {e.message}" for e in errors]

    def _persist_batch(
        self,
        races: pl.DataFrame,
        horse_results: pl.DataFrame,
        jockey_results: pl.DataFrame,
        partition_key: str,
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> None:
        """Persist extracted DataFrames to GCS and BigQuery.

        Args:
            races: Races DataFrame.
            horse_results: Horse results DataFrame.
            jockey_results: Jockey results DataFrame.
            partition_key: Partition key for GCS storage.
            write_disposition: BigQuery write disposition.
        """
        if self._gcs_writer:
            self._gcs_writer.write_parquet(races, "races", partition_key)
            self._gcs_writer.write_parquet(
                horse_results, "horse_results", partition_key
            )
            self._gcs_writer.write_parquet(
                jockey_results, "jockey_results", partition_key
            )

        if self._bq_writer:
            self._bq_writer.write(
                races,
                "races_raw",
                write_disposition=write_disposition,
                partition_field="race_date",
            )
            self._bq_writer.write(
                horse_results,
                "horse_results_raw",
                write_disposition=write_disposition,
                partition_field="race_date",
            )
            self._bq_writer.write(
                jockey_results,
                "jockey_results_raw",
                write_disposition=write_disposition,
                partition_field="race_date",
            )

    def run(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        batch_size: int = 10000,
    ) -> ImportResult:
        """Execute the full import pipeline.

        Loads CSV data, extracts normalized views, validates,
        and persists to storage in batches.

        Args:
            date_from: Optional start date filter (YYYY-MM-DD).
            date_to: Optional end date filter (YYYY-MM-DD).
            batch_size: Number of rows per processing batch.

        Returns:
            ImportResult with counts and validation summary.
        """
        result = ImportResult()

        logger.info(
            "Starting Kaggle import",
            date_from=date_from,
            date_to=date_to,
        )

        # Load full dataset
        df = self._loader.load_race_results(
            date_from=date_from,
            date_to=date_to,
        )

        if df.is_empty():
            logger.warning("No data loaded from Kaggle CSV")
            result.validation_errors.append("No data loaded")
            return result

        # Extract normalized views
        races = self._extract_races(df)
        horse_results = self._extract_horse_results(df)
        jockey_results = self._extract_jockey_results(df)

        logger.info(
            "Extracted views",
            races=len(races),
            horse_results=len(horse_results),
            jockey_results=len(jockey_results),
        )

        # Validate
        race_required = [
            c for c in ["race_id", "race_date", "course"] if c in races.columns
        ]
        hr_required = [c for c in ["horse_id", "race_id"] if c in horse_results.columns]
        jr_required = [
            c for c in ["jockey_id", "race_id"] if c in jockey_results.columns
        ]

        result.validation_errors.extend(self._validate(races, "races", race_required))
        result.validation_errors.extend(
            self._validate(horse_results, "horse_results", hr_required)
        )
        result.validation_errors.extend(
            self._validate(jockey_results, "jockey_results", jr_required)
        )

        # Persist in batches by year for memory efficiency
        if "race_date" in df.columns and not df["race_date"].is_null().all():
            years = (
                df.select(pl.col("race_date").dt.year().alias("year"))
                .unique()
                .sort("year")
                .to_series()
                .to_list()
            )

            first_batch = True
            for year in years:
                if year is None:
                    continue
                year_filter = pl.col("race_date").dt.year() == year

                year_races = races.filter(year_filter)
                year_hr = horse_results.filter(year_filter)
                year_jr = jockey_results.filter(year_filter)

                partition_key = f"kaggle/{year}"
                disposition = "WRITE_TRUNCATE" if first_batch else "WRITE_APPEND"
                self._persist_batch(
                    year_races, year_hr, year_jr, partition_key, disposition
                )
                first_batch = False

                logger.info(
                    "Persisted year batch",
                    year=year,
                    races=len(year_races),
                    horse_results=len(year_hr),
                    jockey_results=len(year_jr),
                )
        else:
            # No date info: persist all at once
            self._persist_batch(races, horse_results, jockey_results, "kaggle/all")

        result.races_count = len(races)
        result.horse_results_count = len(horse_results)
        result.jockey_results_count = len(jockey_results)

        logger.info(
            "Kaggle import completed",
            races=result.races_count,
            horse_results=result.horse_results_count,
            jockey_results=result.jockey_results_count,
            errors=len(result.validation_errors),
        )

        return result
