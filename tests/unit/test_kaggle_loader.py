"""Unit tests for KaggleDataLoader and KaggleImporter.

Tests CSV loading, column mapping, ID generation, finish position
parsing, date filtering, race extraction, and import orchestration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data_collector.importers.kaggle_importer import KaggleImporter
from src.data_collector.kaggle_loader import (
    KaggleDataLoader,
    _generate_id,
    _parse_corner_positions,
    _parse_horse_weight,
    _parse_sex_age,
)
from src.data_collector.schemas import (
    HORSE_RESULT_SCHEMA,
    JOCKEY_RESULT_SCHEMA,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "kaggle_sample.csv"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGenerateId:
    """Tests for deterministic ID generation."""

    def test_deterministic(self) -> None:
        """Same input always produces same ID."""
        assert _generate_id("テストホースA") == _generate_id("テストホースA")

    def test_different_names_different_ids(self) -> None:
        """Different inputs produce different IDs."""
        assert _generate_id("テストホースA") != _generate_id("テストホースB")

    def test_returns_12_chars(self) -> None:
        """ID is exactly 12 hex characters."""
        result = _generate_id("SomeName")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)


class TestParseSexAge:
    """Tests for sex/age parsing."""

    def test_normal_male(self) -> None:
        sex, age = _parse_sex_age("牡3")
        assert sex == "牡"
        assert age == 3

    def test_normal_female(self) -> None:
        sex, age = _parse_sex_age("牝4")
        assert sex == "牝"
        assert age == 4

    def test_gelding(self) -> None:
        sex, age = _parse_sex_age("セ5")
        assert sex == "セ"
        assert age == 5

    def test_none_input(self) -> None:
        sex, age = _parse_sex_age(None)
        assert sex is None
        assert age is None

    def test_empty_string(self) -> None:
        sex, age = _parse_sex_age("")
        assert sex is None
        assert age is None


class TestParseCornerPositions:
    """Tests for corner position parsing."""

    def test_four_positions(self) -> None:
        result = _parse_corner_positions("3-3-2-1")
        assert result == (3, 3, 2, 1)

    def test_two_positions(self) -> None:
        result = _parse_corner_positions("1-1")
        assert result == (1, 1, None, None)

    def test_none_input(self) -> None:
        result = _parse_corner_positions(None)
        assert result == (None, None, None, None)

    def test_empty_string(self) -> None:
        result = _parse_corner_positions("")
        assert result == (None, None, None, None)


class TestParseHorseWeight:
    """Tests for horse weight parsing."""

    def test_with_positive_change(self) -> None:
        weight, change = _parse_horse_weight("480(+4)")
        assert weight == 480.0
        assert change == 4.0

    def test_with_negative_change(self) -> None:
        weight, change = _parse_horse_weight("460(-2)")
        assert weight == 460.0
        assert change == -2.0

    def test_with_zero_change(self) -> None:
        weight, change = _parse_horse_weight("470(0)")
        assert weight == 470.0
        assert change == 0.0

    def test_no_change(self) -> None:
        weight, change = _parse_horse_weight("480")
        assert weight == 480.0
        assert change is None

    def test_none_input(self) -> None:
        weight, change = _parse_horse_weight(None)
        assert weight is None
        assert change is None


# ---------------------------------------------------------------------------
# KaggleDataLoader tests
# ---------------------------------------------------------------------------


class TestKaggleDataLoader:
    """Tests for KaggleDataLoader CSV loading and transformation."""

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_load_race_results(self, mock_settings: MagicMock) -> None:
        """Loads sample CSV and returns a non-empty DataFrame."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        assert not df.is_empty()
        assert len(df) == 20

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_column_mapping(self, mock_settings: MagicMock) -> None:
        """Japanese columns are mapped to internal English names."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        # Core columns should be present
        assert "race_id" in df.columns
        assert "horse_id" in df.columns
        assert "jockey_id" in df.columns
        assert "finish_position" in df.columns
        assert "course" in df.columns
        assert "distance" in df.columns
        assert "race_date" in df.columns

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_horse_id_deterministic(self, mock_settings: MagicMock) -> None:
        """horse_id is deterministic for the same horse name."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        # テストホースA appears in row 0 and row 8 (two different races)
        horse_a_rows = df.filter(pl.col("horse_name") == "テストホースA")
        assert len(horse_a_rows) == 2
        ids = horse_a_rows["horse_id"].unique()
        assert len(ids) == 1  # Same name → same ID

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_jockey_id_deterministic(self, mock_settings: MagicMock) -> None:
        """jockey_id is deterministic for the same jockey name."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        # テスト騎手A appears in multiple races
        jockey_a_rows = df.filter(pl.col("jockey_name") == "テスト騎手A")
        assert len(jockey_a_rows) >= 2
        ids = jockey_a_rows["jockey_id"].unique()
        assert len(ids) == 1

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_finish_position_parse_numeric(self, mock_settings: MagicMock) -> None:
        """Numeric finish positions are parsed correctly."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        # First row: finish_position = 1
        first_row = df.row(0, named=True)
        assert first_row["finish_position"] == 1

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_finish_position_non_numeric_to_null(
        self, mock_settings: MagicMock
    ) -> None:
        """Non-numeric finish positions (取消, 除外, 中止, 失格) become null."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        # Row 3 has 取消, row 6 has 除外, row 11 has 中止, row 13 has 失格
        null_count = df["finish_position"].null_count()
        assert null_count == 4

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_date_filter_from(self, mock_settings: MagicMock) -> None:
        """date_from filter works correctly."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results(date_from="2021-02-01")

        # Should exclude Jan races (7 rows), include Feb and Mar
        assert len(df) == 13

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_date_filter_to(self, mock_settings: MagicMock) -> None:
        """date_to filter works correctly."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results(date_to="2021-01-31")

        # Should include only Jan races (7 rows)
        assert len(df) == 7

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_date_filter_range(self, mock_settings: MagicMock) -> None:
        """Combined date_from + date_to filter."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results(date_from="2021-02-01", date_to="2021-02-28")

        # Should include only Feb races (7 rows)
        assert len(df) == 7

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_sex_age_parsed(self, mock_settings: MagicMock) -> None:
        """sex_age column is split into sex and age."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        assert "sex" in df.columns
        assert "age" in df.columns
        assert "sex_age" not in df.columns

        first_row = df.row(0, named=True)
        assert first_row["sex"] == "牡"
        assert first_row["age"] == 3

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_corner_positions_parsed(self, mock_settings: MagicMock) -> None:
        """Corner positions string is split into 4 columns."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        assert "corner_position_1" in df.columns
        assert "corner_position_4" in df.columns
        assert "corner_positions" not in df.columns

        first_row = df.row(0, named=True)
        assert first_row["corner_position_1"] == 3
        assert first_row["corner_position_4"] == 1

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_weight_parsed(self, mock_settings: MagicMock) -> None:
        """Horse weight string is parsed into weight and change."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        assert "weight" in df.columns
        assert "horse_weight_change" in df.columns
        assert "horse_weight" not in df.columns

        first_row = df.row(0, named=True)
        assert first_row["weight"] == 480.0
        assert first_row["horse_weight_change"] == 4.0

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_file_not_found_raises(self, mock_settings: MagicMock) -> None:
        """Raises FileNotFoundError when CSV doesn't exist."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "nonexistent.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        with pytest.raises(FileNotFoundError):
            loader.load_race_results()


# ---------------------------------------------------------------------------
# KaggleImporter tests
# ---------------------------------------------------------------------------


class TestKaggleImporter:
    """Tests for KaggleImporter orchestration."""

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_run_extracts_all_views(self, mock_settings: MagicMock) -> None:
        """Import run produces correct counts for all three views."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        importer = KaggleImporter(loader=loader)

        result = importer.run()

        # 6 unique races in sample
        assert result.races_count == 6
        # 20 total rows = 20 horse results
        assert result.horse_results_count == 20
        # 20 jockey result rows
        assert result.jockey_results_count == 20

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_run_with_date_filter(self, mock_settings: MagicMock) -> None:
        """Date filters are passed through to the loader."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        importer = KaggleImporter(loader=loader)

        result = importer.run(date_from="2021-02-01", date_to="2021-02-28")

        # Feb has 2 races (4 and 3 entries)
        assert result.races_count == 2
        assert result.horse_results_count == 7

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_run_calls_gcs_writer(self, mock_settings: MagicMock) -> None:
        """GCS writer is called for each data type."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        mock_gcs = MagicMock()
        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        importer = KaggleImporter(loader=loader, gcs_writer=mock_gcs)

        importer.run()

        # Should write races, horse_results, jockey_results for each year
        assert mock_gcs.write_parquet.call_count >= 3

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_run_calls_bq_writer(self, mock_settings: MagicMock) -> None:
        """BigQuery writer is called with WRITE_TRUNCATE."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        mock_bq = MagicMock()
        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        importer = KaggleImporter(loader=loader, bq_writer=mock_bq)

        importer.run()

        # Should write 3 tables per year batch
        assert mock_bq.write.call_count >= 3
        # Check WRITE_TRUNCATE is used
        for call in mock_bq.write.call_args_list:
            assert call.kwargs.get("write_disposition") == "WRITE_TRUNCATE" or (
                len(call.args) >= 3 and call.args[2] == "WRITE_TRUNCATE"
            )

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_race_deduplication(self, mock_settings: MagicMock) -> None:
        """Races are deduplicated by race_id."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        importer = KaggleImporter(loader=loader)
        races = importer._extract_races(df)

        # 6 unique race IDs in the sample
        assert len(races) == 6
        assert races["race_id"].n_unique() == 6

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_race_num_entries(self, mock_settings: MagicMock) -> None:
        """num_entries is correctly computed per race."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        importer = KaggleImporter(loader=loader)
        races = importer._extract_races(df)

        assert "num_entries" in races.columns
        # First race (202101010101) has 4 entries
        race1 = races.filter(pl.col("race_id") == "202101010101")
        assert race1["num_entries"][0] == 4

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_horse_result_schema_columns(self, mock_settings: MagicMock) -> None:
        """Horse results contain expected schema columns."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        importer = KaggleImporter(loader=loader)
        hr = importer._extract_horse_results(df)

        for col in HORSE_RESULT_SCHEMA:
            if col in df.columns:
                assert col in hr.columns

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_jockey_result_schema_columns(self, mock_settings: MagicMock) -> None:
        """Jockey results contain expected schema columns."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        df = loader.load_race_results()

        importer = KaggleImporter(loader=loader)
        jr = importer._extract_jockey_results(df)

        for col in JOCKEY_RESULT_SCHEMA:
            if col in df.columns:
                assert col in jr.columns

    @patch("src.data_collector.kaggle_loader.get_settings")
    def test_empty_data_returns_zero_counts(self, mock_settings: MagicMock) -> None:
        """Import with no matching data returns zero counts."""
        settings = MagicMock()
        settings.kaggle.data_dir = str(FIXTURES_DIR)
        settings.kaggle.race_result_file = "kaggle_sample.csv"
        mock_settings.return_value = settings

        loader = KaggleDataLoader(data_dir=FIXTURES_DIR)
        importer = KaggleImporter(loader=loader)

        # Filter to a date range with no data
        result = importer.run(date_from="2030-01-01", date_to="2030-12-31")

        assert result.races_count == 0
        assert result.horse_results_count == 0
        assert result.jockey_results_count == 0
