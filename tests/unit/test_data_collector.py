"""Unit tests for the data collector module.

Tests JRA client, collectors, validators, and storage writers
using mocks for all external API and GCP calls.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data_collector.collectors.horse_collector import HorseCollector
from src.data_collector.collectors.jockey_collector import JockeyCollector
from src.data_collector.collectors.race_collector import RaceCollector
from src.data_collector.jra_client import (
    JRAAuthError,
    JRAClient,
    JRAClientError,
)
from src.data_collector.storage.bq_writer import BQWriter
from src.data_collector.storage.gcs_writer import GCSWriter
from src.data_collector.validators.data_validator import DataValidator

# ---------------------------------------------------------------------------
# JRAClient tests
# ---------------------------------------------------------------------------


class TestJRAClient:
    """Tests for JRAClient request handling."""

    @patch("src.data_collector.jra_client.httpx.Client")
    @patch("src.data_collector.jra_client.get_settings")
    def test_get_request_success(self, mock_settings, mock_http_cls):
        settings = MagicMock()
        settings.jra_api_key = "test-key"
        mock_settings.return_value = settings

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"races": [{"race_id": "R001"}]}

        mock_http = MagicMock()
        mock_http.request.return_value = mock_response
        mock_http_cls.return_value = mock_http

        client = JRAClient(api_key="test-key")
        result = client.get("/races", params={"date": "2024-01-01"})

        assert result == {"races": [{"race_id": "R001"}]}
        mock_http.request.assert_called_once()

    @patch("src.data_collector.jra_client.httpx.Client")
    @patch("src.data_collector.jra_client.get_settings")
    def test_auth_error_raises(self, mock_settings, mock_http_cls):
        settings = MagicMock()
        settings.jra_api_key = "bad-key"
        mock_settings.return_value = settings

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_http = MagicMock()
        mock_http.request.return_value = mock_response
        mock_http_cls.return_value = mock_http

        client = JRAClient(api_key="bad-key")
        with pytest.raises(JRAAuthError):
            client.get("/races")

    @patch("src.data_collector.jra_client.httpx.Client")
    @patch("src.data_collector.jra_client.get_settings")
    def test_get_races(self, mock_settings, mock_http_cls):
        settings = MagicMock()
        settings.jra_api_key = "test-key"
        mock_settings.return_value = settings

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "races": [
                {"race_id": "R001", "race_name": "Test Race"},
                {"race_id": "R002", "race_name": "Test Race 2"},
            ]
        }

        mock_http = MagicMock()
        mock_http.request.return_value = mock_response
        mock_http_cls.return_value = mock_http

        client = JRAClient(api_key="test-key")
        races = client.get_races(date="2024-01-01")

        assert len(races) == 2
        assert races[0]["race_id"] == "R001"

    @patch("src.data_collector.jra_client.httpx.Client")
    @patch("src.data_collector.jra_client.get_settings")
    def test_server_error_raises_client_error(self, mock_settings, mock_http_cls):
        settings = MagicMock()
        settings.jra_api_key = "test-key"
        mock_settings.return_value = settings

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http = MagicMock()
        mock_http.request.return_value = mock_response
        mock_http_cls.return_value = mock_http

        client = JRAClient(api_key="test-key")
        with pytest.raises(JRAClientError):
            client.get("/races")


# ---------------------------------------------------------------------------
# DataValidator tests
# ---------------------------------------------------------------------------


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_data_no_errors(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001", "R002"],
                "race_name": ["Race 1", "Race 2"],
            }
        )
        validator = DataValidator(required_columns=["race_id", "race_name"])
        errors = validator.validate(df)
        assert len(errors) == 0

    def test_missing_column(self):
        df = pl.DataFrame({"race_id": ["R001"]})
        validator = DataValidator(required_columns=["race_id", "race_name"])
        errors = validator.validate(df)

        error_types = [e.error_type for e in errors]
        assert "missing_column" in error_types

    def test_high_null_rate(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001", None, None, None, None],
            }
        )
        validator = DataValidator(
            required_columns=["race_id"],
            max_null_rate=0.5,
        )
        errors = validator.validate(df)

        error_types = [e.error_type for e in errors]
        assert "high_null_rate" in error_types

    def test_empty_dataframe(self):
        df = pl.DataFrame({"race_id": []})
        validator = DataValidator(required_columns=["race_id"])
        errors = validator.validate(df)

        error_types = [e.error_type for e in errors]
        assert "empty_dataframe" in error_types

    def test_null_rate_within_threshold(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001", "R002", None],
            }
        )
        validator = DataValidator(
            required_columns=["race_id"],
            max_null_rate=0.5,
        )
        errors = validator.validate(df)
        # 1/3 = 0.33 < 0.5 threshold, should not trigger
        error_types = [e.error_type for e in errors]
        assert "high_null_rate" not in error_types


# ---------------------------------------------------------------------------
# RaceCollector tests
# ---------------------------------------------------------------------------


class TestRaceCollector:
    """Tests for RaceCollector."""

    def test_collect_by_date_returns_dataframe(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_races.return_value = [
            {"race_id": "R001"},
            {"race_id": "R002"},
        ]
        mock_client.get_race_detail.side_effect = [
            {
                "race_id": "R001",
                "race_date": "2024-01-01",
                "race_name": "New Year Race",
                "race_number": 1,
                "course": "Tokyo",
                "distance": 1600,
                "track_type": "Turf",
                "track_condition": "Good",
                "weather": "Sunny",
                "grade": "G1",
                "entries": [{"horse_id": "H001"}, {"horse_id": "H002"}],
            },
            {
                "race_id": "R002",
                "race_date": "2024-01-01",
                "race_name": "Second Race",
                "race_number": 2,
                "course": "Tokyo",
                "distance": 2000,
                "track_type": "Turf",
                "track_condition": "Good",
                "weather": "Sunny",
                "grade": "",
                "entries": [],
            },
        ]

        collector = RaceCollector(client=mock_client)
        result = collector.collect_by_date(date(2024, 1, 1))

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "race_id" in result.columns
        assert "distance" in result.columns

    def test_collect_by_date_empty(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_races.return_value = []

        collector = RaceCollector(client=mock_client)
        result = collector.collect_by_date(date(2024, 1, 1))

        assert result.is_empty()


# ---------------------------------------------------------------------------
# HorseCollector tests
# ---------------------------------------------------------------------------


class TestHorseCollector:
    """Tests for HorseCollector."""

    def test_collect_horse_results(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_horse_results.return_value = [
            {
                "race_id": "R001",
                "race_date": "2024-01-01",
                "course": "Tokyo",
                "distance": 1600,
                "track_condition": "Good",
                "finish_position": 1,
                "time": "1:33.5",
                "weight": 480,
                "jockey_id": "J001",
            },
        ]

        collector = HorseCollector(client=mock_client)
        result = collector.collect_horse_results("H001")

        assert len(result) == 1
        assert result["horse_id"][0] == "H001"
        assert result["finish_position"][0] == 1

    def test_collect_horse_results_empty(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_horse_results.return_value = []

        collector = HorseCollector(client=mock_client)
        result = collector.collect_horse_results("H999")

        assert result.is_empty()


# ---------------------------------------------------------------------------
# JockeyCollector tests
# ---------------------------------------------------------------------------


class TestJockeyCollector:
    """Tests for JockeyCollector."""

    def test_collect_jockey_results(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_jockey_results.return_value = [
            {
                "race_id": "R001",
                "race_date": "2024-01-01",
                "course": "Tokyo",
                "distance": 1600,
                "horse_id": "H001",
                "finish_position": 2,
            },
        ]

        collector = JockeyCollector(client=mock_client)
        result = collector.collect_jockey_results("J001")

        assert len(result) == 1
        assert result["jockey_id"][0] == "J001"

    def test_collect_jockeys_from_race_deduplicates(self):
        mock_client = MagicMock(spec=JRAClient)
        mock_client.get_jockey_results.return_value = [
            {
                "race_id": "R001",
                "race_date": "2024-01-01",
                "course": "Tokyo",
                "distance": 1600,
                "horse_id": "H001",
                "finish_position": 1,
            },
        ]

        collector = JockeyCollector(client=mock_client, bq_writer=None, gcs_writer=None)
        entries = [
            {"jockey_id": "J001"},
            {"jockey_id": "J001"},  # duplicate
            {"jockey_id": "J002"},
        ]
        result = collector.collect_jockeys_from_race(entries, save=False)

        # J001 and J002 (deduplicated), each with 1 result row
        assert len(result) == 2
        assert mock_client.get_jockey_results.call_count == 2


# ---------------------------------------------------------------------------
# GCSWriter tests
# ---------------------------------------------------------------------------


class TestGCSWriter:
    """Tests for GCSWriter."""

    @patch("src.data_collector.storage.gcs_writer.get_settings")
    @patch("src.data_collector.storage.gcs_writer.GCSClient")
    def test_write_parquet(self, mock_gcs_cls, mock_settings):
        settings = MagicMock()
        settings.gcs.bucket_raw = "test-bucket"
        mock_settings.return_value = settings

        mock_gcs = MagicMock()
        mock_gcs.upload_file.return_value = (
            "gs://test-bucket/races/2024/01/01/races.parquet"
        )
        mock_gcs_cls.return_value = mock_gcs

        writer = GCSWriter()
        df = pl.DataFrame({"race_id": ["R001"], "race_date": ["2024-01-01"]})
        uri = writer.write_parquet(df, "races", "2024-01-01")

        assert uri.startswith("gs://")
        mock_gcs.upload_file.assert_called_once()

    @patch("src.data_collector.storage.gcs_writer.get_settings")
    @patch("src.data_collector.storage.gcs_writer.GCSClient")
    def test_write_parquet_empty_skipped(self, mock_gcs_cls, mock_settings):
        settings = MagicMock()
        settings.gcs.bucket_raw = "test-bucket"
        mock_settings.return_value = settings
        mock_gcs_cls.return_value = MagicMock()

        writer = GCSWriter()
        result = writer.write_parquet(pl.DataFrame(), "races", "2024-01-01")

        assert result == ""


# ---------------------------------------------------------------------------
# BQWriter tests
# ---------------------------------------------------------------------------


class TestBQWriter:
    """Tests for BQWriter."""

    @patch("src.data_collector.storage.bq_writer.BigQueryClient")
    def test_write_to_bigquery(self, mock_bq_cls):
        mock_bq = MagicMock()
        mock_bq_cls.return_value = mock_bq

        writer = BQWriter()
        df = pl.DataFrame({"race_id": ["R001"], "value": [100]})
        writer.write(df, "races_raw")

        mock_bq.load_dataframe.assert_called_once()
        call_args = mock_bq.load_dataframe.call_args
        assert call_args[0][1] == "races_raw"

    @patch("src.data_collector.storage.bq_writer.BigQueryClient")
    def test_write_empty_skipped(self, mock_bq_cls):
        mock_bq = MagicMock()
        mock_bq_cls.return_value = mock_bq

        writer = BQWriter()
        writer.write(pl.DataFrame(), "races_raw")

        mock_bq.load_dataframe.assert_not_called()
