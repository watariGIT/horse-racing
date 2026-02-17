"""Unit tests for MLflow Cloud Run authentication plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.common.mlflow_auth import CloudRunRequestHeaderProvider


class TestCloudRunRequestHeaderProvider:
    """Tests for CloudRunRequestHeaderProvider."""

    def test_not_in_context_when_file_uri(self) -> None:
        provider = CloudRunRequestHeaderProvider()
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "file:./mlruns"}):
            assert provider.in_context() is False

    def test_not_in_context_when_empty(self) -> None:
        provider = CloudRunRequestHeaderProvider()
        with patch.dict("os.environ", {}, clear=True):
            assert provider.in_context() is False

    def test_in_context_when_https_uri(self) -> None:
        provider = CloudRunRequestHeaderProvider()
        with patch.dict(
            "os.environ",
            {"MLFLOW_TRACKING_URI": "https://mlflow-ui-dev-xyz.run.app"},
        ):
            assert provider.in_context() is True

    def test_request_headers_empty_for_file_uri(self) -> None:
        provider = CloudRunRequestHeaderProvider()
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "file:./mlruns"}):
            assert provider.request_headers() == {}

    @patch("google.oauth2.id_token.fetch_id_token")
    @patch("google.auth.transport.requests.Request")
    def test_request_headers_with_https_uri(
        self, mock_request_cls: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.return_value = "test-token-123"
        provider = CloudRunRequestHeaderProvider()
        uri = "https://mlflow-ui-dev-xyz.run.app"
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": uri}):
            headers = provider.request_headers()
            assert headers == {"Authorization": "Bearer test-token-123"}
            mock_fetch.assert_called_once_with(mock_request_cls.return_value, uri)

    @patch("google.oauth2.id_token.fetch_id_token")
    @patch("google.auth.transport.requests.Request")
    def test_request_headers_returns_empty_on_token_error(
        self, mock_request_cls: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.side_effect = Exception("credentials not found")
        provider = CloudRunRequestHeaderProvider()
        uri = "https://mlflow-ui-dev-xyz.run.app"
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": uri}):
            headers = provider.request_headers()
            assert headers == {}
