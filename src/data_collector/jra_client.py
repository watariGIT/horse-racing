"""JRA API client with rate limiting and retry logic.

Handles authentication, request construction, and response parsing
for the JRA public data API.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

# API base URL
_BASE_URL = "https://api.jra.jp/v1"

# Rate limiting defaults
_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests


class JRAClientError(Exception):
    """Base error for JRA API client."""


class JRAAuthError(JRAClientError):
    """Authentication failure."""


class JRARateLimitError(JRAClientError):
    """Rate limit exceeded."""


class JRAClient:
    """Client for the JRA public data API.

    Manages authentication, rate limiting, and retries for all API requests.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.jra_api_key
        if not self._api_key:
            logger.warning("JRA API key not configured; requests will fail")
        self._last_request_time: float = 0.0
        self._client = httpx.Client(
            base_url=_BASE_URL,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> JRAClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, JRARateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make an API request with retry and rate-limit handling.

        Args:
            method: HTTP method (GET, POST).
            path: API endpoint path.
            params: Optional query parameters.

        Returns:
            Parsed JSON response body.

        Raises:
            JRAAuthError: On 401/403 responses.
            JRARateLimitError: On 429 responses (will be retried).
            JRAClientError: On other non-2xx responses.
        """
        self._throttle()

        logger.debug("API request", method=method, path=path, params=params)
        response = self._client.request(method, path, params=params)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            logger.warning("Rate limited, retry after %ds", retry_after)
            time.sleep(retry_after)
            raise JRARateLimitError("Rate limit exceeded")

        if response.status_code in (401, 403):
            raise JRAAuthError(
                f"Authentication failed: {response.status_code} {response.text}"
            )

        if response.status_code >= 400:
            raise JRAClientError(f"API error {response.status_code}: {response.text}")

        return response.json()  # type: ignore[no-any-return]

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GET request to the JRA API.

        Args:
            path: API endpoint path (e.g. "/races").
            params: Optional query parameters.

        Returns:
            Parsed JSON response.
        """
        return self._request("GET", path, params=params)

    def get_races(
        self,
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch race listings.

        Args:
            date: Single date (YYYY-MM-DD) to query.
            date_from: Start date for range query.
            date_to: End date for range query.

        Returns:
            List of race data dictionaries.
        """
        params: dict[str, Any] = {}
        if date:
            params["date"] = date
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        data = self.get("/races", params=params)
        return data.get("races", [])

    def get_race_detail(self, race_id: str) -> dict[str, Any]:
        """Fetch detailed information for a single race.

        Args:
            race_id: Unique race identifier.

        Returns:
            Race detail dictionary including entries.
        """
        return self.get(f"/races/{race_id}")

    def get_horse(self, horse_id: str) -> dict[str, Any]:
        """Fetch horse profile and past performance.

        Args:
            horse_id: Unique horse identifier.

        Returns:
            Horse data dictionary.
        """
        return self.get(f"/horses/{horse_id}")

    def get_horse_results(self, horse_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch past race results for a horse.

        Args:
            horse_id: Unique horse identifier.
            limit: Maximum number of results to return.

        Returns:
            List of past race result dictionaries.
        """
        data = self.get(f"/horses/{horse_id}/results", params={"limit": limit})
        return data.get("results", [])

    def get_jockey(self, jockey_id: str) -> dict[str, Any]:
        """Fetch jockey profile and statistics.

        Args:
            jockey_id: Unique jockey identifier.

        Returns:
            Jockey data dictionary.
        """
        return self.get(f"/jockeys/{jockey_id}")

    def get_jockey_results(
        self, jockey_id: str, year: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch race results for a jockey.

        Args:
            jockey_id: Unique jockey identifier.
            year: Optional year filter.

        Returns:
            List of jockey result dictionaries.
        """
        params: dict[str, Any] = {}
        if year:
            params["year"] = year
        data = self.get(f"/jockeys/{jockey_id}/results", params=params)
        return data.get("results", [])
