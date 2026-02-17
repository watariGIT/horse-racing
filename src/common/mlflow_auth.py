"""MLflow authentication for Cloud Run IAM-protected tracking servers.

Provides a RequestHeaderProvider plugin that automatically injects
GCP OIDC ID tokens into MLflow HTTP requests when the tracking URI
points to a Cloud Run service.
"""

from __future__ import annotations

import os

import structlog
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)

logger = structlog.get_logger(__name__)


class CloudRunRequestHeaderProvider(RequestHeaderProvider):
    """Inject Authorization header for IAM-protected Cloud Run MLflow servers.

    Active only when ``MLFLOW_TRACKING_URI`` starts with ``https://``.
    Uses Application Default Credentials to fetch a GCP ID token
    with the tracking URI as the audience.
    """

    @property
    def _tracking_uri(self) -> str:
        return os.environ.get("MLFLOW_TRACKING_URI", "")

    def in_context(self) -> bool:
        return self._tracking_uri.startswith("https://")

    def request_headers(self) -> dict[str, str]:
        uri = self._tracking_uri
        if not uri.startswith("https://"):
            return {}

        try:
            import google.auth.transport.requests
            import google.oauth2.id_token

            request = google.auth.transport.requests.Request()
            token = google.oauth2.id_token.fetch_id_token(request, uri)
            return {"Authorization": f"Bearer {token}"}
        except Exception:
            logger.warning("mlflow_auth_token_fetch_failed", tracking_uri=uri)
            return {}
