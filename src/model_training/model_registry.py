"""Model registry for versioned model storage on GCS.

Handles saving, loading, and versioning of trained models.
Models are stored as pickle files with date+hash versioning.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import get_settings
from src.common.gcp_client import GCSClient
from src.common.logging import get_logger
from src.model_training.models.base import BaseModel

logger = get_logger(__name__)

_REGISTRY_PREFIX = "models"
_METADATA_FILE = "metadata.json"


class ModelRegistry:
    """Versioned model storage backed by GCS.

    Stores models with date+hash version identifiers.
    Provides listing, loading, and latest-model retrieval.

    Args:
        gcs_client: Optional pre-configured GCS client.
        bucket_name: GCS bucket for model storage.
            Defaults to the configured models bucket.
    """

    def __init__(
        self,
        gcs_client: GCSClient | None = None,
        bucket_name: str | None = None,
    ) -> None:
        self._gcs = gcs_client or GCSClient()
        settings = get_settings()
        self._bucket = bucket_name or settings.gcs.bucket_models

    def save_model(
        self,
        model: BaseModel,
        model_name: str,
        metrics: dict[str, float] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a model to GCS with version metadata.

        Creates a versioned directory: models/{model_name}/{version}/
        containing the model pickle and a metadata.json.

        Args:
            model: Trained model instance.
            model_name: Logical model name (e.g. "win_classifier").
            metrics: Optional evaluation metrics to store.
            extra_metadata: Optional additional metadata.

        Returns:
            The version string (e.g. "20260215_abc123").
        """
        version = self._generate_version()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model.save(model_path)

            # Upload model file
            blob_path = f"{_REGISTRY_PREFIX}/{model_name}/{version}/model.pkl"
            self._gcs.upload_file(self._bucket, model_path, blob_path)

            # Upload metadata
            metadata = {
                "model_name": model_name,
                "model_type": model.model_type,
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "params": model.params,
                "metrics": metrics or {},
                **(extra_metadata or {}),
            }
            metadata_blob = (
                f"{_REGISTRY_PREFIX}/{model_name}/{version}/{_METADATA_FILE}"
            )
            self._gcs.upload_json(self._bucket, metadata, metadata_blob)

        logger.info(
            "Model saved to registry",
            model_name=model_name,
            version=version,
        )
        return version

    def load_model(self, model_name: str, version: str) -> BaseModel:
        """Load a specific model version from GCS.

        Args:
            model_name: Logical model name.
            version: Model version string.

        Returns:
            Loaded BaseModel instance.
        """
        blob_path = f"{_REGISTRY_PREFIX}/{model_name}/{version}/model.pkl"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "model.pkl"
            self._gcs.download_file(self._bucket, blob_path, local_path)
            model = BaseModel.load(local_path)

        logger.info(
            "Model loaded from registry",
            model_name=model_name,
            version=version,
        )
        return model

    def get_latest_version(self, model_name: str) -> str | None:
        """Get the latest version string for a model.

        Versions are sorted lexicographically (date prefix ensures
        chronological order).

        Args:
            model_name: Logical model name.

        Returns:
            Latest version string, or None if no versions exist.
        """
        versions = self.list_versions(model_name)
        if not versions:
            return None
        return versions[-1]

    def load_latest_model(self, model_name: str) -> BaseModel:
        """Load the most recent version of a model.

        Args:
            model_name: Logical model name.

        Returns:
            Loaded BaseModel instance.

        Raises:
            FileNotFoundError: If no versions exist.
        """
        version = self.get_latest_version(model_name)
        if version is None:
            raise FileNotFoundError(f"No model versions found for '{model_name}'")
        return self.load_model(model_name, version)

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions of a model.

        Args:
            model_name: Logical model name.

        Returns:
            Sorted list of version strings.
        """
        prefix = f"{_REGISTRY_PREFIX}/{model_name}/"
        blobs = self._gcs.list_blobs(self._bucket, prefix=prefix)

        versions: set[str] = set()
        for blob_name in blobs:
            parts = blob_name.replace(prefix, "").split("/")
            if parts:
                versions.add(parts[0])

        return sorted(versions)

    def get_metadata(self, model_name: str, version: str) -> dict[str, Any]:
        """Load metadata for a specific model version.

        Args:
            model_name: Logical model name.
            version: Model version string.

        Returns:
            Metadata dict.
        """
        blob_path = f"{_REGISTRY_PREFIX}/{model_name}/{version}/{_METADATA_FILE}"
        return self._gcs.download_json(self._bucket, blob_path)

    @staticmethod
    def _generate_version() -> str:
        """Generate a version string: YYYYMMDD_<short_hash>."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        time_hash = hashlib.sha256(now.isoformat().encode()).hexdigest()[:8]
        return f"{date_str}_{time_hash}"
