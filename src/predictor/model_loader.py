"""Model loader with LRU caching.

Loads trained models from the GCS model registry with in-memory
caching to avoid repeated downloads.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.model_training.model_registry import ModelRegistry
from src.model_training.models.base import BaseModel

logger = get_logger(__name__)


class ModelLoader:
    """Loads models from the registry with LRU caching.

    Caches loaded models in memory to avoid repeated GCS downloads
    when the same model version is requested multiple times.

    Args:
        registry: Optional pre-configured ModelRegistry.
        cache_size: Maximum number of cached models.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        cache_size: int = 4,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._cache_size = cache_size
        self._cache: dict[str, BaseModel] = {}

    def load(
        self,
        model_name: str,
        version: str | None = None,
    ) -> BaseModel:
        """Load a model, using cache when available.

        Args:
            model_name: Logical model name in the registry.
            version: Specific version to load. If None, loads latest.

        Returns:
            Loaded BaseModel instance.
        """
        if version is None:
            version = self._registry.get_latest_version(model_name)
            if version is None:
                raise FileNotFoundError(f"No model versions found for '{model_name}'")

        cache_key = f"{model_name}:{version}"

        if cache_key in self._cache:
            logger.debug("Model loaded from cache", key=cache_key)
            return self._cache[cache_key]

        model = self._registry.load_model(model_name, version)

        # Evict oldest entry if cache is full
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Cache evicted", key=oldest_key)

        self._cache[cache_key] = model
        logger.info("Model loaded and cached", key=cache_key)
        return model

    def get_metadata(
        self,
        model_name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Get metadata for a model version.

        Args:
            model_name: Logical model name.
            version: Specific version. If None, uses latest.

        Returns:
            Metadata dict.
        """
        if version is None:
            version = self._registry.get_latest_version(model_name)
            if version is None:
                raise FileNotFoundError(f"No model versions found for '{model_name}'")
        return self._registry.get_metadata(model_name, version)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._cache.clear()
        logger.info("Model cache cleared")
