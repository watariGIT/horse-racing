"""Feature pipeline with Strategy pattern composition.

Orchestrates feature extractors and transformers in a scikit-learn
compatible fit/transform API. Extractors can be dynamically configured.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from src.common.logging import get_logger
from src.feature_engineering.extractors.base import BaseFeatureExtractor
from src.feature_engineering.extractors.horse_features import HorseFeatureExtractor
from src.feature_engineering.extractors.jockey_features import JockeyFeatureExtractor
from src.feature_engineering.extractors.race_features import RaceFeatureExtractor
from src.feature_engineering.extractors.running_style_features import (
    RunningStyleFeatureExtractor,
)
from src.feature_engineering.transformers.encoders import CategoryEncoder
from src.feature_engineering.transformers.scalers import FeatureScaler

logger = get_logger(__name__)

# Registry of available extractors by name
_EXTRACTOR_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {
    "race": RaceFeatureExtractor,
    "horse": HorseFeatureExtractor,
    "jockey": JockeyFeatureExtractor,
    "running_style": RunningStyleFeatureExtractor,
}


class FeaturePipeline:
    """Composable feature extraction and transformation pipeline.

    Uses the Strategy pattern to combine multiple feature extractors,
    then applies optional encoding and scaling transformations.
    Provides a scikit-learn compatible fit/transform API.

    Args:
        extractors: List of feature extractor instances to apply.
        encoder: Optional category encoder.
        scaler: Optional feature scaler.
    """

    def __init__(
        self,
        extractors: list[BaseFeatureExtractor] | None = None,
        encoder: CategoryEncoder | None = None,
        scaler: FeatureScaler | None = None,
    ) -> None:
        self._extractors: list[BaseFeatureExtractor] = extractors or []
        self._encoder = encoder
        self._scaler = scaler
        self._fitted = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeaturePipeline:
        """Build a pipeline from a configuration dictionary.

        Config format:
            {
                "extractors": ["race", "horse", "jockey"],
                "encoder": {"columns": [...], "strategy": "label"},
                "scaler": {"columns": [...], "strategy": "standard"},
            }

        Args:
            config: Pipeline configuration dictionary.

        Returns:
            Configured FeaturePipeline instance.
        """
        extractor_names = config.get("extractors", [])
        extractors: list[BaseFeatureExtractor] = []
        for name in extractor_names:
            if name in _EXTRACTOR_REGISTRY:
                extractors.append(_EXTRACTOR_REGISTRY[name]())
            else:
                logger.warning("Unknown extractor '%s', skipping", name)

        encoder = None
        encoder_cfg = config.get("encoder")
        if encoder_cfg:
            encoder = CategoryEncoder(
                columns=encoder_cfg.get("columns", []),
                strategy=encoder_cfg.get("strategy", "label"),
            )

        scaler = None
        scaler_cfg = config.get("scaler")
        if scaler_cfg:
            scaler = FeatureScaler(
                columns=scaler_cfg.get("columns", []),
                strategy=scaler_cfg.get("strategy", "standard"),
            )

        return cls(extractors=extractors, encoder=encoder, scaler=scaler)

    @property
    def feature_names(self) -> list[str]:
        """All feature names produced by the pipeline extractors."""
        names: list[str] = []
        for ext in self._extractors:
            names.extend(ext.feature_names)
        return names

    def add_extractor(self, extractor: BaseFeatureExtractor) -> FeaturePipeline:
        """Add a feature extractor to the pipeline.

        Args:
            extractor: Feature extractor instance.

        Returns:
            Self for method chaining.
        """
        self._extractors.append(extractor)
        return self

    def fit(self, df: pl.DataFrame) -> FeaturePipeline:
        """Fit the pipeline transformers on training data.

        Runs all extractors first, then fits the encoder and scaler
        on the extracted features.

        Args:
            df: Training DataFrame.

        Returns:
            Self for method chaining.
        """
        # Run extractors (stateless, no fit needed)
        extracted = self._run_extractors(df)

        # Fit encoder
        if self._encoder:
            self._encoder.fit(extracted)

        # Apply encoder before fitting scaler
        encoded = extracted
        if self._encoder and self._encoder.is_fitted:
            encoded = self._encoder.transform(extracted)

        # Fit scaler
        if self._scaler:
            self._scaler.fit(encoded)

        self._fitted = True
        logger.info(
            "FeaturePipeline fitted",
            extractors=[type(e).__name__ for e in self._extractors],
            features=len(self.feature_names),
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data through the full pipeline.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with all features extracted and transformed.

        Raises:
            RuntimeError: If transform is called before fit
                (when encoder/scaler are configured).
        """
        if (self._encoder or self._scaler) and not self._fitted:
            raise RuntimeError(
                "Pipeline must be fitted before transform "
                "when encoder or scaler is configured"
            )

        result = self._run_extractors(df)

        if self._encoder and self._encoder.is_fitted:
            result = self._encoder.transform(result)

        if self._scaler and self._scaler.is_fitted:
            result = self._scaler.transform(result)

        return result

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def get_feature_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract only the feature columns from a transformed DataFrame.

        Args:
            df: Transformed DataFrame (output of transform()).

        Returns:
            DataFrame containing only feature columns.
        """
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        return df.select(feature_cols)

    def _run_extractors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all extractors sequentially.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with features from all extractors.
        """
        result = df
        for extractor in self._extractors:
            result = extractor.extract(result)
        return result
