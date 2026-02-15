"""Base feature extractor with ABC interface.

All feature extractors inherit from BaseFeatureExtractor and implement
the extract() method following the Strategy pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Subclasses must implement extract() and feature_names.
    This follows the Strategy pattern -- extractors can be swapped
    in and out of the FeaturePipeline.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return the list of feature column names produced by this extractor."""

    @abstractmethod
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract features from the input DataFrame.

        Args:
            df: Input DataFrame containing raw or intermediate data.

        Returns:
            DataFrame with the original columns plus new feature columns.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(features={self.feature_names})"
