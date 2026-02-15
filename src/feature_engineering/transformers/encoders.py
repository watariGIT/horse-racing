"""Category encoding transformers.

Provides label encoding and one-hot encoding for categorical features
in Polars DataFrames.
"""

from __future__ import annotations

import polars as pl

from src.common.logging import get_logger

logger = get_logger(__name__)


class CategoryEncoder:
    """Encodes categorical columns to numeric values.

    Supports label encoding (string -> integer mapping) and
    one-hot encoding. Learned mappings are stored for consistent
    transform on new data.

    Args:
        columns: List of column names to encode.
        strategy: "label" for integer encoding, "onehot" for one-hot.
    """

    def __init__(
        self,
        columns: list[str],
        strategy: str = "label",
    ) -> None:
        self._columns = columns
        self._strategy = strategy
        self._mappings: dict[str, dict[str, int]] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pl.DataFrame) -> CategoryEncoder:
        """Learn encoding mappings from training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self for method chaining.
        """
        for col in self._columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found during fit, skipping", col)
                continue
            unique_vals = df[col].drop_nulls().unique().sort().to_list()
            self._mappings[col] = {str(val): idx for idx, val in enumerate(unique_vals)}

        self._fitted = True
        logger.info(
            "CategoryEncoder fitted",
            columns=list(self._mappings.keys()),
            strategy=self._strategy,
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply encoding to a DataFrame.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with encoded columns.

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._fitted:
            raise RuntimeError("CategoryEncoder must be fitted before transform")

        result = df.clone()

        if self._strategy == "label":
            result = self._label_encode(result)
        elif self._strategy == "onehot":
            result = self._onehot_encode(result)

        return result

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def _label_encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply label encoding."""
        for col, mapping in self._mappings.items():
            if col not in df.columns:
                continue
            encoded_col = f"{col}_encoded"
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .replace_strict(mapping, default=-1, return_dtype=pl.Int64)
                .alias(encoded_col)
            )
        return df

    def _onehot_encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply one-hot encoding."""
        for col, mapping in self._mappings.items():
            if col not in df.columns:
                continue
            for val in mapping:
                ohe_col = f"{col}_{val}"
                df = df.with_columns(
                    (pl.col(col).cast(pl.Utf8) == val).cast(pl.Int64).alias(ohe_col)
                )
        return df
