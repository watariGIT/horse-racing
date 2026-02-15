"""Data validation for collected datasets.

Provides column presence, null-rate, and type checks for DataFrames
before they are persisted to storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationError:
    """A single validation issue."""

    column: str
    error_type: str
    message: str


class DataValidator:
    """Validates Polars DataFrames against expected schemas.

    Args:
        required_columns: Columns that must be present.
        max_null_rate: Maximum allowed null fraction per column (0.0-1.0).
    """

    def __init__(
        self,
        required_columns: list[str] | None = None,
        max_null_rate: float = 0.5,
    ) -> None:
        self._required_columns: list[str] = required_columns or []
        self._max_null_rate = max_null_rate

    def validate(self, df: pl.DataFrame) -> list[ValidationError]:
        """Run all validation checks on a DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[ValidationError] = []
        errors.extend(self._check_required_columns(df))
        errors.extend(self._check_null_rates(df))
        errors.extend(self._check_empty(df))
        return errors

    def _check_required_columns(self, df: pl.DataFrame) -> list[ValidationError]:
        """Check that all required columns are present."""
        errors: list[ValidationError] = []
        existing = set(df.columns)
        for col in self._required_columns:
            if col not in existing:
                errors.append(
                    ValidationError(
                        column=col,
                        error_type="missing_column",
                        message=f"Required column '{col}' is missing",
                    )
                )
        return errors

    def _check_null_rates(self, df: pl.DataFrame) -> list[ValidationError]:
        """Check null rates do not exceed the threshold."""
        errors: list[ValidationError] = []
        if df.is_empty():
            return errors

        row_count = len(df)
        for col in df.columns:
            null_count = df[col].null_count()
            null_rate = null_count / row_count
            if null_rate > self._max_null_rate:
                errors.append(
                    ValidationError(
                        column=col,
                        error_type="high_null_rate",
                        message=(
                            f"Column '{col}' has {null_rate:.1%} nulls "
                            f"(threshold: {self._max_null_rate:.1%})"
                        ),
                    )
                )
        return errors

    def _check_empty(self, df: pl.DataFrame) -> list[ValidationError]:
        """Check if the DataFrame is empty."""
        if df.is_empty():
            return [
                ValidationError(
                    column="*",
                    error_type="empty_dataframe",
                    message="DataFrame has no rows",
                )
            ]
        return []
