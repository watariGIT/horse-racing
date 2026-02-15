"""Schema definitions for horse racing data tables.

Centralizes column definitions using Polars DataTypes for consistent
schema management across data loaders and importers.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Core schemas (matching existing internal column conventions)
# ---------------------------------------------------------------------------

RACE_SCHEMA: dict[str, type[pl.DataType]] = {
    "race_id": pl.Utf8,
    "race_date": pl.Date,
    "race_name": pl.Utf8,
    "race_number": pl.Int32,
    "course": pl.Utf8,
    "distance": pl.Int32,
    "track_type": pl.Utf8,
    "track_condition": pl.Utf8,
    "weather": pl.Utf8,
    "grade": pl.Utf8,
    "num_entries": pl.Int32,
}

HORSE_RESULT_SCHEMA: dict[str, type[pl.DataType]] = {
    "horse_id": pl.Utf8,
    "race_id": pl.Utf8,
    "race_date": pl.Date,
    "course": pl.Utf8,
    "distance": pl.Int32,
    "track_condition": pl.Utf8,
    "finish_position": pl.Int32,
    "time": pl.Utf8,
    "weight": pl.Float64,
    "jockey_id": pl.Utf8,
}

JOCKEY_RESULT_SCHEMA: dict[str, type[pl.DataType]] = {
    "jockey_id": pl.Utf8,
    "race_id": pl.Utf8,
    "race_date": pl.Date,
    "course": pl.Utf8,
    "distance": pl.Int32,
    "horse_id": pl.Utf8,
    "finish_position": pl.Int32,
}

# ---------------------------------------------------------------------------
# Extended schema (Kaggle-specific additional columns)
# ---------------------------------------------------------------------------

EXTENDED_HORSE_RESULT_SCHEMA: dict[str, type[pl.DataType]] = {
    **HORSE_RESULT_SCHEMA,
    "sex": pl.Utf8,
    "age": pl.Int32,
    "carried_weight": pl.Float64,
    "win_odds": pl.Float64,
    "win_favorite": pl.Int32,
    "corner_position_1": pl.Int32,
    "corner_position_2": pl.Int32,
    "corner_position_3": pl.Int32,
    "corner_position_4": pl.Int32,
    "last_3f_time": pl.Float64,
    "horse_weight_change": pl.Float64,
    "trainer": pl.Utf8,
    "prize_money": pl.Float64,
}
