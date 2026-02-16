"""Kaggle dataset loader for JRA horse racing data.

Loads CSV data from the Kaggle takamotoki/jra-horse-racing-dataset,
maps Japanese column names to internal English names, and generates
deterministic IDs for horses and jockeys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: Kaggle CSV (Japanese) → internal names
# The dataset may use Japanese or English headers depending on version.
# We try Japanese first, then fall back to English.
# ---------------------------------------------------------------------------

_COLUMN_MAP_JA: dict[str, str] = {
    "レースID": "race_id",
    "着順": "finish_position",
    "枠番": "bracket_number",
    "馬番": "post_position",
    "馬名": "horse_name",
    "性齢": "sex_age",
    "性別": "sex",
    "馬齢": "age",
    "斤量": "carried_weight",
    "騎手": "jockey_name",
    "タイム": "time",
    "着差": "margin",
    "通過": "corner_positions",
    "1コーナー": "corner_position_1",
    "2コーナー": "corner_position_2",
    "3コーナー": "corner_position_3",
    "4コーナー": "corner_position_4",
    "上り": "last_3f_time",
    "単勝": "win_odds",
    "人気": "win_favorite",
    "馬体重": "horse_weight",
    "場体重増減": "horse_weight_change",
    "調教師": "trainer",
    "馬主": "owner",
    "賞金": "prize_money",
    "賞金(万円)": "prize_money",
    "競馬場名": "course",
    "距離": "distance",
    "距離(m)": "distance",
    "天候": "weather",
    "馬場状態": "track_condition",
    "馬場状態1": "track_condition",
    "芝・ダート": "track_type",
    "芝・ダート区分": "track_type",
    "レース名": "race_name",
    "レース番号": "race_number",
    "リステッド・重賞": "grade",
    "リステッド・重賞競走": "grade",
    "日付": "race_date",
    "レース日付": "race_date",
}

_COLUMN_MAP_EN: dict[str, str] = {
    "Race ID": "race_id",
    "FP": "finish_position",
    "Bracket": "bracket_number",
    "PP": "post_position",
    "Horse Name": "horse_name",
    "Sex and Age": "sex_age",
    "Weight(Kg)": "carried_weight",
    "Jockey": "jockey_name",
    "Total Time": "time",
    "Margin": "margin",
    "Position": "corner_positions",
    "Last 3F": "last_3f_time",
    "Win Odds": "win_odds",
    "Fav": "win_favorite",
    "Horse Weight": "horse_weight",
    "Trainer": "trainer",
    "Owner": "owner",
    "Prize Money": "prize_money",
    "Racecourse": "course",
    "Distance": "distance",
    "Weather": "weather",
    "Track Condition": "track_condition",
    "Turf/Dirt": "track_type",
    "Race Name": "race_name",
    "Race Number": "race_number",
    "Grade": "grade",
    "Date": "race_date",
}

# Non-numeric finish position values to treat as None
_NON_FINISH_VALUES = frozenset(
    ["取消", "除外", "中止", "失格", "Cancel", "Exclude", "Stop", "DQ"]
)


def _generate_id(name: str) -> str:
    """Generate a deterministic short ID from a name using SHA-256.

    Args:
        name: Name string (e.g. horse name, jockey name).

    Returns:
        First 12 hex characters of the SHA-256 hash.
    """
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def _parse_sex_age(sex_age: str | None) -> tuple[str | None, int | None]:
    """Parse combined sex+age string (e.g. '牡3' → ('牡', 3)).

    Args:
        sex_age: Combined sex and age string.

    Returns:
        Tuple of (sex, age). Returns (None, None) if unparseable.
    """
    if not sex_age or not isinstance(sex_age, str) or len(sex_age) < 2:
        return None, None
    sex = sex_age[:-1]
    try:
        age = int(sex_age[-1])
    except ValueError:
        # Try parsing multi-digit age
        age_str = ""
        for ch in reversed(sex_age):
            if ch.isdigit():
                age_str = ch + age_str
            else:
                break
        if age_str:
            sex = sex_age[: -len(age_str)]
            age = int(age_str)
        else:
            return sex_age, None
    return sex, age


def _parse_corner_positions(
    positions_str: str | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Parse corner positions string (e.g. '3-3-2-1').

    Args:
        positions_str: Dash-separated position string.

    Returns:
        Tuple of (pos1, pos2, pos3, pos4). Missing values are None.
    """
    if not positions_str or not isinstance(positions_str, str):
        return None, None, None, None
    parts = positions_str.split("-")
    result: list[int | None] = []
    for p in parts[:4]:
        p = p.strip()
        try:
            result.append(int(p))
        except (ValueError, TypeError):
            result.append(None)
    while len(result) < 4:
        result.append(None)
    return result[0], result[1], result[2], result[3]


def _parse_horse_weight(
    weight_str: str | None,
) -> tuple[float | None, float | None]:
    """Parse horse weight string (e.g. '480(+4)' → (480.0, 4.0)).

    Args:
        weight_str: Weight string with optional change in parentheses.

    Returns:
        Tuple of (weight, weight_change).
    """
    if not weight_str or not isinstance(weight_str, str):
        return None, None
    weight_str = weight_str.strip()
    if "(" in weight_str:
        base, change = weight_str.split("(", 1)
        change = change.rstrip(")")
        try:
            return float(base), float(change)
        except ValueError:
            try:
                return float(base), None
            except ValueError:
                return None, None
    try:
        return float(weight_str), None
    except ValueError:
        return None, None


class KaggleDataLoader:
    """Loads and transforms CSV data from the Kaggle JRA dataset.

    Handles column mapping, ID generation, and type conversion.
    Uses lazy evaluation (scan_csv) for memory efficiency.

    Args:
        data_dir: Directory containing Kaggle CSV files.
            Defaults to settings.kaggle.data_dir.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is not None:
            self._data_dir = Path(data_dir)
        else:
            settings = get_settings()
            self._data_dir = Path(settings.kaggle.data_dir)

    def _resolve_csv_path(self) -> Path:
        """Resolve the path to the race result CSV file."""
        settings = get_settings()
        filename = settings.kaggle.race_result_file
        return self._data_dir / filename

    def _read_csv(self, path: Path) -> pl.DataFrame:
        """Read a CSV file with encoding fallback.

        Tries UTF-8 first, then Shift_JIS.

        Args:
            path: Path to the CSV file.

        Returns:
            Raw DataFrame with original column names.
        """
        try:
            df = pl.read_csv(path, encoding="utf8", infer_schema_length=10000)
            logger.debug("Read CSV with UTF-8 encoding", path=str(path))
            return df
        except Exception:
            logger.debug("UTF-8 failed, trying Shift_JIS", path=str(path))
            df = pl.read_csv(
                path,
                encoding="shift_jis",
                infer_schema_length=10000,
            )
            logger.debug("Read CSV with Shift_JIS encoding", path=str(path))
            return df

    def _apply_column_mapping(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename columns from Japanese/English to internal names.

        Args:
            df: DataFrame with original column names.

        Returns:
            DataFrame with renamed columns.
        """
        existing_cols = set(df.columns)

        # Determine which mapping to use based on header language
        ja_matches = sum(1 for k in _COLUMN_MAP_JA if k in existing_cols)
        en_matches = sum(1 for k in _COLUMN_MAP_EN if k in existing_cols)

        if ja_matches >= en_matches:
            col_map = {k: v for k, v in _COLUMN_MAP_JA.items() if k in existing_cols}
        else:
            col_map = {k: v for k, v in _COLUMN_MAP_EN.items() if k in existing_cols}

        logger.debug(
            "Column mapping applied",
            mapped=len(col_map),
            language="ja" if ja_matches >= en_matches else "en",
        )
        return df.rename(col_map)

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all transformations to the raw DataFrame.

        - Parse finish_position (non-numeric → None)
        - Generate horse_id and jockey_id from names
        - Parse sex_age into sex and age
        - Parse corner positions
        - Parse horse weight and weight change
        - Convert race_date to Date type
        - Cast numeric columns

        Args:
            df: DataFrame with internal column names.

        Returns:
            Fully transformed DataFrame.
        """
        # --- finish_position: non-numeric → null ---
        if "finish_position" in df.columns:
            df = df.with_columns(
                pl.col("finish_position").cast(pl.Utf8).alias("finish_position_str")
            )
            df = df.with_columns(
                pl.when(
                    pl.col("finish_position_str").is_in(list(_NON_FINISH_VALUES))
                    | pl.col("finish_position_str").str.contains(r"[^\d]")
                )
                .then(None)
                .otherwise(pl.col("finish_position_str").cast(pl.Int32, strict=False))
                .alias("finish_position")
            ).drop("finish_position_str")

        # --- Generate horse_id from horse_name ---
        if "horse_name" in df.columns:
            df = df.with_columns(
                pl.col("horse_name")
                .map_elements(
                    lambda name: _generate_id(name) if name else None,
                    return_dtype=pl.Utf8,
                )
                .alias("horse_id")
            )

        # --- Generate jockey_id from jockey_name ---
        if "jockey_name" in df.columns:
            df = df.with_columns(
                pl.col("jockey_name")
                .map_elements(
                    lambda name: _generate_id(name) if name else None,
                    return_dtype=pl.Utf8,
                )
                .alias("jockey_id")
            )

        # --- Parse sex_age → sex + age ---
        if "sex_age" in df.columns:
            df = df.with_columns(
                pl.col("sex_age")
                .cast(pl.Utf8)
                .map_elements(
                    lambda x: _parse_sex_age(x)[0],
                    return_dtype=pl.Utf8,
                )
                .alias("sex"),
                pl.col("sex_age")
                .cast(pl.Utf8)
                .map_elements(
                    lambda x: _parse_sex_age(x)[1],
                    return_dtype=pl.Int32,
                )
                .alias("age"),
            ).drop("sex_age")

        # --- Cast separate sex/age columns if present ---
        if "age" in df.columns and "sex_age" not in df.columns:
            df = df.with_columns(
                pl.col("age").cast(pl.Int32, strict=False).alias("age")
            )

        # --- Parse corner_positions → corner_position_1..4 ---
        if "corner_positions" in df.columns:
            for i, alias in enumerate(
                [
                    "corner_position_1",
                    "corner_position_2",
                    "corner_position_3",
                    "corner_position_4",
                ]
            ):
                idx = i

                df = df.with_columns(
                    pl.col("corner_positions")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda x, _idx=idx: _parse_corner_positions(x)[_idx],  # type: ignore[misc]
                        return_dtype=pl.Int32,
                    )
                    .alias(alias)
                )
            df = df.drop("corner_positions")

        # --- Cast pre-split corner columns to Int32 ---
        for col_name in [
            "corner_position_1",
            "corner_position_2",
            "corner_position_3",
            "corner_position_4",
        ]:
            if col_name in df.columns and "corner_positions" not in df.columns:
                df = df.with_columns(
                    pl.col(col_name).cast(pl.Int32, strict=False).alias(col_name)
                )

        # --- Parse horse_weight → weight + horse_weight_change ---
        if "horse_weight" in df.columns:
            weight_exprs = [
                pl.col("horse_weight")
                .cast(pl.Utf8)
                .map_elements(
                    lambda x: _parse_horse_weight(x)[0],
                    return_dtype=pl.Float64,
                )
                .alias("weight"),
            ]
            # Only derive horse_weight_change from parsing if not already present
            if "horse_weight_change" not in df.columns:
                weight_exprs.append(
                    pl.col("horse_weight")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda x: _parse_horse_weight(x)[1],
                        return_dtype=pl.Float64,
                    )
                    .alias("horse_weight_change"),
                )
            df = df.with_columns(weight_exprs).drop("horse_weight")

        # --- Cast horse_weight_change to float if already present ---
        if "horse_weight_change" in df.columns:
            df = df.with_columns(
                pl.col("horse_weight_change")
                .cast(pl.Float64, strict=False)
                .alias("horse_weight_change")
            )

        # --- Convert race_date to Date type ---
        if "race_date" in df.columns:
            df = df.with_columns(
                pl.col("race_date")
                .cast(pl.Utf8)
                .str.to_date(format="%Y-%m-%d", strict=False)
                .alias("race_date")
            )

        # --- Cast numeric columns ---
        int_cols = ["race_number", "distance", "win_favorite"]
        for col in int_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Int32, strict=False).alias(col)
                )

        float_cols = ["carried_weight", "win_odds", "last_3f_time", "prize_money"]
        for col in float_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )

        # --- Ensure race_id is string ---
        if "race_id" in df.columns:
            df = df.with_columns(pl.col("race_id").cast(pl.Utf8).alias("race_id"))

        return df

    def load_race_results(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pl.DataFrame:
        """Load and transform race results from the Kaggle CSV.

        Args:
            date_from: Optional start date filter (YYYY-MM-DD).
            date_to: Optional end date filter (YYYY-MM-DD).

        Returns:
            Transformed DataFrame with all race result data.
        """
        csv_path = self._resolve_csv_path()
        if not csv_path.exists():
            logger.error("CSV file not found", path=str(csv_path))
            raise FileNotFoundError(f"Kaggle CSV not found: {csv_path}")

        logger.info("Loading Kaggle CSV", path=str(csv_path))
        df = self._read_csv(csv_path)
        logger.info("Raw CSV loaded", rows=len(df), columns=df.columns)

        df = self._apply_column_mapping(df)
        df = self._transform(df)

        # Apply date filters
        if date_from and "race_date" in df.columns:
            from_date = pl.Series([date_from]).str.to_date("%Y-%m-%d")[0]
            df = df.filter(pl.col("race_date") >= from_date)

        if date_to and "race_date" in df.columns:
            to_date = pl.Series([date_to]).str.to_date("%Y-%m-%d")[0]
            df = df.filter(pl.col("race_date") <= to_date)

        logger.info(
            "Kaggle data loaded and transformed",
            rows=len(df),
            date_from=date_from,
            date_to=date_to,
        )
        return df
