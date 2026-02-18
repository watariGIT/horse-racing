# Feature Engineering Architecture

## Current Feature Count

Check `config/base.yaml` `model.feature_version` and CLAUDE.md Feature Settings for current version.

## Key Files

| File | Role |
|---|---|
| `src/data_collector/schemas.py` | `HORSE_RESULT_SCHEMA`, `EXTENDED_HORSE_RESULT_SCHEMA` - BigQuery column definitions |
| `src/data_collector/importers/kaggle_importer.py` | `_extract_horse_results()` - selects columns for BigQuery import |
| `src/data_collector/kaggle_loader.py` | `_COLUMN_MAP_JA` / `_COLUMN_MAP_EN` - CSV to internal name mapping |
| `src/pipeline/data_preparer.py` | `DataPreparer` - rolling history aggregation (leakage-free) |
| `src/feature_engineering/extractors/base.py` | `BaseFeatureExtractor` ABC - Strategy pattern interface |
| `src/feature_engineering/extractors/race_features.py` | `RaceFeatureExtractor` - race-level features |
| `src/feature_engineering/extractors/horse_features.py` | `HorseFeatureExtractor` - horse-level features |
| `src/feature_engineering/extractors/jockey_features.py` | `JockeyFeatureExtractor` - jockey-level features |
| `src/feature_engineering/extractors/running_style_features.py` | `RunningStyleFeatureExtractor` - historical running style |
| `src/feature_engineering/pipeline.py` | `FeaturePipeline` + `_EXTRACTOR_REGISTRY` |
| `src/pipeline/orchestrator.py` | `PipelineOrchestrator.prepare_features()` - wires extractors |

## Extractor Pattern

All extractors follow this pattern:

```python
class MyFeatureExtractor(BaseFeatureExtractor):
    _FEATURES = ["feat_my_feature"]

    @property
    def feature_names(self) -> list[str]:
        return self._FEATURES

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        if "source_col" in df.columns:
            result = result.with_columns(
                pl.col("source_col").cast(pl.Float64).alias("feat_my_feature")
            )
        else:
            result = result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("feat_my_feature")
            )
        return result
```

Conventions:
- All feature columns prefixed with `feat_`
- Missing source columns produce `None` (not error)
- Categorical encoding uses `replace_strict(MAP, default=-1)`
- Extractors are stateless (no fit)

## DataPreparer Leakage-Free Pattern

```python
# In _compute_horse_history(), add:
pl.col("in_race_column")
    .cast(pl.Float64)
    .shift(1)                                    # exclude current race
    .rolling_mean(window_size=n, min_samples=1)  # past N races
    .over("horse_id")                            # per horse
    .alias("avg_in_race_column")
```

## Registering a New Extractor

1. Add to `_EXTRACTOR_REGISTRY` in `pipeline.py`
2. Add to extractors list in `orchestrator.py` `prepare_features()`
3. Update test assertions for total feature count
