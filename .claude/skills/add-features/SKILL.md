---
name: add-features
description: Add new features to the ML prediction pipeline and validate accuracy. Use when the user wants to add new features, extend feature extractors, or improve model accuracy by adding columns from the dataset. Covers the full workflow from feature design through implementation, testing, and Cloud Run accuracy verification.
---

# Add Features

## Workflow

### 1. Analyze requirements

- Identify target features from Issue or user request
- Check data availability in `src/data_collector/kaggle_loader.py` (`_COLUMN_MAP_JA` / `_COLUMN_MAP_EN`)
- Check BigQuery schema in `src/data_collector/schemas.py`
- Run leakage check (see below)

### 2. Leakage check

Classify each candidate feature:

| Category | Rule | Examples |
|---|---|---|
| Safe (pre-race) | Use directly | win_odds, bracket_number, carried_weight, sex, weight |
| Leakage (in-race) | Use historical avg via DataPreparer `shift(1).rolling_mean().over("horse_id")` | corner_position, last_3f_time, finish_time |
| Derived | Compute from history in DataPreparer | win_rate, jockey_win_rate |

### 3. Implement

See [references/architecture.md](references/architecture.md) for file locations and patterns.

Modification order:
1. **Schema** (`src/data_collector/schemas.py`) - add new columns if missing
2. **Importer** (`src/data_collector/importers/kaggle_importer.py`) - ensure schema used includes new columns
3. **DataPreparer** (`src/pipeline/data_preparer.py`) - add rolling aggregations for leakage-prone or derived features
4. **Extractor** - extend existing or create new `BaseFeatureExtractor` subclass in `src/feature_engineering/extractors/`
5. **Pipeline registry** (`src/feature_engineering/pipeline.py`) - register new extractor in `_EXTRACTOR_REGISTRY`
6. **Orchestrator** (`src/pipeline/orchestrator.py`) - wire new extractor into `prepare_features()`
7. **Config** (`config/base.yaml`) - bump `model.feature_version`
8. **Tests** - add/update in `tests/unit/test_feature_engineering.py` and `tests/unit/test_pipeline.py`
9. **CLAUDE.md** - update Feature Settings and Module Structure sections

### 4. Verify locally

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run black --check src/ tests/
uv run mypy src/
```

Do NOT run model training locally (resource constraint).

### 5. PR & accuracy verification

1. Create branch, commit, push, create PR (Japanese title/description, Closes #N)
2. Add `preview-deploy` label, wait for CI + dev deployment
3. Run `/preview-report` to execute Cloud Run backtest and post results as PR comment
4. Add v(N-1) vs v(N) comparison table to PR comment:

```markdown
### v{old} -> v{new} comparison
| Metric | v{old} | v{new} | Change |
|---|---|---|---|
| Win Accuracy | ... | ... | ... |
| AUC-ROC | ... | ... | ... |
| NDCG | ... | ... | ... |
| F1 | ... | ... | ... |
```

Criteria: Win Accuracy / AUC-ROC / NDCG should maintain or improve.

### 6. Post-merge

File GitHub Issues for deferred features (Phase 2 items) per issue-creation rule.
