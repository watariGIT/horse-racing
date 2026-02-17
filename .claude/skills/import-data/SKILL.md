---
name: import-data
description: Download Kaggle JRA horse racing dataset and import into BigQuery. Use when the user wants to import or re-import race data, update BigQuery tables, or set up data for the first time.
---

# Import Data

## Prerequisites

- `kaggle` CLI configured (`~/.kaggle/kaggle.json`)
- `gcloud` CLI authenticated (`gcloud auth application-default login`)
- `uv` installed
- Terraform datasets/tables created

## Workflow

1. **Download Kaggle CSV**
   ```bash
   kaggle datasets download -d takamotoki/jra-horse-racing-dataset -p data/raw/kaggle/ --unzip
   ```

2. **Dry run (validation only)**: Validate data without saving to storage
   ```bash
   uv run python -m src.data_collector --source kaggle --dry-run
   ```
   Confirm no errors before proceeding.

3. **Import to BigQuery**
   ```bash
   # prod: All data
   ENVIRONMENT=prod GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle

   # prod: Last 10 years (recommended)
   ENVIRONMENT=prod GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle --date-from 2012-01-01

   # dev
   ENVIRONMENT=dev GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle
   ```
   - `GCP_PROJECT_ID` is required (referenced in config/prod.yaml)
   - `ENVIRONMENT` switches the target dataset

4. **Verify import**: Check row counts in BigQuery
   ```bash
   uv run python .claude/skills/import-data/scripts/verify_import.py --env prod
   uv run python .claude/skills/import-data/scripts/verify_import.py --env dev
   ```

## Environments

| Env | Environment Variable | BigQuery Dataset |
|-----|---------------------|-----------------|
| prod | `ENVIRONMENT=prod` | `horse_racing` |
| dev | `ENVIRONMENT=dev` | `horse_racing_dev` |

After import, the pipeline reads from BigQuery by default (`uv run python -m src.pipeline`).

## Imported Tables

| Table | Content | Partition |
|-------|---------|-----------|
| `races_raw` | Race info (1 row per race) | `race_date` |
| `horse_results_raw` | Per-horse race results | `race_date` |
| `jockey_results_raw` | Per-jockey race results | `race_date` |
