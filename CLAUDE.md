# CLAUDE.md - Horse Racing ML Prediction System

## Project Overview

Low-cost horse racing prediction ML system on GCP. Imports historical data (1986-2021) from [Kaggle dataset](https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset) and predicts race finishing positions using LightGBM. Target: <$1/month operating cost.

## Architecture

```
Data Import -> Feature Engineering -> Training -> Prediction -> Evaluation
(Kaggle CSV)   (Cloud Run Jobs)      (Cloud Run Jobs) (Cloud Run Jobs) (Cloud Run Jobs)
      |              |                    |              |              |
      v              v                    v              v              v
   GCS(raw)    BigQuery(features)     GCS(models)    BigQuery       BigQuery
```

### GCP Services
- **Cloud Storage (GCS)**: Raw data, processed data, models
- **BigQuery**: Feature tables, predictions, raw race data
- **Cloud Run Jobs**: Batch processing (feature eng, training, prediction, evaluation)
- **Cloud Functions**: Lightweight triggers (scheduler)
- **Artifact Registry**: Docker images (keep latest 5, auto-delete older)
- **Secret Manager**: API key management
- **Workload Identity Federation**: GitHub Actions CI/CD auth

### MLflow Experiment Tracking
- **Tracking URI**: Local file store (dev) / GCS `gs://{project}-models/mlruns` (prod)
- **Run naming**: `{model_type}_{YYYYMMDD_HHmmss}` with environment/model/feature tags
- **Artifacts**: Feature importance (PNG/JSON), backtest results (JSON), evaluation metrics
- **Model Registry linkage**: `mlflow_run_id` stored in GCS model metadata
- **Storage**: Evaluation results stored exclusively in MLflow (no separate BigQuery table)
- **Comparison**: `uv run python -m src.model_training.compare_experiments --last 5`

## Tech Stack

- **Language**: Python 3.10+
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Data**: Polars (fast), Pandas (GCP integration)
- **ML**: LightGBM, scikit-learn, MLflow
- **Config**: Pydantic Settings + YAML
- **Logging**: structlog (JSON / text)
- **Infra**: Terraform, GitHub Actions
- **Linting**: Ruff, Black, mypy

## Module Structure

```
src/
├── common/                 # Shared utilities (config, GCP clients, logging)
│   ├── config.py           # Pydantic Settings + YAML config
│   ├── gcp_client.py       # GCS/BigQuery client wrapper
│   └── logging.py          # structlog setup
├── data_collector/         # Data collection/import (Kaggle CSV / JRA API)
├── feature_engineering/    # Feature extraction pipeline
├── model_training/         # Model training + experiment tracking (MLflow)
├── predictor/              # Prediction execution
└── evaluator/              # Backtesting + evaluation
```

## Configuration

Config loading priority (later overrides earlier):
1. Defaults (Pydantic models)
2. `config/base.yaml` (shared)
3. `config/{environment}.yaml` (env-specific: dev / prod)
4. Environment variables

Switch environments via `ENVIRONMENT` env var (`dev` / `prod`).

### MLflow Settings

| Key | Default | Description |
|-----|---------|-------------|
| `mlflow.tracking_uri` | `file:./mlruns` | MLflow tracking backend URI |
| `mlflow.experiment_name` | `horse-racing-prediction` | Experiment name |
| `mlflow.enabled` | `true` | Enable/disable MLflow tracking |

## Development

### Setup

```bash
uv sync --all-extras
cp .env.example .env  # Edit .env
gcloud auth application-default login
```

### Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=xml -v  # With coverage
```

### Linting

```bash
uv run ruff check src/ tests/           # Lint check
uv run ruff check --fix src/ tests/     # Auto-fix
uv run black src/ tests/                # Format
uv run black --check src/ tests/        # Format check
uv run mypy src/                        # Type check
```

### CI/CD

- **On PR**: Auto-run tests (test.yaml), lint (lint.yaml), Docker build validation (preview-deploy.yaml)
- **On PR (label)**: `preview-deploy` label triggers dev environment deployment (preview-deploy.yaml)
- **On main push**: Test + lint → prod deploy to Cloud Run Jobs (deploy.yaml)
- **Deploy method**: `docker build/push` → Artifact Registry → Cloud Run Jobs
- **Auth**: Workload Identity Federation (scoped to `watariGIT/horse-racing`)

### Environments

| Env | Trigger | ENVIRONMENT | BigQuery Dataset | Purpose |
|-----|---------|-------------|-----------------|---------|
| dev | PR preview | `dev` | `horse_racing_dev` | PR validation |
| prod | main merge | `prod` | `horse_racing` | Production |

### Terraform

State managed in GCS (`horse-racing-ml-dev-terraform-state`). Terraform Workspaces for dev/prod.

```bash
cd infrastructure/terraform
terraform init

# dev
terraform workspace select dev
terraform plan -var-file=dev.tfvars
terraform apply -var-file=dev.tfvars

# prod
terraform workspace select prod
terraform plan -var-file=prod.tfvars
terraform apply -var-file=prod.tfvars
```

## Coding Standards

- **Formatter**: Black (line-length=88)
- **Linter**: Ruff (E, F, I, N, W, UP rules)
- **Type check**: mypy (disallow_untyped_defs=true)
- **Style**: PEP 8, mandatory type hints, Google-style docstrings
- **Files**: ≤500 lines, single responsibility
- **Commits**: Conventional Commits (feat:, fix:, docs:, refactor:, test:, chore:)

## Branch Strategy

- `main`: Production. No direct push.
- Feature branches: Named by feature/change (e.g., `fix-deploy-permissions`). Merge to main via PR.
- Squash merge.
