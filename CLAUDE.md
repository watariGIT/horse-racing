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

### MLflow UI
- **Hosting**: Cloud Run Service (Gen2, min-instances=0, scales to zero when idle)
- **Backend Store**: GCS bucket mounted via Cloud Run GCS FUSE volume (`/mlruns/mlruns`)
- **Resources**: 1 vCPU, 1Gi memory (GCS FUSE requires Gen2 + additional memory)
- **Access**: Authenticated GCP users only (IAM `roles/run.invoker` required)
- **Image**: `infrastructure/mlflow/Dockerfile`
- **Deploy**: Image auto-built on `infrastructure/mlflow/**` changes via `deploy-mlflow.yaml`; Cloud Run Service managed by Terraform (`mlflow.tf`)
- **Artifact Store**: GCS (`gs://{project}-models/mlartifacts`) に `--serve-artifacts` でプロキシ
- **Enable**: `mlflow_ui_enabled = true` in tfvars (dev/prod 両方有効)
- **Cost**: ~$0 when idle (scales to zero)
- **Service naming**: `mlflow-ui-{env}` (e.g., `mlflow-ui-dev`, `mlflow-ui-prod`)
- **Access URL**: `gcloud run services describe mlflow-ui-{env} --region us-central1 --format 'value(status.url)'`
- **Proxy access**: `gcloud run services proxy mlflow-ui-{env} --region us-central1 --port 5000`

### MLflow Experiment Tracking
- **Tracking URI**: HTTP via Cloud Run MLflow server (Cloud Run Job に `MLFLOW_TRACKING_URI` 環境変数で注入)
- **Authentication**: `RequestHeaderProvider` プラグイン (`src/common/mlflow_auth.py`) が GCP OIDC ID トークンを自動付与
- **Run naming**: `{model_type}_{YYYYMMDD_HHmmss}` with environment/model/feature tags
- **GitHub traceability**: `github.pr_number`, `github.commit_sha`, `github.branch`, `github.repository` タグで MLflow Run ↔ GitHub PR の双方向追跡が可能（CI/CD ワークフローから環境変数経由で注入）
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
│   ├── logging.py          # structlog setup
│   └── mlflow_auth.py      # MLflow Cloud Run IAM auth plugin (RequestHeaderProvider)
├── data_collector/         # Data collection/import (Kaggle CSV / JRA API)
├── feature_engineering/    # Feature extraction pipeline (v2: 26 features)
│   ├── extractors/
│   │   ├── race_features.py          # 6 features (distance, track, weather, etc.)
│   │   ├── horse_features.py         # 14 features (stats, odds, gate, weight, sex)
│   │   ├── jockey_features.py        # 4 features (win rate, top3, experience)
│   │   └── running_style_features.py # 2 features (avg corner pos, avg last 3F)
│   └── pipeline.py                   # FeaturePipeline (Strategy pattern)
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

### Model Settings

| Key | Default | Description |
|-----|---------|-------------|
| `model.feature_version` | `v2` | Feature version tag (logged to MLflow for experiment comparison) |
| `model.calibration_method` | `isotonic` | 確率キャリブレーション手法 (`isotonic` or `sigmoid`) |
| `model.optimize_threshold` | `true` | F1 最大化の閾値探索を有効化 |

### Feature Settings

| Key | Default | Description |
|-----|---------|-------------|
| `feature_pipeline.extractors` | `["race", "horse", "jockey", "running_style"]` | エクストラクタ名リスト (`_EXTRACTOR_REGISTRY` キーと一致) |

Feature version history:
- **v1**: 13 features (race 6 + horse 7)
- **v2**: 26 features (race 6 + horse 14 + jockey 4 + running_style 2)

### MLflow Settings

| Key | Default | Description |
|-----|---------|-------------|
| `mlflow.tracking_uri` | `file:./mlruns` | MLflow tracking URI (Cloud Run Job では `MLFLOW_TRACKING_URI` 環境変数で上書き) |
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
- **On PR (label)**: `preview-deploy` label triggers dev environment deployment (preview-deploy.yaml) — ML pipeline + MLflow UI イメージのビルド・デプロイ
- **On main push**: Test + lint → prod deploy to Cloud Run Jobs (deploy.yaml)
- **On main push (mlflow)**: `infrastructure/mlflow/**` changes → MLflow UI image build+push (deploy-mlflow.yaml)
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
