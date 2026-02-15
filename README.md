# Horse Racing ML Prediction System

<!-- Badges -->
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Lint](https://img.shields.io/badge/lint-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

GCP上に構築する低コスト競馬予測MLシステム。JRAのレースデータを収集・分析し、LightGBMベースの機械学習モデルで着順予測を行います。

**月間運用コスト $1 以下** を目標としたコスト効率の高い設計です。

---

GCP-based machine learning system for horse racing prediction. Collects and analyzes JRA race data, predicting race outcomes using LightGBM. Designed to operate under $1/month.

## ドキュメント / Documentation

| ドキュメント | 内容 |
|---|---|
| [セットアップガイド](docs/setup.md) | 環境構築・GCP認証・Terraform手順 |
| [アーキテクチャ](docs/architecture.md) | GCP構成図・データフロー・コスト試算 |
| [コーディング規約](rules.md) | Python規約・Git規則・レビュー基準 |
| [開発ガイドライン](CLAUDE.md) | プロジェクト方針・技術スタック・コスト管理 |

## 技術スタック / Tech Stack

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3.10+ |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |
| データ処理 | Polars, Pandas |
| 機械学習 | LightGBM, scikit-learn |
| 実験管理 | MLflow |
| クラウド | GCP (GCS, BigQuery, Cloud Run Jobs, Cloud Functions) |
| インフラ | Terraform |
| CI/CD | GitHub Actions |
| リンター | Ruff, Black, mypy |

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- GCP account with billing enabled
- Terraform >= 1.5.0

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd horse-racing
uv sync              # Production dependencies
uv sync --all-extras # + Development dependencies
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your GCP project ID and API keys
```

### 4. GCP Authentication

```bash
gcloud auth application-default login
```

### 5. Infrastructure Setup (Terraform)

```bash
cd infrastructure/terraform
terraform init
terraform plan -var="project_id=horse-racing-ml-dev"
terraform apply -var="project_id=horse-racing-ml-dev"
```

詳細は [セットアップガイド](docs/setup.md) を参照してください。

## Project Structure

```
horse-racing/
├── config/                 # YAML configuration (base, dev, prod)
├── src/
│   ├── common/             # Shared utilities (config, GCP clients, logging)
│   ├── data_collector/     # JRA API data collection
│   ├── feature_engineering/# Feature extraction pipeline
│   ├── model_training/     # Model training and experiment tracking
│   ├── predictor/          # Prediction execution
│   └── evaluator/          # Backtesting and evaluation
├── tests/                  # Unit and integration tests
├── scripts/                # Operational scripts
├── docs/                   # Documentation
│   ├── setup.md            # Setup guide
│   └── architecture.md     # Architecture details
├── infrastructure/         # Terraform (GCP resources)
└── .github/workflows/      # CI/CD (GitHub Actions)
```

## Development

### Run tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=xml -v  # With coverage
```

### Run linting

```bash
uv run ruff check src/ tests/
uv run black --check src/ tests/
uv run mypy src/
```

### Format code

```bash
uv run black src/ tests/
uv run ruff check --fix src/ tests/
```

## CI/CD

- **PR**: Automated tests and linting via GitHub Actions
- **main push**: Automated deployment to GCP Cloud Run Jobs

## Cost Estimate

目標: **月間 $1 以下** / Target: **under $1/month**

| Service           | Estimated Cost |
|-------------------|---------------|
| Cloud Storage     | ~$0.01/month  |
| BigQuery          | Free tier     |
| Cloud Run Jobs    | Free tier     |
| Cloud Functions   | Free tier     |
| Secret Manager    | Free tier     |

詳細は [アーキテクチャドキュメント](docs/architecture.md#コスト試算表) を参照してください。
