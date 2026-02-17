# セットアップガイド

## 前提条件

- **Python**: 3.10 以上
- **GCPアカウント**: 課金が有効なプロジェクト
- **uv**: Python パッケージマネージャ
- **Terraform**: 1.5.0 以上
- **gcloud CLI**: GCP認証用

## 1. uv のインストール

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール確認:

```bash
uv --version
```

## 2. リポジトリのクローンと依存関係インストール

```bash
git clone <repository-url>
cd horse-racing
```

### 依存関係のインストール

```bash
# 本番依存関係のみ
uv sync

# 開発依存関係を含む（テスト、リンター等）
uv sync --all-extras
```

`uv sync` は仮想環境の作成と依存関係のインストールを自動で行う。

## 3. 環境変数の設定

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```bash
# GCP Configuration
GCP_PROJECT_ID=horse-racing-ml-dev
GCP_REGION=us-central1

# JRA API
JRA_API_KEY=your-jra-api-key-here

# Environment (dev / prod)
ENVIRONMENT=dev

# BigQuery
BQ_DATASET=horse_racing_dev

# GCS Buckets (プロジェクトIDから自動生成されるため通常は設定不要)
# GCS_BUCKET_RAW=horse-racing-ml-dev-raw-data
# GCS_BUCKET_PROCESSED=horse-racing-ml-dev-processed
# GCS_BUCKET_MODELS=horse-racing-ml-dev-models
```

### 必須項目

| 変数名 | 説明 |
|---|---|
| `GCP_PROJECT_ID` | GCPプロジェクトID |
| `ENVIRONMENT` | 実行環境 (`dev` / `prod`) |

### オプション項目

| 変数名 | 説明 | デフォルト |
|---|---|---|
| `GCP_REGION` | GCPリージョン | `us-central1` |
| `BQ_DATASET` | BigQueryデータセット名 | `horse_racing` (prodの場合) |
| `GCS_BUCKET_RAW` | 生データバケット | `{project_id}-raw-data` |
| `GCS_BUCKET_PROCESSED` | 加工済みデータバケット | `{project_id}-processed` |
| `GCS_BUCKET_MODELS` | モデルバケット | `{project_id}-models` |
| `MLFLOW_TRACKING_URI` | MLflow Tracking Server URL | `file:./mlruns` (ローカル) |

GCSバケット名は `GCP_PROJECT_ID` から自動生成されるため、通常は設定不要。

`MLFLOW_TRACKING_URI` は Cloud Run Job 実行時に自動設定されるため、ローカル開発時は設定不要（デフォルトの `file:./mlruns` が使用される）。

## 4. GCP認証

### ローカル開発（Application Default Credentials）

```bash
# gcloud CLI でログイン
gcloud auth login

# Application Default Credentials を設定
gcloud auth application-default login
```

### サービスアカウント（本番環境）

1. GCPコンソールでサービスアカウントを作成（または Terraform で自動作成済み）
2. JSONキーファイルをダウンロード
3. 環境変数を設定:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### 必要な権限

MLパイプラインのサービスアカウント (`ml-pipeline-sa`) に付与されている権限:

| ロール | 用途 |
|---|---|
| `roles/storage.objectAdmin` | GCSバケットの読み書き |
| `roles/bigquery.dataEditor` | BigQueryテーブルの読み書き |
| `roles/bigquery.jobUser` | BigQueryジョブの実行 |
| `roles/secretmanager.secretAccessor` | Secret Managerからの読み取り |
| `roles/run.admin` | Cloud Run Jobsのデプロイ・実行 |
| `roles/iam.serviceAccountUser` | サービスアカウントのimpersonate |
| `roles/artifactregistry.writer` | コンテナイメージのpush |
| `roles/cloudbuild.builds.editor` | Cloud Build実行 |
| `roles/logging.logWriter` | ログ書き込み |

## 5. Terraformによるインフラ構築

### 初期化

```bash
cd infrastructure/terraform
terraform init
```

### プランの確認

```bash
terraform plan -var="project_id=horse-racing-ml-dev"
```

### リソースの作成

```bash
terraform apply -var="project_id=horse-racing-ml-dev"
```

### 環境別のデプロイ

```bash
# 開発環境
terraform apply \
  -var="project_id=horse-racing-ml-dev" \
  -var="environment=dev"

# 本番環境
terraform apply \
  -var="project_id=YOUR_PROD_PROJECT_ID" \
  -var="environment=prod"
```

### 作成されるリソース

| リソース | 説明 |
|---|---|
| GCS バケット x3 | raw-data, processed, models |
| BigQuery データセット | horse_racing (prod) / horse_racing_dev (dev) |
| BigQuery テーブル x3 | races_raw, features, predictions |
| Cloud Run Service | MLflow UI (`mlflow_ui_enabled=true` の場合) |
| Secret Manager シークレット | jra-api-key |
| サービスアカウント | ml-pipeline-sa (10個のIAMロール付与) |
| Workload Identity Federation | GitHub Actions用（`watariGIT/horse-racing` リポジトリに制限） |

### リソースの削除

```bash
terraform destroy -var="project_id=horse-racing-ml-dev"
```

## 6. テストの実行

```bash
# 全テスト実行
uv run pytest tests/ -v

# ユニットテストのみ
uv run pytest tests/unit/ -v

# カバレッジ付き
uv run pytest tests/ --cov=src --cov-report=xml -v

# 特定テストの実行
uv run pytest tests/unit/test_config.py -v
```

## 7. リント・フォーマット

```bash
# リントチェック
uv run ruff check src/ tests/

# リント自動修正
uv run ruff check --fix src/ tests/

# フォーマットチェック
uv run black --check src/ tests/

# フォーマット実行
uv run black src/ tests/

# 型チェック
uv run mypy src/
```

## 8. 設定の確認

YAML設定ファイルは `config/` ディレクトリにある:

- `config/base.yaml`: 全環境共通の設定
- `config/dev.yaml`: 開発環境の上書き設定
- `config/prod.yaml`: 本番環境の上書き設定

設定は以下の優先順位で適用される（後が優先）:
1. Pydanticデフォルト値
2. `base.yaml`
3. `{environment}.yaml`
4. 環境変数

## トラブルシューティング

### uv sync が失敗する

Python 3.10以上がインストールされているか確認:

```bash
python --version
```

### GCP認証エラー

```bash
# 認証状態の確認
gcloud auth list

# Application Default Credentials のリセット
gcloud auth application-default revoke
gcloud auth application-default login
```

### Terraform のState管理

Terraform stateはGCSバケット (`horse-racing-ml-dev-terraform-state`) でリモート管理されている。
`main.tf` の backend ブロックで設定済み:

```hcl
backend "gcs" {
  bucket = "horse-racing-ml-dev-terraform-state"
  prefix = "horse-racing"
}
```
