# CLAUDE.md - Horse Racing ML Prediction System

## プロジェクト概要

GCP上に構築する低コスト競馬予測MLシステム。Kaggle データセット（[takamotoki/jra-horse-racing-dataset](https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset)）から1986〜2021年の過去データを一括インポートし、LightGBMベースの機械学習モデルで着順予測を行う。月間運用コスト$1以下を目標とする。

## システムアーキテクチャ概要

```
データインポート -> 特徴量生成 -> モデル学習 -> 予測 -> 評価
(Kaggle CSV)       (Cloud Run Jobs)  (Cloud Run Jobs)  (Cloud Run Jobs)  (Cloud Run Jobs)
      |                |              |            |            |
      v                v              v            v            v
   GCS(raw)      BigQuery(features)  GCS(models)  BigQuery   BigQuery
```

### GCPサービス構成
- **Cloud Storage (GCS)**: 生データ・加工済みデータ・モデルの保存
- **BigQuery**: 特徴量テーブル・予測結果・レース生データの格納
- **Cloud Run Jobs**: バッチ処理（特徴量生成、学習、予測、評価）
- **Cloud Functions**: 軽量トリガー（データ収集スケジューラ）
- **Artifact Registry**: Dockerイメージの保存（直近5イメージを保持、古いものは自動削除）
- **Secret Manager**: APIキー管理
- **Workload Identity Federation**: GitHub Actions CI/CD認証

## 技術スタック

- **言語**: Python 3.10+
- **パッケージマネージャ**: [uv](https://docs.astral.sh/uv/)
- **データ処理**: Polars (高速), Pandas (GCP連携用)
- **ML**: LightGBM, scikit-learn
- **実験管理**: MLflow
- **設定管理**: Pydantic Settings + YAML
- **ログ**: structlog (JSON / テキスト)
- **インフラ**: Terraform
- **CI/CD**: GitHub Actions
- **リンター/フォーマッター**: Ruff, Black, mypy

## モジュール構成

```
src/
├── common/                 # 共通ユーティリティ（設定・GCPクライアント・ログ）
│   ├── config.py           # Pydantic Settings + YAML設定管理
│   ├── gcp_client.py       # GCS/BigQueryクライアントラッパー
│   └── logging.py          # structlog設定
├── data_collector/         # データ収集・インポート（Kaggle CSV / JRA API）
├── feature_engineering/    # 特徴量抽出パイプライン
├── model_training/         # モデル学習・実験管理（MLflow）
├── predictor/              # 予測実行
└── evaluator/              # バックテスト・評価
```

## 設定管理

設定は以下の優先順位で読み込まれる（後が優先）:
1. デフォルト値（Pydanticモデル）
2. `config/base.yaml`（共通設定）
3. `config/{environment}.yaml`（環境別設定: dev / prod）
4. 環境変数

環境は `ENVIRONMENT` 環境変数で切り替え（`dev` / `prod`）。

## 開発ガイドライン

### セットアップ

```bash
# 依存関係インストール
uv sync --all-extras

# 環境変数設定
cp .env.example .env
# .env を編集

# GCP認証
gcloud auth application-default login

# Kaggle データセットのダウンロード（要 kaggle CLI）
kaggle datasets download -d takamotoki/jra-horse-racing-dataset -p data/raw/kaggle/ --unzip
```

### Kaggle データインポート

```bash
# 全データをインポート（ドライラン: ストレージ保存なし）
uv run python -m src.data_collector --source kaggle --dry-run

# 日付範囲を指定してインポート
uv run python -m src.data_collector --source kaggle --date-from 2020-01-01 --date-to 2021-07-31

# GCS/BigQuery に保存（GCP認証が必要）
uv run python -m src.data_collector --source kaggle
```

### テスト

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=xml -v  # カバレッジ付き
```

### リント・フォーマット

```bash
uv run ruff check src/ tests/           # リントチェック
uv run ruff check --fix src/ tests/     # リント自動修正
uv run black src/ tests/                # フォーマット
uv run black --check src/ tests/        # フォーマットチェック
uv run mypy src/                        # 型チェック
```

### CI/CD

- **PR時**: GitHub Actionsでテスト（test.yaml）とリント（lint.yaml）を自動実行
- **PR時**: Docker Buildの検証を全PRで自動実行（preview-deploy.yaml）
- **PR時（ラベル）**: `preview-deploy` ラベル付与でdev環境へのデプロイを実行（preview-deploy.yaml）
- **main push時**: テスト・リント後、GCP Cloud Run Jobsへprod環境デプロイ（deploy.yaml）
- **デプロイ方式**: CIランナー上で `docker build/push` → Artifact Registry → `--image` で Cloud Run Jobs にデプロイ
- 認証: Workload Identity Federation（`watariGIT/horse-racing` リポジトリに制限）

#### Preview Deploy（PRプレビュー検証）

PRマージ前にデプロイエラーを検知するための仕組み。

- **Docker Build検証（全PR自動実行）**: `docker build` のみ実行し、依存パッケージ不足（libgomp, db-dtypes等）をビルド時に検出
- **dev環境デプロイ（`preview-deploy` ラベル付与時）**: PRに `preview-deploy` ラベルを付与すると、dev環境（`ENVIRONMENT=dev`）でCloud Run Job `ml-pipeline-preview` にデプロイ
- **パイプライン実行+レポート**: preview-report skill（`.claude/skills/preview-report/`）で手動実行。パイプラインを実行し、精度メトリクスをPRコメントに投稿

#### 環境の使い分け

| 環境 | トリガー | ENVIRONMENT | BigQueryデータセット | 用途 |
|------|---------|-------------|---------------------|------|
| dev | PRプレビュー | `dev` | `horse_racing_dev` | PR検証・テスト |
| prod | mainマージ | `prod` | `horse_racing` | 本番運用 |

### Terraform

Terraform stateはGCS (`horse-racing-ml-dev-terraform-state`) で管理。

```bash
cd infrastructure/terraform
terraform init
terraform plan -var="project_id=horse-racing-ml-dev"
terraform apply -var="project_id=horse-racing-ml-dev"
```

## コーディング規約

- **フォーマッター**: Black (line-length=88)
- **リンター**: Ruff (E, F, I, N, W, UP ルール有効)
- **型チェック**: mypy (disallow_untyped_defs=true)
- **スタイル**: PEP 8準拠、型ヒント必須
- **Docstring**: Google style
- **ファイル**: 500行以内、単一責務
- **コミット**: Conventional Commits (feat:, fix:, docs:, refactor:, test:, chore:)

## コスト管理方針

**目標: 月間 $1 以下**

| サービス | 想定コスト | 備考 |
|---|---|---|
| Cloud Storage | ~$0.20/月 | Nearlineへのライフサイクル移行 (90日) |
| BigQuery | ~$0.50/月 | オンデマンド課金、10GB以下 |
| Cloud Run Jobs | Free tier | 月間200万リクエスト無料 |
| Cloud Functions | Free tier | 月間200万回無料 |
| Secret Manager | Free tier | 月間10,000アクセス無料 |

### コスト削減の工夫
- GCSライフサイクルルールで古いデータをNearlineに移行
- BigQueryはパーティション＋クラスタリングで読み取りコスト最小化
- Cloud Run Jobsでバッチ実行（実行時のみ課金）
- Terraformで `force_destroy` は dev環境のみ有効

## ブランチ戦略

- `main`: プロダクション。直接プッシュ禁止。
- 開発ブランチ: 機能名・変更名でブランチを作成（例: `fix-deploy-permissions`, `add-lint-to-deploy`）。PRでmainにマージ。
- Squash mergeを使用。
