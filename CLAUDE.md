# CLAUDE.md - Horse Racing ML Prediction System

## プロジェクト概要

GCP上に構築する低コスト競馬予測MLシステム。JRAのレースデータを収集・分析し、LightGBMベースの機械学習モデルで着順予測を行う。月間運用コスト$1以下を目標とする。

## システムアーキテクチャ概要

```
データ収集 -> 特徴量生成 -> モデル学習 -> 予測 -> 評価
(Cloud Functions)  (Cloud Run)   (Cloud Run)  (Cloud Run)  (Cloud Run)
      |                |              |            |            |
      v                v              v            v            v
   GCS(raw)      BigQuery(features)  GCS(models)  BigQuery   BigQuery
```

### GCPサービス構成
- **Cloud Storage (GCS)**: 生データ・加工済みデータ・モデルの保存
- **BigQuery**: 特徴量テーブル・予測結果・レース生データの格納
- **Cloud Run**: バッチ処理（特徴量生成、学習、予測、評価）
- **Cloud Functions**: 軽量トリガー（データ収集スケジューラ）
- **Secret Manager**: APIキー管理

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
├── data_collector/         # JRA APIからのデータ収集
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
- **main push時**: テスト後、GCP Cloud Runへ自動デプロイ（deploy.yaml）
- 認証: Workload Identity Federation

### Terraform

```bash
cd infrastructure/terraform
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
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
| Cloud Run | Free tier | 月間200万リクエスト無料 |
| Cloud Functions | Free tier | 月間200万回無料 |
| Secret Manager | Free tier | 月間10,000アクセス無料 |

### コスト削減の工夫
- GCSライフサイクルルールで古いデータをNearlineに移行
- BigQueryはパーティション＋クラスタリングで読み取りコスト最小化
- Cloud Runは最大インスタンス数1、最小0（完全スケールダウン）
- Terraformで `force_destroy` は dev環境のみ有効

## ブランチ戦略

- `main`: プロダクション。直接プッシュ禁止。
- `feature/*`: 機能開発ブランチ。PRでmainにマージ。
- Squash mergeを使用。
