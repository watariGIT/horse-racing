# レビュー例外事項（既知の設計判断）

以下の項目は意図的な設計判断として承認済みであり、レビューで指摘対象外とする。

---

## Infrastructure

### `infrastructure/terraform/mlflow.tf` - GCS ボリュームマウントの `read_only = false`

MLflow server はバックエンドストア（`/mlruns/mlruns`）に実験・Run データを書き込むため、
GCS FUSE ボリュームマウントは `read_only = false` が必須。

### `infrastructure/terraform/prod.tfvars` - `project_id = horse-racing-ml-dev`

dev/prod は同一 GCP プロジェクト上で Terraform Workspace により分離する設計。
`prod.tfvars` の `project_id` が `horse-racing-ml-dev` であるのは意図通り。

### `.github/workflows/*.yaml` - `GCP_PROJECT_ID` のハードコード

単一プロジェクト構成のため、ワークフロー内に `GCP_PROJECT_ID: horse-racing-ml-dev` を直接記載している。
マルチプロジェクト構成への移行時に Secrets/vars への移行を検討する（Issue #48）。

---

## Python / MLflow

### `src/common/mlflow_auth.py` - `google.auth` の遅延インポート

`request_headers()` 内での `google.auth` インポートは意図的な設計。
`in_context()` が `False`（ローカル開発・CI 環境）の場合に不要なインポートを回避するためのパフォーマンス最適化。
MLflow の `RequestHeaderProvider` プラグインはプロセス起動時にロードされるため、
トップレベルインポートにすると非 HTTPS 環境でも `google.auth` が常にインポートされる。
