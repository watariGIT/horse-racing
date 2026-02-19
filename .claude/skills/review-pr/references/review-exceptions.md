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

---

## Python / MLflow

### `src/common/mlflow_auth.py` - `google.auth` の遅延インポート

`request_headers()` 内での `google.auth` インポートは意図的な設計。
`in_context()` が `False`（ローカル開発・CI 環境）の場合に不要なインポートを回避するためのパフォーマンス最適化。
MLflow の `RequestHeaderProvider` プラグインはプロセス起動時にロードされるため、
トップレベルインポートにすると非 HTTPS 環境でも `google.auth` が常にインポートされる。

---

## Data Preparer / Feature Engineering

### `src/pipeline/data_preparer.py` - `days_since_last_race` の `.over("horse_id")` は正しい

`race_date.shift(1).over("horse_id")` により馬ごとの前走日付を正しく取得している。
外側の `race_date - ...` で当該レース日との差分を計算するため、リーケージなし。

### `src/feature_engineering/extractors/horse_features.py` - `win_odds` / `win_favorite` はレース前データ

Kaggle データの `win_odds`（確定オッズ）と `win_favorite`（人気順位）は発走直前に確定する情報であり、
レース結果（着順）ではない。競馬予測 ML で一般的に使用される pre-race 特徴量。

### `src/pipeline/data_preparer.py` - `race_date` のみのソートで十分

同一馬の同日複数出走は極めて稀であり、`rolling_mean` は微小な順序差に対して非感受的。
副キー追加による複雑化に見合うメリットがない。

### `src/pipeline/orchestrator.py` - SQL テーブル名の文字列フォーマット

`_load_from_bigquery` のテーブル名は `settings.bigquery.dataset`（内部設定値）由来であり、
ユーザー入力ではない。SQL インジェクションリスクなし。

### `src/pipeline/data_preparer.py` - `age` → `horse_age` リネーム

DataPreparer はデータ正規化の責務を持ち、KaggleLoader の出力カラム名 (`age`) を
HorseFeatureExtractor の期待 (`horse_age`) に合わせる適切な場所。

### `src/pipeline/data_preparer.py` - walk-forward バックテストでの CV リーケージなし

本システムは walk-forward バックテストを使用しており、DataPreparer 段階での
ローリング集計が fold 間リーケージを引き起こすことはない。
将来 k-fold CV を導入する場合は再設計が必要（その時点で対応）。

### `src/evaluator/backtest_engine.py` - バックテスト再学習タイミング

BacktestEngine は各 walk-forward ステップで新規モデルを学習する設計。
既存動作であり PR #50 の変更対象外。

### `src/common/config.py` - `FeaturePipelineConfig` のデフォルト値と `base.yaml` の二重定義

Pydantic モデルのデフォルト値は YAML 設定ファイルが読み込めない場合のフォールバックとして機能する。
`base.yaml` が Source of Truth であり、Pydantic デフォルトはセーフティネット。
`ModelConfig`、`BacktestConfig`、`BigQueryConfig` 等すべての設定クラスで同じパターンを使用しており、
一貫性のある設計判断。

### `src/model_training/trainer.py` - train メトリクスが最適閾値で計算される

閾値最適化後の `predict` は validation で最適化した閾値を使用するため、
train メトリクス（F1, precision, recall）も同じ閾値で計算される。
これは閾値最適化の本質的な動作であり、train/val 両方のメトリクスを
同じ閾値で比較すること自体は妥当。train メトリクスの意味が変わるが、
モデルの訓練データに対する性能を同一条件で評価するために意図的な設計。
