# Import Data Skill

Kaggle データセットをダウンロードし、BigQuery にインポートする。

## 前提条件

- `kaggle` CLIが設定済みであること（`~/.kaggle/kaggle.json`）
- `gcloud` CLIでGCP認証済みであること（`gcloud auth application-default login`）
- `uv` がインストール済みであること
- Terraformでデータセット・テーブルが作成済みであること

## 手順

1. **Kaggle データダウンロード**: CSVファイルをダウンロードして展開する
   ```bash
   kaggle datasets download -d takamotoki/jra-horse-racing-dataset -p data/raw/kaggle/ --unzip
   ```

2. **ドライラン（検証のみ）**: データの読み込みとバリデーションのみ実行し、ストレージには保存しない
   ```bash
   uv run python -m src.data_collector --source kaggle --dry-run
   ```
   - エラーがないことを確認してから本番インポートに進む

3. **BigQuery インポート実行**: データをBigQueryとGCSに保存する
   ```bash
   # prod環境: 全データをインポート
   ENVIRONMENT=prod GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle

   # prod環境: 直近10年のみインポート（推奨）
   ENVIRONMENT=prod GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle --date-from 2012-01-01

   # dev環境: インポート
   ENVIRONMENT=dev GCP_PROJECT_ID=horse-racing-ml-dev \
     uv run python -m src.data_collector --source kaggle
   ```
   - `GCP_PROJECT_ID` 環境変数の設定が必須（config/prod.yaml で参照）
   - `ENVIRONMENT` で対象データセットを切り替え

4. **インポート結果確認**: BigQueryでデータ件数を確認する
   ```bash
   # prod環境
   uv run python -c "
   from google.cloud import bigquery
   client = bigquery.Client(project='horse-racing-ml-dev')
   result = client.query('SELECT COUNT(*) as cnt FROM horse_racing.horse_results_raw').result()
   for row in result: print(f'horse_results_raw: {row.cnt} rows')
   "

   # dev環境
   uv run python -c "
   from google.cloud import bigquery
   client = bigquery.Client(project='horse-racing-ml-dev')
   result = client.query('SELECT COUNT(*) as cnt FROM horse_racing_dev.horse_results_raw').result()
   for row in result: print(f'horse_results_raw: {row.cnt} rows')
   "
   ```

## 環境の切り替え

| 環境 | 環境変数 | BigQueryデータセット |
|------|---------|---------------------|
| prod | `ENVIRONMENT=prod`（デフォルト） | `horse_racing` |
| dev | `ENVIRONMENT=dev` | `horse_racing_dev` |

**注意**: パイプライン実行時のデフォルトデータソースは `bigquery` です。インポート完了後はそのまま `uv run python -m src.pipeline` でBigQueryからデータを読み込みます。

## インポートされるテーブル

| テーブル名 | 内容 | パーティション |
|-----------|------|--------------|
| `races_raw` | レース情報（1レース1行） | `race_date` |
| `horse_results_raw` | 馬ごとのレース結果 | `race_date` |
| `jockey_results_raw` | 騎手ごとのレース結果 | `race_date` |
