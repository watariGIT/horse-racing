# Run Pipeline Skill

prod環境の `ml-pipeline` Cloud Run Job を実行し、バックテストレポートを表示する。

## 前提条件

- `gcloud` CLIでGCP認証済みであること
- BigQueryデータセット `horse_racing` にデータがインポート済みであること（`import-data` skill参照）

## 手順

1. **パイプライン実行**: 以下のコマンドでCloud Run Jobを実行し完了を待機する
   ```bash
   # デフォルト（全データ使用）
   gcloud run jobs execute ml-pipeline \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --wait

   # 日付範囲を指定して実行（推奨: 直近10年）
   gcloud run jobs execute ml-pipeline \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --args="uv,run,python,-m,src.pipeline,--stage,full,--date-from,2012-01-01" \
     --wait
   ```
   - `--args` を使用する場合は `uv,run,python,-m,src.pipeline` から全て指定する（CMD をオーバーライドするため）

2. **実行結果確認**: コマンドの終了コードで成功/失敗を判定する

3. **失敗時のログ確認**: パイプラインが失敗した場合、以下でエラーログを取得する
   ```bash
   gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=ml-pipeline" \
     --project horse-racing-ml-dev \
     --limit 50 --freshness=30m
   ```

4. **レポート取得**: GCSからバックテストレポートJSONを取得する
   ```bash
   uv run python -c "
   from google.cloud import storage
   import json
   client = storage.Client(project='horse-racing-ml-dev')
   blob = client.bucket('horse-racing-ml-dev-processed').blob('reports/backtest_report.json')
   data = json.loads(blob.download_as_text())
   print(data['report'])
   "
   ```
   - JSONフォーマット: `{"report": "<markdown>"}`
   - 取得失敗時は空JSONとして扱う

5. **メトリクス表示**: レポートのmarkdownから `| Metric | Value |` テーブルを解析し、主要メトリクス（Win Accuracy, AUC ROC等）をフォーマットして表示する

## 出力フォーマット

```markdown
## Pipeline Execution Report

**Status**: Success / Failed

### Key Metrics
| Metric | Value |
|--------|-------|
| Win Accuracy | 0.0823 |
| Auc Roc | 0.6012 |

<details>
<summary>Full Backtest Report</summary>
{レポート全文}
</details>
```

## よくあるエラーと対処法

| エラー | 原因 | 対処法 |
|--------|------|--------|
| `Dataset not found` | BigQueryデータセット未作成 | `terraform apply` でデータセット作成 |
| `Table not found: horse_results_raw` | データ未インポート | `import-data` skill でデータインポート |
| `Memory limit exceeded` | メモリ不足 | Cloud Run Jobのメモリ設定を増やす |
