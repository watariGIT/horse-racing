# Preview Report Skill

dev環境にデプロイ済みの `ml-pipeline-preview` Cloud Run Jobを実行し、バックテストレポートをPRコメントに投稿する。

## 前提条件

- 現在のブランチにPRが存在すること
- `preview-deploy` ラベルによりdev環境へのデプロイが完了していること（GitHub Actions）
- `gcloud` CLIでGCP認証済みであること

## 手順

1. **PR情報取得**: `gh pr view --json number,headRefOid` で現在のブランチのPR番号とHEAD commitを取得する

2. **パイプライン実行**: 以下のコマンドでCloud Run Jobを実行し完了を待機する
   ```bash
   gcloud run jobs execute ml-pipeline-preview \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --wait
   ```

3. **レポート取得**: GCSからバックテストレポートJSONを取得する
   ```bash
   gsutil cat gs://horse-racing-ml-dev-processed/reports/backtest_report.json
   ```
   - JSONフォーマット: `{"report": "<markdown>"}`
   - 取得失敗時は空JSONとして扱う

4. **メトリクス抽出**: レポートのmarkdownから `| Metric | Value |` テーブルを解析し、主要メトリクス（Win Accuracy, AUC ROC等）を抽出する

5. **PRコメント投稿/更新**: `gh api` を使用してPRコメントを投稿する
   - HTMLマーカー `<!-- preview-deploy-report -->` で既存コメントを検索
   - 既存コメントがあればPATCHで更新、なければPOSTで新規作成
   - これにより同一PRに重複コメントを防止する

6. **失敗時の処理**: パイプライン実行が失敗した場合も、Status: Failed としてPRコメントを投稿する

## PRコメントフォーマット

```markdown
<!-- preview-deploy-report -->
## Pipeline Preview Report

**Status**: Success / Failed
**Commit**: `{short_sha}`

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
