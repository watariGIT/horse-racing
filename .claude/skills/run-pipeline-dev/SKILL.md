---
name: run-pipeline-dev
description: Execute the dev environment ML pipeline on Cloud Run and post a backtest metrics report as a PR comment. Use when the user wants to validate pipeline changes before merging or check model accuracy on a PR.
---

# Preview Report

## Prerequisites

- PR exists on current branch
- `preview-deploy` label added and dev deployment completed (GitHub Actions)
- `gcloud` CLI authenticated

## Workflow

1. **Get PR info**
   ```bash
   gh pr view --json number,headRefOid
   ```

2. **Execute Cloud Run Job**
   ```bash
   gcloud run jobs execute ml-pipeline-preview \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --wait
   ```

3. **Fetch metrics from MLflow**

   Start MLflow proxy and retrieve the latest run's backtest metrics:

   ```bash
   gcloud run services proxy mlflow-ui-dev --region us-central1 --project horse-racing-ml-dev --port 5000 &
   PROXY_PID=$!
   sleep 3
   ```

   ```bash
   uv run python -c "
   import mlflow, json
   mlflow.set_tracking_uri('http://localhost:5000')
   runs = mlflow.search_runs(
       experiment_names=['horse-racing-prediction'],
       max_results=1,
       order_by=['start_time DESC'],
   )
   if not runs.empty:
       row = runs.iloc[0]
       metrics = {}
       for c in sorted(runs.columns):
           if c.startswith('metrics.backtest_overall_'):
               name = c.replace('metrics.backtest_overall_', '')
               metrics[name] = round(row[c], 4)
       result = {
           'run_id': row.get('run_id', ''),
           'run_name': row.get('tags.mlflow.runName', ''),
           'experiment_id': row.get('experiment_id', ''),
           'metrics': metrics,
       }
       print(json.dumps(result, indent=2))
   "
   ```

   ```bash
   kill $PROXY_PID 2>/dev/null
   ```

   **トラブルシューティング**: proxy 接続エラー時は `lsof -i :5000` でポート競合を確認し、別ポート (`--port 5001`) で再試行する。

4. **Extract metrics**: Build the `| Metric | Value |` table from the MLflow metrics JSON

5. **Post/update PR comment**:
   - Search for existing comment with marker `<!-- preview-deploy-report -->`:
     ```bash
     COMMENT_ID=$(gh api repos/{owner}/{repo}/issues/{pr_number}/comments \
       --jq '[.[] | select(.body | test("preview-deploy-report")) | .id] | first // empty')
     ```
   - If found: PATCH to update
     ```bash
     gh api repos/{owner}/{repo}/issues/comments/$COMMENT_ID -X PATCH -f body="..."
     ```
   - If not found: create new with `gh pr comment`
     ```bash
     gh pr comment {pr_number} --body "..."
     ```

6. **On failure**: Post a "Status: Failed" comment even if the pipeline fails (同じ冪等パターンで `gh pr comment` を使用)

## PR Comment Format

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

### MLflow Run
**Run**: `{run_name}` (`{run_id}`)

<details>
<summary>MLflow UI アクセス方法</summary>

gcloud run services proxy mlflow-ui-dev --region us-central1 --project horse-racing-ml-dev --port 5000
# Open: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}

</details>

<details>
<summary>Full Backtest Report</summary>
{full report}
</details>
```
