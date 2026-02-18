---
name: run-pipeline
description: Execute the prod environment ML pipeline Cloud Run Job and display backtest results. Use when the user wants to run the full ML pipeline in production, check model performance, or generate a backtest report.
---

# Run Pipeline

## Prerequisites

- `gcloud` CLI authenticated
- BigQuery dataset `horse_racing` has imported data (see `import-data` skill)

## Workflow

1. **Execute Cloud Run Job**
   ```bash
   # Default (last 5 years from BigQuery)
   gcloud run jobs execute ml-pipeline \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --wait

   # Custom date range (overrides entire CMD)
   gcloud run jobs execute ml-pipeline \
     --region us-central1 \
     --project horse-racing-ml-dev \
     --args="uv,run,python,-m,src.pipeline,--stage,full,--date-from,2012-01-01" \
     --wait
   ```
   - Default: uses last 5 years (2016-08-01 to 2021-07-31)
   - With `--args`: specify full command from `uv,run,python,...` (overrides CMD)

2. **Check result**: Use exit code to determine success/failure

3. **On failure, get logs**:
   ```bash
   gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=ml-pipeline" \
     --project horse-racing-ml-dev \
     --limit 50 --freshness=30m
   ```

4. **Fetch metrics from MLflow**

   Start MLflow proxy and retrieve the latest run's backtest metrics:

   ```bash
   gcloud run services proxy mlflow-ui-prod --region us-central1 --project horse-racing-ml-dev --port 5000 &
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
       print(json.dumps(metrics, indent=2))
   "
   ```

   ```bash
   kill $PROXY_PID 2>/dev/null
   ```

   **トラブルシューティング**: proxy 接続エラー時は `lsof -i :5000` でポート競合を確認し、別ポート (`--port 5001`) で再試行する。

5. **Display metrics**: Build `| Metric | Value |` table from the MLflow metrics JSON and display formatted results

## Output Format

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
{full report}
</details>
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Dataset not found` | BigQuery dataset not created | Run `terraform apply` |
| `Table not found: horse_results_raw` | Data not imported | Run `import-data` skill |
| `Memory limit exceeded` | Insufficient memory | Increase Cloud Run Job memory |
