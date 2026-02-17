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

4. **Fetch report from GCS**
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

5. **Display metrics**: Parse `| Metric | Value |` table and display formatted results

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
