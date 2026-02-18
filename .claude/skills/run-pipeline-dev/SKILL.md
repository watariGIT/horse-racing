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

3. **Fetch report from GCS**
   ```bash
   gsutil cat gs://horse-racing-ml-dev-processed/reports/backtest_report.json
   ```
   JSON format: `{"report": "<markdown>"}`

4. **Extract metrics**: Parse the `| Metric | Value |` table from the markdown report

5. **Post/update PR comment** using `gh api`:
   - Search for existing comment with HTML marker `<!-- preview-deploy-report -->`
   - PATCH to update if found, POST to create if not (prevents duplicates)

6. **On failure**: Post a "Status: Failed" comment even if the pipeline fails

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

<details>
<summary>Full Backtest Report</summary>
{full report}
</details>
```
