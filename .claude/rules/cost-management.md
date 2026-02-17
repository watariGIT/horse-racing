# Cost Management

**Target: <$1/month**

| Service | Estimated Cost | Notes |
|---------|---------------|-------|
| Cloud Storage | ~$0.20/mo | Nearline lifecycle transition (90 days) |
| BigQuery | ~$0.50/mo | On-demand billing, <10GB |
| Cloud Run Jobs | Free tier | 2M requests/month free |
| Cloud Functions | Free tier | 2M invocations/month free |
| Secret Manager | Free tier | 10,000 accesses/month free |

## Cost Reduction Strategies
- GCS lifecycle rules: move old data to Nearline
- BigQuery: partitioning + clustering to minimize read costs
- Cloud Run Jobs: batch execution (pay only during execution)
- Terraform: `force_destroy` enabled only for dev environment
