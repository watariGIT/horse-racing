output "gcs_bucket_raw" {
  description = "GCS bucket for raw data"
  value       = google_storage_bucket.raw_data.name
}

output "gcs_bucket_processed" {
  description = "GCS bucket for processed data"
  value       = google_storage_bucket.processed.name
}

output "gcs_bucket_models" {
  description = "GCS bucket for ML models"
  value       = google_storage_bucket.models.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.horse_racing.dataset_id
}

output "service_account_email" {
  description = "ML pipeline service account email"
  value       = google_service_account.ml_pipeline.email
}

output "workload_identity_provider" {
  description = "Workload Identity Provider for GitHub Actions"
  value       = google_iam_workload_identity_pool_provider.github.name
}
