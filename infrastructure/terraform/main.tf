terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "horse-racing-ml-dev-terraform-state"
    prefix = "horse-racing"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# -------------------------------------------------------------------
# GCS Buckets
# -------------------------------------------------------------------

resource "google_storage_bucket" "raw_data" {
  name     = "${var.project_id}-raw-data"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev"

  lifecycle_rule {
    condition {
      age = var.gcs_lifecycle_days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "google_storage_bucket" "processed" {
  name     = "${var.project_id}-processed"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev"

  lifecycle_rule {
    condition {
      age = var.gcs_lifecycle_days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "google_storage_bucket" "models" {
  name     = "${var.project_id}-models"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev"

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# -------------------------------------------------------------------
# Artifact Registry (Docker images for Cloud Run)
# -------------------------------------------------------------------

resource "google_artifact_registry_repository" "ml_pipeline" {
  location      = var.region
  repository_id = "ml-pipeline"
  format        = "DOCKER"

  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"

    most_recent_versions {
      keep_count = 5
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# -------------------------------------------------------------------
# BigQuery Dataset
# -------------------------------------------------------------------

resource "google_bigquery_dataset" "horse_racing" {
  dataset_id = "horse_racing${var.environment == "dev" ? "_dev" : ""}"
  location   = var.bigquery_location

  delete_contents_on_destroy = var.environment == "dev"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# BigQuery Tables

resource "google_bigquery_table" "races_raw" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "races_raw"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "race_date"
  }

  labels = {
    environment = var.environment
  }

  schema = jsonencode([
    { name = "race_id", type = "STRING", mode = "NULLABLE" },
    { name = "race_date", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "race_name", type = "STRING", mode = "NULLABLE" },
    { name = "race_number", type = "INTEGER", mode = "NULLABLE" },
    { name = "course", type = "STRING", mode = "NULLABLE" },
    { name = "distance", type = "INTEGER", mode = "NULLABLE" },
    { name = "track_type", type = "STRING", mode = "NULLABLE" },
    { name = "track_condition", type = "STRING", mode = "NULLABLE" },
    { name = "weather", type = "STRING", mode = "NULLABLE" },
    { name = "grade", type = "STRING", mode = "NULLABLE" },
    { name = "num_entries", type = "INTEGER", mode = "NULLABLE" },
  ])
}

resource "google_bigquery_table" "features" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "features"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "race_date"
  }

  clustering = ["feature_version"]

  labels = {
    environment = var.environment
  }

  schema = jsonencode([
    { name = "race_id", type = "STRING", mode = "REQUIRED" },
    { name = "horse_id", type = "STRING", mode = "REQUIRED" },
    { name = "race_date", type = "DATE", mode = "REQUIRED" },
    { name = "feature_version", type = "STRING", mode = "REQUIRED" },
    { name = "features", type = "JSON", mode = "REQUIRED" },
    { name = "created_at", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "predictions"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "race_date"
  }

  labels = {
    environment = var.environment
  }

  schema = jsonencode([
    { name = "race_id", type = "STRING", mode = "REQUIRED" },
    { name = "horse_id", type = "STRING", mode = "REQUIRED" },
    { name = "race_date", type = "DATE", mode = "REQUIRED" },
    { name = "model_version", type = "STRING", mode = "REQUIRED" },
    { name = "prediction", type = "FLOAT64", mode = "REQUIRED" },
    { name = "predicted_at", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "horse_results_raw" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "horse_results_raw"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "race_date"
  }

  labels = {
    environment = var.environment
  }

  # Schema matches auto-detected types from load_table_from_dataframe:
  # - race_date: TIMESTAMP (pandas datetime64 -> BQ TIMESTAMP)
  # - finish_position: FLOAT (nullable Int32 -> BQ FLOAT)
  # - All columns NULLABLE (pandas default)
  schema = jsonencode([
    { name = "horse_id", type = "STRING", mode = "NULLABLE" },
    { name = "race_id", type = "STRING", mode = "NULLABLE" },
    { name = "race_date", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "course", type = "STRING", mode = "NULLABLE" },
    { name = "distance", type = "INTEGER", mode = "NULLABLE" },
    { name = "track_condition", type = "STRING", mode = "NULLABLE" },
    { name = "finish_position", type = "FLOAT", mode = "NULLABLE" },
    { name = "time", type = "STRING", mode = "NULLABLE" },
    { name = "weight", type = "FLOAT64", mode = "NULLABLE" },
    { name = "jockey_id", type = "STRING", mode = "NULLABLE" },
    { name = "sex", type = "STRING", mode = "NULLABLE" },
    { name = "age", type = "INTEGER", mode = "NULLABLE" },
    { name = "carried_weight", type = "FLOAT64", mode = "NULLABLE" },
    { name = "win_odds", type = "FLOAT64", mode = "NULLABLE" },
    { name = "win_favorite", type = "INTEGER", mode = "NULLABLE" },
    { name = "corner_position_1", type = "INTEGER", mode = "NULLABLE" },
    { name = "corner_position_2", type = "INTEGER", mode = "NULLABLE" },
    { name = "corner_position_3", type = "INTEGER", mode = "NULLABLE" },
    { name = "corner_position_4", type = "INTEGER", mode = "NULLABLE" },
    { name = "last_3f_time", type = "FLOAT64", mode = "NULLABLE" },
    { name = "horse_weight_change", type = "FLOAT64", mode = "NULLABLE" },
    { name = "trainer", type = "STRING", mode = "NULLABLE" },
    { name = "prize_money", type = "FLOAT64", mode = "NULLABLE" },
  ])
}

resource "google_bigquery_table" "jockey_results_raw" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "jockey_results_raw"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "race_date"
  }

  labels = {
    environment = var.environment
  }

  # Schema matches auto-detected types from load_table_from_dataframe
  schema = jsonencode([
    { name = "jockey_id", type = "STRING", mode = "NULLABLE" },
    { name = "race_id", type = "STRING", mode = "NULLABLE" },
    { name = "race_date", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "course", type = "STRING", mode = "NULLABLE" },
    { name = "distance", type = "INTEGER", mode = "NULLABLE" },
    { name = "horse_id", type = "STRING", mode = "NULLABLE" },
    { name = "finish_position", type = "FLOAT", mode = "NULLABLE" },
  ])
}

resource "google_bigquery_table" "evaluation_results" {
  dataset_id          = google_bigquery_dataset.horse_racing.dataset_id
  table_id            = "evaluation_results"
  deletion_protection = var.environment == "prod"

  labels = {
    environment = var.environment
  }

  schema = jsonencode([
    { name = "model_name", type = "STRING", mode = "REQUIRED" },
    { name = "win_accuracy", type = "FLOAT64", mode = "NULLABLE" },
    { name = "place_accuracy", type = "FLOAT64", mode = "NULLABLE" },
    { name = "top3_accuracy", type = "FLOAT64", mode = "NULLABLE" },
    { name = "ndcg", type = "FLOAT64", mode = "NULLABLE" },
    { name = "ndcg_at_3", type = "FLOAT64", mode = "NULLABLE" },
    { name = "auc_roc", type = "FLOAT64", mode = "NULLABLE" },
    { name = "log_loss", type = "FLOAT64", mode = "NULLABLE" },
    { name = "precision", type = "FLOAT64", mode = "NULLABLE" },
    { name = "recall", type = "FLOAT64", mode = "NULLABLE" },
    { name = "f1", type = "FLOAT64", mode = "NULLABLE" },
  ])
}

# -------------------------------------------------------------------
# Secret Manager
# -------------------------------------------------------------------

resource "google_secret_manager_secret" "jra_api_key" {
  secret_id = "jra-api-key"

  replication {
    auto {}
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# -------------------------------------------------------------------
# Service Account for Cloud Run / Cloud Functions
# -------------------------------------------------------------------

resource "google_service_account" "ml_pipeline" {
  account_id   = "ml-pipeline-sa"
  display_name = "ML Pipeline Service Account"
}

# GCS access
resource "google_project_iam_member" "ml_pipeline_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

# BigQuery access
resource "google_project_iam_member" "ml_pipeline_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

resource "google_project_iam_member" "ml_pipeline_bigquery_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

# Secret Manager access
resource "google_secret_manager_secret_iam_member" "ml_pipeline_jra_key" {
  secret_id = google_secret_manager_secret.jra_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

# -------------------------------------------------------------------
# Workload Identity Federation (for GitHub Actions)
# -------------------------------------------------------------------

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions Pool"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Actions Provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "attribute.repository == \"${var.github_repository}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# WIF IAM binding: allow GitHub Actions to impersonate the SA
resource "google_service_account_iam_member" "wif_sa_binding" {
  service_account_id = google_service_account.ml_pipeline.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# -------------------------------------------------------------------
# Cloud Run deploy IAM roles for the service account
# -------------------------------------------------------------------

resource "google_project_iam_member" "ml_pipeline_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

resource "google_project_iam_member" "ml_pipeline_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

resource "google_project_iam_member" "ml_pipeline_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

resource "google_project_iam_member" "ml_pipeline_cloudbuild_editor" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

resource "google_project_iam_member" "ml_pipeline_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ml_pipeline.email}"
}
