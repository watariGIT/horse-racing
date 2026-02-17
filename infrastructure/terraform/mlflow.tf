# -------------------------------------------------------------------
# MLflow UI - Cloud Run Service (scales to zero)
# -------------------------------------------------------------------

resource "google_cloud_run_v2_service" "mlflow_ui" {
  count    = var.mlflow_ui_enabled ? 1 : 0
  name     = "mlflow-ui-${local.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    volumes {
      name = "mlruns"
      gcs {
        bucket    = "${var.project_id}-models"
        read_only = false
      }
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/ml-pipeline/mlflow-ui:${var.mlflow_ui_image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "/mlruns/mlruns"
      }

      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${var.project_id}-models/mlartifacts"
      }

      volume_mounts {
        name       = "mlruns"
        mount_path = "/mlruns"
      }

      startup_probe {
        http_get {
          path = "/"
          port = 8080
        }
        initial_delay_seconds = 15
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 10
      }
    }

    service_account = google_service_account.ml_pipeline.email
  }

  labels = {
    environment = local.environment
    managed_by  = "terraform"
  }
}

# IAM: Deny unauthenticated access (no allUsers binding)
# Only the ML pipeline service account can invoke MLflow UI
resource "google_cloud_run_v2_service_iam_member" "mlflow_ui_invoker" {
  count    = var.mlflow_ui_enabled ? 1 : 0
  name     = google_cloud_run_v2_service.mlflow_ui[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.ml_pipeline.email}"
}

# Allow authenticated users in the project to access MLflow UI via gcloud proxy
# Cloud Run v2 does not support projectEditor as a member type.
# Users need `roles/run.invoker` on this service to use `gcloud run services proxy`.
# Add individual user IAM bindings as needed:
#   gcloud run services add-iam-policy-binding mlflow-ui-{env} \
#     --region us-central1 --member="user:EMAIL" --role="roles/run.invoker"
