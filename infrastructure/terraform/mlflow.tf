# -------------------------------------------------------------------
# MLflow UI - Cloud Run Service (scales to zero)
# -------------------------------------------------------------------

resource "google_cloud_run_v2_service" "mlflow_ui" {
  count    = var.mlflow_ui_enabled ? 1 : 0
  name     = "mlflow-ui"
  location = var.region

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/ml-pipeline/mlflow-ui:${var.mlflow_ui_image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "gs://${var.project_id}-models/mlruns"
      }
    }

    service_account = google_service_account.ml_pipeline.email
  }

  labels = {
    environment = local.environment
    managed_by  = "terraform"
  }
}
