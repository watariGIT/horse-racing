"""MLflow experiment tracking integration.

Provides a context-manager interface for recording experiments,
including parameters, metrics, and model artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from src.common.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracker.

    Manages experiment lifecycle: start, log, and end runs.
    Uses local file store by default.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking server URI.
            Defaults to local file store.
    """

    def __init__(
        self,
        experiment_name: str = "horse-racing-prediction",
        tracking_uri: str = "file:./mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run: mlflow.ActiveRun | None = None

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

    @property
    def is_active(self) -> bool:
        return self._run is not None

    @property
    def run_id(self) -> str | None:
        if self._run is not None:
            return self._run.info.run_id
        return None

    def start_run(self, run_name: str | None = None) -> ExperimentTracker:
        """Start a new MLflow run.

        Args:
            run_name: Optional human-readable name for the run.

        Returns:
            Self for method chaining.
        """
        self._run = mlflow.start_run(run_name=run_name)
        logger.info(
            "MLflow run started",
            run_id=self._run.info.run_id,
            run_name=run_name,
        )
        return self

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._run is not None:
            mlflow.end_run()
            logger.info("MLflow run ended", run_id=self._run.info.run_id)
            self._run = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        if self._run is not None:
            mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters at once.

        Args:
            params: Dict of parameter name -> value.
        """
        if self._run is not None:
            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if self._run is not None:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dict of metric name -> value.
            step: Optional step number.
        """
        if self._run is not None:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a local file as an artifact.

        Args:
            local_path: Path to the file to log.
        """
        if self._run is not None:
            mlflow.log_artifact(str(local_path))
            logger.info("Artifact logged", path=str(local_path))

    def log_model_artifact(
        self, model_path: str | Path, artifact_name: str = "model"
    ) -> None:
        """Log a model file as a named artifact.

        Args:
            model_path: Path to the model file.
            artifact_name: Subdirectory name in artifacts.
        """
        if self._run is not None:
            mlflow.log_artifact(str(model_path), artifact_path=artifact_name)

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
