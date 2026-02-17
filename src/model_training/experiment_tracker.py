"""MLflow experiment tracking integration.

Provides a context-manager interface for recording experiments,
including parameters, metrics, and model artifacts.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.common.config import MLflowConfig

logger = get_logger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracker.

    Manages experiment lifecycle: start, log, and end runs.
    Supports local file store and GCS backends.

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

    @classmethod
    def from_config(cls, config: MLflowConfig) -> ExperimentTracker:
        """Create a tracker from MLflowConfig.

        Args:
            config: MLflow configuration object.

        Returns:
            Configured ExperimentTracker instance.
        """
        return cls(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
        )

    @staticmethod
    def generate_run_name(model_type: str) -> str:
        """Generate a run name with timestamp.

        Format: ``{model_type}_{YYYYMMDD_HHmmss}``

        Args:
            model_type: Model type identifier.

        Returns:
            Generated run name string.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{ts}"

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

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set multiple tags on the active run.

        Args:
            tags: Dict of tag name -> value.
        """
        if self._run is not None:
            mlflow.set_tags(tags)

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

    def log_dict_artifact(self, data: dict[str, Any], filename: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to serialize and log.
            filename: Name for the artifact file.
        """
        if self._run is None:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = Path(f.name)
        try:
            mlflow.log_artifact(str(temp_path))
            logger.info("Dict artifact logged", filename=filename)
        finally:
            temp_path.unlink(missing_ok=True)

    def log_figure(self, fig: Any, filename: str) -> None:
        """Log a matplotlib figure as a PNG artifact.

        Args:
            fig: Matplotlib figure object.
            filename: Name for the artifact file.
        """
        if self._run is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
        try:
            fig.savefig(str(temp_path), dpi=100, bbox_inches="tight")
            mlflow.log_artifact(str(temp_path))
            logger.info("Figure artifact logged", filename=filename)
        finally:
            temp_path.unlink(missing_ok=True)

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
