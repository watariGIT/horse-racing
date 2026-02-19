"""Configuration management using Pydantic Settings.

Provides environment-aware configuration with YAML file support,
environment variable overrides, and GCP Secret Manager integration.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEV = "dev"
    PROD = "prod"


class GCPConfig(BaseModel):
    project_id: str = ""
    region: str = "us-central1"
    credentials_path: str = ""


class BigQueryConfig(BaseModel):
    dataset: str = "horse_racing"
    location: str = "US"


class GCSConfig(BaseModel):
    bucket_raw: str = ""
    bucket_processed: str = ""
    bucket_models: str = ""
    lifecycle_days: int = 90


class CloudRunConfig(BaseModel):
    memory: str = "512Mi"
    cpu: str = "1"
    concurrency: int = 1
    timeout: int = 300


class KaggleConfig(BaseModel):
    data_dir: str = "data/raw/kaggle"
    race_result_file: str = "race_result.csv"


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "horse-racing-prediction"
    enabled: bool = True


class BacktestConfig(BaseModel):
    train_window_days: int = 365
    test_window_days: int = 30
    step_days: int = 30


class ModelConfig(BaseModel):
    default_type: str = "lgbm_classifier"
    feature_version: str = "v1"
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    optimize_threshold: bool = True


class FeaturePipelineConfig(BaseModel):
    extractors: list[str] = Field(default=["race", "horse", "jockey", "running_style"])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class AppSettings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    environment: Environment = Environment.DEV
    gcp_project_id: str = ""
    google_application_credentials: str = ""
    jra_api_key: str = ""

    gcp: GCPConfig = Field(default_factory=GCPConfig)
    bigquery: BigQueryConfig = Field(default_factory=BigQueryConfig)
    gcs: GCSConfig = Field(default_factory=GCSConfig)
    cloud_run: CloudRunConfig = Field(default_factory=CloudRunConfig)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    feature_pipeline: FeaturePipelineConfig = Field(
        default_factory=FeaturePipelineConfig
    )
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${VAR} placeholders in config values from environment."""
    import re

    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, str):
            has_unresolved = False

            def _replace(m: re.Match[str]) -> str:
                nonlocal has_unresolved
                env_val = os.getenv(m.group(1))
                if env_val is None:
                    has_unresolved = True
                    return ""
                return env_val

            resolved = re.sub(r"\$\{(\w+)\}", _replace, value)
            # Skip values with unresolved env vars so Pydantic defaults apply
            if not has_unresolved:
                result[key] = resolved
        else:
            result[key] = value
    return result


def _load_yaml_config(env: Environment) -> dict[str, Any]:
    """Load and merge YAML config files (base + environment-specific)."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "config"

    base_path = config_dir / "base.yaml"
    env_path = config_dir / f"{env.value}.yaml"

    config: dict[str, Any] = {}

    if base_path.exists():
        with open(base_path) as f:
            base_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, base_config)

    if env_path.exists():
        with open(env_path) as f:
            env_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, env_config)

    return _resolve_env_vars(config)


def _resolve_gcs_buckets(settings: AppSettings) -> AppSettings:
    """Auto-generate GCS bucket names from project ID if not explicitly set."""
    project_id = settings.gcp_project_id or settings.gcp.project_id
    if project_id:
        if not settings.gcs.bucket_raw:
            settings.gcs.bucket_raw = f"{project_id}-raw-data"
        if not settings.gcs.bucket_processed:
            settings.gcs.bucket_processed = f"{project_id}-processed"
        if not settings.gcs.bucket_models:
            settings.gcs.bucket_models = f"{project_id}-models"
        settings.gcp.project_id = project_id
    return settings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings.

    Loads configuration in this order (later overrides earlier):
    1. Default values
    2. base.yaml
    3. {environment}.yaml
    4. Environment variables
    """
    env_str = os.getenv("ENVIRONMENT", "dev")
    env = Environment(env_str)

    yaml_config = _load_yaml_config(env)
    settings = AppSettings(environment=env, **yaml_config)
    settings = _resolve_gcs_buckets(settings)

    return settings


def get_secret(secret_id: str, project_id: str | None = None) -> str:
    """Retrieve a secret from GCP Secret Manager.

    Args:
        secret_id: The secret ID in Secret Manager.
        project_id: GCP project ID. Defaults to configured project.

    Returns:
        The secret value as a string.
    """
    from google.cloud import secretmanager

    settings = get_settings()
    project = project_id or settings.gcp.project_id

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})

    return response.payload.data.decode("utf-8")
