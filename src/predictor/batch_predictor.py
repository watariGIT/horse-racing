"""Batch predictor for processing multiple races.

Provides batch prediction with result persistence to
BigQuery and GCS. Can be invoked as a CLI module:
    python -m src.predictor.batch_predictor
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import polars as pl

from src.common.config import get_settings
from src.common.gcp_client import BigQueryClient, GCSClient
from src.common.logging import get_logger, setup_logging
from src.feature_engineering.pipeline import FeaturePipeline
from src.predictor.model_loader import ModelLoader
from src.predictor.prediction_pipeline import PredictionPipeline

logger = get_logger(__name__)


class BatchPredictor:
    """Batch prediction executor for multiple races.

    Loads model, processes races in batches, and persists
    results to BigQuery and/or GCS.

    Args:
        model_name: Model name in the registry.
        model_version: Specific version (None for latest).
        feature_config: Feature pipeline configuration dict.
        model_loader: Optional pre-configured model loader.
    """

    def __init__(
        self,
        model_name: str = "win_classifier",
        model_version: str | None = None,
        feature_config: dict[str, Any] | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._feature_config = feature_config or {
            "extractors": ["race", "horse", "jockey"],
        }
        self._loader = model_loader or ModelLoader()

    def predict_batch(
        self,
        race_data: pl.DataFrame,
        save_to_bq: bool = False,
        save_to_gcs: bool = False,
    ) -> pd.DataFrame:
        """Run predictions on a batch of race data.

        Args:
            race_data: DataFrame with race entries to predict.
            save_to_bq: Persist results to BigQuery.
            save_to_gcs: Persist results to GCS.

        Returns:
            Prediction results DataFrame.
        """
        model = self._loader.load(self._model_name, self._model_version)
        feature_pipeline = FeaturePipeline.from_config(self._feature_config)

        # Fit pipeline on the input data (for encoding/scaling)
        feature_pipeline.fit(race_data)

        pipeline = PredictionPipeline(
            model=model,
            feature_pipeline=feature_pipeline,
        )

        results = pipeline.predict_with_ranking(race_data)

        # Add metadata
        results["predicted_at"] = datetime.now(timezone.utc).isoformat()
        results["model_name"] = self._model_name
        results["model_version"] = self._model_version or "latest"

        if save_to_bq:
            self._save_to_bigquery(results)

        if save_to_gcs:
            self._save_to_gcs(results)

        logger.info(
            "Batch prediction completed",
            n_races=results["race_id"].nunique() if "race_id" in results.columns else 0,
            n_predictions=len(results),
        )

        return results

    def _save_to_bigquery(self, results: pd.DataFrame) -> None:
        """Save prediction results to BigQuery."""
        bq_client = BigQueryClient()
        bq_client.load_dataframe(
            results,
            "predictions",
            write_disposition="WRITE_APPEND",
        )
        logger.info("Results saved to BigQuery", rows=len(results))

    def _save_to_gcs(self, results: pd.DataFrame) -> None:
        """Save prediction results to GCS as JSON."""
        settings = get_settings()
        gcs_client = GCSClient()

        now = datetime.now(timezone.utc)
        blob_path = (
            f"predictions/{now.strftime('%Y/%m/%d')}/"
            f"predictions_{now.strftime('%Y%m%d_%H%M%S')}.json"
        )
        gcs_client.upload_json(
            settings.gcs.bucket_processed,
            results.to_dict(orient="records"),
            blob_path,
        )
        logger.info("Results saved to GCS", path=blob_path)


def main() -> None:
    """CLI entry point for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Run batch predictions for horse racing"
    )
    parser.add_argument(
        "--model-name",
        default="win_classifier",
        help="Model name in the registry",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version (default: latest)",
    )
    parser.add_argument(
        "--save-bq",
        action="store_true",
        help="Save results to BigQuery",
    )
    parser.add_argument(
        "--save-gcs",
        action="store_true",
        help="Save results to GCS",
    )
    args = parser.parse_args()

    setup_logging()

    BatchPredictor(
        model_name=args.model_name,
        model_version=args.model_version,
    )

    logger.info(
        "Batch predictor initialized",
        model_name=args.model_name,
        model_version=args.model_version,
    )
    # In production, race_data would be loaded from BigQuery/GCS
    # For now, log that the predictor is ready
    logger.info("Batch predictor ready. Load race data to run predictions.")


if __name__ == "__main__":
    main()
