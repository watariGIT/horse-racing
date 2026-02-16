"""Pipeline orchestrator for end-to-end ML execution.

Coordinates data loading, feature engineering, model training,
evaluation, and result persistence across the full pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import polars as pl

from src.common.config import get_settings
from src.common.logging import get_logger
from src.evaluator.backtest_engine import BacktestEngine, BacktestResult
from src.evaluator.reporter import Reporter
from src.feature_engineering.extractors.horse_features import HorseFeatureExtractor
from src.feature_engineering.extractors.race_features import RaceFeatureExtractor
from src.feature_engineering.pipeline import FeaturePipeline
from src.model_training.models.lgbm_classifier import LGBMClassifierModel
from src.model_training.trainer import ModelTrainer
from src.pipeline.data_preparer import DataPreparer

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full ML pipeline from data to evaluation.

    Stages:
        1. import_data: Load raw data from BigQuery or CSV
        2. prepare_features: Historical aggregation + feature extraction
        3. train_model: Train LightGBM model with temporal split
        4. evaluate_model: Walk-forward backtest
        5. save_results: Persist reports and metrics

    Args:
        date_from: Start date filter (YYYY-MM-DD).
        date_to: End date filter (YYYY-MM-DD).
        train_window: Training window in days for backtest.
        test_window: Test window in days for backtest.
        model_name: Model identifier for registry.
        data_source: Data source (``bigquery`` or ``csv``).
    """

    def __init__(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        train_window: int = 365,
        test_window: int = 30,
        model_name: str = "win_classifier",
        data_source: str | None = None,
    ) -> None:
        self._date_from = date_from
        self._date_to = date_to
        self._train_window = train_window
        self._test_window = test_window
        self._model_name = model_name
        self._data_source = data_source or os.getenv("PIPELINE_DATA_SOURCE", "csv")
        self._settings = get_settings()

        # Pipeline state
        self._raw_df: pl.DataFrame | None = None
        self._feature_df: pl.DataFrame | None = None
        self._model: LGBMClassifierModel | None = None
        self._train_metrics: dict[str, Any] = {}
        self._backtest_result: BacktestResult | None = None
        self._report_content: str = ""

    def run_full(self) -> dict[str, Any]:
        """Execute the full pipeline end-to-end.

        Returns:
            Dict with summary metrics and report path.
        """
        logger.info(
            "Starting full pipeline",
            data_source=self._data_source,
            date_from=self._date_from,
            date_to=self._date_to,
        )

        self.import_data()
        self.prepare_features()
        self.train_model()
        self.evaluate_model()
        result = self.save_results()

        logger.info("Full pipeline completed")
        return result

    def import_data(self) -> pl.DataFrame:
        """Load raw race data from configured source.

        Reads from BigQuery when ``data_source='bigquery'``,
        otherwise from local Kaggle CSV files.

        Returns:
            Raw race results DataFrame.
        """
        logger.info("Importing data", source=self._data_source)

        if self._data_source == "bigquery":
            self._raw_df = self._load_from_bigquery()
        else:
            self._raw_df = self._load_from_csv()

        logger.info("Data imported", rows=len(self._raw_df))
        return self._raw_df

    def prepare_features(self) -> pl.DataFrame:
        """Run history aggregation and feature extraction.

        Returns:
            DataFrame with all features and targets.

        Raises:
            RuntimeError: If import_data has not been called.
        """
        if self._raw_df is None:
            raise RuntimeError("import_data() must be called first")

        logger.info("Preparing features")

        # Step 1: Historical aggregation
        preparer = DataPreparer(n_past_races=5)
        prepared_df = preparer.prepare_for_training(self._raw_df)

        # Step 2: Feature extraction pipeline
        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor(), HorseFeatureExtractor()]
        )
        self._feature_df = pipeline.fit_transform(prepared_df)

        feature_cols = [c for c in self._feature_df.columns if c.startswith("feat_")]
        logger.info(
            "Features prepared",
            rows=len(self._feature_df),
            n_features=len(feature_cols),
            features=feature_cols,
        )
        return self._feature_df

    def train_model(self) -> dict[str, Any]:
        """Train LightGBM classifier with temporal validation.

        Returns:
            Training result dict with model and metrics.

        Raises:
            RuntimeError: If prepare_features has not been called.
        """
        if self._feature_df is None:
            raise RuntimeError("prepare_features() must be called first")

        logger.info("Training model", model_name=self._model_name)

        feature_cols = [c for c in self._feature_df.columns if c.startswith("feat_")]
        X = self._feature_df.select(feature_cols).to_pandas()
        y = self._feature_df["is_win"].to_pandas()

        race_dates: pd.Series | None = None
        if "race_date" in self._feature_df.columns:
            race_dates = self._feature_df["race_date"].to_pandas()

        self._model = LGBMClassifierModel()
        trainer = ModelTrainer()

        train_result = trainer.train(
            model=self._model,
            X=X,
            y=y,
            race_dates=race_dates,
        )
        self._train_metrics = train_result["metrics"]

        # Cross-validation
        cv_result = trainer.cross_validate(
            model=LGBMClassifierModel(),
            X=X,
            y=y,
            n_splits=5,
        )

        logger.info(
            "Training complete",
            train_metrics={
                k: round(v, 4)
                for k, v in self._train_metrics.items()
                if k.startswith("val_")
            },
            cv_mean_metrics={
                k: round(v, 4) for k, v in cv_result["mean_metrics"].items()
            },
        )

        return {
            "train_metrics": self._train_metrics,
            "cv_metrics": cv_result["mean_metrics"],
        }

    def evaluate_model(self) -> BacktestResult:
        """Run walk-forward backtest evaluation.

        Returns:
            BacktestResult with per-period and overall metrics.

        Raises:
            RuntimeError: If train_model has not been called.
        """
        if self._model is None or self._feature_df is None:
            raise RuntimeError("train_model() must be called first")

        logger.info(
            "Evaluating model",
            train_window=self._train_window,
            test_window=self._test_window,
        )

        engine = BacktestEngine(
            train_window_days=self._train_window,
            test_window_days=self._test_window,
            step_days=self._test_window,
        )
        self._backtest_result = engine.run(
            df=self._feature_df,
            model=LGBMClassifierModel(),
        )

        # Generate report
        reporter = Reporter(
            title="Horse Racing Model Backtest Report",
            model_name=self._model_name,
        )
        self._report_content = reporter.generate_backtest_report(self._backtest_result)

        logger.info(
            "Evaluation complete",
            n_periods=len(self._backtest_result.periods),
            overall_metrics=self._backtest_result.overall_metrics.to_dict(),
        )
        return self._backtest_result

    def save_results(self) -> dict[str, Any]:
        """Persist model, reports, and metrics to GCS/BigQuery.

        Saves:
            - Trained model to ModelRegistry (GCS)
            - Backtest report markdown to GCS
            - Feature importances JSON to GCS
            - Evaluation metrics to BigQuery

        Returns:
            Dict with saved artifact paths and summary.
        """
        if self._backtest_result is None:
            raise RuntimeError("evaluate_model() must be called first")

        result: dict[str, Any] = {
            "overall_metrics": self._backtest_result.overall_metrics.to_dict(),
            "n_periods": len(self._backtest_result.periods),
            "train_metrics": self._train_metrics,
        }

        # Log report to stdout even if GCP save fails
        logger.info(
            "Backtest report generated",
            report_length=len(self._report_content),
        )
        print(self._report_content)

        try:
            result.update(self._save_to_gcp())
        except Exception as e:
            logger.warning("GCP save failed (non-fatal)", error=str(e))
            result["gcp_save_error"] = str(e)

        return result

    def _save_to_gcp(self) -> dict[str, str]:
        """Upload artifacts to GCS and BigQuery.

        Returns:
            Dict of artifact type to GCS URI or table ID.
        """
        from src.common.gcp_client import BigQueryClient, GCSClient
        from src.model_training.model_registry import ModelRegistry

        gcs = GCSClient()
        saved: dict[str, str] = {}

        # Save model
        if self._model is not None:
            registry = ModelRegistry(gcs_client=gcs)
            version = registry.save_model(
                model=self._model,
                model_name=self._model_name,
                metrics=self._train_metrics,
            )
            saved["model_version"] = version

        # Save backtest report
        bucket = self._settings.gcs.bucket_processed
        if bucket and self._report_content:
            uri = gcs.upload_json(
                bucket_name=bucket,
                data={"report": self._report_content},
                destination_blob="reports/backtest_report.json",
            )
            saved["report_uri"] = uri

        # Save feature importances
        if self._model is not None and self._model.is_fitted:
            feature_cols = [
                c
                for c in (self._feature_df or pl.DataFrame()).columns
                if c.startswith("feat_")
            ]
            try:
                importances = self._model.feature_importances
                importance_data = dict(zip(feature_cols, importances.tolist()))
                uri = gcs.upload_json(
                    bucket_name=bucket,
                    data=importance_data,
                    destination_blob="reports/feature_importances.json",
                )
                saved["feature_importances_uri"] = uri
            except RuntimeError:
                logger.debug("Feature importances not available")

        # Save metrics to BigQuery
        if self._backtest_result is not None:
            try:
                bq = BigQueryClient()
                metrics_df = pd.DataFrame(
                    [self._backtest_result.overall_metrics.to_dict()]
                )
                metrics_df["model_name"] = self._model_name
                bq.load_dataframe(
                    df=metrics_df,
                    table_id="evaluation_results",
                    write_disposition="WRITE_APPEND",
                )
                saved["metrics_table"] = "evaluation_results"
            except Exception as e:
                logger.warning("BigQuery metrics save failed", error=str(e))

        return saved

    def _load_from_bigquery(self) -> pl.DataFrame:
        """Load raw data from BigQuery horse_results_raw table.

        Returns:
            Polars DataFrame of race results.
        """
        from src.common.gcp_client import BigQueryClient

        bq = BigQueryClient()
        dataset = self._settings.bigquery.dataset
        project = self._settings.gcp.project_id

        sql = f"SELECT * FROM `{project}.{dataset}.horse_results_raw`"
        conditions: list[str] = []

        if self._date_from:
            conditions.append(f"race_date >= '{self._date_from}'")
        if self._date_to:
            conditions.append(f"race_date <= '{self._date_to}'")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY race_date"

        logger.info("Loading from BigQuery", sql=sql[:200])
        pandas_df = bq.query(sql)
        return pl.from_pandas(pandas_df)

    def _load_from_csv(self) -> pl.DataFrame:
        """Load raw data from Kaggle CSV files.

        Returns:
            Polars DataFrame of race results.
        """
        from src.data_collector.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader()
        return loader.load_race_results(
            date_from=self._date_from,
            date_to=self._date_to,
        )
