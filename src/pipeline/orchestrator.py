"""Pipeline orchestrator for end-to-end ML execution.

Coordinates data loading, feature engineering, model training,
evaluation, and result persistence across the full pipeline.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import polars as pl

from src.common.config import get_settings
from src.common.logging import get_logger
from src.evaluator.backtest_engine import BacktestEngine, BacktestResult
from src.evaluator.reporter import Reporter
from src.feature_engineering.pipeline import FeaturePipeline
from src.model_training.experiment_tracker import ExperimentTracker
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
        train_window: Training window in days for backtest (default: from config).
        test_window: Test window in days for backtest (default: from config).
        model_name: Model identifier for registry.
        data_source: Data source (``bigquery`` or ``csv``).
    """

    def __init__(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        train_window: int | None = None,
        test_window: int | None = None,
        model_name: str = "win_classifier",
        data_source: str | None = None,
    ) -> None:
        self._date_from = date_from
        self._date_to = date_to
        self._model_name = model_name
        self._data_source = data_source or os.getenv("PIPELINE_DATA_SOURCE", "bigquery")
        self._settings = get_settings()

        # Resolve backtest parameters from config when not explicitly provided
        self._train_window = (
            train_window
            if train_window is not None
            else self._settings.backtest.train_window_days
        )
        self._test_window = (
            test_window
            if test_window is not None
            else self._settings.backtest.test_window_days
        )
        self._step_days = self._settings.backtest.step_days

        # Apply default date range (last 5 years) when not specified
        if self._date_from is None and self._date_to is None:
            default_to = date(2021, 7, 31)
            default_from = default_to - timedelta(days=365 * 5)
            self._date_from = default_from.isoformat()
            self._date_to = default_to.isoformat()
            logger.info(
                "No date range specified, using default period",
                date_from=self._date_from,
                date_to=self._date_to,
            )

        # Pipeline state
        self._raw_df: pl.DataFrame | None = None
        self._feature_df: pl.DataFrame | None = None
        self._model: LGBMClassifierModel | None = None
        self._train_metrics: dict[str, Any] = {}
        self._backtest_result: BacktestResult | None = None
        self._report_content: str = ""

        # MLflow tracking
        self._tracker: ExperimentTracker | None = None
        if self._settings.mlflow.enabled:
            self._tracker = ExperimentTracker.from_config(self._settings.mlflow)

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
        """Run history aggregation and feature extraction."""
        if self._raw_df is None:
            raise RuntimeError("import_data() must be called first")

        logger.info("Preparing features")

        # Step 1: Historical aggregation
        preparer = DataPreparer(n_past_races=5)
        prepared_df = preparer.prepare_for_training(self._raw_df)

        # Step 2: Feature extraction pipeline (v2)
        pipeline = FeaturePipeline.from_config(
            {"extractors": self._settings.feature_pipeline.extractors}
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

        if self._tracker:
            run_name = ExperimentTracker.generate_run_name(self._model.model_type)
            self._tracker.start_run(run_name=run_name)
            self._tracker.set_tags(
                {
                    "environment": self._settings.environment.value,
                    "model_type": self._model.model_type,
                    "feature_version": self._settings.model.feature_version,
                    "date_from": self._date_from or "",
                    "date_to": self._date_to or "",
                    "github.pr_number": os.getenv("GITHUB_PR_NUMBER", ""),
                    "github.commit_sha": os.getenv("GITHUB_COMMIT_SHA", ""),
                    "github.branch": os.getenv("GITHUB_BRANCH", ""),
                    "github.repository": os.getenv("GITHUB_REPOSITORY", ""),
                }
            )
            self._tracker.log_params(
                {
                    "n_samples": len(X),
                    "n_features": len(feature_cols),
                    "date_from": self._date_from or "",
                    "date_to": self._date_to or "",
                    "calibration_method": self._settings.model.calibration_method,
                    "optimize_threshold": self._settings.model.optimize_threshold,
                }
            )

        trainer = ModelTrainer(tracker=self._tracker)

        train_result = trainer.train(
            model=self._model,
            X=X,
            y=y,
            race_dates=race_dates,
            calibration_method=self._settings.model.calibration_method,
            optimize_threshold=self._settings.model.optimize_threshold,
        )
        self._train_metrics = train_result["metrics"]

        # Cross-validation
        cv_result = trainer.cross_validate(
            model=LGBMClassifierModel(),
            X=X,
            y=y,
            n_splits=5,
        )

        self._log_training_to_mlflow(feature_cols)

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
            step_days=self._step_days,
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

        self._log_backtest_to_mlflow()

        logger.info(
            "Evaluation complete",
            n_periods=len(self._backtest_result.periods),
            overall_metrics=self._backtest_result.overall_metrics.to_dict(),
        )
        return self._backtest_result

    def save_results(self) -> dict[str, Any]:
        """Persist model, reports, and metrics to GCS/BigQuery."""
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

        if self._tracker:
            self._tracker.end_run()

        return result

    def _save_to_gcp(self) -> dict[str, str]:
        """Upload model to GCS via ModelRegistry.

        Backtest reports and feature importances are stored exclusively
        in MLflow; evaluation metrics are logged as MLflow metrics.
        """
        from src.common.gcp_client import GCSClient
        from src.model_training.model_registry import ModelRegistry

        gcs = GCSClient()
        saved: dict[str, str] = {}

        if self._model is not None:
            registry = ModelRegistry(gcs_client=gcs)
            extra_metadata: dict[str, Any] = {}
            if self._tracker and self._tracker.run_id:
                extra_metadata["mlflow_run_id"] = self._tracker.run_id
            version = registry.save_model(
                model=self._model,
                model_name=self._model_name,
                metrics=self._train_metrics,
                extra_metadata=extra_metadata,
            )
            saved["model_version"] = version

        return saved

    def _log_training_to_mlflow(self, feature_cols: list[str]) -> None:
        """Log feature importances and chart to MLflow."""
        if not self._tracker or self._model is None or not self._model.is_fitted:
            return

        try:
            importances = self._model.feature_importances
            importance_data = dict(zip(feature_cols, importances.tolist()))
            self._tracker.log_dict_artifact(importance_data, "feature_importances.json")
        except RuntimeError:
            logger.debug("Feature importances not available for MLflow")
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sorted_pairs = sorted(
                importance_data.items(), key=lambda x: x[1], reverse=True
            )[:20]
            names = [p[0] for p in sorted_pairs]
            values = [p[1] for p in sorted_pairs]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(names[::-1], values[::-1])
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Feature Importances")
            fig.tight_layout()

            self._tracker.log_figure(fig, "feature_importances.png")
            plt.close(fig)
        except ImportError:
            logger.debug("matplotlib not available, skipping chart")

    def _log_backtest_to_mlflow(self) -> None:
        """Log backtest results to MLflow."""
        if not self._tracker or self._backtest_result is None:
            return

        for period in self._backtest_result.periods:
            pm = {
                f"backtest_{k}": v
                for k, v in period.metrics.to_dict().items()
                if isinstance(v, (int, float))
            }
            self._tracker.log_metrics(pm, step=period.period_index)

        overall = self._backtest_result.overall_metrics.to_dict()
        self._tracker.log_metrics(
            {
                f"backtest_overall_{k}": v
                for k, v in overall.items()
                if isinstance(v, (int, float))
            },
        )
        self._tracker.log_params(
            {
                "backtest_train_window": self._train_window,
                "backtest_test_window": self._test_window,
                "backtest_step_days": self._step_days,
                "backtest_n_periods": len(self._backtest_result.periods),
            }
        )
        self._tracker.log_dict_artifact(
            {
                "overall_metrics": overall,
                "periods": [
                    {
                        "period": p.period_index,
                        "test_range": f"{p.test_start} - {p.test_end}",
                        "n_test": p.n_test,
                        "metrics": p.metrics.to_dict(),
                    }
                    for p in self._backtest_result.periods
                ],
            },
            "backtest_results.json",
        )

    def _load_from_bigquery(self) -> pl.DataFrame:
        """Load raw data from BigQuery horse_results_raw table."""
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
        """Load raw data from Kaggle CSV files."""
        from src.data_collector.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader()
        return loader.load_race_results(
            date_from=self._date_from,
            date_to=self._date_to,
        )
