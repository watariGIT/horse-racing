"""CLI tool for comparing MLflow experiment runs.

Fetches recent runs from an MLflow experiment and displays a formatted
comparison table of parameters and metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import mlflow
import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger

__all__ = ["fetch_runs", "format_comparison_table"]

logger = get_logger(__name__)

PARAM_COLUMNS = ["model_type", "n_samples", "n_features"]
METRIC_COLUMNS = [
    "val_accuracy",
    "val_auc",
    "val_f1",
    "backtest_overall_win_accuracy",
    "backtest_overall_auc_roc",
]


def fetch_runs(experiment_name: str, max_results: int = 5) -> pd.DataFrame:
    """Fetch recent runs from an MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        max_results: Maximum number of runs to return.

    Returns:
        DataFrame of runs with run info, parameters, and metrics.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment not found", experiment_name=experiment_name)
        return pd.DataFrame()

    runs_df = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
            output_format="pandas",
        ),
    )
    if runs_df.empty:
        logger.info("No runs found", experiment_name=experiment_name)
    return runs_df


def format_comparison_table(
    runs: pd.DataFrame,
    sort_metric: str | None = None,
) -> pd.DataFrame:
    """Build a comparison table from raw MLflow run data.

    Args:
        runs: DataFrame returned by ``fetch_runs``.
        sort_metric: Metric column to sort by (descending). If the column
            is not present the original order is kept.

    Returns:
        Formatted DataFrame with selected columns.
    """
    if runs.empty:
        return pd.DataFrame()

    selected: dict[str, str] = {"run_name": "run_name", "start_time": "start_time"}
    for col in PARAM_COLUMNS:
        key = f"params.{col}"
        if key in runs.columns:
            selected[key] = col
    for col in METRIC_COLUMNS:
        key = f"metrics.{col}"
        if key in runs.columns:
            selected[key] = col

    available = {k: v for k, v in selected.items() if k in runs.columns}
    table = runs[list(available.keys())].rename(columns=available)

    if sort_metric and sort_metric in table.columns:
        table = table.sort_values(sort_metric, ascending=False, ignore_index=True)

    return table


def main(argv: list[str] | None = None) -> None:
    """Entry point for the compare-experiments CLI."""
    parser = argparse.ArgumentParser(
        description="Compare recent MLflow experiment runs.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of recent runs to fetch (default: 5).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Sort results by this metric (descending).",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export comparison table to a CSV file.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    runs = fetch_runs(settings.mlflow.experiment_name, max_results=args.last)
    if runs.empty:
        print("No runs found.")
        return

    table = format_comparison_table(runs, sort_metric=args.metric)
    print(table.to_string(index=False))

    if args.export:
        export_path = Path(args.export)
        table.to_csv(export_path, index=False)
        print(f"\nExported to {export_path}")


if __name__ == "__main__":
    main()
