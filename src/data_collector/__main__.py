"""CLI entry point for the data collector module.

Usage:
    uv run python -m src.data_collector --source kaggle
    uv run python -m src.data_collector --source kaggle \
        --date-from 2020-01-01 --date-to 2021-07-31
"""

from __future__ import annotations

import argparse
import sys

from src.common.logging import get_logger

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Horse Racing Data Collector",
        prog="python -m src.data_collector",
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "jra"],
        default="kaggle",
        help="Data source to use (default: kaggle)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate without persisting to storage",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the data collector CLI.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = _parse_args(argv)

    if args.source == "kaggle":
        from src.data_collector.importers.kaggle_importer import KaggleImporter
        from src.data_collector.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader()

        if args.dry_run:
            importer = KaggleImporter(loader=loader)
        else:
            from src.data_collector.storage.bq_writer import BQWriter
            from src.data_collector.storage.gcs_writer import GCSWriter

            gcs_writer = GCSWriter()
            bq_writer = BQWriter()
            importer = KaggleImporter(
                loader=loader,
                gcs_writer=gcs_writer,
                bq_writer=bq_writer,
            )

        result = importer.run(
            date_from=args.date_from,
            date_to=args.date_to,
        )

        logger.info(
            "Import complete",
            races=result.races_count,
            horse_results=result.horse_results_count,
            jockey_results=result.jockey_results_count,
            errors=len(result.validation_errors),
        )

        if result.validation_errors:
            for err in result.validation_errors:
                logger.warning("Validation", error=err)

        return 0

    elif args.source == "jra":
        logger.error("JRA API source requires jra_api_key configuration")
        logger.info("Use --source kaggle for free Kaggle dataset import")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
