"""CLI entry point for the pipeline orchestrator.

Usage:
    uv run python -m src.pipeline --stage full
    uv run python -m src.pipeline --stage train \
        --date-from 2015-01-01 --date-to 2021-07-31
"""

from __future__ import annotations

import argparse
import sys

from src.common.logging import get_logger, setup_logging

logger = get_logger(__name__)

_STAGES = ("import", "features", "train", "evaluate", "full")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Horse Racing ML Pipeline Orchestrator",
        prog="python -m src.pipeline",
    )
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        default="full",
        help="Pipeline stage to run (default: full)",
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
        "--train-window",
        type=int,
        default=365,
        help="Training window in days for backtest (default: 365)",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=30,
        help="Test window in days for backtest (default: 30)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="win_classifier",
        help="Model name for registry (default: win_classifier)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline CLI.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    setup_logging()
    args = _parse_args(argv)

    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        date_from=args.date_from,
        date_to=args.date_to,
        train_window=args.train_window,
        test_window=args.test_window,
        model_name=args.model_name,
    )

    try:
        if args.stage == "full":
            result = orchestrator.run_full()
        elif args.stage == "import":
            orchestrator.import_data()
            result = {"stage": "import", "status": "complete"}
        elif args.stage == "features":
            orchestrator.import_data()
            orchestrator.prepare_features()
            result = {"stage": "features", "status": "complete"}
        elif args.stage == "train":
            orchestrator.import_data()
            orchestrator.prepare_features()
            result = orchestrator.train_model()
        elif args.stage == "evaluate":
            result = orchestrator.run_full()
        else:
            logger.error("Unknown stage", stage=args.stage)
            return 1

        logger.info("Pipeline completed", stage=args.stage, result=result)
        return 0

    except Exception as e:
        logger.error("Pipeline failed", stage=args.stage, error=str(e))
        raise


if __name__ == "__main__":
    sys.exit(main())
