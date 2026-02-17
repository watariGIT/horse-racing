"""Report generation for backtest and evaluation results.

Produces Markdown-formatted reports with summary tables
and period-wise performance trends.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.common.logging import get_logger
from src.evaluator.backtest_engine import BacktestResult
from src.evaluator.metrics import MetricsResult
from src.evaluator.monitor import MonitorReport

logger = get_logger(__name__)


class Reporter:
    """Generates Markdown evaluation reports.

    Args:
        title: Report title.
        model_name: Name/identifier of the model being evaluated.
    """

    def __init__(
        self,
        title: str = "Horse Racing Model Evaluation Report",
        model_name: str = "unnamed",
    ) -> None:
        self._title = title
        self._model_name = model_name

    def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a full backtest report in Markdown.

        Args:
            result: BacktestResult from BacktestEngine.run().

        Returns:
            Markdown-formatted report string.
        """
        lines: list[str] = []
        lines.append(f"# {self._title}")
        lines.append("")
        lines.append(f"**Model**: {self._model_name}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Periods**: {len(result.periods)}")
        lines.append("")

        # Overall metrics summary
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append(self._metrics_table(result.overall_metrics))
        lines.append("")

        # Period-by-period table
        if result.periods:
            lines.append("## Period-wise Results")
            lines.append("")
            lines.append(self._period_table(result))
            lines.append("")

        return "\n".join(lines)

    def generate_metrics_report(self, metrics: MetricsResult) -> str:
        """Generate a summary report for a single metrics evaluation.

        Args:
            metrics: MetricsResult to report.

        Returns:
            Markdown-formatted report string.
        """
        lines: list[str] = []
        lines.append(f"# {self._title}")
        lines.append("")
        lines.append(f"**Model**: {self._model_name}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Metrics Summary")
        lines.append("")
        lines.append(self._metrics_table(metrics))
        lines.append("")

        return "\n".join(lines)

    def generate_monitor_report(self, report: MonitorReport) -> str:
        """Generate a monitoring report in Markdown.

        Args:
            report: MonitorReport from PerformanceMonitor.check().

        Returns:
            Markdown-formatted report string.
        """
        lines: list[str] = []
        lines.append("# Performance Monitoring Report")
        lines.append("")
        lines.append(f"**Model**: {self._model_name}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        status = "HEALTHY" if report.is_healthy else "ALERT"
        lines.append(f"**Status**: {status}")
        lines.append("")

        if report.alerts:
            lines.append("## Alerts")
            lines.append("")
            lines.append("| Type | Metric | Current | Threshold | Message |")
            lines.append("|------|--------|---------|-----------|---------|")
            for alert in report.alerts:
                lines.append(
                    f"| {alert.alert_type} | {alert.metric_name} "
                    f"| {alert.current_value:.4f} | {alert.threshold:.4f} "
                    f"| {alert.message} |"
                )
            lines.append("")
        else:
            lines.append("No alerts. All metrics within acceptable ranges.")
            lines.append("")

        return "\n".join(lines)

    def save(self, content: str, path: str | Path) -> Path:
        """Write report content to a file.

        Args:
            content: Markdown report string.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        logger.info("Report saved", path=str(out))
        return out

    def generate_leakage_report(
        self,
        result: BacktestResult,
        train_auc: float | None = None,
    ) -> str:
        """Generate a temporal leakage verification report in Markdown.

        Analyzes per-period AUC-ROC values to detect anomalous spikes
        that may indicate data leakage, and optionally compares train
        vs validation AUC.

        Args:
            result: BacktestResult from BacktestEngine.run().
            train_auc: Optional training AUC-ROC for gap analysis.

        Returns:
            Markdown-formatted leakage verification report.
        """
        lines: list[str] = []
        lines.append("# Temporal Leakage Verification Report")
        lines.append("")
        lines.append(f"**Model**: {self._model_name}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Periods**: {len(result.periods)}")
        lines.append("")

        # Per-period AUC table
        auc_values: list[float] = []
        lines.append("## Per-period AUC")
        lines.append("")
        lines.append("| Period | Test Range | N Test | AUC ROC |")
        lines.append("|--------|------------|--------|---------|")
        for p in result.periods:
            auc = p.metrics.values.get("auc_roc", 0.0)
            auc_values.append(auc)
            lines.append(
                f"| {p.period_index} "
                f"| {p.test_start} ~ {p.test_end} "
                f"| {p.n_test} | {auc:.4f} |"
            )
        lines.append("")

        # Summary statistics
        warnings: list[str] = []

        if auc_values:
            import statistics

            mean_auc = statistics.mean(auc_values)
            std_auc = statistics.stdev(auc_values) if len(auc_values) > 1 else 0.0
            min_auc = min(auc_values)
            max_auc = max(auc_values)
            cv = std_auc / mean_auc if mean_auc > 0 else 0.0

            lines.append("## AUC Summary Statistics")
            lines.append("")
            lines.append(f"- **Mean**: {mean_auc:.4f}")
            lines.append(f"- **Std**: {std_auc:.4f}")
            lines.append(f"- **Min**: {min_auc:.4f}")
            lines.append(f"- **Max**: {max_auc:.4f}")
            lines.append(f"- **Coefficient of Variation**: {cv:.4f}")
            lines.append("")

            # Anomaly detection
            lines.append("## Anomaly Detection")
            lines.append("")
            threshold = mean_auc + 2 * std_auc
            anomalies_found = False
            for p, auc in zip(result.periods, auc_values):
                if auc > threshold and std_auc > 0:
                    msg = (
                        f"WARNING: Period {p.period_index} AUC "
                        f"{auc:.4f} > {threshold:.4f} "
                        f"(mean + 2*std) - potential leakage"
                    )
                    lines.append(msg)
                    warnings.append(msg)
                    anomalies_found = True
            if not anomalies_found:
                lines.append("No anomalous AUC spikes detected.")
            lines.append("")

            # Train vs Val AUC assessment
            if train_auc is not None:
                lines.append("## Train vs Val AUC Assessment")
                lines.append("")
                gap = train_auc - mean_auc
                lines.append(f"- **Train AUC**: {train_auc:.4f}")
                lines.append(f"- **Val AUC (mean)**: {mean_auc:.4f}")
                lines.append(f"- **Gap**: {gap:.4f}")
                if gap < 0.1:
                    lines.append("- Within acceptable range.")
                else:
                    msg = "WARNING: Large gap may indicate " "overfitting or leakage"
                    lines.append(f"- {msg}")
                    warnings.append(msg)
                lines.append("")

        # Verdict
        lines.append("## Verdict")
        lines.append("")
        if not warnings:
            lines.append("No significant temporal data leakage detected.")
        else:
            lines.append("Potential issues detected:")
            lines.append("")
            for w in warnings:
                lines.append(f"- {w}")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_table(metrics: MetricsResult) -> str:
        """Format metrics as a Markdown table."""
        lines: list[str] = []

        accuracy_keys = ["win_accuracy", "place_accuracy", "top3_accuracy"]
        ml_keys = [
            "auc_roc",
            "precision",
            "recall",
            "f1",
            "log_loss",
            "ndcg",
            "ndcg_at_3",
        ]
        profit_keys = ["recovery_rate", "roi", "expected_value"]

        all_values = metrics.to_dict()

        sections: list[tuple[str, list[str]]] = [
            ("Accuracy", accuracy_keys),
            ("ML Performance", ml_keys),
            ("Profitability", profit_keys),
        ]

        for section_name, keys in sections:
            present = {k: all_values[k] for k in keys if k in all_values}
            if not present:
                continue
            lines.append(f"### {section_name}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in present.items():
                display_name = k.replace("_", " ").title()
                lines.append(f"| {display_name} | {v:.4f} |")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _period_table(result: BacktestResult) -> str:
        """Format period results as a Markdown table."""
        if not result.periods:
            return "No periods to display."

        # Collect common metric keys from first period
        metric_keys = list(result.periods[0].metrics.to_dict().keys())
        display_keys = metric_keys[:6]  # Limit columns for readability

        header = (
            "| Period | Test Range | N Test | "
            + " | ".join(k.replace("_", " ").title() for k in display_keys)
            + " |"
        )
        separator = (
            "|--------|------------|--------|"
            + "|".join("-------" for _ in display_keys)
            + "|"
        )

        lines: list[str] = [header, separator]
        for p in result.periods:
            values = p.metrics.to_dict()
            metric_cells = " | ".join(f"{values.get(k, 0):.4f}" for k in display_keys)
            lines.append(
                f"| {p.period_index} | {p.test_start} ~ {p.test_end} "
                f"| {p.n_test} | {metric_cells} |"
            )

        return "\n".join(lines)
