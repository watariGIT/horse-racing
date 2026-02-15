"""Performance monitoring and drift detection.

Tracks model performance over time, detects degradation against
configurable thresholds, and identifies data drift through basic
statistical comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from src.common.logging import get_logger
from src.evaluator.metrics import MetricsResult

logger = get_logger(__name__)


@dataclass
class Alert:
    """A single performance or drift alert."""

    alert_type: str  # "degradation" | "drift"
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MonitorReport:
    """Summary of a monitoring check."""

    alerts: list[Alert] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    is_healthy: bool = True


class PerformanceMonitor:
    """Monitors model performance over time.

    Keeps a running history of metric snapshots and checks for
    degradation against configurable thresholds.

    Args:
        thresholds: Dict mapping metric name to minimum acceptable value.
            If a metric falls below its threshold, an alert is raised.
        window_size: Number of recent snapshots used for trend analysis.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        window_size: int = 5,
    ) -> None:
        self._thresholds: dict[str, float] = thresholds or {
            "win_accuracy": 0.05,
            "place_accuracy": 0.20,
            "auc_roc": 0.55,
            "recovery_rate": 0.70,
        }
        self._window_size = window_size
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the full metrics history."""
        return self._history.copy()

    def record(self, metrics: MetricsResult, label: str = "") -> None:
        """Record a metrics snapshot.

        Args:
            metrics: Computed metrics to record.
            label: Optional label for the snapshot (e.g. date range).
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
        }
        entry.update(metrics.to_dict())
        self._history.append(entry)
        logger.info("Metrics recorded", label=label, n_history=len(self._history))

    def check(self) -> MonitorReport:
        """Run degradation checks on the latest recorded metrics.

        Returns:
            MonitorReport with any alerts found.
        """
        alerts: list[Alert] = []

        if not self._history:
            return MonitorReport(
                alerts=alerts,
                metrics_history=self._history.copy(),
                is_healthy=True,
            )

        latest = self._history[-1]

        # Threshold check
        for metric_name, threshold in self._thresholds.items():
            value = latest.get(metric_name)
            if value is not None and value < threshold:
                alerts.append(
                    Alert(
                        alert_type="degradation",
                        metric_name=metric_name,
                        current_value=value,
                        threshold=threshold,
                        message=(
                            f"{metric_name} = {value:.4f} is below "
                            f"threshold {threshold:.4f}"
                        ),
                    )
                )

        # Trend check: compare recent window average vs previous window
        if len(self._history) >= self._window_size * 2:
            recent = self._history[-self._window_size :]
            previous = self._history[-self._window_size * 2 : -self._window_size]

            for metric_name in self._thresholds:
                recent_vals = [h[metric_name] for h in recent if metric_name in h]
                prev_vals = [h[metric_name] for h in previous if metric_name in h]

                if recent_vals and prev_vals:
                    recent_avg = float(np.mean(recent_vals))
                    prev_avg = float(np.mean(prev_vals))
                    if prev_avg > 0:
                        change = (recent_avg - prev_avg) / prev_avg
                        if change < -0.1:
                            alerts.append(
                                Alert(
                                    alert_type="degradation",
                                    metric_name=metric_name,
                                    current_value=recent_avg,
                                    threshold=prev_avg,
                                    message=(
                                        f"{metric_name} dropped {abs(change):.1%} "
                                        f"over recent {self._window_size} periods "
                                        f"(from {prev_avg:.4f} to {recent_avg:.4f})"
                                    ),
                                )
                            )

        is_healthy = len(alerts) == 0
        if not is_healthy:
            logger.warning("Performance alerts detected", n_alerts=len(alerts))

        return MonitorReport(
            alerts=alerts,
            metrics_history=self._history.copy(),
            is_healthy=is_healthy,
        )

    def detect_data_drift(
        self,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        columns: list[str] | None = None,
        threshold: float = 0.1,
    ) -> list[Alert]:
        """Detect data drift by comparing summary statistics.

        Compares mean and standard deviation of numeric columns between
        a reference distribution and current data.

        Args:
            reference_df: Baseline feature DataFrame.
            current_df: Current feature DataFrame.
            columns: Columns to check. Defaults to all numeric columns.
            threshold: Relative change threshold to trigger alert.

        Returns:
            List of drift alerts.
        """
        alerts: list[Alert] = []

        if columns is None:
            columns = [
                c
                for c in reference_df.columns
                if reference_df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]

        for col in columns:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_mean = reference_df[col].mean()
            cur_mean = current_df[col].mean()
            ref_std = reference_df[col].std()
            cur_std = current_df[col].std()

            if ref_mean is None or cur_mean is None:
                continue

            # Mean drift check
            if abs(ref_mean) > 1e-10:
                mean_change = abs(cur_mean - ref_mean) / abs(ref_mean)
                if mean_change > threshold:
                    alerts.append(
                        Alert(
                            alert_type="drift",
                            metric_name=f"{col}_mean",
                            current_value=float(cur_mean),
                            threshold=float(ref_mean),
                            message=(
                                f"Column '{col}' mean shifted by {mean_change:.1%} "
                                f"(ref={float(ref_mean):.4f}, cur={float(cur_mean):.4f})"
                            ),
                        )
                    )

            # Std drift check
            if ref_std is not None and cur_std is not None and abs(ref_std) > 1e-10:
                std_change = abs(cur_std - ref_std) / abs(ref_std)
                if std_change > threshold:
                    alerts.append(
                        Alert(
                            alert_type="drift",
                            metric_name=f"{col}_std",
                            current_value=float(cur_std),
                            threshold=float(ref_std),
                            message=(
                                f"Column '{col}' std shifted by {std_change:.1%} "
                                f"(ref={float(ref_std):.4f}, cur={float(cur_std):.4f})"
                            ),
                        )
                    )

        if alerts:
            logger.warning("Data drift detected", n_drifted=len(alerts))

        return alerts

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
