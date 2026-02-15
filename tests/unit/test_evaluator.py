"""Unit tests for the evaluator module.

Tests metrics accuracy with known inputs/outputs, backtest engine
basic operations, performance monitor, and reporter.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluator.backtest_engine import BacktestEngine, BacktestResult
from src.evaluator.metrics import MetricsResult, RacingMetrics
from src.evaluator.monitor import PerformanceMonitor
from src.evaluator.reporter import Reporter

# ---------------------------------------------------------------------------
# RacingMetrics tests
# ---------------------------------------------------------------------------


class TestWinAccuracy:
    """Tests for win_accuracy metric."""

    def test_perfect_prediction(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R2", "R2"],
                "horse_id": ["H1", "H2", "H3", "H4"],
                "predicted_rank": [1, 2, 1, 2],
                "actual_position": [1, 2, 1, 2],
            }
        )
        assert RacingMetrics.win_accuracy(df) == 1.0

    def test_zero_accuracy(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R2", "R2"],
                "horse_id": ["H1", "H2", "H3", "H4"],
                "predicted_rank": [1, 2, 1, 2],
                "actual_position": [2, 1, 2, 1],
            }
        )
        assert RacingMetrics.win_accuracy(df) == 0.0

    def test_partial_accuracy(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R2", "R2"],
                "horse_id": ["H1", "H2", "H3", "H4"],
                "predicted_rank": [1, 2, 1, 2],
                "actual_position": [1, 2, 3, 1],
            }
        )
        # R1 correct, R2 wrong -> 0.5
        assert RacingMetrics.win_accuracy(df) == pytest.approx(0.5)

    def test_empty_dataframe(self):
        df = pl.DataFrame(
            {
                "predicted_rank": pl.Series([], dtype=pl.Int64),
                "actual_position": pl.Series([], dtype=pl.Int64),
            }
        )
        assert RacingMetrics.win_accuracy(df) == 0.0


class TestPlaceAccuracy:
    """Tests for place_accuracy metric."""

    def test_place_top3(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "horse_id": ["H1", "H2", "H3"],
                "predicted_rank": [1, 2, 3],
                "actual_position": [3, 1, 2],
            }
        )
        # Top pick (predicted_rank=1) finished 3rd -> within top 3
        assert RacingMetrics.place_accuracy(df, top_n=3) == 1.0

    def test_place_outside_top3(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R1", "R1", "R1"],
                "horse_id": ["H1", "H2", "H3", "H4"],
                "predicted_rank": [1, 2, 3, 4],
                "actual_position": [4, 1, 2, 3],
            }
        )
        # Top pick finished 4th -> outside top 3
        assert RacingMetrics.place_accuracy(df, top_n=3) == 0.0


class TestTopNAccuracy:
    """Tests for top_n_accuracy metric."""

    def test_perfect_top3(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1"] * 5,
                "horse_id": ["H1", "H2", "H3", "H4", "H5"],
                "predicted_rank": [1, 2, 3, 4, 5],
                "actual_position": [1, 2, 3, 4, 5],
            }
        )
        assert RacingMetrics.top_n_accuracy(df, n=3) == pytest.approx(1.0)

    def test_partial_top3(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1"] * 5,
                "horse_id": ["H1", "H2", "H3", "H4", "H5"],
                "predicted_rank": [1, 2, 3, 4, 5],
                "actual_position": [1, 4, 5, 2, 3],
            }
        )
        # Predicted top-3: H1, H2, H3 -> actual top-3: H1, H4, H5
        # Overlap: {H1} -> 1/3
        assert RacingMetrics.top_n_accuracy(df, n=3) == pytest.approx(1 / 3)


class TestRecoveryRate:
    """Tests for recovery_rate and ROI."""

    def test_break_even(self):
        df = pl.DataFrame(
            {
                "bet_amount": [100.0, 100.0],
                "payout": [150.0, 50.0],
            }
        )
        assert RacingMetrics.recovery_rate(df) == pytest.approx(1.0)

    def test_profit(self):
        df = pl.DataFrame(
            {
                "bet_amount": [100.0, 100.0],
                "payout": [300.0, 0.0],
            }
        )
        assert RacingMetrics.recovery_rate(df) == pytest.approx(1.5)
        assert RacingMetrics.roi(df) == pytest.approx(0.5)

    def test_total_loss(self):
        df = pl.DataFrame(
            {
                "bet_amount": [100.0, 100.0],
                "payout": [0.0, 0.0],
            }
        )
        assert RacingMetrics.recovery_rate(df) == pytest.approx(0.0)
        assert RacingMetrics.roi(df) == pytest.approx(-1.0)

    def test_zero_wagered(self):
        df = pl.DataFrame(
            {
                "bet_amount": [0.0],
                "payout": [0.0],
            }
        )
        assert RacingMetrics.recovery_rate(df) == 0.0
        assert RacingMetrics.roi(df) == 0.0


class TestExpectedValue:
    """Tests for expected_value metric."""

    def test_positive_ev(self):
        df = pl.DataFrame(
            {
                "predicted_prob": [0.5],
                "odds": [3.0],
            }
        )
        # EV = 0.5 * 3.0 - 1 = 0.5
        assert RacingMetrics.expected_value(df) == pytest.approx(0.5)

    def test_negative_ev(self):
        df = pl.DataFrame(
            {
                "predicted_prob": [0.1],
                "odds": [3.0],
            }
        )
        # EV = 0.1 * 3.0 - 1 = -0.7
        assert RacingMetrics.expected_value(df) == pytest.approx(-0.7)

    def test_empty(self):
        df = pl.DataFrame(
            {
                "predicted_prob": pl.Series([], dtype=pl.Float64),
                "odds": pl.Series([], dtype=pl.Float64),
            }
        )
        assert RacingMetrics.expected_value(df) == 0.0


class TestAucRoc:
    """Tests for AUC-ROC metric."""

    def test_perfect_separation(self):
        df = pl.DataFrame(
            {
                "is_win": [1, 1, 0, 0, 0],
                "predicted_prob": [0.9, 0.8, 0.3, 0.2, 0.1],
            }
        )
        assert RacingMetrics.auc_roc(df) == pytest.approx(1.0)

    def test_random_performance(self):
        np.random.seed(42)
        n = 1000
        y = np.random.randint(0, 2, n)
        probs = np.random.rand(n)
        df = pl.DataFrame({"is_win": y, "predicted_prob": probs})
        auc = RacingMetrics.auc_roc(df)
        # Should be approximately 0.5 for random predictions
        assert 0.4 < auc < 0.6

    def test_all_same_class(self):
        df = pl.DataFrame(
            {
                "is_win": [0, 0, 0],
                "predicted_prob": [0.5, 0.3, 0.7],
            }
        )
        assert RacingMetrics.auc_roc(df) == 0.0


class TestPrecisionRecallF1:
    """Tests for precision, recall, F1 metrics."""

    def test_perfect_predictions(self):
        df = pl.DataFrame(
            {
                "is_win": [1, 0, 1, 0],
                "predicted_win": [1, 0, 1, 0],
            }
        )
        result = RacingMetrics.precision_recall_f1(df)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_no_true_positives(self):
        df = pl.DataFrame(
            {
                "is_win": [0, 0, 1, 1],
                "predicted_win": [1, 1, 0, 0],
            }
        )
        result = RacingMetrics.precision_recall_f1(df)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_known_values(self):
        # TP=2, FP=1, FN=1
        df = pl.DataFrame(
            {
                "is_win": [1, 1, 1, 0, 0],
                "predicted_win": [1, 1, 0, 1, 0],
            }
        )
        result = RacingMetrics.precision_recall_f1(df)
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(2 / 3)


class TestLogLoss:
    """Tests for log_loss metric."""

    def test_confident_correct(self):
        df = pl.DataFrame(
            {
                "is_win": [1, 0],
                "predicted_prob": [0.99, 0.01],
            }
        )
        loss = RacingMetrics.log_loss(df)
        assert loss < 0.1

    def test_confident_wrong(self):
        df = pl.DataFrame(
            {
                "is_win": [1, 0],
                "predicted_prob": [0.01, 0.99],
            }
        )
        loss = RacingMetrics.log_loss(df)
        assert loss > 2.0

    def test_uncertain(self):
        df = pl.DataFrame(
            {
                "is_win": [1, 0],
                "predicted_prob": [0.5, 0.5],
            }
        )
        loss = RacingMetrics.log_loss(df)
        assert loss == pytest.approx(0.6931, rel=1e-3)


class TestNDCG:
    """Tests for NDCG metric."""

    def test_perfect_ranking(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1"] * 5,
                "predicted_rank": [1, 2, 3, 4, 5],
                "actual_position": [1, 2, 3, 4, 5],
            }
        )
        assert RacingMetrics.ndcg(df) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1"] * 3,
                "predicted_rank": [1, 2, 3],
                "actual_position": [3, 2, 1],
            }
        )
        score = RacingMetrics.ndcg(df)
        # Reversed ranking should be less than 1.0
        assert score < 1.0
        assert score > 0.0

    def test_ndcg_at_k(self):
        df = pl.DataFrame(
            {
                "race_id": ["R1"] * 5,
                "predicted_rank": [1, 2, 3, 4, 5],
                "actual_position": [1, 2, 3, 4, 5],
            }
        )
        assert RacingMetrics.ndcg(df, k=3) == pytest.approx(1.0)

    def test_empty(self):
        df = pl.DataFrame(
            {
                "race_id": pl.Series([], dtype=pl.Utf8),
                "predicted_rank": pl.Series([], dtype=pl.Int64),
                "actual_position": pl.Series([], dtype=pl.Int64),
            }
        )
        assert RacingMetrics.ndcg(df) == 0.0


class TestComputeAll:
    """Tests for the compute_all convenience method."""

    def test_returns_metrics_result(self, sample_prediction_df: pl.DataFrame):
        result = RacingMetrics.compute_all(sample_prediction_df, has_betting=True)
        assert isinstance(result, MetricsResult)
        assert "win_accuracy" in result.values
        assert "place_accuracy" in result.values
        assert "auc_roc" in result.values
        assert "recovery_rate" in result.values
        assert "roi" in result.values

    def test_without_betting(self, sample_prediction_df: pl.DataFrame):
        result = RacingMetrics.compute_all(sample_prediction_df, has_betting=False)
        assert "recovery_rate" not in result.values
        assert "roi" not in result.values


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    """Tests for BacktestEngine with mock model."""

    def test_basic_backtest(self, sample_features_df: pl.DataFrame, mock_model):
        engine = BacktestEngine(
            train_window_days=14,
            test_window_days=7,
            step_days=7,
        )
        result = engine.run(sample_features_df, mock_model)

        assert isinstance(result, BacktestResult)

    def test_backtest_with_predictions(self):
        """Test backtest with controlled data spanning multiple periods."""
        np.random.seed(42)
        n_per_race = 5
        dates = []
        race_ids = []
        horse_ids = []
        features = []
        targets = []
        positions = []

        for day_offset in range(120):
            date_str = f"2024-{1 + day_offset // 30:02d}-{1 + day_offset % 28:02d}"
            race_id = f"R{day_offset:04d}"
            for h in range(n_per_race):
                dates.append(date_str)
                race_ids.append(race_id)
                horse_ids.append(f"H{h:03d}")
                features.append(np.random.randn(3).tolist())
                is_win = 1 if h == 0 else 0
                targets.append(is_win)
                positions.append(h + 1)

        feat_arrays = np.array(features)
        df = pl.DataFrame(
            {
                "race_date": dates,
                "race_id": race_ids,
                "horse_id": horse_ids,
                "feat_a": feat_arrays[:, 0],
                "feat_b": feat_arrays[:, 1],
                "feat_c": feat_arrays[:, 2],
                "is_win": targets,
                "actual_position": positions,
            }
        )

        from tests.conftest import MockModel

        model = MockModel(seed=42)

        engine = BacktestEngine(
            train_window_days=30,
            test_window_days=14,
            step_days=14,
        )
        result = engine.run(df, model)

        assert isinstance(result, BacktestResult)
        assert len(result.periods) > 0
        for period in result.periods:
            assert period.n_train > 0
            assert period.n_test > 0
            assert "win_accuracy" in period.metrics.values

    def test_expanding_window(self):
        """Test with expanding window (train_window_days=None)."""
        np.random.seed(42)
        dates = []
        race_ids = []
        horse_ids = []

        for day_offset in range(180):
            date_str = f"2024-{1 + day_offset // 30:02d}-{1 + day_offset % 28:02d}"
            race_id = f"R{day_offset:04d}"
            for h in range(3):
                dates.append(date_str)
                race_ids.append(race_id)
                horse_ids.append(f"H{h:03d}")

        n = len(dates)
        df = pl.DataFrame(
            {
                "race_date": dates,
                "race_id": race_ids,
                "horse_id": horse_ids,
                "feat_x": np.random.randn(n),
                "is_win": [1 if i % 3 == 0 else 0 for i in range(n)],
                "actual_position": [(i % 3) + 1 for i in range(n)],
            }
        )

        from tests.conftest import MockModel

        engine = BacktestEngine(
            train_window_days=None,
            test_window_days=30,
            step_days=30,
        )
        result = engine.run(df, MockModel(seed=42))
        assert len(result.periods) > 0

    def test_summary_df(self):
        """Test that summary_df returns a valid DataFrame."""
        np.random.seed(42)
        n_per_race = 4
        dates = []
        race_ids = []
        horse_ids = []

        for day_offset in range(90):
            date_str = f"2024-{1 + day_offset // 30:02d}-{1 + day_offset % 28:02d}"
            race_id = f"R{day_offset:04d}"
            for h in range(n_per_race):
                dates.append(date_str)
                race_ids.append(race_id)
                horse_ids.append(f"H{h:03d}")

        n = len(dates)
        df = pl.DataFrame(
            {
                "race_date": dates,
                "race_id": race_ids,
                "horse_id": horse_ids,
                "feat_x": np.random.randn(n),
                "is_win": [1 if i % n_per_race == 0 else 0 for i in range(n)],
                "actual_position": [(i % n_per_race) + 1 for i in range(n)],
            }
        )

        from tests.conftest import MockModel

        engine = BacktestEngine(
            train_window_days=30,
            test_window_days=14,
            step_days=14,
        )
        result = engine.run(df, MockModel(seed=42))
        summary = result.summary_df()

        assert isinstance(summary, pl.DataFrame)
        if not summary.is_empty():
            assert "period" in summary.columns
            assert "n_test" in summary.columns


# ---------------------------------------------------------------------------
# PerformanceMonitor tests
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_healthy_check(self):
        monitor = PerformanceMonitor(thresholds={"win_accuracy": 0.05})
        monitor.record(MetricsResult(values={"win_accuracy": 0.10}), label="period_1")
        report = monitor.check()
        assert report.is_healthy

    def test_degradation_alert(self):
        monitor = PerformanceMonitor(thresholds={"win_accuracy": 0.10})
        monitor.record(MetricsResult(values={"win_accuracy": 0.05}), label="period_1")
        report = monitor.check()
        assert not report.is_healthy
        assert len(report.alerts) == 1
        assert report.alerts[0].alert_type == "degradation"

    def test_trend_degradation(self):
        monitor = PerformanceMonitor(
            thresholds={"win_accuracy": 0.01},
            window_size=3,
        )
        # Record good performance
        for i in range(3):
            monitor.record(
                MetricsResult(values={"win_accuracy": 0.20}), label=f"good_{i}"
            )
        # Record degraded performance
        for i in range(3):
            monitor.record(
                MetricsResult(values={"win_accuracy": 0.10}), label=f"bad_{i}"
            )

        report = monitor.check()
        # Should detect trend degradation (50% drop)
        degradation_alerts = [a for a in report.alerts if "dropped" in a.message]
        assert len(degradation_alerts) > 0

    def test_data_drift_detection(self):
        monitor = PerformanceMonitor()
        ref_df = pl.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        # Drifted data: feat_a shifted significantly
        cur_df = pl.DataFrame(
            {
                "feat_a": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        alerts = monitor.detect_data_drift(ref_df, cur_df, threshold=0.1)
        drifted_cols = {a.metric_name for a in alerts}
        assert "feat_a_mean" in drifted_cols
        # feat_b should not drift
        assert "feat_b_mean" not in drifted_cols

    def test_no_drift(self):
        monitor = PerformanceMonitor()
        df = pl.DataFrame({"feat_a": [1.0, 2.0, 3.0]})
        alerts = monitor.detect_data_drift(df, df)
        assert len(alerts) == 0

    def test_reset(self):
        monitor = PerformanceMonitor()
        monitor.record(MetricsResult(values={"win_accuracy": 0.10}))
        assert len(monitor.history) == 1
        monitor.reset()
        assert len(monitor.history) == 0


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------


class TestReporter:
    """Tests for Reporter Markdown generation."""

    def test_metrics_report(self):
        reporter = Reporter(model_name="test_model")
        metrics = MetricsResult(
            values={
                "win_accuracy": 0.12,
                "place_accuracy": 0.35,
                "auc_roc": 0.72,
                "log_loss": 0.45,
            }
        )
        md = reporter.generate_metrics_report(metrics)

        assert "# Horse Racing Model Evaluation Report" in md
        assert "test_model" in md
        assert "Win Accuracy" in md
        assert "0.1200" in md

    def test_backtest_report(self):
        from src.evaluator.backtest_engine import BacktestPeriod

        period = BacktestPeriod(
            period_index=0,
            train_start="2024-01-01",
            train_end="2024-01-31",
            test_start="2024-02-01",
            test_end="2024-02-14",
            n_train=100,
            n_test=20,
            metrics=MetricsResult(values={"win_accuracy": 0.15, "auc_roc": 0.68}),
        )
        result = BacktestResult(
            periods=[period],
            overall_metrics=MetricsResult(
                values={"win_accuracy": 0.15, "auc_roc": 0.68}
            ),
        )

        reporter = Reporter(model_name="backtest_model")
        md = reporter.generate_backtest_report(result)

        assert "Overall Metrics" in md
        assert "Period-wise Results" in md
        assert "backtest_model" in md

    def test_monitor_report_healthy(self):
        from src.evaluator.monitor import MonitorReport

        report = MonitorReport(alerts=[], is_healthy=True)
        reporter = Reporter()
        md = reporter.generate_monitor_report(report)

        assert "HEALTHY" in md
        assert "No alerts" in md

    def test_monitor_report_with_alerts(self):
        from src.evaluator.monitor import Alert, MonitorReport

        alert = Alert(
            alert_type="degradation",
            metric_name="win_accuracy",
            current_value=0.03,
            threshold=0.05,
            message="win_accuracy = 0.0300 is below threshold 0.0500",
        )
        report = MonitorReport(alerts=[alert], is_healthy=False)
        reporter = Reporter()
        md = reporter.generate_monitor_report(report)

        assert "ALERT" in md
        assert "degradation" in md
        assert "win_accuracy" in md

    def test_save_report(self, tmp_path):
        reporter = Reporter()
        content = "# Test Report\nHello"
        path = reporter.save(content, tmp_path / "report.md")

        assert path.exists()
        assert path.read_text(encoding="utf-8") == content
