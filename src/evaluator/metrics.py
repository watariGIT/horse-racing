"""Evaluation metrics specialized for horse racing prediction.

Provides accuracy metrics (win rate, place rate, top-N),
profitability metrics (ROI, recovery rate), and standard ML
metrics (AUC-ROC, Precision/Recall/F1, NDCG, Log Loss).
All functions accept polars DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class MetricsResult:
    """Container for a collection of computed metrics."""

    values: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def to_dict(self) -> dict[str, float]:
        return self.values.copy()


class RacingMetrics:
    """Horse racing specific evaluation metrics.

    All public methods accept ``pl.DataFrame`` inputs and return
    either a scalar float or a ``MetricsResult`` bundle.
    """

    # ------------------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------------------

    @staticmethod
    def win_accuracy(df: pl.DataFrame) -> float:
        """Compute win prediction accuracy (predicted 1st == actual 1st).

        Args:
            df: DataFrame with columns ``predicted_rank`` and
                ``actual_position``.  Rows where ``predicted_rank == 1``
                are the model's top pick per race.

        Returns:
            Fraction of races where the top pick finished 1st.
        """
        top_picks = df.filter(pl.col("predicted_rank") == 1)
        if top_picks.is_empty():
            return 0.0
        correct = top_picks.filter(pl.col("actual_position") == 1)
        return len(correct) / len(top_picks)

    @staticmethod
    def place_accuracy(df: pl.DataFrame, top_n: int = 3) -> float:
        """Compute place prediction accuracy (predicted 1st finishes within top N).

        Args:
            df: DataFrame with ``predicted_rank`` and ``actual_position``.
            top_n: Number of places considered (default 3 = fukusho).

        Returns:
            Fraction of races where the top pick finished in top N.
        """
        top_picks = df.filter(pl.col("predicted_rank") == 1)
        if top_picks.is_empty():
            return 0.0
        correct = top_picks.filter(pl.col("actual_position") <= top_n)
        return len(correct) / len(top_picks)

    @staticmethod
    def top_n_accuracy(df: pl.DataFrame, n: int = 3) -> float:
        """Fraction of actual top-N horses captured in predicted top-N.

        For each race, compares the set of horses predicted in top N
        with the set of horses that actually finished in top N.

        Args:
            df: DataFrame with ``race_id``, ``predicted_rank``, ``actual_position``.
            n: Number of top positions to consider.

        Returns:
            Average hit rate across races.
        """
        races = df["race_id"].unique().to_list()
        if not races:
            return 0.0

        hits: list[float] = []
        for race_id in races:
            race_df = df.filter(pl.col("race_id") == race_id)
            predicted_top = set(
                race_df.filter(pl.col("predicted_rank") <= n)["horse_id"].to_list()
            )
            actual_top = set(
                race_df.filter(pl.col("actual_position") <= n)["horse_id"].to_list()
            )
            if not actual_top:
                continue
            hits.append(len(predicted_top & actual_top) / len(actual_top))

        return float(np.mean(hits)) if hits else 0.0

    # ------------------------------------------------------------------
    # Profitability metrics
    # ------------------------------------------------------------------

    @staticmethod
    def recovery_rate(df: pl.DataFrame) -> float:
        """Compute recovery rate (total payout / total wagered).

        Args:
            df: DataFrame with ``bet_amount`` and ``payout`` columns.
                Each row represents a single bet.

        Returns:
            Recovery rate (1.0 means break-even).
        """
        total_wagered = df["bet_amount"].sum()
        if total_wagered == 0:
            return 0.0
        total_payout = df["payout"].sum()
        return float(total_payout / total_wagered)

    @staticmethod
    def roi(df: pl.DataFrame) -> float:
        """Compute ROI (Return on Investment).

        Args:
            df: DataFrame with ``bet_amount`` and ``payout``.

        Returns:
            ROI as a fraction (0.0 = break-even, > 0 = profit).
        """
        total_wagered = df["bet_amount"].sum()
        if total_wagered == 0:
            return 0.0
        total_payout = df["payout"].sum()
        return float((total_payout - total_wagered) / total_wagered)

    @staticmethod
    def expected_value(
        df: pl.DataFrame,
        prob_col: str = "predicted_prob",
        odds_col: str = "odds",
    ) -> float:
        """Compute average expected value per bet.

        EV = mean(predicted_prob * odds - 1).

        Args:
            df: DataFrame with probability and odds columns.
            prob_col: Column name for predicted win probability.
            odds_col: Column name for offered odds.

        Returns:
            Average expected value across all entries.
        """
        if df.is_empty():
            return 0.0
        ev = (df[prob_col] * df[odds_col] - 1.0).mean()
        return float(ev) if ev is not None else 0.0

    # ------------------------------------------------------------------
    # Standard ML metrics
    # ------------------------------------------------------------------

    @staticmethod
    def auc_roc(
        df: pl.DataFrame, target_col: str = "is_win", prob_col: str = "predicted_prob"
    ) -> float:
        """Compute Area Under the ROC Curve.

        Uses a simple trapezoidal implementation to avoid sklearn dependency.

        Args:
            df: DataFrame with binary target and predicted probability.
            target_col: Column with binary labels (0/1).
            prob_col: Column with predicted probabilities.

        Returns:
            AUC-ROC score.
        """
        y_true = df[target_col].to_numpy().astype(float)
        y_score = df[prob_col].to_numpy().astype(float)

        # Sort by score descending
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0

        tp = 0.0
        fp = 0.0
        auc = 0.0

        for label in y_true_sorted:
            if label == 1.0:
                tp += 1
            else:
                fp += 1
                auc += tp  # rectangle area contribution

        return float(auc / (n_pos * n_neg))

    @staticmethod
    def precision_recall_f1(
        df: pl.DataFrame,
        target_col: str = "is_win",
        pred_col: str = "predicted_win",
    ) -> dict[str, float]:
        """Compute Precision, Recall, and F1-score.

        Args:
            df: DataFrame with binary target and binary prediction.
            target_col: Column with actual binary labels.
            pred_col: Column with predicted binary labels.

        Returns:
            Dict with keys "precision", "recall", "f1".
        """
        y_true = df[target_col].to_numpy().astype(float)
        y_pred = df[pred_col].to_numpy().astype(float)

        tp = float(((y_true == 1) & (y_pred == 1)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def log_loss(
        df: pl.DataFrame,
        target_col: str = "is_win",
        prob_col: str = "predicted_prob",
        eps: float = 1e-15,
    ) -> float:
        """Compute binary log loss (cross-entropy).

        Args:
            df: DataFrame with binary target and predicted probability.
            target_col: Column with actual binary labels.
            prob_col: Column with predicted probabilities.
            eps: Clipping epsilon to avoid log(0).

        Returns:
            Log loss value.
        """
        y_true = df[target_col].to_numpy().astype(float)
        y_prob = df[prob_col].to_numpy().astype(float)

        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(loss.mean())

    @staticmethod
    def ndcg(df: pl.DataFrame, k: int | None = None) -> float:
        """Compute Normalized Discounted Cumulative Gain per race, averaged.

        Uses ``actual_position`` to derive relevance and ``predicted_rank``
        for the predicted ordering.

        Args:
            df: DataFrame with ``race_id``, ``predicted_rank``, ``actual_position``.
            k: Truncation level. None means use all entries.

        Returns:
            Mean NDCG across all races.
        """
        races = df["race_id"].unique().to_list()
        if not races:
            return 0.0

        ndcg_scores: list[float] = []

        for race_id in races:
            race_df = df.filter(pl.col("race_id") == race_id)
            n_entries = len(race_df)
            max_pos = n_entries

            # Relevance: higher is better (max_pos - actual_position + 1)
            relevance = (max_pos - race_df["actual_position"].to_numpy() + 1).astype(
                float
            )
            predicted_order = race_df["predicted_rank"].to_numpy().astype(int)

            # Sort relevance by predicted rank
            sorted_idx = np.argsort(predicted_order)
            predicted_relevance = relevance[sorted_idx]

            # Ideal relevance: sorted descending
            ideal_relevance = np.sort(relevance)[::-1]

            cutoff = min(k, n_entries) if k is not None else n_entries

            dcg = _dcg(predicted_relevance[:cutoff])
            idcg = _dcg(ideal_relevance[:cutoff])

            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    # ------------------------------------------------------------------
    # Convenience: compute all metrics at once
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        df: pl.DataFrame,
        has_betting: bool = False,
    ) -> MetricsResult:
        """Compute all available metrics on a prediction DataFrame.

        The DataFrame must contain at minimum:
            race_id, horse_id, predicted_rank, actual_position,
            predicted_prob, is_win, predicted_win

        If ``has_betting`` is True, also requires:
            bet_amount, payout, odds

        Args:
            df: Prediction results DataFrame.
            has_betting: Whether betting data is available.

        Returns:
            MetricsResult with all computed values.
        """
        result: dict[str, float] = {}

        result["win_accuracy"] = cls.win_accuracy(df)
        result["place_accuracy"] = cls.place_accuracy(df)
        result["top3_accuracy"] = cls.top_n_accuracy(df, n=3)
        result["ndcg"] = cls.ndcg(df)
        result["ndcg_at_3"] = cls.ndcg(df, k=3)

        if "predicted_prob" in df.columns and "is_win" in df.columns:
            result["auc_roc"] = cls.auc_roc(df)
            result["log_loss"] = cls.log_loss(df)

        if "predicted_win" in df.columns and "is_win" in df.columns:
            prf = cls.precision_recall_f1(df)
            result.update(prf)

        if has_betting:
            result["recovery_rate"] = cls.recovery_rate(df)
            result["roi"] = cls.roi(df)
            if "predicted_prob" in df.columns and "odds" in df.columns:
                result["expected_value"] = cls.expected_value(df)

        return MetricsResult(values=result)


def _dcg(relevance: np.ndarray) -> float:
    """Compute Discounted Cumulative Gain."""
    positions = np.arange(1, len(relevance) + 1)
    return float(np.sum(relevance / np.log2(positions + 1)))
