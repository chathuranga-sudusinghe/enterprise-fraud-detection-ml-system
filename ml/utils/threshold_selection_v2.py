from __future__ import annotations

from typing import Any

import pandas as pd

from ml.utils.threshold_evaluation_v2 import (
    DEFAULT_MODEL_V2_THRESHOLDS,
    evaluate_model_v2_thresholds,
)


def select_model_v2_operating_threshold(
    y_true: Any,
    y_proba: Any,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MODEL_V2_THRESHOLDS,
    min_recall: float = 0.80,
    max_alert_rate: float = 0.20,
) -> dict[str, Any]:
    """
    Select a candidate Model v2 operating threshold from validation predictions.

    The selection is advisory and side-effect free. It favors thresholds that
    keep recall high, improve precision through F1, and stay within a manageable
    alert-rate target when possible.
    """

    comparison = evaluate_model_v2_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=thresholds,
    )
    recommendation, selection_rule = _select_recommendation(
        comparison,
        min_recall=min_recall,
        max_alert_rate=max_alert_rate,
    )

    return {
        "recommended_threshold": float(recommendation["threshold"]),
        "selection_rule": selection_rule,
        "min_recall": float(min_recall),
        "max_alert_rate": float(max_alert_rate),
        "recommended_metrics": _row_to_dict(recommendation),
        "threshold_comparison": comparison,
    }


def _select_recommendation(
    comparison: pd.DataFrame,
    *,
    min_recall: float,
    max_alert_rate: float,
) -> tuple[pd.Series, str]:
    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be between 0 and 1.")
    if not 0 <= max_alert_rate <= 1:
        raise ValueError("max_alert_rate must be between 0 and 1.")

    eligible = comparison[
        (comparison["recall"] >= min_recall)
        & (comparison["alert_rate"] <= max_alert_rate)
    ]
    if not eligible.empty:
        return (
            _best_by_f1_precision_recall(eligible),
            "max_f1_with_min_recall_and_max_alert_rate",
        )

    recall_eligible = comparison[comparison["recall"] >= min_recall]
    if not recall_eligible.empty:
        return (
            _best_by_alert_rate_precision_f1_recall(recall_eligible),
            "max_f1_with_min_recall_alert_rate_relaxed",
        )

    return (
        _best_by_f1_precision_recall(comparison),
        "max_f1_recall_requirement_not_met",
    )


def _best_by_f1_precision_recall(candidates: pd.DataFrame) -> pd.Series:
    sorted_candidates = candidates.sort_values(
        by=["f1_score", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return sorted_candidates.iloc[0]


def _best_by_alert_rate_precision_f1_recall(candidates: pd.DataFrame) -> pd.Series:
    sorted_candidates = candidates.sort_values(
        by=["alert_rate", "precision", "f1_score", "recall", "threshold"],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    return sorted_candidates.iloc[0]


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    result = row.to_dict()
    for key, value in result.items():
        if hasattr(value, "item"):
            result[key] = value.item()
    return result
