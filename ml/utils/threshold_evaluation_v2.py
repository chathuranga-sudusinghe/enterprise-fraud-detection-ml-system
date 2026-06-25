from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL_V2_THRESHOLDS: tuple[float, ...] = tuple(
    round(threshold, 2) for threshold in np.arange(0.10, 1.00, 0.10)
)

THRESHOLD_EVALUATION_COLUMNS: list[str] = [
    "threshold",
    "precision",
    "recall",
    "f1_score",
    "false_positives",
    "false_negatives",
    "true_positives",
    "true_negatives",
    "alert_rate",
    "fraud_capture_rate",
]


def evaluate_model_v2_thresholds(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    y_proba: Iterable[float] | np.ndarray | pd.Series,
    thresholds: Iterable[float] = DEFAULT_MODEL_V2_THRESHOLDS,
) -> pd.DataFrame:
    """
    Build a threshold comparison table for Model v2 fraud probabilities.

    This utility is intentionally side-effect free. It does not write artifacts,
    update production thresholds, or change v1 inference behavior.
    """

    y_true_array, y_proba_array = _validate_threshold_inputs(y_true, y_proba)
    threshold_values = _validate_thresholds(thresholds)

    rows = []
    for threshold in threshold_values:
        y_pred = y_proba_array >= threshold
        positive = y_true_array == 1
        negative = y_true_array == 0

        true_positives = int(np.sum(y_pred & positive))
        false_positives = int(np.sum(y_pred & negative))
        true_negatives = int(np.sum(~y_pred & negative))
        false_negatives = int(np.sum(~y_pred & positive))

        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)
        alert_rate = _safe_divide(true_positives + false_positives, len(y_true_array))

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "alert_rate": alert_rate,
                "fraud_capture_rate": recall,
            }
        )

    return pd.DataFrame(rows, columns=THRESHOLD_EVALUATION_COLUMNS)


def _validate_threshold_inputs(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    y_proba: Iterable[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(list(y_true) if not hasattr(y_true, "__array__") else y_true)
    y_proba_array = np.asarray(
        list(y_proba) if not hasattr(y_proba, "__array__") else y_proba,
        dtype=float,
    )

    if y_true_array.ndim != 1 or y_proba_array.ndim != 1:
        raise ValueError("Model v2 threshold evaluation expects one-dimensional inputs.")
    if y_true_array.size == 0:
        raise ValueError("Model v2 threshold evaluation requires non-empty inputs.")
    if y_true_array.shape[0] != y_proba_array.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if not np.isin(y_true_array, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if np.isnan(y_proba_array).any():
        raise ValueError("y_proba must not contain NaN values.")
    if np.any((y_proba_array < 0) | (y_proba_array > 1)):
        raise ValueError("y_proba values must be between 0 and 1.")

    return y_true_array.astype(int), y_proba_array


def _validate_thresholds(thresholds: Iterable[float]) -> list[float]:
    threshold_values = [float(threshold) for threshold in thresholds]
    if not threshold_values:
        raise ValueError("At least one threshold is required.")
    if any(threshold < 0 or threshold > 1 for threshold in threshold_values):
        raise ValueError("Thresholds must be between 0 and 1.")

    return threshold_values


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
