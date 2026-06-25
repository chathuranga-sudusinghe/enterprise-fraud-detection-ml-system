import pandas as pd
import pytest

from ml.utils.threshold_evaluation_v2 import (
    DEFAULT_MODEL_V2_THRESHOLDS,
    THRESHOLD_EVALUATION_COLUMNS,
    evaluate_model_v2_thresholds,
)


def test_evaluate_model_v2_thresholds_returns_default_threshold_table():
    y_true = pd.Series([0, 1, 1, 0])
    y_proba = pd.Series([0.05, 0.20, 0.80, 0.70])

    table = evaluate_model_v2_thresholds(y_true, y_proba)

    assert list(table.columns) == THRESHOLD_EVALUATION_COLUMNS
    assert table["threshold"].tolist() == list(DEFAULT_MODEL_V2_THRESHOLDS)
    assert len(table) == 9


def test_evaluate_model_v2_thresholds_calculates_confusion_counts():
    y_true = [0, 1, 1, 0]
    y_proba = [0.05, 0.20, 0.80, 0.70]

    table = evaluate_model_v2_thresholds(y_true, y_proba, thresholds=[0.5])
    row = table.iloc[0]

    assert row["threshold"] == 0.5
    assert row["true_positives"] == 1
    assert row["false_positives"] == 1
    assert row["true_negatives"] == 1
    assert row["false_negatives"] == 1


def test_evaluate_model_v2_thresholds_calculates_precision_recall_alert_rate():
    y_true = [0, 1, 1, 0]
    y_proba = [0.05, 0.20, 0.80, 0.70]

    table = evaluate_model_v2_thresholds(y_true, y_proba, thresholds=[0.5])
    row = table.iloc[0]

    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["fraud_capture_rate"] == 0.5
    assert row["f1_score"] == 0.5
    assert row["alert_rate"] == 0.5


def test_evaluate_model_v2_thresholds_shows_threshold_tradeoff():
    y_true = [0, 1, 1, 0]
    y_proba = [0.05, 0.20, 0.80, 0.70]

    table = evaluate_model_v2_thresholds(y_true, y_proba, thresholds=[0.1, 0.9])
    low_threshold = table[table["threshold"] == 0.1].iloc[0]
    high_threshold = table[table["threshold"] == 0.9].iloc[0]

    assert low_threshold["alert_rate"] > high_threshold["alert_rate"]
    assert low_threshold["fraud_capture_rate"] > high_threshold["fraud_capture_rate"]


def test_evaluate_model_v2_thresholds_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        evaluate_model_v2_thresholds([0, 1], [0.1])


def test_evaluate_model_v2_thresholds_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_model_v2_thresholds([0, 2], [0.1, 0.9])


def test_evaluate_model_v2_thresholds_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_model_v2_thresholds([0, 1], [0.1, 1.2])


def test_evaluate_model_v2_thresholds_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_model_v2_thresholds([0, 1], [0.1, 0.9], thresholds=[-0.1])
