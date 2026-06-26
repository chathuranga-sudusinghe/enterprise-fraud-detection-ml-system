import numpy as np
import pandas as pd
import pytest

import ml.experiments.model_v2_recall_improvement as recall_experiment
from ml.experiments.model_v2_recall_improvement import (
    calculate_scale_pos_weight,
    run_model_v2_recall_improvement_experiment,
    summarize_false_negatives,
)


def test_calculate_scale_pos_weight_uses_non_fraud_to_fraud_ratio():
    y_train = pd.Series([0, 0, 0, 1])

    assert calculate_scale_pos_weight(y_train) == 3.0


def test_calculate_scale_pos_weight_rejects_zero_fraud_rows():
    with pytest.raises(ValueError, match="zero fraud"):
        calculate_scale_pos_weight(pd.Series([0, 0, 0]))


def test_calculate_scale_pos_weight_rejects_zero_non_fraud_rows():
    with pytest.raises(ValueError, match="zero non-fraud"):
        calculate_scale_pos_weight(pd.Series([1, 1, 1]))


def test_summarize_false_negatives_counts_missed_fraud_at_thresholds():
    summary = summarize_false_negatives(
        pd.Series([1, 1, 0, 0]),
        [0.05, 0.30, 0.20, 0.90],
        thresholds=(0.10, 0.20),
    )

    assert summary == [
        {
            "threshold": 0.10,
            "false_negatives": 1,
            "fraud_count": 2,
            "missed_fraud_rate": 0.5,
        },
        {
            "threshold": 0.20,
            "false_negatives": 1,
            "fraud_count": 2,
            "missed_fraud_rate": 0.5,
        },
    ]


def test_recall_improvement_experiment_compares_baseline_and_weighted(
    monkeypatch,
):
    train_calls = []

    monkeypatch.setattr(
        recall_experiment,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: pd.DataFrame({"unused": []}),
    )
    monkeypatch.setattr(
        recall_experiment,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(
        recall_experiment,
        "FeatureEngineeringV2",
        FakeTransformer,
    )

    def fake_train_lightgbm_v2(
        X_train,
        y_train,
        X_val,
        y_val,
        categorical_cols,
        scale_pos_weight=None,
    ):
        train_calls.append(scale_pos_weight)
        if scale_pos_weight is None:
            return FakeModel([0.05, 0.80]), np.array([0.10, 0.70])
        return FakeModel([0.30, 0.85]), np.array([0.30, 0.80])

    monkeypatch.setattr(recall_experiment, "train_lightgbm_v2", fake_train_lightgbm_v2)

    summary = run_model_v2_recall_improvement_experiment()

    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["scale_pos_weight"] == 3.0
    assert train_calls == [None, 3.0]
    assert summary["baseline_v2"]["candidate"] == "baseline_v2"
    assert summary["weighted_v2"]["candidate"] == "weighted_v2_scale_pos_weight"
    assert summary["baseline_v2"]["validation_threshold_comparison"]
    assert summary["weighted_v2"]["validation_false_negative_analysis"]


class FakeTransformer:
    categorical_columns_ = []
    feature_names_ = ["feature"]

    def fit_transform(self, X):
        return X[["feature"]].copy()

    def transform(self, X):
        return X[["feature"]].copy()


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        probabilities = self.probabilities[: len(X)]
        return np.array([[1 - probability, probability] for probability in probabilities])


def _synthetic_splits():
    return {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        "y_train": pd.Series([0, 0, 0, 1]),
        "X_val": pd.DataFrame({"feature": [5.0, 6.0]}),
        "y_val": pd.Series([0, 1]),
        "X_test": pd.DataFrame({"feature": [7.0, 8.0]}),
        "y_test": pd.Series([0, 1]),
    }
