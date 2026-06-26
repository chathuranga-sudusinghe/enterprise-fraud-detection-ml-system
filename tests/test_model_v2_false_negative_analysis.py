import pandas as pd
import pytest

import ml.experiments.model_v2_false_negative_analysis as fn_analysis
from ml.experiments.model_v2_false_negative_analysis import (
    analyze_false_negatives,
    build_outcome_frame,
    summarize_false_negative_groups,
)


def test_build_outcome_frame_extracts_false_negatives():
    X = pd.DataFrame(
        {
            "ProductCD": ["W", "C", "W", "H"],
            "TransactionAmt": [10.0, 200.0, 40.0, 700.0],
        }
    )
    y_true = pd.Series([1, 1, 0, 1])
    y_proba = [0.10, 0.80, 0.05, 0.19]

    outcome = build_outcome_frame(
        X=X,
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.20,
    )

    false_negatives = outcome[outcome["_is_false_negative"]]
    assert len(false_negatives) == 2
    assert list(false_negatives["ProductCD"]) == ["W", "H"]
    assert int(outcome["_is_true_positive"].sum()) == 1


def test_build_outcome_frame_rejects_length_mismatch():
    with pytest.raises(ValueError, match="matching row counts"):
        build_outcome_frame(
            X=pd.DataFrame({"ProductCD": ["W", "C"]}),
            y_true=pd.Series([1]),
            y_proba=[0.10, 0.20],
            threshold=0.20,
        )


def test_analyze_false_negatives_builds_grouped_summary():
    X = pd.DataFrame(
        {
            "ProductCD": ["W", "W", "C", "C", "C"],
            "card1": [1001, 1001, 1002, 1002, 1003],
            "TransactionAmt": [10.0, 20.0, 300.0, 400.0, 900.0],
            "TransactionDT": [3600, 7200, 43200, 68400, 80000],
            "id_01": [None, None, 1.0, None, None],
            "id_02": [None, None, 2.0, None, None],
        }
    )
    y_true = pd.Series([1, 1, 1, 0, 0])
    y_proba = [0.10, 0.15, 0.80, 0.05, 0.01]

    analysis = analyze_false_negatives(
        X=X,
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.20,
        group_columns=["ProductCD", "TransactionAmt_band", "identity_missing_count_band"],
        top_n=5,
    )

    assert analysis["summary"]["fraud_count"] == 3
    assert analysis["summary"]["false_negative_count"] == 2
    product_rows = [
        row
        for row in analysis["top_false_negative_groups"]
        if row["group_column"] == "ProductCD"
    ]
    assert product_rows[0]["group_value"] == "W"
    assert product_rows[0]["false_negatives"] == 2
    assert product_rows[0]["missed_fraud_rate"] == 1.0


def test_summarize_false_negative_groups_compares_against_non_fraud():
    outcome = pd.DataFrame(
        {
            "DeviceType": ["mobile", "mobile", "desktop", "desktop"],
            "_y_true": [1, 0, 1, 0],
            "_is_false_negative": [True, False, False, False],
            "_is_true_positive": [False, False, True, False],
        }
    )

    rows = summarize_false_negative_groups(
        outcome,
        group_columns=["DeviceType"],
    )

    assert rows == [
        {
            "group_column": "DeviceType",
            "group_value": "mobile",
            "false_negatives": 1,
            "true_positives": 0,
            "fraud_count": 1,
            "non_fraud_count": 1,
            "total_count": 2,
            "missed_fraud_rate": 1.0,
            "false_negative_share": 1.0,
            "fraud_rate": 0.5,
        }
    ]


def test_false_negative_analysis_experiment_is_non_artifact_workflow(monkeypatch):
    monkeypatch.setattr(
        fn_analysis,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: None,
    )
    monkeypatch.setattr(
        fn_analysis,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(fn_analysis, "FeatureEngineeringV2", FakeTransformer)
    monkeypatch.setattr(
        fn_analysis,
        "train_lightgbm_v2",
        lambda **kwargs: (FakeModel([0.10, 0.80]), [0.10, 0.80]),
    )

    summary = fn_analysis.run_model_v2_false_negative_analysis()

    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["model_family"] == "lightgbm"
    assert summary["scale_pos_weight"] == 5.0
    assert summary["threshold"] == 0.20
    assert summary["validation_analysis"]["summary"]["false_negative_count"] == 1


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
        import numpy as np

        probabilities = self.probabilities[: len(X)]
        return np.array([[1 - probability, probability] for probability in probabilities])


def _synthetic_splits():
    return {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0], "ProductCD": ["W", "C"]}),
        "y_train": pd.Series([0, 1]),
        "X_val": pd.DataFrame({"feature": [3.0, 4.0], "ProductCD": ["W", "C"]}),
        "y_val": pd.Series([1, 0]),
        "X_test": pd.DataFrame({"feature": [5.0, 6.0], "ProductCD": ["W", "C"]}),
        "y_test": pd.Series([1, 0]),
    }
