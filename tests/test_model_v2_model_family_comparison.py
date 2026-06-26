import builtins

import pytest

import ml.experiments.model_v2_model_family_comparison as family
from ml.experiments.model_v2_model_family_comparison import (
    build_model_family_policy_rows,
    get_optional_model_family_availability,
    run_model_v2_model_family_comparison_experiment,
    select_best_model_family_policy,
)


def test_optional_model_family_availability_marks_missing_dependencies(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"xgboost", "catboost"}:
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    availability = get_optional_model_family_availability()

    assert availability["lightgbm"]["available"] is True
    assert availability["xgboost"]["available"] is False
    assert availability["catboost"]["available"] is False


def test_build_model_family_policy_rows_skips_untrained_candidates():
    rows = build_model_family_policy_rows(
        [
            _trained_candidate("lightgbm", recall=0.60, precision=0.50, f1_score=0.55),
            {"model_family": "xgboost", "candidate": "xgboost_default_v2", "status": "skipped"},
        ]
    )

    assert len(rows) == 1
    assert rows[0]["model_family"] == "lightgbm"
    assert rows[0]["threshold"] == 0.20


def test_select_best_model_family_policy_uses_alert_constraint_and_recall_priority():
    rows = [
        _policy_row("lightgbm", recall=0.60, precision=0.70, f1_score=0.64, alert_rate=0.04),
        _policy_row("xgboost", recall=0.75, precision=0.55, f1_score=0.63, alert_rate=0.05),
        _policy_row("catboost", recall=0.90, precision=0.30, f1_score=0.45, alert_rate=0.12),
    ]

    selected = select_best_model_family_policy(rows, max_alert_rate=0.05)

    assert selected["model_family"] == "xgboost"
    assert selected["recall"] == 0.75


def test_model_family_comparison_experiment_skips_missing_optional_candidates(
    monkeypatch,
):
    monkeypatch.setattr(
        family,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: None,
    )
    monkeypatch.setattr(
        family,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(family, "FeatureEngineeringV2", FakeTransformer)
    monkeypatch.setattr(
        family,
        "get_optional_model_family_availability",
        lambda: {
            "lightgbm": {"available": True, "reason": "available"},
            "xgboost": {"available": False, "reason": "missing xgboost"},
            "catboost": {"available": False, "reason": "missing catboost"},
        },
    )
    monkeypatch.setattr(
        family,
        "train_lightgbm_v2",
        lambda **kwargs: (FakeModel([0.10, 0.80]), [0.10, 0.80]),
    )

    summary = run_model_v2_model_family_comparison_experiment()

    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["benchmark"]["scale_pos_weight"] == 5.0
    assert summary["candidates"][0]["status"] == "trained"
    assert summary["candidates"][1]["status"] == "skipped"
    assert summary["candidates"][2]["status"] == "skipped"
    assert summary["policy_rows"]


def _trained_candidate(model_family, recall, precision, f1_score):
    return {
        "model_family": model_family,
        "candidate": f"{model_family}_default_v2",
        "status": "trained",
        "scale_pos_weight": None,
        "validation_threshold_comparison": [
            _policy_row(
                model_family,
                threshold=0.20,
                recall=recall,
                precision=precision,
                f1_score=f1_score,
                alert_rate=0.04,
            )
        ],
    }


def _policy_row(
    model_family,
    *,
    threshold=0.20,
    recall,
    precision,
    f1_score,
    alert_rate,
):
    return {
        "model_family": model_family,
        "candidate": f"{model_family}_default_v2",
        "scale_pos_weight": None,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "alert_rate": alert_rate,
        "false_positives": 10,
        "false_negatives": 2,
        "true_positives": 8,
        "true_negatives": 80,
    }


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
    import pandas as pd

    return {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
        "y_train": pd.Series([0, 1]),
        "X_val": pd.DataFrame({"feature": [3.0, 4.0]}),
        "y_val": pd.Series([0, 1]),
        "X_test": pd.DataFrame({"feature": [5.0, 6.0]}),
        "y_test": pd.Series([0, 1]),
    }
