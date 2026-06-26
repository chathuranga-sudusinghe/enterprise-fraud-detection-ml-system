import pandas as pd

import ml.experiments.model_v2_final_policy_validation as final_validation
from ml.experiments.model_v2_final_policy_validation import (
    PROMOTION_DO_NOT_PROMOTE,
    PROMOTION_PROMOTE_CANDIDATE,
    get_final_policy_candidates,
    run_model_v2_final_policy_validation,
    select_final_policy_decision,
)


def test_final_policy_candidates_match_required_definitions():
    candidates = get_final_policy_candidates()
    by_name = {candidate["candidate"]: candidate for candidate in candidates}

    assert by_name["lightgbm_constrained"]["model_family"] == "lightgbm"
    assert by_name["lightgbm_constrained"]["scale_pos_weight"] == 1.0
    assert by_name["lightgbm_constrained"]["threshold"] == 0.10
    assert by_name["lightgbm_high_recall"]["scale_pos_weight"] == 5.0
    assert by_name["lightgbm_high_recall"]["threshold"] == 0.20
    assert by_name["catboost_default"]["model_family"] == "catboost"
    assert by_name["catboost_default"]["threshold"] == 0.10
    assert by_name["catboost_auto_class_weights_balanced"]["params"] == {
        "auto_class_weights": "Balanced"
    }
    assert by_name["catboost_auto_class_weights_balanced"]["threshold"] == 0.70


def test_final_decision_selects_highest_test_recall_under_alert_constraint():
    candidates = [
        _trained_candidate(
            "lightgbm_constrained",
            validation_alert_rate=0.04,
            test_alert_rate=0.04,
            test_recall=0.60,
            test_precision=0.70,
            test_f1_score=0.64,
        ),
        _trained_candidate(
            "catboost_default",
            validation_alert_rate=0.03,
            test_alert_rate=0.03,
            test_recall=0.66,
            test_precision=0.62,
            test_f1_score=0.64,
        ),
        _trained_candidate(
            "lightgbm_too_many_alerts",
            validation_alert_rate=0.06,
            test_alert_rate=0.04,
            test_recall=0.90,
            test_precision=0.30,
            test_f1_score=0.45,
        ),
    ]

    decision = select_final_policy_decision(candidates, max_alert_rate=0.05)

    assert decision["recommended_candidate"] == "catboost_default"
    assert decision["promotion_recommendation"] == PROMOTION_PROMOTE_CANDIDATE
    assert decision["selected_policy"]["test_recall"] == 0.66


def test_final_decision_uses_precision_then_f1_as_tie_breakers():
    candidates = [
        _trained_candidate(
            "candidate_a",
            validation_alert_rate=0.04,
            test_alert_rate=0.04,
            test_recall=0.60,
            test_precision=0.50,
            test_f1_score=0.55,
        ),
        _trained_candidate(
            "candidate_b",
            validation_alert_rate=0.04,
            test_alert_rate=0.04,
            test_recall=0.60,
            test_precision=0.55,
            test_f1_score=0.54,
        ),
    ]

    decision = select_final_policy_decision(candidates, max_alert_rate=0.05)

    assert decision["recommended_candidate"] == "candidate_b"


def test_final_decision_returns_do_not_promote_when_alert_rule_fails():
    candidates = [
        _trained_candidate(
            "lightgbm_high_recall",
            validation_alert_rate=0.08,
            test_alert_rate=0.04,
            test_recall=0.80,
            test_precision=0.40,
            test_f1_score=0.53,
        )
    ]

    decision = select_final_policy_decision(candidates, max_alert_rate=0.05)

    assert decision["recommended_candidate"] is None
    assert decision["promotion_recommendation"] == PROMOTION_DO_NOT_PROMOTE
    assert "No trained candidate" in decision["reason"]


def test_final_policy_validation_is_non_artifact_workflow(monkeypatch):
    monkeypatch.setattr(
        final_validation,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: None,
    )
    monkeypatch.setattr(
        final_validation,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(final_validation, "FeatureEngineeringV2", FakeTransformer)
    monkeypatch.setattr(
        final_validation,
        "get_catboost_availability",
        lambda: {"available": False, "reason": "missing catboost"},
    )
    monkeypatch.setattr(
        final_validation,
        "train_lightgbm_v2",
        lambda **kwargs: (FakeModel([0.05, 0.80, 0.30, 0.90]), [0.05, 0.80, 0.30, 0.90]),
    )

    summary = run_model_v2_final_policy_validation()

    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["feature_count"] == 1
    assert len(summary["candidates"]) == 4
    assert summary["candidates"][0]["status"] == "trained"
    assert summary["candidates"][2]["status"] == "skipped"
    assert "/predict" not in str(summary)


def _trained_candidate(
    candidate,
    *,
    validation_alert_rate,
    test_alert_rate,
    test_recall,
    test_precision,
    test_f1_score,
):
    return {
        "candidate": candidate,
        "model_family": "lightgbm",
        "status": "trained",
        "scale_pos_weight": 1.0,
        "threshold": 0.10,
        "validation_metrics": {
            "alert_rate": validation_alert_rate,
            "recall": 0.50,
            "precision": 0.50,
            "f1_score": 0.50,
        },
        "test_metrics": {
            "alert_rate": test_alert_rate,
            "recall": test_recall,
            "precision": test_precision,
            "f1_score": test_f1_score,
        },
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
    return {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        "y_train": pd.Series([0, 1, 0, 1]),
        "X_val": pd.DataFrame({"feature": [5.0, 6.0, 7.0, 8.0]}),
        "y_val": pd.Series([0, 1, 0, 1]),
        "X_test": pd.DataFrame({"feature": [9.0, 10.0, 11.0, 12.0]}),
        "y_test": pd.Series([0, 1, 0, 1]),
    }
