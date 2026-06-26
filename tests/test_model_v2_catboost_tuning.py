import builtins

import pytest

import ml.experiments.model_v2_catboost_tuning as catboost_tuning
from ml.experiments.model_v2_catboost_tuning import (
    build_catboost_policy_rows,
    generate_catboost_tuning_candidates,
    get_catboost_availability,
    select_best_catboost_policy,
)


def test_generate_catboost_tuning_candidates_contains_controlled_configs():
    candidates = generate_catboost_tuning_candidates()
    names = [candidate["candidate"] for candidate in candidates]

    assert "catboost_default" in names
    assert "catboost_class_weights_moderate" in names
    assert "catboost_auto_class_weights_balanced" in names
    assert len(candidates) >= 5


def test_catboost_availability_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "catboost":
            raise ImportError("No module named catboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    availability = get_catboost_availability()

    assert availability["available"] is False
    assert "catboost" in availability["reason"]


def test_build_catboost_policy_rows_skips_untrained_candidates():
    rows = build_catboost_policy_rows(
        [
            _trained_candidate("catboost_default", recall=0.60, precision=0.50),
            {
                "candidate": "catboost_missing",
                "model_family": "catboost",
                "status": "skipped",
            },
        ]
    )

    assert len(rows) == 1
    assert rows[0]["candidate"] == "catboost_default"


def test_select_best_catboost_policy_uses_alert_constraint_and_recall_priority():
    rows = [
        _policy_row("catboost_default", recall=0.60, precision=0.70, f1_score=0.64, alert_rate=0.04),
        _policy_row("catboost_weighted", recall=0.75, precision=0.55, f1_score=0.63, alert_rate=0.05),
        _policy_row("catboost_too_many_alerts", recall=0.90, precision=0.30, f1_score=0.45, alert_rate=0.12),
    ]

    selected = select_best_catboost_policy(rows, max_alert_rate=0.05)

    assert selected["candidate"] == "catboost_weighted"
    assert selected["recall"] == 0.75


def test_train_or_skip_catboost_candidate_skips_when_dependency_missing():
    result = catboost_tuning._train_or_skip_catboost_candidate(
        availability={"available": False, "reason": "missing catboost"},
        candidate_config={"candidate": "catboost_default", "params": {}},
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        categorical_cols=[],
    )

    assert result["status"] == "skipped"
    assert result["skip_reason"] == "missing catboost"


def test_catboost_params_use_gpu_when_requested():
    params = catboost_tuning._catboost_params({}, task_type="GPU")

    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"


def test_catboost_params_use_cpu_without_gpu_devices():
    params = catboost_tuning._catboost_params({}, task_type="CPU")

    assert params["task_type"] == "CPU"
    assert "devices" not in params


def test_train_or_skip_catboost_candidate_uses_gpu_when_training_succeeds(monkeypatch):
    calls = []

    def fake_fit_catboost_candidate(**kwargs):
        calls.append(kwargs)
        return {
            "candidate": kwargs["candidate"],
            "status": "trained",
            "execution_device": kwargs["execution_device"],
            "params": kwargs["params"],
        }

    monkeypatch.setattr(
        catboost_tuning,
        "_fit_catboost_candidate",
        fake_fit_catboost_candidate,
    )

    result = catboost_tuning._train_or_skip_catboost_candidate(
        availability={"available": True, "reason": "available"},
        candidate_config={"candidate": "catboost_default", "params": {}},
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        categorical_cols=[],
    )

    assert result["execution_device"] == "GPU"
    assert result["params"]["task_type"] == "GPU"
    assert result["params"]["devices"] == "0"
    assert len(calls) == 1


def test_train_or_skip_catboost_candidate_falls_back_to_cpu(monkeypatch):
    calls = []

    def fake_fit_catboost_candidate(**kwargs):
        calls.append(kwargs)
        if kwargs["execution_device"] == "GPU":
            raise RuntimeError("GPU unavailable")
        return {
            "candidate": kwargs["candidate"],
            "status": "trained",
            "execution_device": kwargs["execution_device"],
            "params": kwargs["params"],
        }

    monkeypatch.setattr(
        catboost_tuning,
        "_fit_catboost_candidate",
        fake_fit_catboost_candidate,
    )

    result = catboost_tuning._train_or_skip_catboost_candidate(
        availability={"available": True, "reason": "available"},
        candidate_config={"candidate": "catboost_default", "params": {}},
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        categorical_cols=[],
    )

    assert result["execution_device"] == "CPU fallback"
    assert result["params"]["task_type"] == "CPU"
    assert "devices" not in result["params"]
    assert result["gpu_error"] == "GPU unavailable"
    assert [call["execution_device"] for call in calls] == ["GPU", "CPU fallback"]


def _trained_candidate(candidate, recall, precision):
    return {
        "candidate": candidate,
        "model_family": "catboost",
        "status": "trained",
        "validation_threshold_comparison": [
            _policy_row(candidate, recall=recall, precision=precision)
        ],
    }


def _policy_row(
    candidate,
    *,
    recall,
    precision,
    f1_score=0.55,
    alert_rate=0.04,
):
    return {
        "candidate": candidate,
        "threshold": 0.10,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "alert_rate": alert_rate,
        "false_positives": 10,
        "false_negatives": 2,
        "true_positives": 8,
        "true_negatives": 80,
    }
