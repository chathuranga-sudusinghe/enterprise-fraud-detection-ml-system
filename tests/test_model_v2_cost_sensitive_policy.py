import pytest

import ml.experiments.model_v2_cost_sensitive_policy as policy_module
from ml.experiments.model_v2_cost_sensitive_policy import (
    build_cost_sensitive_policy_rows,
    evaluate_cost_sensitive_policies,
    run_model_v2_cost_sensitive_policy_experiment,
    select_best_cost_sensitive_policy,
)


def test_build_cost_sensitive_policy_rows_flattens_candidate_thresholds():
    rows = build_cost_sensitive_policy_rows([_candidate(1.0), _candidate(3.0)])

    assert len(rows) == 4
    assert rows[0]["scale_pos_weight"] == 1.0
    assert rows[0]["threshold"] == 0.10
    assert rows[0]["false_negatives"] == 2
    assert rows[2]["scale_pos_weight"] == 3.0


def test_select_best_policy_respects_alert_rate_then_maximizes_recall():
    rows = [
        _policy_row(scale_pos_weight=1.0, recall=0.60, precision=0.70, f1_score=0.64, alert_rate=0.04),
        _policy_row(scale_pos_weight=3.0, recall=0.75, precision=0.55, f1_score=0.63, alert_rate=0.05),
        _policy_row(scale_pos_weight=5.0, recall=0.90, precision=0.30, f1_score=0.45, alert_rate=0.12),
    ]

    selected = select_best_cost_sensitive_policy(rows, max_alert_rate=0.05)

    assert selected["scale_pos_weight"] == 3.0
    assert selected["recall"] == 0.75


def test_select_best_policy_tie_breaks_on_precision_then_f1():
    rows = [
        _policy_row(scale_pos_weight=3.0, recall=0.70, precision=0.50, f1_score=0.58, alert_rate=0.04),
        _policy_row(scale_pos_weight=5.0, recall=0.70, precision=0.60, f1_score=0.55, alert_rate=0.04),
    ]

    selected = select_best_cost_sensitive_policy(rows, max_alert_rate=0.05)

    assert selected["scale_pos_weight"] == 5.0


def test_select_best_policy_returns_none_when_no_policy_meets_alert_limit():
    rows = [_policy_row(scale_pos_weight=1.0, recall=0.80, precision=0.40, f1_score=0.53, alert_rate=0.08)]

    assert select_best_cost_sensitive_policy(rows, max_alert_rate=0.05) is None


def test_select_best_policy_rejects_invalid_alert_rate_limit():
    with pytest.raises(ValueError, match="max_alert_rate"):
        select_best_cost_sensitive_policy([], max_alert_rate=1.1)


def test_evaluate_cost_sensitive_policies_reports_each_constraint():
    rows = [_policy_row(scale_pos_weight=1.0, recall=0.80, precision=0.40, f1_score=0.53, alert_rate=0.08)]

    evaluations = evaluate_cost_sensitive_policies(
        rows,
        max_alert_rates=(0.05, 0.10),
    )

    assert evaluations[0]["policy_found"] is False
    assert evaluations[1]["policy_found"] is True
    assert evaluations[1]["best_policy"]["scale_pos_weight"] == 1.0


def test_cost_sensitive_policy_experiment_uses_controlled_weights(monkeypatch):
    calls = []

    def fake_weight_search(candidate_weights):
        calls.append(candidate_weights)
        return {
            "candidate_weights": candidate_weights,
            "full_scale_pos_weight": 27.4343,
            "candidates": [_candidate(1.0), _candidate(3.0)],
        }

    monkeypatch.setattr(
        policy_module,
        "run_model_v2_controlled_weight_search_experiment",
        fake_weight_search,
    )

    summary = run_model_v2_cost_sensitive_policy_experiment()

    assert calls == [[1.0, 3.0, 5.0, 5.2378, 10.0]]
    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["candidate_weights"] == [1.0, 3.0, 5.0, 5.2378, 10.0]
    assert summary["policy_rows"]
    assert summary["policy_evaluations"]


def _candidate(scale_pos_weight):
    return {
        "candidate": f"scale_pos_weight_{scale_pos_weight:.4f}",
        "scale_pos_weight": scale_pos_weight,
        "validation_threshold_comparison": [
            _policy_row(
                scale_pos_weight=scale_pos_weight,
                threshold=0.10,
                recall=0.70,
                precision=0.40,
                f1_score=0.51,
                alert_rate=0.08,
            ),
            _policy_row(
                scale_pos_weight=scale_pos_weight,
                threshold=0.20,
                recall=0.60,
                precision=0.50,
                f1_score=0.55,
                alert_rate=0.04,
            ),
        ],
    }


def _policy_row(
    *,
    scale_pos_weight,
    threshold=0.10,
    recall,
    precision,
    f1_score,
    alert_rate,
):
    return {
        "scale_pos_weight": scale_pos_weight,
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
