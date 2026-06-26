from scripts import generate_model_v2_final_policy_validation_report as report_script


def test_generate_model_v2_final_policy_validation_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_final_policy_validation",
        _fake_final_policy_summary,
    )
    output_path = tmp_path / "final_policy.md"

    result = report_script.generate_model_v2_final_policy_validation_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["decision"]["recommended_candidate"] == "catboost_default"
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Final Policy Validation Report" in report
    assert "Candidate Policy Results" in report
    assert "Final Decision" in report
    assert "catboost_default" in report
    assert "promote_candidate" in report


def _fake_final_policy_summary():
    decision = {
        "recommended_candidate": "catboost_default",
        "promotion_recommendation": "promote_candidate",
        "reason": "Selected by conservative test recall rule.",
        "risks": ["Requires separate artifact promotion review."],
        "selected_policy": {
            "candidate": "catboost_default",
            "model_family": "catboost",
            "scale_pos_weight": None,
            "threshold": 0.10,
            "validation_alert_rate": 0.04,
            "test_alert_rate": 0.04,
            "test_recall": 0.64,
            "test_precision": 0.52,
            "test_f1_score": 0.57,
        },
    }
    return {
        "artifacts_written": False,
        "max_alert_rate": 0.05,
        "feature_count": 831,
        "categorical_feature_count": 25,
        "catboost_availability": {"available": True, "reason": "available"},
        "candidates": [
            {
                "candidate": "catboost_default",
                "model_family": "catboost",
                "status": "trained",
                "execution_device": "GPU",
                "scale_pos_weight": None,
                "threshold": 0.10,
                "validation_metrics": _metrics(alert_rate=0.04),
                "test_metrics": _metrics(alert_rate=0.04),
            }
        ],
        "decision": decision,
    }


def _metrics(*, alert_rate):
    return {
        "roc_auc": 0.90,
        "pr_auc": 0.70,
        "precision": 0.52,
        "recall": 0.64,
        "f1_score": 0.57,
        "alert_rate": alert_rate,
        "confusion_matrix": {
            "fp": 10,
            "fn": 4,
            "tp": 7,
            "tn": 90,
        },
    }
