from scripts import generate_model_v2_model_family_comparison_report as report_script


def test_generate_model_v2_model_family_comparison_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_model_family_comparison_experiment",
        _fake_family_summary,
    )
    output_path = tmp_path / "family_report.md"

    result = report_script.generate_model_v2_model_family_comparison_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["best_policy"]["model_family"] == "lightgbm"
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Model-Family Comparison Report" in report
    assert "Candidate Availability" in report
    assert "Best Policy Under Alert Constraint" in report
    assert "missing xgboost" in report


def _fake_family_summary():
    policy = {
        "model_family": "lightgbm",
        "candidate": "lightgbm_scale_pos_weight_5.0000",
        "scale_pos_weight": 5.0,
        "threshold": 0.20,
        "precision": 0.60,
        "recall": 0.65,
        "f1_score": 0.62,
        "alert_rate": 0.049,
        "false_positives": 10,
        "false_negatives": 3,
        "true_positives": 7,
        "true_negatives": 90,
    }
    return {
        "artifacts_written": False,
        "max_alert_rate": 0.05,
        "benchmark": {
            "model_family": "lightgbm",
            "scale_pos_weight": 5.0,
            "threshold": 0.20,
        },
        "availability": {
            "lightgbm": {"available": True, "reason": "available"},
            "xgboost": {"available": False, "reason": "missing xgboost"},
            "catboost": {"available": False, "reason": "missing catboost"},
        },
        "candidates": [
            {
                "model_family": "lightgbm",
                "candidate": "lightgbm_scale_pos_weight_5.0000",
                "status": "trained",
                "selected_threshold": 0.20,
                "validation_metrics": {
                    "roc_auc": 0.90,
                    "pr_auc": 0.80,
                    "precision": 0.60,
                    "recall": 0.65,
                    "f1_score": 0.62,
                    "alert_rate": 0.049,
                },
            },
            {
                "model_family": "xgboost",
                "candidate": "xgboost_default_v2",
                "status": "skipped",
                "skip_reason": "missing xgboost",
            },
        ],
        "policy_rows": [policy],
        "best_policy": policy,
    }
