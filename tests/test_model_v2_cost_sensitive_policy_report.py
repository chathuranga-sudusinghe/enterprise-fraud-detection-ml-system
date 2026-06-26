from scripts import generate_model_v2_cost_sensitive_policy_report as report_script


def test_generate_model_v2_cost_sensitive_policy_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_cost_sensitive_policy_experiment",
        _fake_policy_summary,
    )
    output_path = tmp_path / "policy_report.md"

    result = report_script.generate_model_v2_cost_sensitive_policy_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["candidate_weights"] == [1.0, 3.0]
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Cost-Sensitive Operating Policy Report" in report
    assert "Best Policy By Alert-Rate Constraint" in report
    assert "Full Policy Search Table" in report
    assert "0.0500" in report


def _fake_policy_summary():
    policy = {
        "candidate": "scale_pos_weight_3.0000",
        "scale_pos_weight": 3.0,
        "threshold": 0.20,
        "precision": 0.50,
        "recall": 0.70,
        "f1_score": 0.58,
        "alert_rate": 0.05,
        "false_positives": 10,
        "false_negatives": 3,
        "true_positives": 7,
        "true_negatives": 90,
    }
    return {
        "artifacts_written": False,
        "candidate_weights": [1.0, 3.0],
        "full_scale_pos_weight": 27.4343,
        "policy_rows": [policy],
        "policy_evaluations": [
            {
                "max_alert_rate": 0.05,
                "policy_found": True,
                "best_policy": policy,
            }
        ],
    }
