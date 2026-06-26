from scripts import generate_model_v2_false_negative_analysis_report as report_script


def test_generate_model_v2_false_negative_analysis_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_false_negative_analysis",
        _fake_false_negative_summary,
    )
    output_path = tmp_path / "false_negative_report.md"

    result = report_script.generate_model_v2_false_negative_analysis_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["threshold"] == 0.20
    assert result["scale_pos_weight"] == 5.0
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 False-Negative Analysis Report" in report
    assert "Selected Policy" in report
    assert "Validation Top False-Negative Groups" in report
    assert "ProductCD" in report
    assert "visa" in report


def _fake_false_negative_summary():
    group_row = {
        "group_column": "ProductCD",
        "group_value": "visa",
        "false_negatives": 5,
        "true_positives": 4,
        "fraud_count": 9,
        "non_fraud_count": 20,
        "total_count": 29,
        "missed_fraud_rate": 5 / 9,
        "false_negative_share": 0.25,
        "fraud_rate": 9 / 29,
    }
    split_analysis = {
        "summary": {
            "row_count": 100,
            "fraud_count": 20,
            "false_negative_count": 8,
            "true_positive_count": 12,
            "missed_fraud_rate": 0.40,
        },
        "top_false_negative_groups": [group_row],
    }
    return {
        "artifacts_written": False,
        "model_family": "lightgbm",
        "scale_pos_weight": 5.0,
        "threshold": 0.20,
        "feature_count": 818,
        "categorical_feature_count": 20,
        "validation_analysis": split_analysis,
        "test_analysis": split_analysis,
    }
