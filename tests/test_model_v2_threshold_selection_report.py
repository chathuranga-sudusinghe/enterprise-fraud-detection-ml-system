from scripts import generate_model_v2_threshold_selection_report as report_script


def test_generate_model_v2_threshold_selection_report_uses_non_artifact_training(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_run_training_pipeline_v2(*, write_artifacts=False):
        calls.append(write_artifacts)
        return {
            "model_type": "lightgbm",
            "feature_engineering_version": "v2",
            "feature_count": 818,
            "would_write_artifacts": False,
            "artifacts_written": False,
            "threshold_selection": {
                "recommended_threshold": 0.40,
                "selection_rule": "max_f1_with_min_recall_and_max_alert_rate",
                "min_recall": 0.80,
                "max_alert_rate": 0.20,
                "recommended_metrics": {
                    "precision": 0.30,
                    "recall": 0.85,
                    "f1_score": 0.44,
                    "alert_rate": 0.12,
                },
            },
            "validation_threshold_comparison": [_threshold_row(0.40)],
            "test_threshold_comparison": [_threshold_row(0.40)],
        }

    monkeypatch.setattr(
        report_script,
        "run_training_pipeline_v2",
        fake_run_training_pipeline_v2,
    )
    output_path = tmp_path / "threshold_report.md"

    result = report_script.generate_model_v2_threshold_selection_report(
        output_path=output_path,
    )

    assert calls == [False]
    assert result["recommended_threshold"] == 0.40
    assert result["would_write_artifacts"] is False
    assert result["artifacts_written"] is False
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Threshold Selection Report" in report
    assert "Recommended threshold: 0.40" in report
    assert "Validation Threshold Comparison" in report
    assert "Test Threshold Comparison" in report


def _threshold_row(threshold):
    return {
        "threshold": threshold,
        "precision": 0.30,
        "recall": 0.85,
        "f1_score": 0.44,
        "alert_rate": 0.12,
        "false_positives": 10,
        "false_negatives": 3,
        "true_positives": 17,
        "true_negatives": 70,
    }
