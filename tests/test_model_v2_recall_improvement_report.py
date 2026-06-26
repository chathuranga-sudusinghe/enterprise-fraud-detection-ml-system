from scripts import generate_model_v2_recall_improvement_report as report_script


def test_generate_model_v2_recall_improvement_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_recall_improvement_experiment",
        _fake_experiment_summary,
    )
    output_path = tmp_path / "recall_report.md"

    result = report_script.generate_model_v2_recall_improvement_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["scale_pos_weight"] == 3.0
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Recall Improvement Experiment" in report
    assert "baseline_v2" in report
    assert "weighted_v2_scale_pos_weight" in report
    assert "False Negative Analysis" in report


def _fake_experiment_summary():
    candidate = {
        "candidate": "baseline_v2",
        "selected_threshold": 0.20,
        "validation_metrics": _metrics(),
        "test_metrics": _metrics(),
        "validation_threshold_comparison": [_threshold_row()],
        "test_threshold_comparison": [_threshold_row()],
        "validation_false_negative_analysis": [_false_negative_row()],
        "test_false_negative_analysis": [_false_negative_row()],
    }
    weighted = {**candidate, "candidate": "weighted_v2_scale_pos_weight"}
    return {
        "feature_count": 818,
        "train_fraud_count": 10,
        "train_non_fraud_count": 30,
        "scale_pos_weight": 3.0,
        "artifacts_written": False,
        "baseline_v2": candidate,
        "weighted_v2": weighted,
    }


def _metrics():
    return {
        "roc_auc": 0.90,
        "pr_auc": 0.80,
        "precision": 0.50,
        "recall": 0.70,
        "f1_score": 0.58,
        "alert_rate": 0.10,
        "confusion_matrix": {"tn": 90, "fp": 5, "fn": 3, "tp": 7},
        "threshold": 0.20,
    }


def _threshold_row():
    return {
        "threshold": 0.10,
        "precision": 0.40,
        "recall": 0.80,
        "f1_score": 0.53,
        "alert_rate": 0.20,
        "false_positives": 10,
        "false_negatives": 2,
        "true_positives": 8,
        "true_negatives": 80,
    }


def _false_negative_row():
    return {
        "threshold": 0.10,
        "false_negatives": 2,
        "fraud_count": 10,
        "missed_fraud_rate": 0.20,
    }
