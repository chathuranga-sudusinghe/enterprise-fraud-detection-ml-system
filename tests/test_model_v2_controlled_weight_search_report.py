from scripts import generate_model_v2_controlled_weight_search_report as report_script


def test_generate_model_v2_controlled_weight_search_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_controlled_weight_search_experiment",
        _fake_search_summary,
    )
    output_path = tmp_path / "controlled_weight_search.md"

    result = report_script.generate_model_v2_controlled_weight_search_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["candidate_weights"] == [1.0, 3.0]
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Controlled Weight Search Report" in report
    assert "scale_pos_weight_1.0000" in report
    assert "scale_pos_weight_3.0000" in report
    assert "Validation Threshold Comparison" in report


def _fake_search_summary():
    candidates = [_candidate(1.0), _candidate(3.0)]
    return {
        "feature_count": 818,
        "train_fraud_count": 10,
        "train_non_fraud_count": 30,
        "full_scale_pos_weight": 3.0,
        "candidate_weights": [1.0, 3.0],
        "artifacts_written": False,
        "candidate_summary": [
            {
                "candidate": candidate["candidate"],
                "scale_pos_weight": candidate["scale_pos_weight"],
                "selected_threshold": candidate["selected_threshold"],
                "validation_roc_auc": 0.90,
                "validation_pr_auc": 0.80,
                "validation_precision": 0.50,
                "validation_recall": 0.70,
                "validation_f1_score": 0.58,
                "validation_alert_rate": 0.10,
                "validation_false_negatives": 3,
                "validation_recall_delta": 0.0,
                "validation_pr_auc_delta": 0.0,
                "validation_precision_delta": 0.0,
                "validation_alert_rate_delta": 0.0,
            }
            for candidate in candidates
        ],
        "candidates": candidates,
    }


def _candidate(weight):
    return {
        "candidate": f"scale_pos_weight_{weight:.4f}",
        "scale_pos_weight": weight,
        "selected_threshold": 0.20,
        "validation_threshold_comparison": [_threshold_row()],
        "test_threshold_comparison": [_threshold_row()],
        "validation_false_negative_analysis": [_false_negative_row()],
        "test_false_negative_analysis": [_false_negative_row()],
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
