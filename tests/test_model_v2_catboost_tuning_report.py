from scripts import generate_model_v2_catboost_tuning_report as report_script


def test_generate_model_v2_catboost_tuning_report_is_non_artifact_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_catboost_tuning_experiment",
        _fake_catboost_tuning_summary,
    )
    output_path = tmp_path / "catboost_tuning.md"

    result = report_script.generate_model_v2_catboost_tuning_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["best_catboost_policy"]["candidate"] == "catboost_default"
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 CatBoost Tuning Report" in report
    assert "LightGBM Benchmark" in report
    assert "CatBoost Candidate Summary" in report
    assert "catboost_default" in report
    assert "GPU" in report


def _fake_catboost_tuning_summary():
    policy = {
        "candidate": "catboost_default",
        "threshold": 0.10,
        "precision": 0.51,
        "recall": 0.61,
        "f1_score": 0.56,
        "alert_rate": 0.04,
        "false_positives": 10,
        "false_negatives": 4,
        "true_positives": 6,
        "true_negatives": 90,
    }
    return {
        "artifacts_written": False,
        "lightgbm_benchmark": {
            "candidate": "lightgbm_scale_pos_weight_5.0000",
            "scale_pos_weight": 5.0,
            "threshold": 0.20,
            "validation_metrics": {
                "precision": 0.45,
                "recall": 0.65,
                "f1_score": 0.53,
                "alert_rate": 0.05,
            },
        },
        "catboost_availability": {"available": True, "reason": "available"},
        "catboost_candidates": [
            {
                "candidate": "catboost_default",
                "status": "trained",
                "execution_device": "GPU",
                "selected_threshold": 0.10,
                "validation_metrics": {
                    "roc_auc": 0.90,
                    "pr_auc": 0.80,
                    "precision": 0.51,
                    "recall": 0.61,
                    "f1_score": 0.56,
                    "alert_rate": 0.04,
                },
            }
        ],
        "best_catboost_policy": policy,
        "policy_rows": [policy],
    }
