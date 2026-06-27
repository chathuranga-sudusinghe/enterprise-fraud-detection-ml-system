from scripts import generate_model_v2_deep_learning_baseline_report as report_script


def test_generate_deep_learning_baseline_report_handles_trained_summary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_deep_learning_baseline",
        _fake_trained_summary,
    )
    output_path = tmp_path / "deep_learning_report.md"

    result = report_script.generate_model_v2_deep_learning_baseline_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["status"] == "trained"
    report = output_path.read_text(encoding="utf-8")
    assert "# Model v2 Deep Learning Baseline Report" in report
    assert "CatBoost Validated Baseline" in report
    assert "Neural Baseline Metrics" in report
    assert "catboost_remains_benchmark" in report


def test_generate_deep_learning_baseline_report_handles_skipped_summary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        report_script,
        "run_model_v2_deep_learning_baseline",
        _fake_skipped_summary,
    )
    output_path = tmp_path / "deep_learning_skipped.md"

    result = report_script.generate_model_v2_deep_learning_baseline_report(
        output_path=output_path,
    )

    assert result["artifacts_written"] is False
    assert result["status"] == "skipped"
    report = output_path.read_text(encoding="utf-8")
    assert "Skipped: missing torch" in report
    assert "PyTorch is unavailable" in report


def _fake_trained_summary():
    decision = {
        "recommended_candidate": "catboost_default",
        "beats_catboost_baseline": False,
        "recommendation": "catboost_remains_benchmark",
        "selected_threshold": 0.20,
        "reason": "CatBoost remains stronger.",
        "risks": ["Requires separate deployment review."],
    }
    return {
        "artifacts_written": False,
        "status": "trained",
        "torch_availability": {
            "available": True,
            "reason": "available",
            "version": "test",
            "cuda_available": False,
            "device": "cpu",
        },
        "catboost_validated_baseline": {
            "candidate": "catboost_default",
            "threshold": 0.10,
            "test_alert_rate": 0.0493,
            "test_precision": 0.4103,
            "test_recall": 0.5806,
            "test_f1_score": 0.4808,
        },
        "feature_count": 831,
        "mlp_input_dim": 900,
        "device": "cpu",
        "selected_threshold": 0.20,
        "training": {
            "epochs_completed": 3,
            "best_validation_pr_auc": 0.40,
        },
        "validation_metrics": _metrics(alert_rate=0.04),
        "test_metrics": _metrics(alert_rate=0.04),
        "validation_threshold_comparison": [
            {
                "threshold": 0.20,
                "precision": 0.40,
                "recall": 0.50,
                "f1_score": 0.44,
                "alert_rate": 0.04,
                "false_positives": 10,
                "false_negatives": 5,
                "true_positives": 5,
                "true_negatives": 90,
            }
        ],
        "decision": decision,
    }


def _fake_skipped_summary():
    return {
        "artifacts_written": False,
        "status": "skipped",
        "skip_reason": "missing torch",
        "torch_availability": {
            "available": False,
            "reason": "missing torch",
            "version": None,
            "cuda_available": False,
            "device": None,
        },
        "catboost_validated_baseline": {
            "candidate": "catboost_default",
            "threshold": 0.10,
            "test_alert_rate": 0.0493,
            "test_precision": 0.4103,
            "test_recall": 0.5806,
            "test_f1_score": 0.4808,
        },
        "decision": {
            "recommended_candidate": None,
            "beats_catboost_baseline": False,
            "reason": "PyTorch is unavailable, so the neural baseline was skipped.",
        },
    }


def _metrics(*, alert_rate):
    return {
        "roc_auc": 0.80,
        "pr_auc": 0.45,
        "precision": 0.40,
        "recall": 0.50,
        "f1_score": 0.44,
        "alert_rate": alert_rate,
        "confusion_matrix": {
            "fp": 10,
            "fn": 5,
            "tp": 5,
            "tn": 90,
        },
    }
