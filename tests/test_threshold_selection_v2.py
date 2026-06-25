import pytest

from ml.utils.threshold_selection_v2 import select_model_v2_operating_threshold


def test_select_model_v2_operating_threshold_returns_comparison_table():
    result = select_model_v2_operating_threshold(
        y_true=[0, 1, 1, 0],
        y_proba=[0.05, 0.20, 0.80, 0.70],
        thresholds=(0.10, 0.50, 0.90),
        min_recall=0.50,
        max_alert_rate=0.50,
    )

    assert result["recommended_threshold"] == 0.50
    assert result["selection_rule"] == "max_f1_with_min_recall_and_max_alert_rate"
    assert result["threshold_comparison"]["threshold"].tolist() == [0.10, 0.50, 0.90]


def test_select_model_v2_operating_threshold_relaxes_alert_rate_if_needed():
    result = select_model_v2_operating_threshold(
        y_true=[0, 1, 1, 0],
        y_proba=[0.05, 0.20, 0.80, 0.70],
        thresholds=(0.10, 0.50, 0.90),
        min_recall=0.50,
        max_alert_rate=0.10,
    )

    assert result["recommended_threshold"] == 0.50
    assert result["selection_rule"] == "max_f1_with_min_recall_alert_rate_relaxed"


def test_select_model_v2_operating_threshold_preserves_recall_when_alert_relaxes():
    result = select_model_v2_operating_threshold(
        y_true=[0, 1, 1, 0],
        y_proba=[0.05, 0.20, 0.80, 0.70],
        thresholds=(0.10, 0.50, 0.90),
        min_recall=1.0,
        max_alert_rate=0.10,
    )

    assert result["recommended_threshold"] == 0.10
    assert result["selection_rule"] == "max_f1_with_min_recall_alert_rate_relaxed"


def test_select_model_v2_operating_threshold_falls_back_when_recall_not_met():
    result = select_model_v2_operating_threshold(
        y_true=[1, 0],
        y_proba=[0.05, 0.90],
        thresholds=(0.50,),
        min_recall=0.90,
        max_alert_rate=0.20,
    )

    assert result["recommended_threshold"] == 0.50
    assert result["selection_rule"] == "max_f1_recall_requirement_not_met"


def test_select_model_v2_operating_threshold_rejects_invalid_recall_target():
    with pytest.raises(ValueError, match="min_recall"):
        select_model_v2_operating_threshold(
            y_true=[0, 1],
            y_proba=[0.1, 0.9],
            min_recall=1.1,
        )


def test_select_model_v2_operating_threshold_rejects_invalid_alert_target():
    with pytest.raises(ValueError, match="max_alert_rate"):
        select_model_v2_operating_threshold(
            y_true=[0, 1],
            y_proba=[0.1, 0.9],
            max_alert_rate=-0.1,
        )
