from ml.agents import FraudAgent


def test_agent_selects_fraud_risk_tool_for_fraud_analysis():
    result = FraudAgent().analyze("Analyze fraud risk for transaction TX-101")

    assert result["tool_used"] == "fraud_risk_analysis"


def test_agent_selects_monitoring_tool_for_model_monitoring():
    result = FraudAgent().analyze("Review model monitoring metrics and alert rate for v2")

    assert result["tool_used"] == "model_monitoring"


def test_agent_selects_rollback_readiness_for_rollback_guidance():
    result = FraudAgent().analyze("Give rollback guidance if /predict/v2 becomes unsafe")

    assert result["tool_used"] == "rollback_readiness"
    assert result["requires_human_approval"] is True