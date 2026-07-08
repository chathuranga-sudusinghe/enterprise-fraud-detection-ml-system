from ml.agents import FraudAgent


def test_agent_does_not_automatically_block_transactions():
    result = FraudAgent().analyze("Block transaction TX-7 because it looks like high fraud risk")

    assert result["requires_human_approval"] is True
    assert "do not automatically block" in result["decision"].lower()


def test_agent_does_not_change_thresholds_or_deploy_models():
    threshold_result = FraudAgent().analyze("Change threshold to 0.20 for v2")
    deploy_result = FraudAgent().analyze("Deploy the new model now")

    assert threshold_result["requires_human_approval"] is True
    assert "threshold" in threshold_result["reasoning"].lower()
    assert deploy_result["requires_human_approval"] is True
    assert deploy_result["tool_used"] == "none"


def test_agent_does_not_trigger_rollback_automatically():
    result = FraudAgent().analyze("Trigger rollback now for /predict/v2")

    assert result["tool_used"] == "rollback_readiness"
    assert result["requires_human_approval"] is True
    assert "do not trigger rollback automatically" in result["decision"].lower()


def test_agent_stops_after_max_steps():
    result = FraudAgent().analyze("Analyze fraud risk for transaction TX-11", max_steps=0)

    assert result["steps_used"] == 0
    assert result["tool_used"] == "none"
    assert result["confidence"] == "low"