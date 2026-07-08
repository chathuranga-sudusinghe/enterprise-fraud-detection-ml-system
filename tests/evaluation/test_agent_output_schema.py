from ml.agents import FraudAgent


EXPECTED_KEYS = {
    "decision",
    "risk_level",
    "reasoning",
    "tool_used",
    "requires_human_approval",
    "confidence",
    "limitations",
    "steps_used",
}


def test_agent_output_schema_matches_contract():
    result = FraudAgent().analyze("Analyze fraud risk for transaction TX-101")

    assert set(result) == EXPECTED_KEYS
    assert isinstance(result["decision"], str)
    assert result["risk_level"] in {"low", "medium", "high", "unknown"}
    assert isinstance(result["reasoning"], str)
    assert isinstance(result["tool_used"], str)
    assert isinstance(result["requires_human_approval"], bool)
    assert result["confidence"] in {"low", "medium", "high"}
    assert isinstance(result["limitations"], str)
    assert isinstance(result["steps_used"], int)


def test_failure_returns_structured_safe_fallback():
    result = FraudAgent().analyze("force_tool_failure fraud risk for transaction TX-200")

    assert result["risk_level"] == "unknown"
    assert result["confidence"] == "low"
    assert "safe fallback" in result["decision"].lower()
    assert "failure=" in result["limitations"]
    assert result["steps_used"] == 1