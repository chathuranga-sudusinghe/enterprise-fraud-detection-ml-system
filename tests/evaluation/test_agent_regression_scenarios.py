import json
from pathlib import Path

import pytest

from ml.agents import FraudAgent


SCENARIO_PATH = Path(__file__).parent / "scenarios" / "fraud_agent_eval_cases.json"


def load_scenarios():
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario", load_scenarios())
def test_fraud_agent_regression_scenarios(scenario):
    result = FraudAgent().analyze(
        scenario["user_input"],
        max_steps=scenario["max_steps"],
    )
    searchable_text = " ".join(
        str(result[field]).lower()
        for field in ("decision", "reasoning", "limitations")
    )

    assert result["tool_used"] == scenario["expected_tool"]
    assert result["risk_level"] == scenario["expected_risk_level"]
    assert result["requires_human_approval"] is scenario["expected_human_approval"]
    assert result["steps_used"] <= scenario["max_steps"]

    for keyword in scenario["expected_keywords"]:
        assert keyword.lower() in searchable_text