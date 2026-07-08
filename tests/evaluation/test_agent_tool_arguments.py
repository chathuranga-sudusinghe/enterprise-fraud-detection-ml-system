from ml.agents.tools import extract_tool_arguments


def test_extracts_structured_tool_arguments():
    args = extract_tool_arguments(
        "Analyze transaction TX-123 for v2 from 2026-06-01 to 2026-06-07 "
        "at threshold 0.10 with high risk."
    )

    assert "TX-123" in args["transaction_ids"]
    assert args["date_ranges"] == ["2026-06-01", "2026-06-07"]
    assert args["model_versions"] == ["v2"]
    assert args["thresholds"] == [0.10]
    assert "high" in args["risk_categories"]
    assert args["malformed_or_missing_inputs"] is False


def test_handles_missing_or_malformed_inputs_safely():
    args = extract_tool_arguments("")

    assert args["transaction_ids"] == []
    assert args["date_ranges"] == []
    assert args["model_versions"] == []
    assert args["thresholds"] == []
    assert args["risk_categories"] == []
    assert args["malformed_or_missing_inputs"] is True