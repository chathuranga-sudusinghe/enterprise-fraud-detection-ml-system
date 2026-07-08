from __future__ import annotations

import re
from typing import Any, Dict, List

from ml.agents.schemas import ToolResult


VALID_TOOLS = {
    "fraud_risk_analysis",
    "model_monitoring",
    "rollback_readiness",
    "none",
}


def extract_tool_arguments(user_input: str) -> Dict[str, Any]:
    text = user_input or ""
    lowered = text.lower()

    transaction_ids = re.findall(
        r"(?:transaction(?:\s*id)?|txn)\s*[:#]?\s*([A-Za-z0-9_-]+)",
        text,
        flags=re.IGNORECASE,
    )
    transaction_ids.extend(re.findall(r"\b(?:TX|txn)[-_]?\d+\b", text, flags=re.IGNORECASE))

    date_ranges = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    model_versions = sorted(set(re.findall(r"\bv[12]\b", lowered)))
    thresholds = [float(value) for value in re.findall(r"threshold\s*(?:=|to|at|:)?\s*(0?\.\d+|1(?:\.0+)?)", lowered)]
    risk_categories = [
        category
        for category in ("low", "medium", "high", "critical")
        if re.search(rf"\b{category}\b", lowered)
    ]

    return {
        "transaction_ids": sorted(set(transaction_ids)),
        "date_ranges": date_ranges,
        "model_versions": model_versions,
        "thresholds": thresholds,
        "risk_categories": risk_categories,
        "malformed_or_missing_inputs": not bool(text.strip()),
    }


def run_tool(name: str, arguments: Dict[str, Any], *, fail: bool = False) -> ToolResult:
    if name not in VALID_TOOLS:
        return ToolResult(
            name="none",
            ok=False,
            summary="No supported tool was selected.",
            data={"arguments": arguments},
            error="unsupported_tool",
        )

    if fail:
        return ToolResult(
            name=name,
            ok=False,
            summary="The selected tool failed before producing reliable evidence.",
            data={"arguments": arguments},
            error="tool_failure",
        )

    if name == "fraud_risk_analysis":
        return ToolResult(
            name=name,
            ok=True,
            summary="Prepared deterministic fraud risk analysis context.",
            data={"arguments": arguments},
        )

    if name == "model_monitoring":
        return ToolResult(
            name=name,
            ok=True,
            summary="Prepared deterministic model monitoring context.",
            data={"arguments": arguments},
        )

    if name == "rollback_readiness":
        return ToolResult(
            name=name,
            ok=True,
            summary="Prepared deterministic rollback readiness guidance context.",
            data={"arguments": arguments},
        )

    return ToolResult(
        name="none",
        ok=True,
        summary="No tool required for the request.",
        data={"arguments": arguments},
    )