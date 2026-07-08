from __future__ import annotations

from typing import Any, Dict

from ml.agents.schemas import AgentOutput
from ml.agents.tools import extract_tool_arguments, run_tool


class FraudAgent:
    """Deterministic fraud-operations agent stub for evaluation tests.

    This class is intentionally rule-based. It does not call LLM APIs, mutate model
    state, deploy code, block transactions, change thresholds, or trigger rollback.
    """

    def __init__(self, default_max_steps: int = 3) -> None:
        self.default_max_steps = default_max_steps

    def analyze(self, user_input: str, max_steps: int | None = None) -> Dict[str, Any]:
        allowed_steps = self.default_max_steps if max_steps is None else max_steps
        if allowed_steps <= 0:
            return self._output(
                decision="No action taken because the step budget was exhausted.",
                risk_level="unknown",
                reasoning="The evaluation harness stopped before tool use because max_steps was 0.",
                tool_used="none",
                requires_human_approval=False,
                confidence="low",
                limitations="No evidence was collected due to the configured step limit.",
                steps_used=0,
            ).to_dict()

        text = user_input or ""
        lowered = text.lower()
        tool_name = self._select_tool(lowered)
        arguments = extract_tool_arguments(text)
        requires_approval = self._requires_human_approval(lowered)
        fail_tool = "force_tool_failure" in lowered or "simulate tool failure" in lowered

        tool_result = run_tool(tool_name, arguments, fail=fail_tool)
        steps_used = min(1, allowed_steps)

        if not tool_result.ok:
            return self._output(
                decision="Safe fallback: no reliable fraud decision returned.",
                risk_level="unknown",
                reasoning=(
                    "The selected tool or endpoint failed, so the agent surfaced the "
                    "failure instead of silently returning an unreliable fraud decision."
                ),
                tool_used=tool_result.name,
                requires_human_approval=requires_approval,
                confidence="low",
                limitations=f"failure={tool_result.error}; retry_loop_prevented=True",
                steps_used=steps_used,
            ).to_dict()

        if tool_name == "fraud_risk_analysis":
            return self._fraud_response(lowered, tool_name, requires_approval, arguments, steps_used).to_dict()

        if tool_name == "model_monitoring":
            return self._output(
                decision="Review model monitoring signals before taking operational action.",
                risk_level="medium",
                reasoning=(
                    "Use model monitoring evidence such as alert rate, latency, error rate, "
                    "model version usage, and threshold behavior before making decisions."
                ),
                tool_used=tool_name,
                requires_human_approval=requires_approval,
                confidence="medium",
                limitations="This stub summarizes monitoring intent only and does not query live Prometheus or Grafana.",
                steps_used=steps_used,
            ).to_dict()

        if tool_name == "rollback_readiness":
            return self._output(
                decision="Provide rollback readiness guidance only; do not trigger rollback automatically.",
                risk_level="high" if requires_approval else "medium",
                reasoning=(
                    "Rollback is a release-level operation. The agent can recommend checks for "
                    "artifact validation failures, high error rate, high latency, abnormal alert rate, "
                    "and feature validation failures, but rollback requires human approval."
                ),
                tool_used=tool_name,
                requires_human_approval=True,
                confidence="medium",
                limitations="No deployment state was changed and no rollback command was executed.",
                steps_used=steps_used,
            ).to_dict()

        return self._output(
            decision="Insufficient evidence for a fraud or operations recommendation.",
            risk_level="unknown",
            reasoning="The request did not map to a supported deterministic fraud-agent tool.",
            tool_used="none",
            requires_human_approval=requires_approval,
            confidence="low",
            limitations="Ask for fraud risk analysis, model monitoring, or rollback readiness guidance.",
            steps_used=steps_used,
        ).to_dict()

    def _select_tool(self, lowered: str) -> str:
        if any(term in lowered for term in ("rollback", "roll back", "revert release", "disable v2")):
            return "rollback_readiness"
        if any(term in lowered for term in ("monitor", "metrics", "latency", "alert rate", "error rate", "grafana", "prometheus")):
            return "model_monitoring"
        if any(term in lowered for term in ("fraud", "risk", "transaction", "false positive", "false negative", "threshold")):
            return "fraud_risk_analysis"
        return "none"

    def _requires_human_approval(self, lowered: str) -> bool:
        high_impact_terms = (
            "block transaction",
            "approve transaction",
            "decline transaction",
            "change threshold",
            "set threshold",
            "deploy",
            "promote model",
            "trigger rollback",
            "rollback now",
            "disable /predict/v2",
        )
        return any(term in lowered for term in high_impact_terms)

    def _fraud_response(
        self,
        lowered: str,
        tool_name: str,
        requires_approval: bool,
        arguments: Dict[str, Any],
        steps_used: int,
    ) -> AgentOutput:
        risk_level = self._risk_level(lowered, arguments)
        confidence = "medium" if arguments.get("transaction_ids") else "low"
        limitations = "Evidence is limited to deterministic request parsing; no live model endpoint was called."
        if requires_approval:
            limitations += " High-impact actions are advisory only and require human approval."

        return self._output(
            decision="Return analyst support for fraud risk; do not automatically block or approve the transaction.",
            risk_level=risk_level,
            reasoning=(
                "Explain the decision using recall, precision, false positives, false negatives, "
                "alert rate, and threshold trade-offs. Avoid overconfident claims when evidence "
                "is insufficient or when the model endpoint has not been queried."
            ),
            tool_used=tool_name,
            requires_human_approval=requires_approval,
            confidence=confidence,
            limitations=limitations,
            steps_used=steps_used,
        )

    def _risk_level(self, lowered: str, arguments: Dict[str, Any]) -> str:
        categories = arguments.get("risk_categories", [])
        if "critical" in categories or "high" in categories:
            return "high"
        if "low" in categories:
            return "low"
        if any(term in lowered for term in ("stolen", "chargeback", "account takeover", "very large")):
            return "high"
        if any(term in lowered for term in ("unknown", "missing", "malformed", "insufficient")):
            return "unknown"
        return "medium"

    def _output(
        self,
        *,
        decision: str,
        risk_level: str,
        reasoning: str,
        tool_used: str,
        requires_human_approval: bool,
        confidence: str,
        limitations: str,
        steps_used: int,
    ) -> AgentOutput:
        return AgentOutput(
            decision=decision,
            risk_level=risk_level,
            reasoning=reasoning,
            tool_used=tool_used,
            requires_human_approval=requires_human_approval,
            confidence=confidence,
            limitations=limitations,
            steps_used=steps_used,
        )