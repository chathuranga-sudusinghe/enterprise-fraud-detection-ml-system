from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class AgentOutput:
    decision: str
    risk_level: str
    reasoning: str
    tool_used: str
    requires_human_approval: bool
    confidence: str
    limitations: str
    steps_used: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)