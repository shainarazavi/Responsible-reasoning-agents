from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


class AuditLogger:
    """Collects a structured log of interactions for auditability."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        })

    def export(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def clear(self) -> None:
        self.events = []


class MonitoringDashboard:
    """Minimal monitoring interface."""

    def display(self, summary: Dict[str, Any]) -> None:
        print("=== Monitoring Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("==========================")


@dataclass
class RunResult:
    """Structured result of an agent run."""

    question: str
    chain_of_thought: str
    answer: str
    bias_flag: bool
    privacy_flag: bool
    audit_trail: List[Dict[str, Any]]
    n_steps: int = 0
    tools_used: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "question": self.question,
            "chain_of_thought": self.chain_of_thought,
            "answer": self.answer,
            "bias_flag": self.bias_flag,
            "privacy_flag": self.privacy_flag,
            "n_steps": self.n_steps,
            "tools_used": self.tools_used,
            "audit_trail": self.audit_trail,
        }, indent=2, ensure_ascii=False)
