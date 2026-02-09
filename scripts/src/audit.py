from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


class AuditLogger:
    """Collects a structured log of interactions for auditability (in-memory)."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append({
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        })

    def export(self) -> List[Dict[str, Any]]:
        return self.events

    def clear(self) -> None:
        self.events = []


class MonitoringDashboard:
    """Minimal stub for a monitoring interface (prints summary)."""

    def display(self, summary: Dict[str, Any]) -> None:
        print("=== Monitoring Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
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

    def to_json(self) -> str:
        return json.dumps({
            "question": self.question,
            "chain_of_thought": self.chain_of_thought,
            "answer": self.answer,
            "bias_flag": self.bias_flag,
            "privacy_flag": self.privacy_flag,
            "audit_trail": self.audit_trail,
        }, indent=2)
