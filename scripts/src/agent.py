from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .llm import CoreLLM, BaseLLM
from .tools import ToolAPIProxy
from .policy import PolicyEngine, BiasPrivacyFilter
from .audit import AuditLogger, MonitoringDashboard, RunResult


logger = logging.getLogger(__name__)


class Scratchpad:
    """Simple scratchpad for chain-of-thought-like intermediate steps (linear)."""

    def __init__(self):
        self.entries: List[str] = []

    def add(self, text: str) -> None:
        self.entries.append(text)

    def clear(self) -> None:
        self.entries = []

    def __str__(self) -> str:
        return "\n".join(self.entries)


class ReasoningSampler:
    """Placeholder sampler for self-consistency (demo-only)."""

    def sample(self, model: CoreLLM, prompt: str, n_samples: int = 1) -> List[str]:
        return [model(prompt) for _ in range(n_samples)]

    def select_best(self, samples: List[str]) -> str:
        return samples[0]


class MemoryStore:
    """Simple in-memory key-value memory."""

    def __init__(self):
        self.memory: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.memory.get(key)

    def set(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def clear(self) -> None:
        self.memory = {}


class PlannerExecutor:
    """Planner/executor (single-pass) that can detect toy tool calls.

    Tool call format expected in model output lines:
        call:<tool_name>: <argument>
    """

    def __init__(
        self,
        model: CoreLLM,
        scratchpad: Scratchpad,
        memory: MemoryStore,
        tools: ToolAPIProxy,
        sampler: ReasoningSampler,
    ):
        self.model = model
        self.scratchpad = scratchpad
        self.memory = memory
        self.tools = tools
        self.sampler = sampler

    def plan_and_execute(self, question: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        self.scratchpad.clear()
        prompt = question

        samples = self.sampler.sample(self.model, prompt, n_samples=1)
        reasoning = self.sampler.select_best(samples)

        lines = reasoning.split("\n")
        audit_trail: List[Dict[str, Any]] = []
        final_answer: str = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.lower().startswith("call:"):
                parts = stripped.split(":", 2)
                if len(parts) == 3:
                    tool_name = parts[1].strip()
                    argument = parts[2].strip()
                    success, result, error = self.tools.call(tool_name, argument)

                    self.scratchpad.add(
                        f"Called {tool_name}({argument}) -> {result if success else error}"
                    )
                    audit_trail.append({
                        "type": "tool_call",
                        "tool": tool_name,
                        "argument": argument,
                        "success": success,
                        "result": result,
                        "error": error,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    if success:
                        self.memory.set("last_tool_result", result)
                    continue

            # Otherwise treat as reasoning/answer text
            self.scratchpad.add(stripped)
            audit_trail.append({
                "type": "reasoning",
                "text": stripped,
                "timestamp": datetime.utcnow().isoformat(),
            })

            if stripped.lower().startswith("the result is") or stripped.lower().startswith("answer"):
                final_answer = stripped.split(":", 1)[-1].strip()

        if not final_answer and self.scratchpad.entries:
            final_answer = self.scratchpad.entries[-1]

        return str(self.scratchpad), final_answer, audit_trail


class R2A2Agent:
    """Top-level orchestrator implementing an R2A2-style demo pipeline."""

    def __init__(self, model: Optional[CoreLLM] = None):
        self.model = model or BaseLLM()
        self.scratchpad = Scratchpad()
        self.memory = MemoryStore()
        self.tools = ToolAPIProxy()
        self.sampler = ReasoningSampler()
        self.planner = PlannerExecutor(self.model, self.scratchpad, self.memory, self.tools, self.sampler)
        self.policy = PolicyEngine()
        self.filter = BiasPrivacyFilter()
        self.logger = AuditLogger()
        self.dashboard = MonitoringDashboard()

    def run(self, question: str) -> RunResult:
        self.logger.clear()
        logger.info("Received question: %s", question)

        bias_in, privacy_in = self.filter.check(question)
        self.logger.record("input", {"question": question, "bias": bias_in, "privacy": privacy_in})

        cot, answer, audit_trail = self.planner.plan_and_execute(question)

        bias_ans, privacy_ans = self.filter.check(answer)
        enforced_answer = self.policy.enforce(answer)

        self.logger.record("output", {"answer": enforced_answer, "bias": bias_ans, "privacy": privacy_ans})

        self.dashboard.display({
            "input_bias": bias_in,
            "input_privacy": privacy_in,
            "output_bias": bias_ans,
            "output_privacy": privacy_ans,
            "audit_events": len(audit_trail),
        })

        events = self.logger.export() + audit_trail
        return RunResult(
            question=question,
            chain_of_thought=cot,
            answer=enforced_answer,
            bias_flag=bias_in or bias_ans,
            privacy_flag=privacy_in or privacy_ans,
            audit_trail=events,
        )
