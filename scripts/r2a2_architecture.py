"""
R2A2 Architecture Reference Implementation
========================================

This module provides a reference implementation of a Responsible Reasoning AI Agent (R2A2) architecture as described in the survey
"Responsible Agentic Reasoning via AI Agents".  The goal of this implementation is to demonstrate how the conceptual blueprint shown in
Figures 5 and 6 of the survey. It is **not** a production-ready agent, but rather a fully‑instrumented skeleton that you can extend with your own models,
data stores, and safety modules.

The R2A2 architecture is organized into four layers:

1. **Core LLM Layer**: Performs basic language modeling (tokenization,    embeddings, transformer stack).  In this reference, the `CoreLLM`
   class serves as a stub; you may plug in a pre‑trained model (e.g.,   from HuggingFace or OpenAI) by overriding `generate`.

2. **Reasoning‑Capable LLM Layer**: Augments the core with reasoning capabilities such as chain‑of‑thought scratchpads, tree search, and
   self‑consistency sampling.  Here we provide `Scratchpad` to record intermediate thoughts and `ReasoningSampler` as a placeholder for
   sampling multiple reasoning paths.

3. **Agentic LLM Layer**: Provides planning, memory, reflection, and access to tools.  We implement `PlannerExecutor`, `MemoryStore`,
   and `ToolAPIProxy` to showcase goal decomposition, contextual retrieval, and tool‑augmented execution.

4. **R2A2 Layer**: Enforces governance, bias mitigation, privacy  protection, and audit logging.  The `PolicyEngine`,
   `BiasPrivacyFilter`, and `AuditLogger` classes implement these  safeguards, while `MonitoringDashboard` (a minimal stub) illustrates
   how one might surface metrics to a user or operator.

The top‑level `R2A2Agent` class orchestrates these components.  To use this module in practice, instantiate `R2A2Agent` and call
`agent.run(question)` with an input string.  The agent will perform the following steps:

* **Ingest**: Apply bias and privacy filtering to the raw question;  log the input.
* **Reason**: Use the planner/executor to generate a chain of  thought, optionally using the scratchpad and memory.  Tool calls
  may be invoked via the `ToolAPIProxy`.
* **Act**: Produce a final answer, run through the policy engine for  safety, and format the output.  All steps and tool calls are
  recorded in the audit log.

This implementation includes detailed docstrings and comments to help
users understand the flow and to facilitate extension.  See the
accompanying `README.md` for usage instructions and additional
context.
"""

from __future__ import annotations

import abc
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class CoreLLM(abc.ABC):
    """Abstract base class for the core language model layer.

    The purpose of this class is to encapsulate tokenization,
    embeddings, and the transformer stack.  Subclasses should
    implement the `generate` method to return a model output given an
    input prompt.  In this reference implementation, we provide a
    simple deterministic fallback that echoes the input or returns a
    canned answer when the input contains arithmetic expressions.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a continuation of the given prompt.

        Subclasses may call external models or local models.  The
        return value should contain both the chain of thought and the
        final answer in natural language, separated if desired.  This
        method is expected to be side‑effect free.
        """

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


class BaseLLM(CoreLLM):
    """A fallback language model that demonstrates simple behaviour.

    When the input contains a basic arithmetic operation (e.g.,
    "2+3"), this model responds with a chain of thought explaining
    that it will compute the expression and then supplies the answer.
    Otherwise, it returns a generic placeholder explanation and
    answer.  This class is intended for demonstration only; replace
    it with a real model for practical use.
    """

    def generate(self, prompt: str) -> str:
        # Very naive pattern detection for arithmetic
        expr = self._extract_expression(prompt)
        if expr:
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return f"I need to compute {expr}.\nThe result is {result}."
            except Exception:
                return f"I need to compute {expr}.\nI encountered an error."
        return "I will think about this step by step.\nHere is a placeholder answer."

    @staticmethod
    def _extract_expression(prompt: str) -> Optional[str]:
        # Remove question marks and trim whitespace
        cleaned = prompt.strip().rstrip("?")
        # Naively detect digits and operators; return expression if both present
        operators = ['+', '-', '*', '/']
        if any(op in cleaned for op in operators) and any(ch.isdigit() for ch in cleaned):
            return cleaned
        return None


class Scratchpad:
    """Simple scratchpad for chain‑of‑thought reasoning.

    Stores intermediate thoughts as a list of strings.  The
    scratchpad may be cleared between runs.  In a more sophisticated
    agent, this could support tree structures (tree of thought) or
    graph of thought; here we keep it linear for simplicity.
    """

    def __init__(self):
        self.entries: List[str] = []

    def add(self, text: str) -> None:
        """Append a new reasoning step to the scratchpad."""
        self.entries.append(text)

    def clear(self) -> None:
        """Clear all stored reasoning steps."""
        self.entries = []

    def __str__(self) -> str:
        return "\n".join(self.entries)


class ReasoningSampler:
    """Placeholder for self‑consistency sampling of reasoning paths.

    In a fully‑fledged R2A2 agent, one might sample multiple CoT
    trajectories and select the most coherent answer according to a
    scoring function.  Here we implement a trivial sampler that
    returns a single candidate equal to the model output.
    """

    def sample(self, model: CoreLLM, prompt: str, n_samples: int = 1) -> List[str]:
        return [model(prompt) for _ in range(n_samples)]

    def select_best(self, samples: List[str]) -> str:
        # In this reference implementation, simply return the first candidate.
        return samples[0]


class MemoryStore:
    """Simple key‑value memory for context retention.

    The agent may read from and write to this store during
    execution.  For demonstration, we implement persistent memory
    in memory only; in a real system you could persist to disk or
    an external database.
    """

    def __init__(self):
        self.memory: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.memory.get(key)

    def set(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def clear(self) -> None:
        self.memory = {}


class ToolAPIProxy:
    """Registry and dispatcher for external tools.

    Tools are callables that take a string input and return a result.
    The proxy maintains a mapping from tool names to callables and
    provides a `call` method to dispatch requests.  All tool calls
    return a tuple `(success, result, error)`.  Successful calls
    produce `True` and a result; failures return `False` and an
    error message.
    """

    def __init__(self):
        self.tools: Dict[str, Callable[[str], Any]] = {
            "calculator": self._calculator,
            # Additional tools can be registered here
        }

    def register_tool(self, name: str, func: Callable[[str], Any]) -> None:
        self.tools[name] = func

    def call(self, name: str, argument: str) -> Tuple[bool, Any, Optional[str]]:
        tool = self.tools.get(name)
        if not tool:
            return False, None, f"Tool '{name}' not found"
        try:
            result = tool(argument)
            return True, result, None
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def _calculator(expression: str) -> Any:
        # Use Python’s eval safely for demonstration; restrict builtins.
        return eval(expression, {"__builtins__": {}}, {})


class PlannerExecutor:
    """Planner and executor for agentic tasks.

    This class receives a question and returns a chain of thought and
    final answer.  It uses the scratchpad to record intermediate
    reasoning, the memory store for retrieval/persistence, and the
    tool API proxy for external computations.  The current
    implementation uses a simplistic approach: it forwards the input
    to the reasoning sampler (backed by the model) and then parses
    the response to detect tool invocations.  You may extend this
    class to support structured plans, iterative reflection, and more.
    """

    def __init__(self, model: CoreLLM, scratchpad: Scratchpad, memory: MemoryStore, tools: ToolAPIProxy, sampler: ReasoningSampler):
        self.model = model
        self.scratchpad = scratchpad
        self.memory = memory
        self.tools = tools
        self.sampler = sampler

    def plan_and_execute(self, question: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        # Clear scratchpad for new reasoning
        self.scratchpad.clear()
        # Compose prompt (could include memory context or system prompt)
        prompt = question
        # Sample reasoning paths
        samples = self.sampler.sample(self.model, prompt, n_samples=1)
        reasoning = self.sampler.select_best(samples)
        # Split into lines and process
        lines = reasoning.split("\n")
        audit_trail: List[Dict[str, Any]] = []
        final_answer: str = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Detect if this line requests a tool call (toy pattern: contains 'call:' followed by tool name)
            if stripped.lower().startswith("call:"):
                parts = stripped.split(":", 2)
                if len(parts) == 3:
                    tool_name = parts[1].strip()
                    argument = parts[2].strip()
                    success, result, error = self.tools.call(tool_name, argument)
                    self.scratchpad.add(f"Called {tool_name}({argument}) -> {result if success else error}")
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
                        # Optionally update memory with result
                        self.memory.set("last_tool_result", result)
                    continue
            # Otherwise treat as reasoning or answer
            self.scratchpad.add(stripped)
            audit_trail.append({
                "type": "reasoning",
                "text": stripped,
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Heuristic: if line starts with 'The result is' or 'Answer', treat as final answer
            if stripped.lower().startswith("the result is") or stripped.lower().startswith("answer"):
                final_answer = stripped.split(":", 1)[-1].strip()
        # If no explicit final answer found, default to last line
        if not final_answer and self.scratchpad.entries:
            final_answer = self.scratchpad.entries[-1]
        return str(self.scratchpad), final_answer, audit_trail


class PolicyEngine:
    """Policy engine enforcing constraints on the final output.

    This class can be extended to apply domain-specific rules (e.g.,
    refuse to answer illegal requests).  Here we implement a simple
    refusal if the output contains flagged terms.
    """

    def __init__(self):
        self.refusal_terms = {"racist", "sexist", "hate", "violence"}

    def enforce(self, answer: str) -> str:
        lower = answer.lower()
        if any(term in lower for term in self.refusal_terms):
            return "I’m sorry, but I cannot provide a response to that request."
        return answer


class BiasPrivacyFilter:
    """Filter that flags bias or privacy violations in text.

    `check` returns a tuple `(bias_flag, privacy_flag)`.  In this
    reference implementation, bias is flagged when the text contains
    certain keywords, and privacy is flagged when there are long
    sequences of digits.  Replace these heuristics with real
    detectors for production use.
    """

    def __init__(self):
        self.bias_keywords = {"racist", "sexist", "biased", "slur"}

    def check(self, text: str) -> Tuple[bool, bool]:
        bias_flag = any(word in text.lower() for word in self.bias_keywords)
        import re
        privacy_flag = bool(re.search(r"\d{9,}", text))
        return bias_flag, privacy_flag


class AuditLogger:
    """Collects a structured log of all interactions for auditability.

    The logger stores events in a list and can export them to JSON.
    Each event includes a timestamp and a free‑form payload.  The
    `record` method appends a new event; `export` returns the list of
    events.  This simple logger runs in memory; to persist logs,
    integrate with a database or external logging system.
    """

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
    """Minimal stub for a monitoring interface.

    In a full implementation, this could expose metrics (e.g., via
    Prometheus), provide real‑time dashboards, or send alerts.  Here
    it simply prints a summary of bias/privacy flags and policy
    enforcement actions.
    """

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


class R2A2Agent:
    """Top‑level orchestrator implementing the R2A2 pipeline.

    Combines all subcomponents: a core LLM, scratchpad, sampler,
    planner/executor, memory, tool proxy, policy engine, bias/privacy
    filter, audit logger, and monitoring dashboard.  The `run` method
    performs ingestion (filtering), reasoning, action, and governance
    for a given question.  It returns a `RunResult` containing the
    chain of thought, final answer, bias/privacy flags, and audit
    trail.
    """

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
        """Execute the full R2A2 pipeline for the given question."""
        self.logger.clear()
        logger.info("Received question: %s", question)
        # Ingest phase: bias/privacy filter on the input
        bias_in, privacy_in = self.filter.check(question)
        self.logger.record("input", {"question": question, "bias": bias_in, "privacy": privacy_in})
        # Plan and execute reasoning
        cot, answer, audit_trail = self.planner.plan_and_execute(question)
        # Filter final answer
        bias_ans, privacy_ans = self.filter.check(answer)
        enforced_answer = self.policy.enforce(answer)
        # Record final answer event
        self.logger.record("output", {"answer": enforced_answer, "bias": bias_ans, "privacy": privacy_ans})
        # Display summary in monitoring dashboard
        self.dashboard.display({
            "input_bias": bias_in,
            "input_privacy": privacy_in,
            "output_bias": bias_ans,
            "output_privacy": privacy_ans,
            "audit_events": len(audit_trail),
        })
        # Aggregate audit trail with logged events
        events = self.logger.export() + audit_trail
        return RunResult(
            question=question,
            chain_of_thought=cot,
            answer=enforced_answer,
            bias_flag=bias_in or bias_ans,
            privacy_flag=privacy_in or privacy_ans,
            audit_trail=events,
        )


def demo() -> None:
    """Run the agent on sample questions to illustrate behaviour."""
    agent = R2A2Agent()
    examples = [
        "What is 2 + 3?",
        "Explain the water cycle.",
        "Tell me a racist joke.",
        "Compute 1234567890123 + 1.",
    ]
    for q in examples:
        result = agent.run(q)
        print(result.to_json())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
