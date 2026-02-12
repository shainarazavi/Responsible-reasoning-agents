from __future__ import annotations

import abc
import re
from typing import Optional


class CoreLLM(abc.ABC):
    """Abstract base class for the core language model layer."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


class BaseLLM(CoreLLM):
    """Placeholder LLM (demo only). Replace with HFLLM or an API backend."""

    _EXPR_RE = re.compile(r"([\d][\d\s\+\-\*\/\.\(\)]+[\d\)])")

    def generate(self, prompt: str) -> str:
        expr = self._extract_expression(prompt)
        if expr:
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return f"Thought: I need to compute {expr}.\nFINAL ANSWER: {result}"
            except Exception:
                return f"Thought: I need to compute {expr}.\nFINAL ANSWER: error"
        return "Thought: I will think about this step by step.\nFINAL ANSWER: unknown"

    @classmethod
    def _extract_expression(cls, prompt: str) -> Optional[str]:
        match = cls._EXPR_RE.search(prompt)
        if match:
            return match.group(1).strip()
        return None
