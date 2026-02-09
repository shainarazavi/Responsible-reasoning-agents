from __future__ import annotations

import abc
from typing import Optional


class CoreLLM(abc.ABC):
    """Abstract base class for the core language model layer.

    Subclasses should implement `generate(prompt)` and return a string response.
    """

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a continuation of the given prompt."""
        raise NotImplementedError

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


class BaseLLM(CoreLLM):
    """A placeholder LLM used by default (demo only).

    - If the prompt looks like a basic arithmetic expression, returns a computed result.
    - Otherwise, returns a generic placeholder response.

    Replace this with a real model adapter for practical use.
    """

    def generate(self, prompt: str) -> str:
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
        cleaned = prompt.strip().rstrip("?")
        operators = ['+', '-', '*', '/']
        if any(op in cleaned for op in operators) and any(ch.isdigit() for ch in cleaned):
            return cleaned
        return None
