from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple


class ToolAPIProxy:
    """Registry and dispatcher for external tools.

    Tools are callables that take a string input and return a result.
    `call()` returns a tuple: (success, result, error_message).
    """

    def __init__(self):
        self.tools: Dict[str, Callable[[str], Any]] = {
            "calculator": self._calculator,
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
        # Demo-only calculator. Replace with a proper parser in production.
        return eval(expression, {"__builtins__": {}}, {})
