from __future__ import annotations

import re
from typing import Tuple


class PolicyEngine:
    """Policy engine enforcing constraints on the final output."""

    def __init__(self):
        self.refusal_terms = {"racist", "sexist", "hate", "violence"}

    def enforce(self, answer: str) -> str:
        lower = answer.lower()
        if any(term in lower for term in self.refusal_terms):
            return "I'm sorry, but I cannot provide a response to that request."
        return answer


class BiasPrivacyFilter:
    """Heuristic filter that flags bias or privacy violations in text."""

    _PRIVACY_RE = re.compile(r"\d{9,}")

    def __init__(self):
        self.bias_keywords = {"racist", "sexist", "biased", "slur"}

    def check(self, text: str) -> Tuple[bool, bool]:
        bias_flag = any(word in text.lower() for word in self.bias_keywords)
        privacy_flag = bool(self._PRIVACY_RE.search(text))
        return bias_flag, privacy_flag
