from __future__ import annotations

import logging
import os
import sys

# Allow running demo.py directly without installing as a package.
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from src.agent import R2A2Agent


def demo() -> None:
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
