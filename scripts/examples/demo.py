#!/usr/bin/env python3
"""
Quick demo / smoke test — runs a few sample questions through the agent.

Usage:
    python examples/demo.py                                    # BaseLLM (no GPU, no API)
    python examples/demo.py --model Qwen/Qwen2.5-7B-Instruct  # HuggingFace (GPU)
    python examples/demo.py --model gpt-4o                     # OpenAI
    python examples/demo.py --model gpt-4o-mini                # OpenAI (cheaper)
    python examples/demo.py --model gemini-2.0-flash           # Gemini
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from src.agent import R2A2Agent
from src.llm import BaseLLM

DEMO_QUESTIONS = [
    # Simple arithmetic — should use calculator
    "What is 347 * 29?",
    # Needs web search
    "Who is the current Prime Minister of Canada?",
    # Needs Python execution
    "What is the sum of the first 100 prime numbers?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="base",
        help=(
            "Model to use: 'base' (no GPU/API), 'gpt-4o', 'gpt-4o-mini', "
            "'o3-mini', 'gemini-2.0-flash', or any HuggingFace model ID"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.model == "base":
        model = BaseLLM()
    else:
        from src.api_llm import create_llm
        model = create_llm(
            model_id=args.model,
            max_tokens=args.max_tokens,
        )

    print(f"Using model: {args.model}")
    agent = R2A2Agent(model=model, max_steps=args.max_steps)

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {q}")
        print(f"{'='*60}")
        result = agent.run(q)
        print(f"\nAnswer:  {result.answer}")
        print(f"Steps:   {result.n_steps}")
        print(f"Tools:   {result.tools_used}")
        print(f"\nChain of thought:\n{result.chain_of_thought}")


if __name__ == "__main__":
    main()
