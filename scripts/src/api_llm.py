# src/api_llm.py
"""
API-based LLM backends for OpenAI and Google Gemini.

Usage:
    from src.api_llm import OpenAILLM, GeminiLLM

    model = OpenAILLM("gpt-4o")
    model = OpenAILLM("o3-mini")
    model = GeminiLLM("gemini-2.0-flash")
    model = GeminiLLM("gemini-2.5-pro-preview-05-06")
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from .llm import CoreLLM

logger = logging.getLogger(__name__)

# ── Shared ReAct system prompt (same as hf_llm.py) ───────────────────
DEFAULT_REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by reasoning step-by-step \
and using tools when needed.

Available tools:
- web_search: Search the web for current information.
  Usage:  Action: web_search: <query>
- web_fetch: Retrieve the full text of a web page.
  Usage:  Action: web_fetch: <url>
- python: Execute Python code and return stdout.
  Usage:  Action: python: <code>
- calculator: Evaluate a mathematical expression.
  Usage:  Action: calculator: <expression>
- file_reader: Read and summarise an attached file.
  Usage:  Action: file_reader: <file_path>

Response format — follow this EXACTLY for every turn:
Thought: <your reasoning about what to do next>
Action: <tool_name>: <argument>

When you have the final answer, write EXACTLY:
Thought: <why this is the answer>
FINAL ANSWER: <short, direct answer only>

Rules:
- Use tools to look up facts; do NOT guess or hallucinate.
- Give ONLY ONE Action per turn (the agent loop will show you the Observation).
- FINAL ANSWER must be concise: a number, a name, a short phrase, or a comma-separated list.
- Do NOT include extra explanation after FINAL ANSWER.
"""

# ── Retry helper ──────────────────────────────────────────────────────
def _retry(fn, max_retries: int = 3, backoff: float = 2.0):
    """Call fn() with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = backoff * (2 ** attempt)
            logger.warning(
                "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, e, wait,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


# ═══════════════════════════════════════════════════════════════════════
# OpenAI  (GPT-4o, GPT-4o-mini, o3-mini, o1, etc.)
# ═══════════════════════════════════════════════════════════════════════
class OpenAILLM(CoreLLM):
    """
    OpenAI API backend.

    Requires:  pip install openai
    Env var:   OPENAI_API_KEY

    Examples:
        model = OpenAILLM("gpt-4o")
        model = OpenAILLM("gpt-4o-mini", temperature=0.0)
        model = OpenAILLM("o3-mini")
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_REACT_SYSTEM_PROMPT

        import openai
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn chat (used by ReAct loop)."""
        # Ensure system prompt is first
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        def _call():
            kwargs = dict(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            # o1/o3 models don't support temperature
            if not self.model_id.startswith(("o1", "o3")):
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        return _retry(_call)


# ═══════════════════════════════════════════════════════════════════════
# Google Gemini  (gemini-2.0-flash, gemini-2.5-pro, etc.)
# ═══════════════════════════════════════════════════════════════════════
class GeminiLLM(CoreLLM):
    """
    Google Gemini API backend.

    Requires:  pip install google-genai
    Env var:   GOOGLE_API_KEY  (get from https://aistudio.google.com/apikey)

    Examples:
        model = GeminiLLM("gemini-2.0-flash")
        model = GeminiLLM("gemini-2.5-pro-preview-05-06")
    """

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_REACT_SYSTEM_PROMPT

        from google import genai
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
        )

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn chat (used by ReAct loop)."""
        from google.genai import types

        # Separate system prompt from conversation
        system_text = self.system_prompt
        chat_messages = []

        for m in messages:
            role = m["role"]
            if role == "system":
                system_text = m["content"]
            elif role == "user":
                chat_messages.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=m["content"])],
                    )
                )
            elif role == "assistant":
                chat_messages.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=m["content"])],
                    )
                )

        def _call():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=chat_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_text,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            return response.text.strip()

        return _retry(_call)


# ═══════════════════════════════════════════════════════════════════════
# Factory — create any backend from a string
# ═══════════════════════════════════════════════════════════════════════
def create_llm(
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
) -> CoreLLM:
    """
    Create an LLM backend from a model ID string.

    Examples:
        create_llm("gpt-4o")              → OpenAILLM
        create_llm("o3-mini")             → OpenAILLM
        create_llm("gemini-2.0-flash")    → GeminiLLM
        create_llm("Qwen/Qwen2.5-7B-Instruct")  → HFLLM (needs GPU)
    """
    model_lower = model_id.lower()

    # OpenAI models
    if any(model_lower.startswith(p) for p in ("gpt-", "o1", "o3", "o4")):
        logger.info("Using OpenAI backend: %s", model_id)
        return OpenAILLM(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    # Gemini models
    if "gemini" in model_lower:
        logger.info("Using Gemini backend: %s", model_id)
        return GeminiLLM(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    # HuggingFace models (anything with "/" like "Qwen/Qwen2.5-7B-Instruct")
    if "/" in model_id:
        logger.info("Using HuggingFace backend: %s", model_id)
        from .hf_llm import HFLLM
        return HFLLM(
            model_id=model_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    raise ValueError(
        f"Cannot determine backend for model '{model_id}'. "
        f"Use 'gpt-*' for OpenAI, 'gemini-*' for Gemini, "
        f"or 'org/model' for HuggingFace."
    )
