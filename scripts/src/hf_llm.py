# src/hf_llm.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .llm import CoreLLM

logger = logging.getLogger(__name__)


# ── Early-stop: halt generation once "FINAL ANSWER:" or "Action:" appears ──
class StopOnMarker(StoppingCriteria):
    """Stop generating once a marker appears and a short tail is captured."""

    def __init__(self, tokenizer, prompt_length: int, markers: List[str],
                 tail_tokens: int = 60):
        self.stop_seqs = []
        for m in markers:
            ids = tokenizer.encode(m, add_special_tokens=False)
            if ids:
                self.stop_seqs.append(ids)
        self.prompt_length = prompt_length
        self.tail_tokens = tail_tokens
        self._marker_hit = False
        self._tokens_after = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        new_ids = input_ids[0][self.prompt_length:].tolist()
        if len(new_ids) < 4:
            return False

        if not self._marker_hit:
            for seq in self.stop_seqs:
                seq_len = len(seq)
                for i in range(len(new_ids) - seq_len + 1):
                    if new_ids[i: i + seq_len] == seq:
                        self._marker_hit = True
                        self._tokens_after = len(new_ids) - (i + seq_len)
                        break
                if self._marker_hit:
                    break
        else:
            self._tokens_after += 1

        return self._marker_hit and self._tokens_after >= self.tail_tokens


# ── Default system prompt for ReAct-style agent ────────────────────────
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


class HFLLM(CoreLLM):
    """
    Hugging Face Transformers backend with multi-turn chat support.

    Key changes from v1:
      - Accepts a message history (list of dicts) for multi-turn ReAct
      - Improved system prompt with tool descriptions
      - Stops on both "FINAL ANSWER:" and "Action:" markers
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        system_prompt: str | None = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or DEFAULT_REACT_SYSTEM_PROMPT

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        load_kwargs: Dict = {"device_map": "auto"}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            try:
                load_kwargs["attn_implementation"] = "sdpa"
                logger.info("Using SDPA attention")
            except Exception:
                logger.info("SDPA not available, using default attention")
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # ── Single-string interface (CoreLLM compatibility) ────────────────
    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages)

    # ── Multi-turn chat interface (used by ReAct loop) ─────────────────
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response given a full message history."""

        # Ensure system prompt is first
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Stop on "FINAL ANSWER:" (tail=60) or "Action:" (tail=80)
        stopper = StopOnMarker(
            self.tokenizer, input_length,
            markers=["\nFINAL ANSWER:", "\nFinal Answer:", "\nAction:"],
            tail_tokens=80,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)

        new_tokens = out[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        logger.debug("Generated %d new tokens (prompt=%d)", len(new_tokens), input_length)
        return response
