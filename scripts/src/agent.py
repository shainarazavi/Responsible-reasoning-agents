# src/agent.py
"""
R2A2 Agent – Revised with a multi-turn ReAct loop.


  1. Multi-turn ReAct loop (Thought → Action → Observation → repeat)
  2. Full message history passed to the LLM each turn
  3. Configurable max_steps with graceful fallback
  4. Proper tool dispatch integrated into the loop
  5. Robust answer extraction with multiple fallback strategies

"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .llm import CoreLLM, BaseLLM
from .tools import ToolAPIProxy
from .policy import PolicyEngine, BiasPrivacyFilter
from .audit import AuditLogger, MonitoringDashboard, RunResult

logger = logging.getLogger(__name__)


# ── Markdown cleaning (fixes Gemini formatting issues) ────────────────
def _clean_markdown(text: str) -> str:
    """
    Strip markdown formatting that Gemini (and sometimes other models)
    wraps around ReAct-style outputs.

    Handles:
      - **bold** markers around keywords like Thought/Action/FINAL ANSWER
      - ```code fences``` (with optional language tag)
      - `inline code` backticks
      - Leading/trailing whitespace artifacts
    """
    # Remove bold markers: **text** → text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Remove code fences: ```python ... ``` → contents only
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    # Remove inline code backticks: `text` → text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _clean_answer(text: str) -> str:
    """
    Clean a final answer string by removing residual markdown,
    HTML tags, and other artifacts.
    """
    # Strip markdown
    text = _clean_markdown(text)
    # Remove any HTML tags (e.g. </code>, <br>, etc.)
    text = re.sub(r"<[^>]+>", "", text)
    # Remove leading/trailing quotes
    text = text.strip().strip("\"'").strip()
    # Take only the first line (in case of multi-line junk)
    text = text.split("\n")[0].strip()
    # Remove trailing punctuation that isn't part of the answer
    # (but keep periods that might be part of names/abbreviations)
    return text


# ── Regex helpers ─────────────────────────────────────────────────────
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:FINAL\s+ANSWER|Final\s+Answer)\s*:\s*(.+)",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"(?:^|\n)\s*Action\s*:\s*(\S+?)\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_RE = re.compile(
    r"(?:^|\n)\s*Thought\s*:\s*(.*?)(?=\n\s*(?:Action|FINAL ANSWER)|$)",
    re.IGNORECASE | re.DOTALL,
)


class ReActLoop:
    """
    Multi-turn ReAct executor.

    Each turn:
      1. Send full message history to the LLM
      2. Parse response for Thought / Action / FINAL ANSWER
      3. If Action → execute tool, append Observation, loop
      4. If FINAL ANSWER → return
      5. If max_steps reached → extract best guess from last response
    """

    def __init__(
        self,
        model: CoreLLM,
        tools: ToolAPIProxy,
        max_steps: int = 10,
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self, question: str, file_info: str = ""
    ) -> Tuple[str, str, List[Dict[str, Any]], int, List[str]]:
        """
        Execute the ReAct loop.

        Returns:
            (chain_of_thought, final_answer, audit_trail, n_steps, tools_used)
        """
        # Build initial user message
        user_content = question
        if file_info:
            user_content += f"\n\nAttached file: {file_info}"

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": user_content},
        ]

        audit_trail: List[Dict[str, Any]] = []
        cot_parts: List[str] = []
        tools_used: List[str] = []

        for step in range(1, self.max_steps + 1):
            now = datetime.now(timezone.utc).isoformat()
            logger.info("Step %d/%d", step, self.max_steps)

            # ── Generate ──────────────────────────────────────────────
            if hasattr(self.model, "chat"):
                response = self.model.chat(messages)
            else:
                # Fallback for BaseLLM / simple models
                flat_prompt = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )
                response = self.model(flat_prompt)

            logger.debug("Model response:\n%s", response[:500])
            audit_trail.append({
                "type": "llm_response",
                "step": step,
                "text": response,
                "timestamp": now,
            })

            # ── Clean markdown before parsing ─────────────────────────
            # Keep original response for message history (model expects
            # to see its own formatting), but parse on cleaned version.
            cleaned = _clean_markdown(response)

            # ── Check for FINAL ANSWER ────────────────────────────────
            fa_match = _FINAL_ANSWER_RE.search(cleaned)
            if fa_match:
                answer = _clean_answer(fa_match.group(1))
                thought_match = _THOUGHT_RE.search(cleaned)
                if thought_match:
                    cot_parts.append(f"Step {step}: {thought_match.group(1).strip()}")
                cot_parts.append(f"Step {step}: FINAL ANSWER: {answer}")
                return "\n".join(cot_parts), answer, audit_trail, step, tools_used

            # ── Check for Action ──────────────────────────────────────
            action_match = _ACTION_RE.search(cleaned)
            thought_match = _THOUGHT_RE.search(cleaned)

            if thought_match:
                cot_parts.append(f"Step {step}: {thought_match.group(1).strip()}")

            if action_match:
                tool_name = action_match.group(1).strip().lower()
                tool_arg = action_match.group(2).strip()
                # Clean the argument (take only first line for most tools)
                if tool_name not in ("python",):
                    tool_arg = tool_arg.split("\n")[0].strip()

                logger.info("Calling tool: %s(%s)", tool_name, tool_arg[:100])

                success, result, error = self.tools.call(tool_name, tool_arg)
                tools_used.append(tool_name)

                observation = result if success else f"Error: {error}"
                audit_trail.append({
                    "type": "tool_call",
                    "step": step,
                    "tool": tool_name,
                    "argument": tool_arg[:500],
                    "success": success,
                    "result": observation[:1000],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                cot_parts.append(f"Step {step}: Action: {tool_name}: {tool_arg[:100]}")
                cot_parts.append(f"Step {step}: Observation: {observation[:200]}")

                # Append assistant response + observation to message history
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n"
                               f"Continue reasoning. If you now have enough information, "
                               f"write 'FINAL ANSWER: <answer>'. "
                               f"Otherwise, take another Action.",
                })
            else:
                # Model didn't produce an Action or FINAL ANSWER — nudge it
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "You did not produce a valid Action or FINAL ANSWER. "
                               "Please respond with either:\n"
                               "  Action: <tool_name>: <argument>\n"
                               "or:\n"
                               "  FINAL ANSWER: <your answer>\n"
                               "Do NOT use markdown formatting like **bold** or ```code blocks```.\n"
                               "Try again.",
                })
                cot_parts.append(f"Step {step}: (no action/answer — reprompted)")

        # ── Max steps exhausted: try to extract something ─────────────
        logger.warning("Max steps (%d) exhausted, extracting fallback answer",
                       self.max_steps)
        fallback = self._extract_fallback(messages, response)
        cot_parts.append(f"Fallback: {fallback}")
        return "\n".join(cot_parts), fallback, audit_trail, self.max_steps, tools_used

    def _extract_fallback(self, messages: List[Dict], last_response: str) -> str:
        """Try to extract a reasonable answer from the last model output."""
        # Clean markdown first
        last_response = _clean_markdown(last_response)

        # Look for any "answer is" pattern
        patterns = [
            r"(?:the answer is|answer:)\s*(.+)",
            r"(?:the result is|result:)\s*(.+)",
            r"(?:therefore|thus|so)[,:]?\s*(.+)",
            # Gemini sometimes writes "The final answer is ..."
            r"(?:final answer is)\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, last_response, re.IGNORECASE)
            if m:
                return _clean_answer(m.group(1))

        # Last resort: last non-empty line, skipping junk
        _SKIP_PREFIXES = (
            "thought:", "action:", "observation:",
            "<", "```", "---", "===", "***",
        )
        for line in reversed(last_response.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith(_SKIP_PREFIXES):
                continue
            # Skip lines that are purely HTML tags or markdown artifacts
            if re.match(r"^[<>`*_\-=]+$", line):
                continue
            return _clean_answer(line)

        # Absolute last resort: scan ALL previous responses in message history
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                text = _clean_markdown(msg["content"])
                fa = _FINAL_ANSWER_RE.search(text)
                if fa:
                    return _clean_answer(fa.group(1))
                for pat in patterns:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        return _clean_answer(m.group(1))

        return "unknown"


class R2A2Agent:
    """Top-level orchestrator implementing an R2A2-style ReAct agent."""

    def __init__(
        self,
        model: Optional[CoreLLM] = None,
        max_steps: int = 10,
        tools: Optional[ToolAPIProxy] = None,
    ):
        self.model = model or BaseLLM()
        self.tools = tools or ToolAPIProxy()
        self.react = ReActLoop(self.model, self.tools, max_steps=max_steps)
        self.policy = PolicyEngine()
        self.filter = BiasPrivacyFilter()
        self.logger = AuditLogger()
        self.dashboard = MonitoringDashboard()

    def run(self, question: str, file_path: str = "") -> RunResult:
        self.logger.clear()
        logger.info("Received question: %s", question[:200])

        bias_in, privacy_in = self.filter.check(question)
        self.logger.record("input", {
            "question": question,
            "bias": bias_in,
            "privacy": privacy_in,
        })

        # ── Execute ReAct loop ────────────────────────────────────────
        cot, answer, audit_trail, n_steps, tools_used = self.react.run(
            question, file_info=file_path
        )

        # ── Post-processing ───────────────────────────────────────────
        bias_ans, privacy_ans = self.filter.check(answer)
        enforced_answer = self.policy.enforce(answer)

        self.logger.record("output", {
            "answer": enforced_answer,
            "bias": bias_ans,
            "privacy": privacy_ans,
            "n_steps": n_steps,
            "tools_used": tools_used,
        })

        self.dashboard.display({
            "input_bias": bias_in,
            "input_privacy": privacy_in,
            "output_bias": bias_ans,
            "output_privacy": privacy_ans,
            "n_steps": n_steps,
            "tools_used": tools_used,
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
            n_steps=n_steps,
            tools_used=tools_used,
        )
    