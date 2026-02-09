# R2A2Agent (Reference Implementation)

This repository provides a **lightweight, educational reference** for an “R2A2-style” agent pipeline: a top-level orchestrator that combines a core LLM interface with a planner/executor loop, a scratchpad, tool calling (toy), memory, simple policy enforcement, bias/privacy checks (heuristic), and audit logging.

> **Status:** Demo / skeleton code — not production-ready.

---

## What this is

A small agent scaffold with these components:

- **CoreLLM**: abstract interface for any language model backend (`generate(prompt) -> str`)
- **BaseLLM**: placeholder LLM used by default (echo / arithmetic demo)
- **PlannerExecutor**: “plan-and-execute” loop that can detect *toy* tool calls from model output
- **ToolAPIProxy**: tool registry and dispatcher (includes a `calculator` demo tool)
- **MemoryStore**: in-memory key-value store for context retention
- **ReasoningSampler**: placeholder for self-consistency sampling (currently returns the first sample)
- **PolicyEngine**: simple output constraints (keyword-based)
- **BiasPrivacyFilter**: heuristic bias/privacy flagging
- **AuditLogger + MonitoringDashboard**: structured audit trail + console summary
- **RunResult**: JSON-serializable result object

---

## Quick start

### Requirements
- Python 3.9+ (recommended)

### Run the demo
```bash
python3 r2a2_architecture.py
```

This runs a few example prompts and prints:
- a console monitoring summary, and
- a JSON blob per run (question, chain-of-thought scratchpad, answer, flags, audit events).

---

## Where “inference” happens

Inference (the model call) happens in:

- `R2A2Agent.run(question)` → `PlannerExecutor.plan_and_execute(question)` → `ReasoningSampler.sample(model, prompt)` → `model(prompt)` → `CoreLLM.__call__` → `BaseLLM.generate(prompt)`

By default, the agent uses **BaseLLM** unless you pass a real model into `R2A2Agent(model=...)`.

---

## Model backends (what can be plugged into this agent)

This repository is **model-agnostic**. Any LLM can be used as long as you wrap it in a `CoreLLM` subclass that implements:

```python
def generate(self, prompt: str) -> str:
    ...
```

### Recommended model choices

**Local (Hugging Face / Transformers)**
- `Qwen/Qwen2.5-7B-Instruct` — strong instruction-following, great value
- `meta-llama/Llama-3.1-8B-Instruct` — robust general-purpose baseline
- `mistralai/Mistral-7B-Instruct-v0.3` — fast, lightweight
- `google/gemma-2-9b-it` — strong quality for its size

**Local serving**
- **vLLM** (GPU) or **llama.cpp / GGUF** (CPU/GPU) can be used by implementing a `CoreLLM` adapter that calls their HTTP/CLI interface.

**API-based**
- Hosted models (e.g., OpenAI or other providers) can be integrated by implementing a `CoreLLM` adapter that calls the provider API and returns text.

---

## Tool calling

Tools are only executed if the model outputs lines in this exact format:

```text
call:<tool_name>: <argument>
```

Example:
```text
call:calculator: 2+3
```

> Note: The default `BaseLLM` is a placeholder and may not emit `call:` lines. For real tool use, select an instruction-tuned model and prompt it to follow the `call:<tool>: <arg>` convention.

---

## Safety and limitations (read before using)

This is an educational scaffold, not a secure production agent.

- **Chain-of-thought exposure:** the scratchpad/audit trail can store and print intermediate reasoning. Many real deployments avoid returning internal reasoning traces.
- **`eval()` in calculator:** the demo calculator uses `eval()` with restricted builtins. Treat it as unsafe for untrusted input; replace with a real math parser in production.
- **Heuristic filters:** bias/privacy checks are keyword/regex-based and will produce false positives/negatives.
- **Simple policy engine:** keyword blocking is not robust governance.
- **Single-pass “planning”:** the executor parses one model output; it is not an iterative planner/reflection loop.


