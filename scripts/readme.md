
# R2A2 Agent — ReAct + Tool-Using Scaffold

| Component | Current |
|----------|---------|
| **Agent loop** | Multi-turn ReAct loop (up to 10 steps) |
| **Tools** | Calculator, web search, web fetch, Python exec, file reader |
| **Web search** | DuckDuckGo (free) / SerpAPI / Tavily |
| **Code execution** | Sandboxed Python via subprocess |
| **File handling** | PDF, Excel, CSV, DOCX, PPTX, images, audio |
| **System prompt** | Full ReAct instructions with tool descriptions |
| **Chat format** | Multi-turn message history with observations |
| **Answer matching** | Normalised + numeric + containment matching |
| **GAIA files** | Downloaded via `snapshot_download`, path passed to agent |
| **Output** | JSONL + level, n_steps, tools_used, elapsed_s |



## Architecture

```
User Question + File
       │
       ▼
┌─────────────────────┐
│    R2A2Agent.run()   │
│  ┌───────────────┐   │
│  │  ReAct Loop   │   │   ← up to max_steps iterations
│  │               │   │
│  │  1. LLM chat  │──►│──► Thought: I need to search for X
│  │  2. Parse     │   │    Action: web_search: X
│  │  3. Tool call │──►│──► Observation: [search results]
│  │  4. Append    │   │    (added to message history)
│  │  5. Repeat    │   │
│  │  ...          │   │
│  │  FINAL ANSWER │──►│──► answer extracted
│  └───────────────┘   │
│                       │
│  Policy + Bias check  │
└─────────────────────┘
       │
       ▼
   RunResult (JSONL)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Smoke test (no GPU needed)
python examples/demo.py

# 3. Smoke test with Qwen (needs GPU)
python examples/demo.py --model qwen

# 4. Full GAIA run (SLURM)
sbatch run_qwen-agent.sh
```

## Tips for Higher Accuracy

1. **Search API key**: DuckDuckGo is rate-limited; get a free SerpAPI or Tavily key
   ```bash
   export SERPAPI_KEY="your-key"
   # or
   export TAVILY_API_KEY="your-key"
   ```

2. **Bigger model**: Swap Qwen-7B for Qwen-72B or an API model:
   ```python
   model = HFLLM("Qwen/Qwen2.5-72B-Instruct", max_new_tokens=1024)
   ```

3. **More steps**: Increase `MAX_STEPS` in examples.py (10 → 15 or 20)

4. **Audio transcription**: Install whisper for GAIA's .mp3 questions:
   ```bash
   pip install openai-whisper
   ```

5. **OCR**: Install tesseract for image-based questions:
   ```bash
   apt install tesseract-ocr && pip install pytesseract
   ```

## File Structure

```
agents-revised/
├── src/
│   ├── __init__.py
│   ├── llm.py          # Abstract LLM interface
│   ├── hf_llm.py       # Qwen HF backend + ReAct system prompt
│   ├── agent.py         # ReAct loop + R2A2Agent orchestrator
│   ├── tools.py         # Calculator, web search, Python, file reader
│   ├── policy.py        # Safety policy engine
│   └── audit.py         # Audit logging + RunResult
├── examples/
│   ├── demo.py          # Quick smoke test
│   └── examples.py      # Full GAIA evaluation runner
├── outputs/             # JSONL results written here
├── requirements.txt
├── run_qwen-agent.sh    # SLURM script
└── README.md
```
