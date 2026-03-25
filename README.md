# Responsible Agentic Reasoning and AI Agents: A Critical Survey

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper.pdf)
Shaina Raza*, Ranjan Sapkota*, Manoj Karkee, Christos Emmanouilidis  
Vector Institute · Cornell University · University of Groningen  

---------------

## Overview

### What is R²A²?

Responsible Reasoning AI Agents (R²A²) are large language model--powered
agents that perform multi-step reasoning while embedding
responsibility-aware controls throughout the decision trajectory.

Unlike traditional evaluation approaches that assess only the final
output, R²A² emphasizes:

-   Trace-level transparency\
-   Bias and fairness monitoring\
-   Privacy safeguards\
-   Robustness validation\
-   Structured audit logging

This framework supports safer deployment of agentic systems in
high-stakes domains such as healthcare, finance, governance, and
infrastructure.

------------------------------------------------------------------------

## Why This Work Matters

Recent advances (2024--2026) in reasoning models and autonomous agentic
systems enable:

-   Multi-step planning\
-   Tool use\
-   Persistent memory\
-   Browser and API interaction

However, risks now emerge across the entire reasoning trajectory, not
just the final answer.

This work introduces a survey-driven framework advocating:

-   Continuous trajectory auditing\
-   Policy-aware execution\
-   Human-in-the-loop oversight\
-   Standardized responsible-agent metrics

------------------------------------------------------------------------

## Repository Contents

This repository includes a lightweight research scaffold demonstrating
an R²A²-style agent pipeline:

-   Planner--executor reasoning loop\
-   Tool abstraction layer\
-   Memory module\
-   Policy checks\
-   Heuristic bias/privacy filtering\
-   Structured audit logging\
-   R²A² metric computation utilities

This is intended for research experimentation and benchmarking.


------------------------------------------------------------------------

## Getting Started

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Run Example

``` bash
python scripts/examples/examples.py
```



------------------------------------------------------------------------

## License

This project is released under the MIT License. See `LICENSE` for
details.

------------------------------------------------------------------------

## Contributing

Issues and pull requests are welcome.\
Please ensure contributions align with responsible AI best practices.

------------------------------------------------------------------------

## Key studies 

### Evaluation & Benchmarks
- **Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation** — Kapoor et al. (2025). https://arxiv.org/abs/2510.11977  *(bib: `kapoor2025holisticagentleaderboardmissing`)*
- **BEARCUBS: A benchmark for computer-using web agents** — Song et al. (2025). https://arxiv.org/abs/2503.07919  *(bib: `BEARCUBS2025`)*
- **GAIA: a benchmark for General AI Assistants** — Mialon et al. (2023). https://arxiv.org/abs/2311.12983  *(bib: `mialon2023gaiabenchmarkgeneralai`)*
- **Holistic Evaluation of Language Models (HELM)** — Liang et al. (2023). https://openreview.net/forum?id=iO4LZibEqW  *(bib: `liang2023holistic`)*
- **MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models** — Zhang et al. (2024). http://arxiv.org/abs/2406.07057  *(bib: `zhang_multitrust_2024`)*
- **Putnam-AXIOM: A Functional and Static Benchmark for Higher Level Mathematical Reasoning** — Gulati et al. (2024). https://openreview.net/pdf?id=YXnwlZe0yf  *(bib: `putnam_axiom2024`)*
- **SWE-bench (analysis entry)** — Martinez et al. (2025). https://arxiv.org/abs/2505.04457  *(bib: `martinez2025dissecting`)*
- **MLAgentBench: Evaluating language agents on machine learning experimentation** — Huang et al. (2023). *(no URL in bib)* *(bib: `huang2023mlagentbench`)*

---

### Agent Frameworks & Orchestration
- **AutoGen v0.4: Reimagining the Foundation of Agentic AI** — Microsoft Research (2025). https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/  *(bib: `AutoGen04_2025`)*
- **The AI Agent Index** — Casper et al. (2025). https://arxiv.org/abs/2502.01635  *(bib: `casper2025aiagentindex`)*
- **Watson: A Cognitive Observability Framework for LLM-Powered Agents** — Rombaut et al. (2025). http://arxiv.org/abs/2411.03455  *(bib: `Rombaut2025`)*
- **TRiSM for Agentic AI: Trust, Risk, and Security Management in Agentic Multi-Agent Systems** — Raza et al. (2025). https://arxiv.org/abs/2506.04133  *(bib: `raza2025trismagenticaireview`)*
- **LangChain** — (tooling). https://www.langchain.com/  *(bib: `langchain`)*
- **CrewAI: Orchestrating Role-Playing, Autonomous AI Agents** — (2023). https://github.com/crewAIInc/crewAI  *(bib: `crewAI`)*
- **AutoGPT** — (tooling). https://github.com/Significant-Gravitas/AutoGPT  *(bib: `autogpt`)*

---

### Reasoning & Prompting
- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al. (2023). https://arxiv.org/abs/2210.03629  *(bib: `yao2023reactsynergizingreasoningacting`)*
- **Self-Consistency improves Chain-of-Thought Reasoning** — Wang et al. (2022). https://arxiv.org/abs/2203.11171  *(bib: `wang2022self`)*
- **Tree of Thoughts: Deliberate Problem Solving with LLMs** — Yao et al. (2023). https://arxiv.org/abs/2305.10601  *(bib: `yao2023tree`)*
- **Plan-and-Solve Prompting** — Wang et al. (2023). https://arxiv.org/abs/2305.04091  *(bib: `wang2023plan`)*
- **Claude 3.7 Sonnet (Hybrid Reasoning) Announcement/System Card** — Anthropic (2025). https://www.anthropic.com/news/claude-3-7-sonnet  *(bib: `anthropic2025_claude37_sonnet`)*
- **Alibaba Cloud QwQ-32B blog (compact reasoning model)** — Alibaba (2025). https://www.alibabacloud.com/blog/alibaba-cloud-unveils-qwq-32b-a-compact-reasoning-model-with-cutting-edge-performance  *(bib: `alibaba2025_qwq32b`)*
- **Reflexion: Language agents with verbal reinforcement learning** — Shinn et al. (2023). *(no URL in bib)* *(bib: `shinn2023reflexion`)*

---

### Retrieval & RAG
- **Retrieval-Augmented Generation (RAG)** — Lewis et al. (2020). https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html  *(bib: `lewis2020rag`)*
- **Dense Passage Retrieval (DPR)** — Karpukhin et al. (2020). https://aclanthology.org/2020.emnlp-main.550/  *(bib: `karpukhin2020dpr`)*
- **IBM Granite 3.3: …RAG LoRAs** — IBM (2025). https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras  *(bib: `ibm2025granite33`)*
- **Retrieval augmented generation and understanding in vision: A survey** — (2025). *(no URL in bib)*

---

### Safety, Auditing & Governance
- **EU AI Act (proposal text)** — European Union (2025). https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021PC0206  *(bib: `eu2025ai`)*
- **Auditing large language models: a three-layered approach** — MöKander et al. (2024). *(no URL in bib)* *(bib: `mokander2024auditing`)*
- **Towards a Responsible AI Metrics Catalogue** — Xia et al. (2024). https://arxiv.org/abs/2311.13158  *(bib: `xia2024responsibleaimetricscatalogue`)*
- **PROV-Overview (W3C Provenance)** — W3C (2013). https://www.w3.org/TR/prov-overview/  *(bib: `W3C2013PROVOverview`)*
- **GDPR Article 25: Data protection by design and by default** — EU (2016). https://gdpr-info.eu/art-25-gdpr/  *(bib: `gdpr25`)*
- **HIPAA Privacy Rule — 45 CFR Part 164** — U.S. HHS (2003). https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164  *(bib: `hipaa164`)*
- **Building guardrails for large language models** — (2024). *(no URL in bib)* *(bib: `guardrails2024`)*
- **Constitutional AI: Harmlessness from AI Feedback** — Anthropic (2022). https://arxiv.org/abs/2212.08073  *(bib: `anthropic2022constitutional`)*
- **Trust by Design: Dissecting IBM's Enterprise AI Governance Stack** — Gogia (2025). https://greyhoundresearch.com/trust-by-design-dissecting-ibms-enterprise-ai-governance-stack/  *(bib: `gogia2025trust`)*
- **Jailbreak survey (LLMs→MLLMs→Agents)** — (2025). *(no URL in bib)* *(bib: `jailbreak_survey_2025`)*

---

### Model Documentation
- **gpt-oss-120b & gpt-oss-20b Model Card** — OpenAI (2025). https://openai.com/index/gpt-oss-model-card/  *(bib: `openai2025_gpt_oss_model_card`)*
- **Gemini 2.5 Deep Think Model Card** — Google DeepMind (2025). https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf  *(bib: `google2025_gemini25_deepthink_modelcard`)*

------------------------------------------------------------------------

## Citation

If you use this work, please cite:

``` bibtex
 @article{Raza_2025,
title={Responsible Agentic Reasoning and AI Agents: A Critical Survey},
url={http://dx.doi.org/10.36227/techrxiv.175735299.97215847/v2},
DOI={10.36227/techrxiv.175735299.97215847/v2},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Raza, Shaina and Sapkota, Ranjan and Karkee, Manoj and Emmanouilidis, Christos},
year={2025},
month=nov }

```
------------------------------------------------------------------------

## Contact

Shaina Raza, PhD (shaina.raza@torontomu.ca)
Toronto, Canada
