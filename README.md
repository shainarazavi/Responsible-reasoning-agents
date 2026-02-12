# Responsible Agentic Reasoning and AI Agents: A Critical Survey

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Research%20Scaffold-blue)
![AI](https://img.shields.io/badge/Focus-Responsible%20AI-green)
[![Preprint](https://img.shields.io/badge/Preprint-TechRxiv-blue)](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175735299.97215847)

**Authors:** Shaina Raza*, Ranjan Sapkota*, Manoj Karkee, Christos Emmanouilidis  
**Affiliations:** Vector Institute for Artificial Intelligence; Cornell University; University of Groningen  
**Equal contribution:** Shaina Raza and Ranjan Sapkota
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

## Citation

If you use this work, please cite:

``` bibtex
@article{raza2025responsible,
  author  = {Raza, Shaina and Sapkota, Ranjan and Karkee, Manoj and Emmanouilidis, Christos},
  title   = {Responsible Agentic Reasoning and AI Agents: A Critical Survey},
  journal = {TechRxiv},
  year    = {2025},
  month   = sep,
  day     = {8},
  doi     = {10.36227/techrxiv.175735299.97215847/v1},
  note    = {Preprint}
}
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

## Contact

Shaina Raza, PhD (shaina.raza@torontomu.ca)
Toronto, Canada
