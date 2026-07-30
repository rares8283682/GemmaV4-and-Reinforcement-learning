# AdsAnnotation-OpenWeight-LLM

**Validating Open-Weight LLM Measures of Language Functions in TV Advertising: A Multitrait-Multimethod Analysis**

**Authors:** Rares Olteanu & Christophe Benavent  
**Affiliation:** Université Paris Dauphine–PSL, ACSS (Applied Computational Social Science), Paris, France  

---

## 📌 Project Overview

This repository contains the computational framework, annotation pipelines, evaluation notebooks, and manuscript materials for validating open-weight Large Language Models (LLMs) as computational annotators in computational social science and advertising research.

The study evaluates how reliably open-weight LLMs measure **4 Jakobsonian Language Functions** across a 10-year corpus of French TV automotive advertising:

1. **Informativeness** (*Referential function*): Product-relevant facts, technical attributes, pricing, range, and operational clarity.
2. **Expressiveness** (*Emotive function*): Affective force, brand symbolism, prestige, lifestyle positioning, and emotional intensity.
3. **Phaticness** (*Phatic function*): Direct address, conversational proximity, and audience engagement/relationship maintenance.
4. **Creativeness / Poeticness** (*Poetic function*): Message form, metaphor, linguistic play, stylistic elaboration, and aesthetic construction.

---

## 📊 Dataset & Corpus

* **Corpus Size:** **8,938 French TV Automotive Advertisements** (broadcast between **2014 and 2024**).
* **Source:** Provided by **ARCOM** (Autorité de régulation de la communication audiovisuelle et numérique) archives via ACSS.
* **Powertrain Breakdown:**
  * 5,585 Conventional (ICE) Advertisements (62.5%)
  * 1,380 Hybrid Advertisements (15.4%)
  * 1,973 Electric (EV) Advertisements (22.1%)
* **Input Fields:** Each model receives 6 standardized fields from the ARCOM archive record: `Script` (spoken text), `Titre` (title), `Visuel` (visual description), `Signature` (tagline), `MotsClés` (keywords), and `Thème` (topical classification).

---

## 🛠️ Annotation & Inference Pipeline

* **Inference Engine:** **vLLM** with batched generation (`temperature=0`, greedy decoding, `top_p=1`).
* **Prompting Strategy:** Independent, dimension-specific prompts per language function to eliminate cross-trait cueing and anchoring.
* **Output Format:** Strict JSON schema output on a 1.00–5.00 continuous scale (two-decimal precision).

---

## 🔬 Model Panel & Model Families

The study benchmarked **9 open-weight model configurations across 4 major LLM families**, comparing architectural differences, model scale (8B to 72B parameters), and quantization regimes (BF16 vs. 4-bit AWQ):

| Model Family | Configuration | Size | Precision / Quantization | Analytical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Meta Llama** | Llama 3.1 8B Instruct | 8B | BF16 rerun | Small-model baseline & efficiency sensitivity case |
| | Llama 3.1 70B Instruct | 70B | BF16 | Large standard reference model |
| | Llama 3.3 70B Instruct | 70B | AWQ (4-bit) | Quantized comparison & version delta (3.1 vs 3.3) |
| **Alibaba Qwen** | Qwen 2.5 14B Instruct | 14B | BF16 rerun | Lower-cost Qwen scale comparison |
| | Qwen 2.5 32B Instruct | 32B | BF16 rerun | Mid-scale Qwen comparison |
| | Qwen 2.5 72B Instruct | 72B | BF16 | Large standard reference model |
| | Qwen 2.5 72B Instruct | 72B | AWQ (4-bit) | Controlled quantization comparison (72B BF16 vs AWQ) |
| **Mistral AI** | Mistral Small Instruct | 24B | BF16 | Independent architecture family comparison |
| **Google DeepMind** | Gemma 4 26B | 26B | BF16 (+ Regex Recovery) | Architecture case & parser-recovery sensitivity analysis |

---

## 📐 Validation Methodology (MTMM Framework)

* **Multitrait-Multimethod (MTMM) Matrix:** Treats the 4 language functions as *Traits* and the LLM configurations as *Methods* to evaluate convergent validity, discriminant validity, and method effects.
* **Agreement Diagnostics:** Calculated Lin’s Concordance Correlation Coefficient (CCC), Bland-Altman agreement limits, and Spearman rank correlations.
* **Factor Analytic Validation:** Exploratory Factor Analysis (EFA) and Confirmatory Factor Analysis (CFA) to verify dimensional structure.
* **Substantive Market Analysis:** Regression modeling comparing communication strategies across electric vs. conventional powertrains over a 10-year market transition.

---

## 📂 Repository Structure

```text
AdsAnnotation-OpenWeight-LLM/
├── README.md               # Main repository overview & project metadata
├── paper_work/             # LaTeX manuscript source, paper blueprint, & bibliography
│   └── manuscript/         # Full LaTeX files (main_final_preappendix_revision.tex)
├── analysis_notebooks/     # Jupyter notebooks for MTMM analysis, EFA/CFA, & model comparison
├── model one per model/    # Model-specific execution code & prompts
├── presentation/           # Slides and presentation assets
└── useless_if_you_want_... # Archived legacy notebooks and exploratory tools
```

---

## 📜 Citation & Acknowledgements

The authors acknowledge **ACSS** (Applied Computational Social Science) at Université Paris Dauphine–PSL for providing GPU compute infrastructure, and **ARCOM** for providing the French automotive advertising dataset.
