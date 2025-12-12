# Online Appendix — Bias Ahead: Sensitive Prompts as Early Warnings for Fairness in Large Language Models

This repository contains the complete online appendix for the study
**“Prompt Sensitiveness in Large Language Models”**, including datasets, scripts, experimental configurations, and reproducibility material for RQ1 and RQ2.

The repository is organized into three main components:

1. **Dataset construction** — all sources used to build the SENSY dataset.
2. **RQ1 evaluation** — scripts and raw LLM outputs used to assess adequacy on sensitive prompts.
3. **RQ2 classification pipeline** — code, data, and trained models for automated sensitivity prediction.

---

## 📁 Repository Structure
```text
SENSY/
│
├── dataset/
│   ├── prompts_chatbot_arena.json     # Human-generated prompts annotated as sensitive/non-sensitive
│   ├── prompts_chatgpt.json           # Synthetically generated prompts (ChatGPT)
│   └── SENSY.json                     # Final merged & labeled dataset used in RQ2
│   └── dataset_analysis.py                     # Python script to give details about the dataset SENSY
│
├── RQ1/
│   ├── sample.json                    # Sample of 500 sensitive prompts used in RQ1
│   ├── llama_response.json            # Raw responses from LLaMA (3 runs per prompt)
│   ├── qwen_response.json             # Raw responses from Qwen (3 runs per prompt)
│   ├── deepseek_response.json         # Raw responses from DeepSeek (3 runs per prompt)
│   └── rq1_llm_query.py               # Script used to query local LLMs through LM Studio API
│
├── RQ2/
│   ├── data/                          # Training/test sets automatically derived during experiments
│   ├── models/                        # Saved Random Forest models (optional depending on size)
│   ├── preprocessing/                 # Tokenization, feature extraction, and cleaning utilities
│   ├── samples/                       # Example predictions and error analysis logs
│   ├── common_functions.py            # Shared utilities (loading, metrics, plotting)
│   ├── extract_single.py              # Prompt features extraction
│   ├── predict_sensitive.py           # Module to test the trained classifier
│   └── main.py                        # Full training and evaluation pipeline for the SENSY classifier
│
└── README.md
```
---

## Datasets

### 1. Synthetic prompts (`prompts_chatgpt.json`)

Generated using ChatGPT following the sensitivity definition adopted in the study.Each prompt is annotated as:

- 1 — sensitive
- 0 — non-sensitive

### 2. Chatbot Arena prompts (`prompts_chatbot_arena.json`)

Sampled from the LMSYS *Chatbot Arena Conversations* dataset.
Only first-turn user prompts were retained.
All items were manually annotated using a coding-by-consensus process.

### 3. Final unified dataset (`SENSY.json`)

Used in **RQ2** to train and test the SensY classifier.Contains:

- the sensitivity label
- the domain category
- preprocessing metadata

---

## RQ1 — Evaluating LLM Adequacy on Sensitive Prompts

This study evaluates whether sensitive prompts elicit inadequate or problematic responses from local LLMs.

### Included Files

- `sample.json` — Balanced set of 500 sensitive prompts used for evaluation.
- `{model}_response.json` — The 4,500 total responses (3 models × 500 prompts × 3 runs).
- `rq1_llm_query.py` — Script querying local models via LM Studio REST API.

---

---

## RQ2 — Automatic Prediction of Prompt Sensitiveness

This folder contains the full pipeline for the *SensY* classifier, including preprocessing, feature extraction, model training, and error analysis.

### Running the classifier

Train the model:

```bash
python main.py
```

Use the model:

```bash
python predict_sensitive.py
```

---
