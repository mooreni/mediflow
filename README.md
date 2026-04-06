# MediFlow

**MediFlow** is a medical document translation engine POC designed to help immigrants in Israel understand medical documents in their native language. It is a multi-agent AI system built with [DSPy](https://github.com/stanfordnlp/dspy) and Google Gemini models that translates Hebrew medical documents into Russian with high accuracy on critical medical terminology.

The project is structured around a systematic **benchmarking pipeline** that compares six translation strategies — from a simple baseline to a DSPy-bootstrapped few-shot architecture — and evaluates each one using an LLM-as-a-judge scoring system.

---

## Features

- **6 Translation Scenarios** — from a Google Translate baseline to a DSPy-bootstrapped few-shot pipeline
- **LLM-as-a-Judge Evaluation** — Gemini Pro scores each translation on critical terms, completeness, and semantic accuracy
- **Persistent Results** — all runs stored in a local SQLite database with full cost and latency tracking
- **Interactive HTML Report** — Plotly-based visualisation of benchmark results across all scenarios
- **Concurrent Processing** — documents evaluated in parallel with a configurable thread pool

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Framework | [DSPy](https://github.com/stanfordnlp/dspy) |
| LLM Models | Google Gemini Pro, Gemini Flash (via Vertex AI) |
| Translation Baseline | Google Cloud Translation API v2 |
| Evaluation | Google Vertex AI (Gemini Pro LLM-as-judge) |
| Database | SQLite (via `sqlite3` stdlib) |
| Visualisation | Plotly |
| Dependency Management | `pyproject.toml` / setuptools |

---

## Project Structure

```
mediflow/
├── data/
│   └── informed_consent_forms/
│       ├── pdfs/
│       │   ├── he/          # Hebrew source PDFs
│       │   └── ru/          # Russian reference PDFs
│       └── text/
│           ├── he/          # Extracted Hebrew plain-text
│           └── ru/          # Extracted Russian reference translations
├── scripts/
│   ├── run_benchmark.py     # Main CLI entry point
│   └── test_*.py            # Scenario and integration test scripts
├── src/
│   ├── app/                 # PDF extraction utilities
│   ├── benchmark/
│   │   ├── cost.py          # Per-scenario cost calculation
│   │   ├── dataset.py       # Dataset loading and train/eval split
│   │   ├── db.py            # SQLite persistence layer
│   │   ├── runner.py        # Concurrent scenario runner
│   │   ├── visualize.py     # Plotly report generation
│   │   └── scenarios/
│   │       ├── base.py
│   │       ├── s1_google_translate.py
│   │       ├── s2_gemini_flash_zeroshot.py
│   │       ├── s3_gemini_pro_dspy_predict.py
│   │       ├── s4_gemini_pro_bootstrap.py
│   │       ├── s5_gemini_flash_dspy_predict.py
│   │       └── s6_gemini_flash_cot.py
│   └── evaluation/
│       ├── llm_judge.py     # LLM-as-judge scoring logic
│       └── vertex_client.py # Vertex AI client wrapper
├── env.example
└── pyproject.toml
```

---

## Benchmark Scenarios

| ID | Scenario | Description |
|---|---|---|
| S1 | Google Translate | Cloud Translation API v2 — cost/quality baseline |
| S2 | Gemini Flash Zero-shot | Direct Gemini Flash call, no DSPy, no optimisation |
| S3 | Gemini Pro DSPy Predict | DSPy `Predict` module with structured signature (Gemini Pro) |
| S4 | Gemini Pro Bootstrap | DSPy `BootstrapFewShot` optimiser with few-shot examples (Gemini Pro) |
| S5 | Gemini Flash DSPy Predict | DSPy `Predict` module with structured signature (Gemini Flash) |
| S6 | Gemini Flash ChainOfThought | DSPy `ChainOfThought` module, zero-shot (Gemini Flash) |

Each scenario is evaluated across **40 documents** (10 per document type × 4 types). Scores are computed by a Gemini Pro judge on four dimensions: **critical terms**, **completeness**, **semantic accuracy**, and **overall quality** (0–100 scale).

---

## Installation

**Prerequisites:** Python 3.11+, a Google Cloud project with Vertex AI and Cloud Translation APIs enabled, and authenticated Application Default Credentials (`gcloud auth application-default login`).

**1. Clone the repository**

```bash
git clone <repository-url>
cd mediflow
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install the package and dependencies**

```bash
pip install -e .
pip install dspy-ai google-cloud-translate google-cloud-aiplatform plotly python-dotenv
```

**4. Configure environment variables**

Copy `env.example` to `.env` and fill in your Google Cloud project details:

```bash
cp env.example .env
```

---

## Usage

All commands are run from the project root with the virtual environment active.

**Run the full benchmark (all 5 scenarios)**

```bash
python scripts/run_benchmark.py
```

**Run a single scenario**

```bash
python scripts/run_benchmark.py --scenario s1_google_translate
# Available: s1_google_translate | s2_gemini_flash_zeroshot | s3_gemini_pro_dspy_predict
#            s4_gemini_pro_bootstrap | s5_gemini_flash_dspy_predict | s6_gemini_flash_cot
```

**Generate the HTML visualisation report from existing results**

```bash
python scripts/run_benchmark.py --visualize
```

Results are persisted to `benchmark.db` (SQLite) in the project root. Scenarios already present in the database are automatically skipped on subsequent runs. The visualisation report is written to `benchmark_results.html`.
