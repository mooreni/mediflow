# MediFlow

MediFlow is a medical document translation engine designed to help immigrants in Israel understand medical documents in their native language. It translates Hebrew medical documents (forms, prescriptions, referrals, summaries) into Russian using a multi-step Gemini pipeline: each document is split into labelled clinical sections, every section is translated by Gemini Flash, scored by a Gemini Pro MQM judge, and corrected by Gemini Pro if the quality falls below threshold.

---

## Pipeline Modes

MediFlow has two operating modes:

| Mode | Script | Database | Final MQM Eval |
|---|---|---|---|
| **Production** | `scripts/run.py` | `data/translations.db` | No |
| **Test** | `scripts/run_test.py` | `data/benchmark.db` | Yes |

**Production mode** translates documents and persists results. It does not run a final MQM evaluation pass — the midway judge inside each section task is the quality gate.

**Test mode** translates documents and runs a final MQM evaluation on every section. If a section was corrected by Pro, a fresh judge call scores the corrected output. If not corrected, the midway score is reused — no extra API call. Results are stored with quality scores and error breakdowns for benchmarking.

---

## How the Pipeline Works

```
Document (Hebrew)
    │
    ▼
[Splitter]  Gemini Flash segments the document into labelled clinical sections.
            Each section receives a context header (document type + patient context).
            Split results are cached in data/split_cache.json (test mode only).
    │
    ▼
[Per-section — single ThreadPoolExecutor, 8 workers, doc-first queue order]
    │
    ├─ Step 1: Flash Translate — Gemini Flash produces an initial Hebrew → Russian draft
    ├─ Step 2: Judge Eval     — Gemini Pro scores the draft using MQM error taxonomy
    └─ Step 3: Pro Correct    — if the midway score is below threshold, Gemini Pro
                                 rewrites the section using the judge's error feedback;
                                 otherwise the Flash draft is kept as-is
    │
    ▼
[Test mode only]
    └─ Final Eval — if section was corrected: fresh judge call on the corrected output
                    if not corrected: midway score and errors are reused directly
    │
    ▼
[Persistence]  Section and document results written to SQLite
               Production → data/translations.db
               Test       → data/benchmark.db
[Report]       benchmark_results.html generated via Plotly (test mode)
```

## MQM Scoring Model

Translations are scored using a penalty-based MQM (Multidimensional Quality Metrics) model.

- **5 error categories:** `accuracy`, `terminology`, `audience_appropriateness`, `linguistic_conventions`, `locale_conventions`
- **3 severity levels and their penalties:**
  - `critical` — 0.25
  - `major` — 0.05
  - `minor` — 0.01
- **Score formula:** `score = max(0, 1 − sum_of_penalties)`
- A section with no errors scores 1.0. The document-level score is computed by aggregating all section-level errors and applying the penalty formula once across the full error set.

---

## Project Structure

```
mediflow/
├── data/
│   ├── translations.db           # Production SQLite database
│   ├── benchmark.db              # Test/benchmark SQLite database
│   ├── split_cache.json          # Cached section splits (test mode; auto-built on first run)
│   └── informed_consent_forms/
│       └── text/
│           ├── he/               # Hebrew source documents
│           └── ru/               # Russian reference translations
├── scripts/
│   ├── run.py                    # Production pipeline entry point
│   ├── run_test.py               # Test/benchmark pipeline entry point
│   ├── build_split_cache.py      # Pre-build the split cache
│   └── extract_medical_pdfs.py   # One-time PDF → text extraction utility
├── src/app/
│   ├── clients/
│   │   └── vertex.py             # Thread-local Vertex AI client factory
│   ├── data/
│   │   └── loader.py             # load_documents() with optional type/id filters
│   ├── evaluation/
│   │   └── judge.py              # MQM judge: score_section(), score_document()
│   ├── translation/
│   │   ├── base.py               # SectionTranslationResult, TranslationResult dataclasses
│   │   ├── cost.py               # CostRecord, gemini_cost(), sum_costs()
│   │   ├── splitter.py           # split_document() via Gemini Flash
│   │   └── translator.py         # MedicalTranslator — the per-section pipeline
│   ├── db.py                     # BenchmarkDB — test SQLite persistence layer
│   ├── production_db.py          # ProductionDB — production SQLite persistence layer
│   └── visualize.py              # generate_report() — Plotly HTML report
├── tests/
├── benchmark_results.html        # Generated visualisation report
├── env.example
└── pyproject.toml
```

---

## Installation

**Prerequisites:** Python 3.11+, a Google Cloud project with Vertex AI enabled, and Application Default Credentials configured:

```bash
gcloud auth application-default login
```

**1. Clone and set up the environment**

```bash
git clone <repository-url>
cd mediflow
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

**2. Configure environment variables**

```bash
cp env.example .env
# Edit .env and set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `google-genai` | Vertex AI LLM calls |
| `google-cloud-aiplatform` | Vertex AI platform client |
| `plotly` | HTML report generation |
| `python-dotenv` | `.env` file loading |
| `pdfplumber` | PDF extraction (one-time preprocessing only) |
| `python-bidi` | Hebrew text rendering (preprocessing only) |

---

## CLI Usage

All commands are run from the project root.

```bash
# Production mode — translate all documents, persist to data/translations.db
venv/bin/python3 scripts/run.py

# Production mode — translate only one document type
venv/bin/python3 scripts/run.py --doc-type form        # form | summary | prescript | referral

# Production mode — translate a single document by ID
venv/bin/python3 scripts/run.py --doc-id Form_001

# Test/benchmark mode — translate + final MQM eval, persist to data/benchmark.db
venv/bin/python3 scripts/run_test.py

# Generate the HTML report from existing benchmark DB results (skips translation)
venv/bin/python3 scripts/run_test.py --visualize
```

The split cache (`data/split_cache.json`) is used by test mode and built automatically on first run. To pre-build it explicitly:

```bash
venv/bin/python3 scripts/build_split_cache.py
```

Documents already present in the database are skipped on subsequent runs.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Translation (draft) | Gemini Flash (`gemini-3-flash-preview`) via Vertex AI |
| Translation (correction) | Gemini Pro (`gemini-3.1-pro-preview`) via Vertex AI |
| Evaluation / Judge | Gemini Pro (`gemini-3.1-pro-preview`) via Vertex AI |
| Document splitting | Gemini Flash via Vertex AI |
| LLM client | `google-genai` |
| Persistence | SQLite via `sqlite3` stdlib |
| Concurrency | `ThreadPoolExecutor` (8 workers, single shared pool) |
| Reporting | Plotly (HTML) |
| Languages | Hebrew → Russian |
