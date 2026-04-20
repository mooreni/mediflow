# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MediFlow** is a medical document translation engine POC designed to help immigrants in Israel understand medical documents in their native language. It translates Hebrew medical documents (forms, prescriptions, referrals, summaries) into Russian using a multi-step Gemini pipeline with MQM-based quality evaluation.

## Architecture

The core per-section pipeline lives in `MedicalTranslator` (`src/app/translation/translator.py`). There are two operating modes:

### Pipeline Modes

| Mode | Script | Database |
|---|---|---|
| Production | `scripts/run.py` | `data/translations.db` |
| Test | `scripts/run_test.py` | `data/benchmark.db` |

### 1. Document Splitting (`src/app/translation/splitter.py`)
- Gemini Flash segments the document into labelled clinical sections (e.g., "Patient Consent", "Procedure Description")
- Each section is assigned a context header describing document type and patient context
- Split results are cached in `data/split_cache.json` (test mode only; production does not cache)

### 2. Per-Section Translation Pipeline
All sections run concurrently through a single `ThreadPoolExecutor(max_workers=8)`. Sections are submitted doc-first (all sections of Doc1 before any section of Doc2):

1. **Flash Translate** — Gemini Flash produces an initial Hebrew → Russian translation
2. **Judge Eval** — Gemini Pro scores the translation using MQM (5 error categories × 3 severity levels); penalty weights: critical 0.25 / major 0.05 / minor 0.01; score = max(0, 1 − total penalties)
3. **Pro Correct** — if the midway score is below threshold, Gemini Pro rewrites the section using the judge's error feedback; otherwise the Flash output is kept

### 3. Test Mode — Final Evaluation
`scripts/run_test.py` runs `translate_and_evaluate_section` per section inside the shared pool:
- If `was_corrected == True`: calls the judge again on the corrected output
- If `was_corrected == False`: reuses `midway_score` / `midway_errors` directly — no extra API call

### 4. Persistence
- **Production** — `src/app/production_db.py` (`ProductionDB`): stores runs, results, section text. No quality scores.
- **Test** — `src/app/db.py` (`BenchmarkDB`): stores runs, results, section results with quality scores and MQM error breakdowns.
- `src/app/visualize.py` generates an HTML report with 6 charts from `data/benchmark.db`.

## Tech Stack
| Component | Technology |
|---|---|
| Translation (draft) | Gemini 3.0 Flash |
| Translation (correction) | Gemini 3.1 Pro |
| Evaluation / Judge | Gemini 3.1 Pro |
| Document splitting | Gemini 3.0 Flash |
| LLM client | `google-genai` (Vertex AI) |
| Persistence | SQLite via `sqlite3` |
| Concurrency | `ThreadPoolExecutor` (8 workers, single shared pool) |
| Reporting | Plotly (HTML) |
| Languages | Hebrew → Russian |

## Rules and Operating Guide

RULES.md is this project's **primary operating guide**. Read it before writing any code and apply it to every edit. When declining or flagging something, cite the specific rule (e.g., "RULES.md §Error Handling: bare `except` is not allowed").

## Learning from Past Mistakes

Before writing or modifying any `.py` file, read `.claude/review_log.md`. It contains a history of rule violations caught in this project. Avoid repeating any pattern already documented there.

When you catch a mistake yourself — in existing code you are reading, in your own draft before finalizing, or in a bug you are fixing — log it by following the instructions at the top of `.claude/review_log.md`.

## Mandatory Post-Write Code Review

After every response in which you write or modify a `.py` file, YOU **MUST** ASK THE USER if to invoke the `code-reviewer` sub-agent on that file before finishing the code:

```
Agent tool: subagent_type="code-reviewer", prompt="Review and fix: <absolute file path>"
```
- If the reviewer fixes issues, you may note it in one line (e.g., "Reviewer cleaned up error handling."). If nothing was fixed, say nothing.
- If multiple files were written, ask the user if to invoke the reviewer on each one.
