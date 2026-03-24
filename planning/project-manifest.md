<!-- SPLIT_MANIFEST
01-evaluation
02-agent-orchestration
03-xai-transparency
04-ocr-ui
END_MANIFEST -->

# Project Manifest — MediFlow Translation Engine

## Overview

MediFlow is a Hebrew → Russian medical document translation pipeline built with DSPy and Google Gemini. This is a greenfield POC with four milestone-aligned splits, each building on the previous.

The four splits map directly to the four milestones in the requirements. They are sequentially dependent: each split produces outputs consumed by the next.

---

## Splits

### 01-evaluation — Evaluation Mechanism (Milestone 1)

**Purpose:** Build the quality measurement pipeline that all downstream optimization depends on.

**Key components:**
- HuggingFace Comet model pipeline for semantic similarity scoring
- Gemini Pro LLM-as-a-Judge with a medical-specific rubric (1–10, emphasis on terms/dosages/diagnoses)
- Hybrid correlation analysis to select the final DSPy optimization metric

**Inputs:** Existing gold-standard dataset (Hebrew medical docs + Russian reference translations)
**Outputs:** A callable evaluator function/class that accepts (source, hypothesis, reference) and returns a score — ready to be plugged into DSPy as an optimization metric

**No dependencies** (foundational unit)

---

### 02-agent-orchestration — Agent Orchestration & Classification (Milestone 3)

**Purpose:** Build the routing and translation agent system, optimized using the evaluation metric.

**Key components:**
- Router Agent: Gemini Flash classifier → one of 5 categories (Summary, Prescript, Referral, Form, Record)
- 5 DSPy Signatures: per-category translation instructions (including drug-name preservation for Prescript)
- DSPy Optimizer: runs per-agent using the evaluator from 01 to tune prompts

**Inputs:** Evaluator from 01-evaluation; classified Hebrew documents
**Outputs:** A translation pipeline that accepts raw Hebrew text + document type → returns translated Russian text

**Depends on:** 01-evaluation (API contract: evaluator callable)

**Primary risk:** DSPy optimizer tuning — getting the optimization signal to meaningfully improve per-agent translation quality over baseline

---

### 03-xai-transparency — Transparency / XAI (Milestone 4)

**Purpose:** Add per-word confidence signals to translation output so uncertain words can be flagged for the UI.

**Key components:**
- Confidence extraction: either Logprobs from Gemini API per token, or a reflection prompt where the model self-reviews its translation and returns a list of uncertain words
- Tagging mechanism: maps uncertain words to a structured format (e.g., special Markdown tag `{{uncertain:word}}`) in the translated output

**Inputs:** Translation agents from 02-agent-orchestration
**Outputs:** Modified translation output with uncertainty tags embedded inline

**Depends on:** 02-agent-orchestration (extends the translation agent output schema)

---

### 04-ocr-ui — OCR & UI Integration (Milestone 5)

**Purpose:** Build the full end-to-end pipeline from image upload to tagged translated document, plus the frontend.

**Key components:**
- OCR: Google Cloud Vision API (`TEXT_DETECTION`) for Hebrew text extraction from scans/images
- Pipeline glue: `Image → OCR → Router → Specialist Agent (with XAI) → Tagged Translation`
- Backend API endpoint that accepts file uploads and returns structured translation response
- Frontend: file upload UI → loading state → final translated document with uncertain words highlighted (yellow / warning note)

**Inputs:** OCR-extracted Hebrew text; full pipeline from 02 + 03
**Outputs:** Working end-to-end web application

**Depends on:** 02-agent-orchestration (pipeline), 03-xai-transparency (uncertainty tags for UI highlighting)

---

## Execution Order

```
01-evaluation
      |
      v
02-agent-orchestration          ← primary risk: DSPy tuning
      |
      v
03-xai-transparency
      |
      v
04-ocr-ui
```

All splits are sequential. No parallel execution is possible because each builds directly on the previous split's output.

> **Note on parallelism:** The OCR integration sub-task (Google Vision API setup) within 04 could theoretically start after 02 is complete, but the UI highlighting requires XAI tags from 03. Since these are planning units rather than parallel dev tracks, keep 04 as a single sequential unit.

---

## Cross-Cutting Concerns

- **Language pair:** Hebrew (input) → Russian (output), always
- **Drug name handling:** Prescriptions must NOT translate drug names — leave in original (English/Latin). This constraint must be enforced in the DSPy Prescript signature in 02, and preserved through XAI tagging in 03.
- **Gemini model split:** Pro for translation and evaluation; Flash for routing only
- **Dataset:** The gold-standard dataset already exists — no curation work needed in 01

---

## /deep-plan Commands

Run in this order:

```
/deep-plan @planning/01-evaluation/spec.md
/deep-plan @planning/02-agent-orchestration/spec.md
/deep-plan @planning/03-xai-transparency/spec.md
/deep-plan @planning/04-ocr-ui/spec.md
```
