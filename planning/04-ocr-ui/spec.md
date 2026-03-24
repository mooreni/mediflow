# spec.md — 04-ocr-ui: OCR & UI Integration

## What This Split Does

Assembles the full end-to-end pipeline: image/scan → OCR → translation with XAI tags → frontend display. Builds the user-facing web interface where immigrants upload medical documents and receive a translated, annotated Russian version.

## Full Requirements Reference

See `planning/requirements.md` Section 3D (OCR and UI Integration, Milestone 5).

## Scope

1. **OCR Integration** — Google Cloud Vision API (`TEXT_DETECTION`) to extract Hebrew text from uploaded document images/scans
2. **Pipeline glue** — Backend API: `Image Upload → OCR → Router → Specialist Agent (with XAI) → Tagged Translation Response`
3. **Frontend** — File upload UI with loading state and rendered translated document, with uncertain words highlighted (yellow / warning annotation)

## Key Decisions from Interview

- **Language pair:** Hebrew → Russian (all UI copy, translated content, and error messages should reflect this)
- **Target users:** Immigrants in Israel who need Russian translations of Hebrew medical documents
- **Uncertain word display:** Yellow highlight or warning note — exact visual treatment to be decided during planning
- **Document types processed:** Images and scans of real medical documents (ER summaries, prescriptions, referrals, etc.)

## Technical Constraints

- **OCR service:** Google Cloud Vision API only (`TEXT_DETECTION` feature specifically)
- **Pipeline order:** OCR must feed into the Router (not directly to a specific agent) — the router classifies the extracted text
- **XAI tag format:** The frontend must parse the tag format agreed in 03-xai-transparency (e.g., `{{uncertain:word}}`) and render highlights. Confirm exact format before building the parser.
- **POC scope:** Basic web/app interface — not production-grade UX

## Pipeline Architecture

```
User uploads image/scan
        |
        v
Google Cloud Vision API (TEXT_DETECTION)
        |
    Hebrew text
        v
Router Agent (Gemini Flash) → document type
        |
        v
Specialist Translation Agent (Gemini Pro + DSPy) + XAI tagging
        |
    Tagged Russian text (with {{uncertain:word}} markers)
        v
Backend API response (JSON)
        |
        v
Frontend: render translated text with highlights
```

## Uncertainties to Resolve During Planning

- What file formats to support for upload? (PDF, JPG, PNG, TIFF — common medical scan formats)
- Does Google Vision handle multi-page PDFs, or is OCR per-image only?
- How accurate is Google Vision on handwritten Hebrew (common in handwritten prescriptions)?
- Backend framework choice: FastAPI, Flask, or other?
- Frontend technology: React, plain HTML/JS, or other? (POC context suggests minimal)
- How to handle OCR failures or low-confidence extractions?
- Should the API be synchronous (wait for translation) or async (polling/websocket)?

## Dependencies

- **Requires from 02-agent-orchestration:** Router + translation pipeline
- **Requires from 03-xai-transparency:** Tagged translation output with uncertainty markers; confirmed tag format
- **Provides:** Deployed web application (end product of the POC)
