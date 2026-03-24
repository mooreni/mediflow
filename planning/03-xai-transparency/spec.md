# spec.md — 03-xai-transparency: Transparency / XAI

## What This Split Does

Extends the translation agents from 02-agent-orchestration to emit per-word confidence signals, then tags uncertain words in the output so the frontend can highlight them.

## Full Requirements Reference

See `planning/requirements.md` Section 3C (Transparency and Security / XAI, Milestone 4).

## Scope

1. **Confidence extraction** — Two candidate approaches (decide during planning):
   - **Logprobs:** Request token-level log probabilities from the Gemini API; map low-probability tokens to uncertain words
   - **Reflection:** A second LLM call where the model reviews its own translation and returns a list of words it's uncertain about

2. **Tagging mechanism** — Wrap uncertain words in a structured tag format (e.g., `{{uncertain:word}}`) in the translated output so the UI knows what to highlight

## Key Decisions from Interview

- **Output consumers:** The tagged translation is consumed by the 04-ocr-ui frontend, which will render uncertain words highlighted in yellow or with a warning note
- **Drug names:** Already excluded from translation in the Prescript agent — the XAI layer should handle them consistently (don't flag untranslated drug names as "uncertain")

## Technical Constraints

- **Model:** Gemini Pro (same as translation agents in 02)
- **Logprobs availability:** Gemini API logprobs support must be verified — may not be available on all endpoints/models
- **Tag format:** Must be agreed with the frontend team (04-ocr-ui) — use a format that's easy to parse client-side

## Output Contract (for 04-ocr-ui)

Modified translation output with inline uncertainty tags:

```
"הנחיות: קח {{uncertain:Augmentin}} 875mg פעמיים ביום"
→ "Инструкции: принимайте Augmentin 875мг {{uncertain:дважды}} в день"
```

The 04 split needs to know the exact tag format to render highlights correctly.

## Uncertainties to Resolve During Planning

- Does the Gemini Pro API expose logprobs? If not, is the reflection approach acceptable in quality and latency?
- What probability threshold defines "uncertain" for logprobs approach?
- For the reflection approach: how to reliably get the model to return a structured list of uncertain words (JSON schema? function calling?)
- How to handle multi-token words — if a word spans 3 tokens and one is low confidence, is the whole word flagged?
- Performance impact: reflection requires a second LLM call per document — acceptable for POC?

## Dependencies

- **Requires from 02-agent-orchestration:** Translation pipeline (DSPy module or callable) to wrap/extend
- **Provides to 04-ocr-ui:** Tagged translation output with `{{uncertain:word}}` markers (or agreed equivalent)
