# spec.md — 01-evaluation: Evaluation Mechanism

## What This Split Does

Builds the translation quality measurement pipeline for MediFlow. This is the foundational split — its output (a callable evaluator) is what feeds the DSPy optimizer in the next split.

## Full Requirements Reference

See `planning/requirements.md` Section 3A (Evaluation Mechanism, Milestone 1).

## Scope

Build an LLM-as-a-Judge evaluation pipeline that:
1. Sends source, hypothesis, and optional reference to Gemini 3.1 Pro via Vertex AI
2. Scores the translation on three medical-weighted dimensions (0–100 each)
3. Returns a normalized float in [0, 1] for DSPy optimization

## Key Decisions

- **Language pair:** Hebrew → Russian
- **Judge model:** Gemini 3.1 Pro (`gemini-3.1-pro-preview`) via Vertex AI
- **Comet:** Not used — LLM-as-a-Judge is the sole evaluator
- **Reference:** Optional — two separate prompts handle with/without reference cases
- **Greenfield:** No existing code to build on

## Rubric

Three dimensions scored 0–100, combined as a weighted average:

| Dimension | Weight | What it measures |
|---|---|---|
| Critical Terms Accuracy | 40% | Dosages, dates, drug names, diagnoses, medical procedures |
| Completeness | 35% | No dropped content, no hallucinations |
| Semantic Similarity | 25% | Meaning and intent preserved |

The judge returns structured JSON: `critical_terms_score`, `completeness_score`, `semantic_score`, `verbal_evaluation`, `problems`.

## Output Contract (for 02-agent-orchestration)

The evaluator exposes a DSPy-compatible interface:

```python
def evaluate(source: str, hypothesis: str, reference: str | None = None) -> float:
    ...  # returns overall_score / 100.0, i.e. a float in [0, 1]
```

For full structured results (per-dimension scores, verbal feedback, problem list), call `llm_judge.score()` directly, which returns an `EvaluationResult` dataclass.

The next split (02) plugs `evaluate()` directly into `dspy.MIPROv2` or equivalent DSPy optimizer.

## Implementation

- `src/evaluation/vertex_client.py` — Vertex AI client singleton; reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` env vars
- `src/evaluation/llm_judge.py` — `score()` function with retry logic (tenacity, up to 6 attempts, handles 429 and read timeouts)
- `src/evaluation/evaluator.py` — public `evaluate()` wrapper

## Dependencies

- **Requires:** Nothing (this is the foundational split)
- **Provides to 02-agent-orchestration:** A callable `evaluate()` function
