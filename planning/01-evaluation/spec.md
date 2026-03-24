# spec.md — 01-evaluation: Evaluation Mechanism

## What This Split Does

Builds the translation quality measurement pipeline for MediFlow. This is the foundational split — its output (a callable evaluator) is what feeds the DSPy optimizer in the next split.

## Full Requirements Reference

See `planning/requirements.md` Section 3A (Evaluation Mechanism, Milestone 1).

## Scope

Build a hybrid evaluation pipeline that:
1. Runs HuggingFace Comet (semantic similarity) against gold-standard translations
2. Runs Gemini Pro as LLM-as-a-Judge using a medical-specific rubric
3. Correlates both scores to determine which metric (or combination) to use as the DSPy optimization signal

## Key Decisions from Interview

- **Language pair:** Hebrew → Russian (all evaluation is on this pair)
- **Gold standard dataset:** Already exists — annotated set of Hebrew medical documents with reference Russian translations. No curation work needed.
- **Greenfield:** No existing code to build on

## Technical Constraints

- **Comet:** Run via HuggingFace (local pipeline or cloud — decide during planning based on resource constraints)
- **LLM-as-a-Judge rubric:** Must strongly weight medical terms, drug names, dosages, and diagnoses. Scale 1–10 with reasoning output.
- **Judge model:** Gemini Pro (not Flash)

## Output Contract (for 02-agent-orchestration)

The evaluator must expose a callable interface compatible with DSPy's optimization API, e.g.:

```python
def evaluate(source: str, hypothesis: str, reference: str) -> float:
    ...  # returns a score in [0, 1] or [0, 10]
```

The next split (02) plugs this directly into `dspy.MIPROv2` or equivalent DSPy optimizer.

## Uncertainties to Resolve During Planning

- Should Comet run locally (GPU required?) or via a cloud HuggingFace Inference Endpoint?
- Which Comet model variant? (`wmt22-comet-da`, `wmt23-comet-da`, etc.)
- For LLM-as-a-Judge: structured output (JSON score) vs. parsed free-text reasoning?
- How to weight/combine Comet and LLM-Judge scores if correlation is low?
- What does the gold standard dataset format look like? (needed to write the data loader)

## Dependencies

- **Requires:** Nothing (this is the foundational split)
- **Provides to 02-evaluation:** A callable evaluator function/class
