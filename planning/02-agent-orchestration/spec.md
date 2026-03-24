# spec.md — 02-agent-orchestration: Agent Orchestration & Classification

## What This Split Does

Builds the core translation system: a router that classifies incoming Hebrew documents into one of 5 types, 5 specialized DSPy translation agents, and runs the DSPy optimizer on each agent using the evaluator from 01-evaluation.

## Full Requirements Reference

See `planning/requirements.md` Section 3B (Agent Orchestration, Milestone 3).

## Scope

1. **Router Agent** — Gemini Flash classifier that maps a document to exactly one of: `Summary`, `Prescript`, `Referral`, `Form`, `Record`
2. **5 DSPy Signatures** — One per document type with type-specific translation instructions
3. **DSPy Optimizer** — Runs per-agent, using the evaluator from 01-evaluation, to tune each agent's prompt

## Key Decisions from Interview

- **Language pair:** Hebrew → Russian
- **Drug name constraint:** Prescriptions (`Prescript` category) must NOT translate drug names — they stay in their original form (English/Latin). This is a hard rule enforced in the DSPy Prescript signature.
- **Models:** Gemini Flash for routing; Gemini Pro for translation agents
- **Primary risk:** Getting DSPy optimizer tuning to produce meaningful improvements over baseline — this needs careful planning around optimizer choice, number of optimization steps, and evaluation signal quality

## Technical Constraints

- **Framework:** DSPy (not raw prompting)
- **Router:** Must be fast/cheap — Gemini Flash only
- **Optimizer input:** Takes the evaluator callable from 01-evaluation as its metric
- **Dataset:** The gold-standard dataset (from 01-evaluation) must also be classified by document type for per-agent optimization

## Output Contract (for 03-xai-transparency)

The translation pipeline must expose:

```python
def translate(text: str, doc_type: str) -> str:
    ...  # returns translated Russian text
```

Or equivalently, a DSPy module that the XAI split can wrap to intercept outputs and inject confidence tags.

## Uncertainties to Resolve During Planning

- Which DSPy optimizer to use? (`MIPROv2`, `BootstrapFewShot`, `COPRO`, etc.) — tradeoffs between quality and compute cost
- How many optimization examples per agent? (dataset size constraint)
- How to handle the router's confidence — what if a document is ambiguous between types?
- Should the Router itself be a DSPy module or a simple Gemini Flash call?
- How to structure the classified dataset for per-agent optimization runs?

## Dependencies

- **Requires from 01-evaluation:** Callable evaluator `evaluate(source, hypothesis, reference) -> float`
- **Provides to 03-xai-transparency:** Translation pipeline `translate(text, doc_type) -> str` (or DSPy module equivalent)
- **Provides to 04-ocr-ui:** Same translation pipeline (full routing + translation)
