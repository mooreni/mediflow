# PLAN: Section-Based Translation Pipeline

## Overview

Replace the current full-document translation-and-evaluation pipeline with a three-stage section-based pipeline: **Split -> Translate per section -> Evaluate per section**. This applies to scenarios s2-s6 (all LLM-based scenarios). s1 (Google Translate) is unchanged. Each updated scenario gets a `_v2` name suffix for side-by-side comparison with existing results.

## Architecture Summary

```
Document
  |
  v
[Splitter] (Gemini Flash, shared, run upfront for all docs)
  |
  v
SplitResult: context_header + Section[] (numbered, labeled, ~20 sentences max)
  |
  v
[Translator] (per scenario, sequential over sections)
  |  Input: context_header + section label + position + hebrew_text
  |  Output: plain Russian text (no JSON pairs)
  |
  v
SectionTranslationResult[] (per section: russian_text, cost, elapsed)
  |
  v
[Evaluator] (per section, updated prompt)
  |  Input: section_hebrew + section_russian + context_header + position
  |  Output: MQM errors + quality_score per section
  |
  v
Document score = average of section scores
```

---

## Step 1: Data structures and cost utilities

**Files to create/modify:**
- `src/pipeline/__init__.py` (new, empty)
- `src/pipeline/splitter.py` (new — dataclasses only in this step)
- `src/benchmark/scenarios/base.py` (add `SectionTranslationResult`, `supports_sections` property)
- `src/benchmark/cost.py` (add `sum_costs()`)

**What to do:**

1. Create `src/pipeline/splitter.py` with dataclasses only (no LLM logic yet):

```python
@dataclass(frozen=True)
class Section:
    index: int          # 1-based
    label: str          # e.g. "Patient Demographics"
    hebrew_text: str

@dataclass(frozen=True)
class SplitResult:
    context_header: str     # compact clinical context (description, patient sex, primary diagnosis)
    total_sections: int
    sections: list[Section]
```

2. Add to `src/benchmark/scenarios/base.py`:

```python
@dataclass(frozen=True)
class SectionTranslationResult:
    section_index: int
    section_label: str
    hebrew_text: str
    russian_text: str
    cost: CostRecord
    elapsed_sec: float
```

3. Add `supports_sections` property to `TranslationScenario` base class, defaulting to `False`.

4. Add abstract method `translate_section(self, context_header: str, section: Section, total_sections: int) -> SectionTranslationResult` to base class with a default `NotImplementedError` (not abstract — only v2 scenarios override it).

5. Add `sum_costs(costs: list[CostRecord]) -> CostRecord` to `src/benchmark/cost.py`. Sums token counts (treating `None` as 0 when at least one record has a value) and USD.

**Tests:** `tests/test_data_structures.py`
- `SplitResult` and `Section` are frozen and fields accessible.
- `SectionTranslationResult` is frozen and fields accessible.
- `supports_sections` defaults to `False` on the base class.
- `sum_costs()` correctly sums two `CostRecord`s with token fields.
- `sum_costs()` correctly sums a token-based and char-based record (mixed `None`s).
- `sum_costs()` on an empty list returns a zero-cost record.

---

## Step 2: Splitter LLM logic

**Files to modify:**
- `src/pipeline/splitter.py` (add `split_document()` function)

**What to do:**

1. Add the splitter prompt to `src/pipeline/splitter.py`:
   - System instruction telling the LLM to split a Hebrew medical document into sections.
   - Rules: each section <=20 sentences, sub-split longer topics (e.g. "Medications 1/2"), minimum 2 sentences per section (unless standalone heading), respect document structure, preserve all content.
   - Output JSON schema: `{ "context": { "document_description", "patient_sex", "primary_diagnosis" (nullable) }, "total_sections": int, "sections": [{ "index", "label", "hebrew_text" }] }`

2. Implement `split_document(hebrew_text: str) -> SplitResult`:
   - Uses Gemini Flash via `get_translate_client()` (same model as s2).
   - `response_mime_type="application/json"`, `temperature=0.0`.
   - Parses JSON response into `SplitResult`.
   - Validates: `len(sections) == total_sections`, all indices are 1..N contiguous, no section has empty `hebrew_text`.
   - On validation failure, raises `ValueError` with details.

3. Wrap with tenacity retry: `retry_if_exception` for 429/timeout, `stop_after_attempt(3)`, `wait_exponential_jitter(initial=10, max=60)`. On exhaustion, reraise.

**Tests:** `tests/test_splitter.py`
- Mock the Gemini client. Feed a well-formed JSON response; assert `SplitResult` fields match.
- Feed a response with mismatched `total_sections` vs `len(sections)`; assert `ValueError`.
- Feed a response with a gap in indices (1, 3); assert `ValueError`.
- Feed a response with an empty `hebrew_text` in a section; assert `ValueError`.
- Feed invalid JSON; assert `ValueError`.
- Assert that `context_header` is built from `document_description`, `patient_sex`, `primary_diagnosis`.

---

## Step 3: Updated judge prompt and `score_section()`

**Files to modify:**
- `src/evaluation/llm_judge.py`
- `src/evaluation/__init__.py`

**What to do:**

1. Add `_MQM_SECTION_PROMPT` — a variant of `_MQM_PROMPT` with these changes:
   - **Hypothesis format**: plain Russian text (not JSON sentence pairs).
   - **Coverage check**: "If the Russian translation omits significant clinical content present in the Hebrew source section, add a Critical accuracy error with span `[INCOMPLETE TRANSLATION]`." No pair-counting.
   - **Source block**: `SOURCE (Hebrew - section {index} of {total}, "{label}"):\n{section_hebrew}`
   - **Hypothesis block**: `HYPOTHESIS (Russian translation):\n{section_russian}`
   - **New block**: `CLINICAL CONTEXT:\n{context_header}`

2. Add `score_section(section_hebrew: str, section_russian: str, context_header: str, section_index: int, total_sections: int, section_label: str) -> EvaluationResult`:
   - Same async pattern as existing `score()`, same tenacity config.
   - Uses `_MQM_SECTION_PROMPT`.

3. Add `score_document(sections: list[tuple[str, str, str]], context_header: str) -> tuple[float, list[EvaluationResult]]`:
   - Takes list of `(section_hebrew, section_russian, section_label)` tuples.
   - Calls `score_section()` sequentially for each.
   - If one section's evaluation fails after retries, excludes it from average, logs warning.
   - Returns `(average_score, list_of_per_section_results)`.

4. Keep the old `score()` and `_MQM_PROMPT` intact — s1 and old scenarios still use them.

5. Update `src/evaluation/__init__.py` to export `score_section` and `score_document`.

**Tests:** `tests/test_judge_section.py`
- Mock the Gemini client. Verify `score_section()` formats the prompt correctly (contains section index, label, context header, plain-text hypothesis).
- Verify `score_section()` returns an `EvaluationResult` with correct score from mocked response.
- Verify `score_document()` averages scores across 3 mock sections.
- Verify `score_document()` excludes a failing section and averages the remaining.
- Verify the old `score()` function still works unchanged (regression).

---

## Step 4: Database schema — `section_results` table

**Files to modify:**
- `src/benchmark/db.py`

**What to do:**

1. Add `_CREATE_SECTION_RESULTS` SQL:
```sql
CREATE TABLE IF NOT EXISTS section_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES results(id),
    section_index   INTEGER NOT NULL,
    section_label   TEXT NOT NULL,
    section_hebrew  TEXT NOT NULL,
    section_russian TEXT NOT NULL,
    quality_score   REAL NOT NULL,
    errors          TEXT
);
```

2. Call `_CREATE_SECTION_RESULTS` in `create_tables()`.

3. Add `insert_section_result(result_id, section_index, section_label, section_hebrew, section_russian, quality_score, errors)` method.

4. Add `insert_result_with_sections()` method that:
   - Inserts the document-level row into `results` (same as today — concatenated Russian text, averaged score, merged errors).
   - Inserts one row per section into `section_results`.
   - Both within a single transaction.

5. Add `load_section_results(result_id: int) -> list[dict]` for querying per-section data.

**Tests:** `tests/test_db_sections.py`
- Create an in-memory SQLite DB, call `create_tables()`.
- Insert a scenario, insert a result with 3 sections via `insert_result_with_sections()`.
- Assert the `results` row has the averaged score and concatenated translation.
- Assert `load_section_results()` returns 3 rows with correct fields.
- Assert `section_results` rows have correct foreign key to `results.id`.
- Assert the old `insert_result()` still works (regression — s1 uses it).

---

## Step 5: Runner changes — section-based orchestration

**Files to modify:**
- `src/benchmark/runner.py`
- `scripts/run_benchmark.py`

**What to do:**

1. In `runner.py`, add a `split_all_docs(eval_docs: list[DatasetDoc]) -> dict[str, SplitResult]` function:
   - Iterates over all eval docs, calls `split_document()` for each.
   - Returns a dict keyed by `doc_id`.
   - On split failure for a doc, logs a warning and excludes it from the dict (the doc will be skipped by all scenarios).

2. In `run_scenario()`, accept an optional `split_cache: dict[str, SplitResult] | None` parameter.

3. In `_translate_and_evaluate()`, branch on `scenario.supports_sections`:
   - **If False (s1):** existing path unchanged.
   - **If True (v2 scenarios):**
     - Look up `split_cache[doc.doc_id]`.
     - For each section: call `scenario.translate_section(context_header, section, total_sections)`.
     - If any section translation fails after retries, fail the entire document.
     - Call `score_document()` on all section results.
     - Aggregate: concatenate Russian texts, `sum_costs()`, sum elapsed.
     - Return aggregated `TranslationResult` + section-level eval results.

4. In `_persist_result()`, branch:
   - s1: `db.insert_result()` (unchanged).
   - v2: `db.insert_result_with_sections()`.

5. In `scripts/run_benchmark.py`:
   - Add v2 scenarios to `_SCENARIO_REGISTRY`.
   - Add `split_all_docs()` call before the scenario loop.
   - Pass `split_cache` to `run_scenario()`.

**Tests:** `tests/test_runner_sections.py`
- Create a mock scenario with `supports_sections=True` and a mock `translate_section()`.
- Create a mock splitter result (pre-built `split_cache`).
- Run through the section-based path with mocked judge.
- Assert `translate_section()` was called once per section.
- Assert `insert_result_with_sections()` was called with correct aggregated data.
- Assert a scenario with `supports_sections=False` still uses the old `translate()` path.

---

## Step 6: v2 scenario — `s2_gemini_flash_zeroshot_v2.py`

**Files to create:**
- `src/benchmark/scenarios/s2_gemini_flash_zeroshot_v2.py`

**What to do:**

1. Create the v2 scenario class `GeminiFlashZeroShotV2Scenario`:
   - `name` returns `"s2_gemini_flash_zeroshot_v2"`.
   - `supports_sections` returns `True`.
   - `translate_section()` implementation:
     - System instruction: medical translator prompt with glossary (same glossary, same model).
     - User message includes: context header, section label, section position ("section 3 of 5"), section Hebrew text.
     - Output: plain Russian text only — no JSON pairs.
     - Uses `get_translate_client()`, `temperature=0.0`, `max_output_tokens=8192` (sections are smaller).
     - Returns `SectionTranslationResult` with cost from `gemini_cost(..., model="flash")`.
   - `translate()` raises `NotImplementedError` (runner uses `translate_section()` path).

2. Register in `_SCENARIO_REGISTRY` in `run_benchmark.py`.

**Tests:** `tests/test_s2_v2.py`
- Mock the Gemini client.
- Call `translate_section()` with a sample section and context header.
- Assert the prompt sent to the model contains the context header, section label, and section position.
- Assert the returned `SectionTranslationResult` has correct fields.
- Assert `supports_sections` is `True`.
- Assert `translate()` raises `NotImplementedError`.

---

## Step 7: v2 scenario — `s3_gemini_pro_dspy_predict_v2.py`

**Files to create:**
- `src/benchmark/scenarios/s3_gemini_pro_dspy_predict_v2.py`

**What to do:**

1. Create a new DSPy signature `MedicalSectionTranslation`:
   - Input fields: `context_header: str`, `section_label: str`, `section_position: str` (e.g. "3 of 5"), `hebrew_text: str`.
   - Output field: `russian_translation: str` (plain text).
   - Docstring: medical translation instructions + glossary.

2. Create `GeminiProDSPyPredictV2Scenario`:
   - `name` returns `"s3_gemini_pro_dspy_predict_v2"`.
   - `supports_sections` returns `True`.
   - Uses `dspy.Predict(MedicalSectionTranslation)`.
   - `translate_section()` feeds the DSPy signature with section context.
   - Cost via `count_tokens` on the native client (same as s3 v1).

3. Register in `_SCENARIO_REGISTRY`.

**Tests:** `tests/test_s3_v2.py`
- Verify `MedicalSectionTranslation` signature has the expected input/output fields.
- Mock DSPy LM. Call `translate_section()`, assert the prediction was called with context_header, section_label, section_position, hebrew_text.
- Assert `supports_sections` is `True`.

---

## Step 8: v2 scenario — `s4_gemini_pro_bootstrap_v2.py`

**Files to create:**
- `src/benchmark/scenarios/s4_gemini_pro_bootstrap_v2.py`

**What to do:**

1. Create `GeminiProBootstrapV2Scenario`:
   - `name` returns `"s4_gemini_pro_bootstrap_v2"`.
   - `supports_sections` returns `True`.
   - Uses `dspy.ChainOfThought(MedicalSectionTranslation)` (imports signature from s3_v2).
   - `train()`: same BootstrapFewShot pattern as s4 v1 but:
     - Training examples are split into sections first (using the splitter).
     - Metric function: splits the doc, translates each section, evaluates with `score_document()`, returns document-level average.
     - Compiled module saved to `s4_v2_compiled_module.json`.
   - `translate_section()`: uses the compiled module.

2. Register in `_SCENARIO_REGISTRY`.

**Tests:** `tests/test_s4_v2.py`
- Verify `train()` calls the optimizer with the section-based metric.
- Mock the optimizer and verify the compiled module is saved.
- Verify `translate_section()` uses the compiled module.
- Assert `supports_sections` is `True`.

---

## Step 9: v2 scenarios — `s5_gemini_flash_dspy_predict_v2.py` and `s6_gemini_flash_cot_v2.py`

**Files to create:**
- `src/benchmark/scenarios/s5_gemini_flash_dspy_predict_v2.py`
- `src/benchmark/scenarios/s6_gemini_flash_cot_v2.py`

**What to do:**

1. `s5_v2`: Same as s3_v2 but using Gemini Flash model. Uses `dspy.Predict(MedicalSectionTranslation)`. Name: `"s5_gemini_flash_dspy_predict_v2"`.

2. `s6_v2`: Same as s5_v2 but using `dspy.ChainOfThought(MedicalSectionTranslation)`. Name: `"s6_gemini_flash_cot_v2"`.

3. Register both in `_SCENARIO_REGISTRY`.

**Tests:** `tests/test_s5_s6_v2.py`
- For each: assert `supports_sections` is `True`, assert `name` has `_v2` suffix.
- Mock DSPy LM, call `translate_section()`, verify it returns a `SectionTranslationResult`.
- Assert s5_v2 uses `dspy.Predict` and s6_v2 uses `dspy.ChainOfThought`.

---

## Step 10: Integration test and wiring

**Files to modify:**
- `scripts/run_benchmark.py` (add `_DEFAULT_V2_SCENARIOS` list, `--v2` flag)

**What to do:**

1. Add a `--v2` CLI flag to `run_benchmark.py` that runs only the v2 scenarios.
2. Ensure `split_all_docs()` is called once before the scenario loop when any v2 scenario is in the run list.
3. Verify the full pipeline end-to-end with one real document (manual test, not automated — requires API access).

**Tests:** `tests/test_integration.py`
- Full pipeline integration test with all components mocked at the API boundary:
  - Mock `split_document()` to return a 3-section `SplitResult`.
  - Mock `translate_section()` on a fake v2 scenario to return `SectionTranslationResult`.
  - Mock `score_section()` to return fixed `EvaluationResult`s.
  - Use an in-memory SQLite DB.
  - Run `run_scenario()` with 1 eval doc.
  - Assert: `results` table has 1 row with averaged score, `section_results` has 3 rows, costs are summed, Russian texts are concatenated.
- Test that s1 (non-section scenario) still works through the same runner with `split_cache=None`.

---

## File inventory

### New files
- `src/pipeline/__init__.py`
- `src/pipeline/splitter.py`
- `src/benchmark/scenarios/s2_gemini_flash_zeroshot_v2.py`
- `src/benchmark/scenarios/s3_gemini_pro_dspy_predict_v2.py`
- `src/benchmark/scenarios/s4_gemini_pro_bootstrap_v2.py`
- `src/benchmark/scenarios/s5_gemini_flash_dspy_predict_v2.py`
- `src/benchmark/scenarios/s6_gemini_flash_cot_v2.py`
- `tests/__init__.py`
- `tests/test_data_structures.py`
- `tests/test_splitter.py`
- `tests/test_judge_section.py`
- `tests/test_db_sections.py`
- `tests/test_runner_sections.py`
- `tests/test_s2_v2.py`
- `tests/test_s3_v2.py`
- `tests/test_s4_v2.py`
- `tests/test_s5_s6_v2.py`
- `tests/test_integration.py`

### Modified files
- `src/benchmark/scenarios/base.py`
- `src/benchmark/cost.py`
- `src/evaluation/llm_judge.py`
- `src/evaluation/__init__.py`
- `src/benchmark/db.py`
- `src/benchmark/runner.py`
- `scripts/run_benchmark.py`
