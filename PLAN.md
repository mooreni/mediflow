# Milestone 2: Translation Engine Benchmarking

## Context

MediFlow needs to select the best Hebrew→Russian medical translation method before building the full agent pipeline (Milestone 3). This benchmark compares 5 translation scenarios across quality (via the existing LLM judge) and cost, then produces interactive charts to identify the winning approach.

- Evaluation infrastructure from Milestone 1 is complete and reused as-is
- Dataset: 50 Hebrew/Russian informed consent form pairs in `data/informed_consent_forms/text/`
- Train split: Form_001–038 (38 docs) — used only by Scenarios 4 & 5 for DSPy optimization
- Test split: Form_039–050 (12 docs) — all 5 scenarios evaluated here
- Storage: SQLite (`benchmark.db`) + Plotly HTML report

---

## Implementation Steps

After implementing a step, ask the user if you can update the "Status" line.

### Step 1 — Update dependencies

**Status:** ✅ Done

**File:** `requirements.txt`

Add three new dependencies:

```
dspy-ai
google-cloud-translate
plotly
```

---

### Step 2 — Create dataset module

**Status:** ✅ Done

**File:** `src/benchmark/dataset.py`

Create a `DatasetDoc` frozen dataclass and two load functions:

```python
@dataclass(frozen=True)
class DatasetDoc:
    doc_id: str              # e.g. "Form_039"
    hebrew_text: str
    reference_russian: str

TRAIN_IDS = [f"Form_{i:03d}" for i in range(1, 39)]   # 001–038
TEST_IDS  = [f"Form_{i:03d}" for i in range(39, 51)]  # 039–050

def load_doc(doc_id: str, he_dir: Path, ru_dir: Path) -> DatasetDoc: ...
def load_split(doc_ids: list[str], he_dir: Path, ru_dir: Path) -> list[DatasetDoc]: ...
```

File naming convention: `Form_001_HE.txt` / `Form_001_RU.txt`.

Reference pattern: `scripts/test_evaluator.py` for path construction and dotenv setup.

---

### Step 3 — Create cost tracking module

**Status:** ✅ Done

**File:** `src/benchmark/cost.py`

```python
@dataclass(frozen=True)
class CostRecord:
    input_tokens: int | None    # None for Google Translate
    output_tokens: int | None   # None for Google Translate
    input_chars: int | None     # None for Gemini
    output_chars: int | None    # None for Gemini
    cost_usd: float

# Named constants — update if pricing changes
GEMINI_PRO_INPUT_USD_PER_1K_TOKENS    = 0.002
GEMINI_PRO_OUTPUT_USD_PER_1K_TOKENS   = 0.012
GEMINI_FLASH_INPUT_USD_PER_1K_TOKENS  = 0.00025
GEMINI_FLASH_OUTPUT_USD_PER_1K_TOKENS = 0.0005
GOOGLE_TRANSLATE_USD_PER_1M_CHARS     = 20.0

def gemini_cost(input_tokens: int, output_tokens: int, model: Literal["pro", "flash"]) -> CostRecord: ...
def google_translate_cost(input_chars: int, output_chars: int) -> CostRecord: ...
```

---

### Step 4 — Create SQLite database module

**Status:** ⬜ Not started

**File:** `src/benchmark/db.py`

Create a `BenchmarkDB` class wrapping a SQLite connection with this schema:

```sql
CREATE TABLE scenarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    run_at          TEXT NOT NULL,          -- ISO-8601
    train_doc_count INTEGER,               -- NULL for non-trained scenarios
    test_doc_count  INTEGER NOT NULL
);

CREATE TABLE results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id           INTEGER NOT NULL REFERENCES scenarios(id),
    doc_id                TEXT NOT NULL,
    translation           TEXT NOT NULL,
    critical_terms_score  REAL NOT NULL,
    completeness_score    REAL NOT NULL,
    semantic_score        REAL NOT NULL,
    overall_score         REAL NOT NULL,
    verbal_evaluation     TEXT,
    problems              TEXT,            -- JSON array serialized as string
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    input_chars           INTEGER,
    output_chars          INTEGER,
    cost_usd              REAL NOT NULL,
    elapsed_sec           REAL NOT NULL
);

CREATE TABLE training_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id   INTEGER NOT NULL REFERENCES scenarios(id),
    iteration     INTEGER NOT NULL,
    metric_score  REAL NOT NULL,
    examples_used INTEGER NOT NULL
);
```

Key methods: `create_tables()`, `insert_scenario() -> int`, `insert_result()`, `scenario_exists(name) -> bool`, `load_all_results() -> list[dict]`.

Each `insert_result()` commits immediately — partial runs survive crashes.

---

### Step 5 — Create scenario base class

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/base.py`

```python
@dataclass(frozen=True)
class TranslationResult:
    doc_id: str
    translation: str
    cost: CostRecord
    elapsed_sec: float

class TranslationScenario(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...        # e.g. "s1_google_translate"

    @property
    @abstractmethod
    def description(self) -> str: ...

    def train(self, train_docs: list[DatasetDoc]) -> None:
        """No-op by default. Scenarios 4 & 5 override this."""

    @abstractmethod
    def translate(self, doc: DatasetDoc) -> TranslationResult: ...
```

Also create `src/benchmark/__init__.py` and `src/benchmark/scenarios/__init__.py` as empty files.

---

### Step 6 — Implement Scenario 1: Google Translate

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/s1_google_translate.py`

- Client: `google.cloud.translate_v2.Client()` — uses same GCP credentials (mediflow-491208)
- Call: `client.translate(text, target_language="ru", source_language="he")`
- Cost: `google_translate_cost(len(source), len(translation))`
- No `train()` override needed

---

### Step 7 — Implement Scenario 2: Gemini Flash Zero-Shot

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/s2_gemini_flash_zeroshot.py`

- Model constant: `FLASH_MODEL = "gemini-2.0-flash"`
- Reuse `src/evaluation/vertex_client.get_client()` for the genai client
- Direct `client.models.generate_content()` call — no DSPy
- System prompt: translate Hebrew medical form to Russian, preserve all terms, output only Russian
- Token counts: `response.usage_metadata.prompt_token_count` and `candidates_token_count`
- Cost: `gemini_cost(input_tokens, output_tokens, model="flash")`

---

### Step 8 — Implement Scenario 3: Gemini Pro DSPy Predict

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/s3_gemini_pro_dspy_predict.py`

Define the shared DSPy signature (imported by Steps 9 and 10):

```python
class MedicalTranslation(dspy.Signature):
    """Translate a Hebrew medical informed consent form into Russian.
    Preserve all medical terminology, dosages, dates, diagnoses, and document
    structure exactly. Output only Russian text, no commentary.
    """
    hebrew_text: str = dspy.InputField()
    russian_translation: str = dspy.OutputField()
```

DSPy LM configuration:

```python
lm = dspy.LM(
    model="vertex_ai/gemini-3.1-pro-preview",
    temperature=0.0,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)
dspy.configure(lm=lm)
```

Module: `dspy.Predict(MedicalTranslation)`. No `train()` override.

Token counts: read from `dspy.settings.lm.history[-1]` after each forward call.

---

### Step 9 — Implement Scenario 4: DSPy BootstrapFewShot

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/s4_gemini_pro_bootstrap.py`

Import `MedicalTranslation` signature from S3.

Module: `dspy.ChainOfThought(MedicalTranslation)`.

DSPy metric (reuses existing evaluator):

```python
def translation_metric(example, prediction, trace=None) -> float:
    from src.evaluation.evaluator import evaluate
    return evaluate(
        source=example.hebrew_text,
        hypothesis=prediction.russian_translation,
        reference=example.russian_translation,
    )
```

`train()` implementation:

```python
optimizer = dspy.BootstrapFewShot(
    metric=translation_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
    max_rounds=10,
)
self._module = optimizer.compile(ChainOfThoughtTranslatorModule(), trainset=examples)
```

`translate()` must raise `RuntimeError("train() must be called before translate()")` if `self._module` is not set.

---

### Step 10 — Implement Scenario 5: Dual Agent

**Status:** ⬜ Not started

**File:** `src/benchmark/scenarios/s5_dual_agent.py`

Import `MedicalTranslation` from S3. Add a second signature:

```python
class MedicalTranslationReview(dspy.Signature):
    """Review a Hebrew→Russian medical translation and produce a corrected version.
    Check: critical terms, dosages, dates, diagnoses. Fix any issues found.
    """
    hebrew_source: str = dspy.InputField()
    initial_translation: str = dspy.InputField()
    revised_translation: str = dspy.OutputField()
    issues_found: str = dspy.OutputField(desc="Comma-separated issues fixed, or 'none'")
```

Pipeline module:

```python
class DualAgentTranslatorModule(dspy.Module):
    def __init__(self):
        self.translator = dspy.Predict(MedicalTranslation)
        self.critic = dspy.Predict(MedicalTranslationReview)

    def forward(self, hebrew_text: str) -> dspy.Prediction:
        first_pass = self.translator(hebrew_text=hebrew_text)
        review = self.critic(
            hebrew_source=hebrew_text,
            initial_translation=first_pass.russian_translation,
        )
        return dspy.Prediction(russian_translation=review.revised_translation)
```

Same BootstrapFewShot config as S4. Both predictors compiled together.

Cost: 2 Gemini Pro calls per document — sum both token counts into one `CostRecord`.

---

### Step 11 — Create runner

**Status:** ⬜ Not started

**File:** `src/benchmark/runner.py`

```python
def run_scenario(
    scenario: TranslationScenario,
    train_docs: list[DatasetDoc],
    test_docs: list[DatasetDoc],
    db: BenchmarkDB,
) -> None:
    if db.scenario_exists(scenario.name):
        print(f"Skipping {scenario.name} — already in DB")
        return
    scenario_id = db.insert_scenario(scenario.name, scenario.description, ...)
    scenario.train(train_docs)   # no-op for S1, S2, S3
    for doc in test_docs:
        result = scenario.translate(doc)
        eval_result = llm_judge.score(   # use score() not evaluate() for per-dimension data
            source=doc.hebrew_text,
            hypothesis=result.translation,
            reference=doc.reference_russian,
        )
        db.insert_result(scenario_id, result, eval_result)
```

Import `score` from `src.evaluation.llm_judge` (not `evaluator.evaluate`) — we need all dimension scores for the DB, not just the scalar float.

---

### Step 12 — Create CLI entry point

**Status:** ⬜ Not started

**File:** `scripts/run_benchmark.py`

```
python scripts/run_benchmark.py                                # run all 5 scenarios
python scripts/run_benchmark.py --scenario s1_google_translate # run one
python scripts/run_benchmark.py --visualize                    # generate charts from DB
```

- `argparse` for argument parsing
- Load `.env` via `python-dotenv` at startup
- Register all 5 scenario instances in an ordered dict keyed by `scenario.name`
- `--scenario NAME` filters to a single scenario
- `--visualize` calls `generate_report()` without running any scenario
- DB path: `benchmark.db` at project root

---

### Step 13 — Create visualization module

**Status:** ⬜ Not started

**File:** `src/benchmark/visualize.py`

```python
def generate_report(db: BenchmarkDB, output_path: Path) -> None: ...
```

Generates `benchmark_results.html` — a single self-contained Plotly file with 4 charts:

| Chart | Type | What it shows |
|---|---|---|
| Score vs. Cost | Scatter (log x-axis) | One point per scenario; top-left = cheap + accurate = best |
| Per-Dimension Scores | Grouped bar | Critical Terms / Completeness / Semantic per scenario |
| Score Distribution | Box plot | Variance across 12 test docs per scenario |
| Per-Document Heatmap | Heatmap (docs × scenarios) | Which documents are universally hard |

Use `plotly.io.to_html(include_plotlyjs="cdn")` for a lightweight single file.

---

## Critical Files to Reuse

| File | What to reuse |
|---|---|
| `src/evaluation/llm_judge.py` | `score()` + `EvaluationResult` dataclass in runner (Step 11) |
| `src/evaluation/evaluator.py` | `evaluate()` as DSPy metric in Steps 9 & 10 |
| `src/evaluation/vertex_client.py` | `get_client()` pattern for Scenario 2 (Step 7) |
| `scripts/test_evaluator.py` | Reference for path construction, dotenv setup, script structure |

---

## Verification

1. `load_split(TRAIN_IDS, ...)` → 38 docs; `load_split(TEST_IDS, ...)` → 12 docs; no overlap
2. Run S1 on 1 doc: row in DB with `cost_usd > 0`, `input_tokens IS NULL`
3. Run S2 on 1 doc: row in DB with `input_tokens > 0`
4. Run S3 on 1 doc: DSPy connects to Vertex AI, translation returned
5. Run full benchmark: `python scripts/run_benchmark.py` → 12 rows per scenario in DB
6. SQL check: `SELECT name, COUNT(*), AVG(overall_score), SUM(cost_usd) FROM results JOIN scenarios ON results.scenario_id = scenarios.id GROUP BY scenario_id` → 5 rows, each with `count=12`
7. Generate report: `python scripts/run_benchmark.py --visualize` → `benchmark_results.html` opens with 4 charts
8. Resumability: re-run any scenario → skipped, no duplicate rows
