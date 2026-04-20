"""Checks four specific correctness behaviours: nested error unwrapping in
_calc_score(), penalty-summed scoring in score_document(), absence of a
deduplication rule in the evaluation prompt, and that the DB stores the
summed-penalty score rather than an average of per-section scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.translation.cost import gemini_cost
from src.app.db import BenchmarkDB
from src.app.translation.base import SectionTranslationResult
from src.app.evaluation.judge import (
    EvaluationResult,
    _MQM_SECTION_PROMPT,
    _calc_score,
    score_document,
)

_MOCK_PATH = "src.app.evaluation.judge.get_eval_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(errors: list[dict]) -> MagicMock:
    """Build a mock GenAI client that returns a fixed errors payload.

    Args:
        errors: List of MQM error dicts to embed in the mocked response.

    Returns:
        A MagicMock configured to return the given errors from the async
        generate_content method.
    """
    response = MagicMock()
    response.text = json.dumps({"errors": errors})
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    client._api_client = MagicMock()
    return client


def _make_section(index: int, label: str) -> SectionTranslationResult:
    """Return a minimal SectionTranslationResult for the given index and label.

    Args:
        index: 1-based section index.
        label: Human-readable section label.

    Returns:
        A SectionTranslationResult with deterministic placeholder texts and cost.
    """
    return SectionTranslationResult(
        section_index=index,
        section_label=label,
        hebrew_text=f"hebrew {index}",
        russian_text=f"russian {index}",
        cost=gemini_cost(input_tokens=100, output_tokens=50, model="flash"),
        elapsed_sec=1.0,
        was_corrected=False,
        midway_score=1.0,
        midway_errors=[],
    )


# ---------------------------------------------------------------------------
# Fix 1: nested error unwrapping in _calc_score
# ---------------------------------------------------------------------------

def test_calc_score_unwraps_nested_errors() -> None:
    """_calc_score() unwraps {"errors": [...]} wrappers and applies real penalties.

    A nested entry like {"errors": [{"severity": "critical", ...}]} must be
    treated as a critical error (penalty 0.25), not silently skipped.
    """
    errs = [
        {
            "errors": [
                {"span": "x", "category": "accuracy", "severity": "critical", "justification": "bad"},
            ]
        }
    ]
    # Without unwrapping: score = 1.0 (no severity key found → zero penalty)
    # With unwrapping:    score = 1.0 - 0.25 = 0.75
    assert abs(_calc_score(errs) - 0.75) < 1e-9


def test_calc_score_mixed_nested_and_flat() -> None:
    """_calc_score() handles a mix of nested wrappers and flat error dicts."""
    errs = [
        {"span": "a", "category": "accuracy", "severity": "minor", "justification": "ok"},
        {
            "errors": [
                {"span": "b", "category": "accuracy", "severity": "major", "justification": "bad"},
            ]
        },
    ]
    # 1 minor (0.01) + 1 major (0.05) = 0.06 penalty → score 0.94
    assert abs(_calc_score(errs) - 0.94) < 1e-9


def test_calc_score_nested_empty_errors_returns_perfect() -> None:
    """_calc_score() handles {"errors": []} (empty inner list) without crashing."""
    assert _calc_score([{"errors": []}]) == 1.0


# ---------------------------------------------------------------------------
# Fix 2: score_document uses summed penalties, not section score average
# ---------------------------------------------------------------------------

def test_score_document_sums_not_averages() -> None:
    """score_document() must return a summed-penalty score, not an average.

    Section 1: 1 critical error → per-section score 0.75, penalty 0.25
    Section 2: 1 major error    → per-section score 0.95, penalty 0.05
    Average of section scores: (0.75 + 0.95) / 2 = 0.85  ← wrong
    Summed penalty: 0.25 + 0.05 = 0.30 → correct score  = 0.70
    """
    errors_s1 = [{"span": "x", "category": "accuracy", "severity": "critical", "justification": "j"}]
    errors_s2 = [{"span": "y", "category": "terminology", "severity": "major", "justification": "j"}]

    call_count = 0

    def make_client_cycled() -> MagicMock:
        """Return a client pre-loaded with errors_s1 on the first call, errors_s2 on the second.

        Args:
            None

        Returns:
            A MagicMock GenAI client configured with the appropriate error payload.
        """
        nonlocal call_count
        payload = errors_s1 if call_count == 0 else errors_s2
        call_count += 1
        return _make_client(payload)

    with patch(_MOCK_PATH, side_effect=make_client_cycled):
        doc_score, results = score_document(
            sections=[("heb1", "rus1", "S1"), ("heb2", "rus2", "S2")],
            context_header="ctx",
        )

    assert len(results) == 2
    # Must NOT be 0.85 (the per-section average); must be 0.70 (summed penalties).
    assert abs(doc_score - 0.70) < 1e-9


def test_score_document_no_errors_returns_perfect() -> None:
    """score_document() returns 1.0 when all sections have zero errors."""
    call_count = 0

    def make_perfect_client() -> MagicMock:
        """Return a client pre-loaded with an empty errors list (perfect translation).

        Args:
            None

        Returns:
            A MagicMock GenAI client configured to return zero errors.
        """
        nonlocal call_count
        call_count += 1
        return _make_client([])

    with patch(_MOCK_PATH, side_effect=make_perfect_client):
        doc_score, results = score_document(
            sections=[("heb1", "rus1", "S1"), ("heb2", "rus2", "S2")],
            context_header="ctx",
        )

    assert doc_score == 1.0
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Fix 3: section prompt has no deduplication rule
# ---------------------------------------------------------------------------

def test_section_prompt_has_no_deduplication_rule() -> None:
    """The section evaluation prompt must not contain a deduplication rule.

    A dedup rule causes the evaluator to collapse repeated errors into one,
    suppressing legitimate penalties. Every error occurrence must be reported.
    """
    assert "deduplication" not in _MQM_SECTION_PROMPT.lower()
    assert "single error" not in _MQM_SECTION_PROMPT.lower()
    assert "same root cause" not in _MQM_SECTION_PROMPT.lower()


# ---------------------------------------------------------------------------
# Fix 4: db.insert_result_with_sections stores summed-penalty score
# ---------------------------------------------------------------------------

def test_db_stores_summed_penalty_score() -> None:
    """insert_result_with_sections() must persist the summed-penalty doc score.

    Section 1: 1 critical error → penalty 0.25, per-section score 0.75
    Section 2: 2 major errors   → penalty 0.10, per-section score 0.90
    Average of section scores: (0.75 + 0.90) / 2 = 0.825  ← wrong
    Summed penalty: 0.35 → correct doc score = 0.65
    """
    db = BenchmarkDB(Path(":memory:"))
    db.create_tables()
    scenario_id = db.insert_run("s_penalty", "Penalty test", doc_count=1)

    sections = [_make_section(1, "S1"), _make_section(2, "S2")]
    eval1 = EvaluationResult(
        quality_score=0.75,
        errors=[{"span": "x", "category": "accuracy", "severity": "critical", "justification": "j"}],
    )
    eval2 = EvaluationResult(
        quality_score=0.90,
        errors=[
            {"span": "a", "category": "accuracy", "severity": "major", "justification": "j"},
            {"span": "b", "category": "accuracy", "severity": "major", "justification": "j"},
        ],
    )
    cost = gemini_cost(input_tokens=100, output_tokens=50, model="flash")

    db.insert_result_with_sections(
        run_id=scenario_id,
        doc_id="Test_001",
        sections=sections,
        section_eval_results=[eval1, eval2],
        cost=cost,
        elapsed_sec=2.0,
    )

    row = db._conn.execute(
        "SELECT quality_score FROM results WHERE doc_id = 'Test_001'"
    ).fetchone()
    assert row is not None
    # 1 critical (0.25) + 2 major (0.10) = 0.35 penalty → score 0.65
    assert abs(row["quality_score"] - 0.65) < 1e-9
    db.close()
