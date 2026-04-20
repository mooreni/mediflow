"""Checks that BenchmarkDB correctly persists section-level results, computes the
document-level quality score by summing MQM penalties (not averaging), and returns
sections in ascending index order when queried.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app.translation.cost import CostRecord, gemini_cost
from src.app.db import BenchmarkDB
from src.app.translation.base import SectionTranslationResult
from src.app.evaluation.judge import EvaluationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> BenchmarkDB:
    """Create a fresh in-memory BenchmarkDB with all tables initialised.

    Returns:
        A BenchmarkDB backed by an in-memory SQLite database.
    """
    db = BenchmarkDB(Path(":memory:"))
    db.create_tables()
    return db


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


def _make_eval(score: float) -> EvaluationResult:
    """Return an EvaluationResult with the given quality score and no errors.

    Args:
        score: Quality score in [0.0, 1.0].

    Returns:
        An EvaluationResult with the given score and an empty errors list.
    """
    return EvaluationResult(quality_score=score, errors=[])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_tables_creates_section_results_table() -> None:
    """create_tables() must create the section_results table without raising."""
    db = _make_db()
    # Query the sqlite_master table to confirm the table exists.
    row = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='section_results'"
    ).fetchone()
    assert row is not None, "section_results table was not created"
    db.close()


def test_insert_result_with_sections_creates_one_results_row() -> None:
    """insert_result_with_sections() must insert exactly one row into results."""
    db = _make_db()
    run_id = db.insert_run("s_test", "Test", doc_count=1)

    sections = [_make_section(i, f"Section {i}") for i in range(1, 4)]
    evals = [_make_eval(0.8), _make_eval(0.6), _make_eval(1.0)]
    cost = gemini_cost(input_tokens=300, output_tokens=150, model="flash")

    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="Form_001",
        sections=sections,
        section_eval_results=evals,
        cost=cost,
        elapsed_sec=5.0,
    )

    rows = db._conn.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchall()
    assert len(rows) == 1
    db.close()


def test_insert_result_with_sections_sums_penalties_and_concatenates_translation() -> None:
    """The results row must use summed MQM penalties for quality_score and joined Russian text.

    Section 1: 1 major error   → per-section score 0.95, penalty 0.05
    Section 2: 2 minor errors  → per-section score 0.98, penalty 0.02
    Section 3: no errors       → per-section score 1.00, penalty 0.00
    Average of per-section scores would be (0.95 + 0.98 + 1.00) / 3 ≈ 0.977 — wrong.
    Summed penalties: 0.05 + 0.02 + 0.00 = 0.07 → correct doc score = 0.93.
    """
    db = _make_db()
    run_id = db.insert_run("s_agg", "Aggregation test", doc_count=1)

    sections = [_make_section(1, "Intro"), _make_section(2, "Body"), _make_section(3, "Summary")]
    evals = [
        EvaluationResult(
            quality_score=0.95,
            errors=[{"span": "x", "category": "accuracy", "severity": "major", "justification": "j"}],
        ),
        EvaluationResult(
            quality_score=0.98,
            errors=[
                {"span": "a", "category": "linguistic", "severity": "minor", "justification": "j"},
                {"span": "b", "category": "linguistic", "severity": "minor", "justification": "j"},
            ],
        ),
        EvaluationResult(quality_score=1.0, errors=[]),
    ]
    cost = gemini_cost(input_tokens=300, output_tokens=150, model="flash")

    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="Form_002",
        sections=sections,
        section_eval_results=evals,
        cost=cost,
        elapsed_sec=3.0,
    )

    row = db._conn.execute("SELECT * FROM results WHERE doc_id = 'Form_002'").fetchone()
    assert row is not None

    # 1 major (0.05) + 2 minor (0.02) = 0.07 penalty → score 0.93
    assert abs(row["quality_score"] - 0.93) < 1e-9

    expected_translation = "russian 1\n\nrussian 2\n\nrussian 3"
    assert row["translation"] == expected_translation
    db.close()


def test_load_section_results_returns_three_rows_with_correct_fields() -> None:
    """load_section_results() must return one dict per section with correct FK."""
    db = _make_db()
    run_id = db.insert_run("s_load", "Load test", doc_count=1)

    sections = [_make_section(i, f"S{i}") for i in range(1, 4)]
    evals = [_make_eval(0.9), _make_eval(0.7), _make_eval(0.5)]
    cost = gemini_cost(input_tokens=300, output_tokens=150, model="flash")

    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="Form_003",
        sections=sections,
        section_eval_results=evals,
        cost=cost,
        elapsed_sec=4.0,
    )

    result_id = db._conn.execute(
        "SELECT id FROM results WHERE doc_id = 'Form_003'"
    ).fetchone()["id"]

    section_rows = db.load_section_results(result_id)
    assert len(section_rows) == 3

    for i, row in enumerate(section_rows, start=1):
        assert row["result_id"] == result_id
        assert row["section_index"] == i
        assert row["section_label"] == f"S{i}"
        assert row["section_hebrew"] == f"hebrew {i}"
        assert row["section_russian"] == f"russian {i}"
        assert isinstance(row["errors"], list)

    db.close()


def test_load_section_results_ordered_by_section_index() -> None:
    """load_section_results() must return rows in ascending section_index order."""
    db = _make_db()
    run_id = db.insert_run("s_order", "Order test", doc_count=1)

    # Insert sections in reverse order to verify ordering is by section_index.
    sections = [_make_section(3, "Third"), _make_section(1, "First"), _make_section(2, "Second")]
    evals = [_make_eval(0.5), _make_eval(0.9), _make_eval(0.7)]
    cost = gemini_cost(input_tokens=300, output_tokens=150, model="flash")

    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="Form_004",
        sections=sections,
        section_eval_results=evals,
        cost=cost,
        elapsed_sec=2.0,
    )

    result_id = db._conn.execute(
        "SELECT id FROM results WHERE doc_id = 'Form_004'"
    ).fetchone()["id"]
    section_rows = db.load_section_results(result_id)

    assert [r["section_index"] for r in section_rows] == [1, 2, 3]
    db.close()


def test_insert_result_with_sections_mismatched_lengths_raises() -> None:
    """insert_result_with_sections() must raise ValueError on length mismatch."""
    db = _make_db()
    run_id = db.insert_run("s_mismatch", "Mismatch test", doc_count=1)

    sections = [_make_section(1, "Only")]
    evals = [_make_eval(0.8), _make_eval(0.6)]  # one extra
    cost = gemini_cost(input_tokens=100, output_tokens=50, model="flash")

    with pytest.raises(ValueError, match="equal length"):
        db.insert_result_with_sections(
            run_id=run_id,
            doc_id="Form_005",
            sections=sections,
            section_eval_results=evals,
            cost=cost,
            elapsed_sec=1.0,
        )
    db.close()


def test_insert_result_with_sections_empty_raises() -> None:
    """insert_result_with_sections() must raise ValueError when sections is empty."""
    db = _make_db()
    run_id = db.insert_run("s_empty", "Empty test", doc_count=1)
    cost = gemini_cost(input_tokens=0, output_tokens=0, model="flash")

    with pytest.raises(ValueError, match="must not be empty"):
        db.insert_result_with_sections(
            run_id=run_id,
            doc_id="Form_006",
            sections=[],
            section_eval_results=[],
            cost=cost,
            elapsed_sec=0.0,
        )
    db.close()


