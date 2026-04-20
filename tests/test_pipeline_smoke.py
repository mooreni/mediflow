"""Smoke tests exercising the production and test pipeline end-to-end.

These tests use in-memory SQLite and mock section callables — no real API calls.
They verify that the full submit→collect→store flow works correctly under both
full-success and partial-failure conditions.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

import pytest

from scripts.run import PartialDocumentError, run_section_tasks
from src.app.db import BenchmarkDB
from src.app.evaluation.judge import EvaluationResult
from src.app.production_db import ProductionDB
from src.app.translation.base import SectionTranslationResult
from src.app.translation.cost import CostRecord, gemini_cost
from src.app.translation.splitter import Section


def _make_section(index: int, label: str = "Section") -> Section:
    """Return a minimal Section dataclass.

    Args:
        index: 1-based index of the section.
        label: Human-readable label.

    Returns:
        A frozen Section instance.
    """
    return Section(index=index, label=label, hebrew_text=f"עברית {index}")


def _make_section_result(section: Section) -> SectionTranslationResult:
    """Return a minimal SectionTranslationResult for the given section.

    Args:
        section: The source section to wrap.

    Returns:
        A frozen SectionTranslationResult with plausible dummy values.
    """
    return SectionTranslationResult(
        section_index=section.index,
        section_label=section.label,
        hebrew_text=section.hebrew_text,
        russian_text=f"русский {section.index}",
        cost=CostRecord(input_tokens=100, output_tokens=50, cost_usd=0.001),
        elapsed_sec=0.5,
        was_corrected=False,
        midway_score=0.9,
        midway_errors=[],
    )


def _make_eval_result() -> EvaluationResult:
    """Return a minimal EvaluationResult with no errors.

    Returns:
        An EvaluationResult with perfect score and empty error list.
    """
    return EvaluationResult(quality_score=1.0, errors=[])


# ---------------------------------------------------------------------------
# Production pipeline smoke tests
# ---------------------------------------------------------------------------


def test_production_pipeline_full_success() -> None:
    """Production pipeline stores all sections for each document on full success."""
    db = ProductionDB(Path(":memory:"))
    run_id = db.insert_run("smoke_run_prod", "smoke test", doc_count=2)

    sections_doc1 = [_make_section(i) for i in range(1, 3)]
    sections_doc2 = [_make_section(i) for i in range(1, 3)]

    def section_fn(section: Section) -> SectionTranslationResult:
        return _make_section_result(section)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for doc_id, sections in [("Doc1", sections_doc1), ("Doc2", sections_doc2)]:
            results = run_section_tasks(sections, section_fn, pool, doc_id=doc_id)
            total_cost = CostRecord(
                input_tokens=sum(r.cost.input_tokens or 0 for r in results),
                output_tokens=sum(r.cost.output_tokens or 0 for r in results),
                cost_usd=sum(r.cost.cost_usd for r in results),
            )
            db.insert_result_with_sections(
                run_id=run_id,
                doc_id=doc_id,
                sections=results,
                cost=total_cost,
                elapsed_sec=1.0,
            )

    assert db.result_exists(run_id, "Doc1")
    assert db.result_exists(run_id, "Doc2")


def test_production_pipeline_partial_failure() -> None:
    """Production pipeline stores partial document when one section fails."""
    db = ProductionDB(Path(":memory:"))
    run_id = db.insert_run("smoke_run_partial", "partial test", doc_count=1)

    sections = [_make_section(i) for i in range(1, 4)]

    def section_fn(section: Section) -> SectionTranslationResult:
        if section.index == 2:
            raise RuntimeError("forced section failure")
        return _make_section_result(section)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            results = run_section_tasks(sections, section_fn, pool, doc_id="DocP")
        except PartialDocumentError as e:
            results = e.successful_results
            assert len(e.failed_sections) == 1
            assert e.failed_sections[0][0] == 2  # index of failed section

    assert len(results) == 2  # sections 1 and 3 succeeded

    total_cost = CostRecord(
        input_tokens=sum(r.cost.input_tokens or 0 for r in results),
        output_tokens=sum(r.cost.output_tokens or 0 for r in results),
        cost_usd=sum(r.cost.cost_usd for r in results),
    )
    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="DocP",
        sections=results,
        cost=total_cost,
        elapsed_sec=1.0,
    )

    assert db.result_exists(run_id, "DocP"), "Partial document must still be stored"

    # Verify the DB contains exactly 2 section rows for this result
    result_row = db._conn.execute(
        "SELECT id FROM results WHERE doc_id = ?", ("DocP",)
    ).fetchone()
    assert result_row is not None
    section_rows = db._conn.execute(
        "SELECT COUNT(*) FROM section_results WHERE result_id = ?",
        (result_row["id"],),
    ).fetchone()
    assert section_rows[0] == 2, f"Expected 2 section rows; got {section_rows[0]}"


# ---------------------------------------------------------------------------
# Test pipeline smoke tests
# ---------------------------------------------------------------------------


def test_test_pipeline_full_success() -> None:
    """Test pipeline stores all sections with eval scores for each document."""
    db = BenchmarkDB(Path(":memory:"))
    db.create_tables()
    run_id = db.insert_run("smoke_run_bench", "bench smoke test", doc_count=2)

    sections_doc1 = [_make_section(i) for i in range(1, 3)]
    sections_doc2 = [_make_section(i) for i in range(1, 3)]

    def section_fn(
        section: Section,
    ) -> tuple[SectionTranslationResult, EvaluationResult]:
        return _make_section_result(section), _make_eval_result()

    with ThreadPoolExecutor(max_workers=2) as pool:
        for doc_id, sections in [("Doc1", sections_doc1), ("Doc2", sections_doc2)]:
            results = run_section_tasks(sections, section_fn, pool, doc_id=doc_id)
            section_results = [r for r, _ in results]
            eval_results = [ev for _, ev in results]
            total_cost = CostRecord(
                input_tokens=sum(r.cost.input_tokens or 0 for r in section_results),
                output_tokens=sum(r.cost.output_tokens or 0 for r in section_results),
                cost_usd=sum(r.cost.cost_usd for r in section_results),
            )
            db.insert_result_with_sections(
                run_id=run_id,
                doc_id=doc_id,
                sections=section_results,
                section_eval_results=eval_results,
                cost=total_cost,
                elapsed_sec=1.0,
            )

    assert db.result_exists(run_id, "Doc1")
    assert db.result_exists(run_id, "Doc2")


def test_test_pipeline_partial_failure() -> None:
    """Test pipeline stores partial document with eval scores when one section fails."""
    db = BenchmarkDB(Path(":memory:"))
    db.create_tables()
    run_id = db.insert_run("smoke_run_bench_partial", "bench partial test", doc_count=1)

    sections = [_make_section(i) for i in range(1, 4)]

    def section_fn(
        section: Section,
    ) -> tuple[SectionTranslationResult, EvaluationResult]:
        if section.index == 3:
            raise RuntimeError("forced section failure")
        return _make_section_result(section), _make_eval_result()

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            results = run_section_tasks(sections, section_fn, pool, doc_id="DocQ")
        except PartialDocumentError as e:
            results = e.successful_results
            assert len(e.failed_sections) == 1
            assert e.failed_sections[0][0] == 3  # index of failed section

    assert len(results) == 2  # sections 1 and 2 succeeded

    section_results = [r for r, _ in results]
    eval_results = [ev for _, ev in results]
    total_cost = CostRecord(
        input_tokens=sum(r.cost.input_tokens or 0 for r in section_results),
        output_tokens=sum(r.cost.output_tokens or 0 for r in section_results),
        cost_usd=sum(r.cost.cost_usd for r in section_results),
    )
    db.insert_result_with_sections(
        run_id=run_id,
        doc_id="DocQ",
        sections=section_results,
        section_eval_results=eval_results,
        cost=total_cost,
        elapsed_sec=1.0,
    )

    assert db.result_exists(run_id, "DocQ"), "Partial document must still be stored"

    result_row = db._conn.execute(
        "SELECT id FROM results WHERE doc_id = ?", ("DocQ",)
    ).fetchone()
    assert result_row is not None
    section_rows = db._conn.execute(
        "SELECT COUNT(*) FROM section_results WHERE result_id = ?",
        (result_row["id"],),
    ).fetchone()
    assert section_rows[0] == 2, f"Expected 2 section rows; got {section_rows[0]}"
