"""Tests for run_section_tasks and PartialDocumentError in scripts/run.py."""

from __future__ import annotations

import os
import threading

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.run import PartialDocumentError, run_section_tasks
from src.app.translation.splitter import Section


def _make_section(index: int, label: str = "Section") -> Section:
    """Return a minimal Section for use in tests.

    Args:
        index: 1-based section index.
        label: Human-readable label for the section.

    Returns:
        A frozen Section dataclass instance.
    """
    return Section(index=index, label=label, hebrew_text="text")


def test_all_sections_succeed_returns_results() -> None:
    """run_section_tasks returns results in original section order on full success."""
    sections = [_make_section(i) for i in range(1, 4)]

    def section_fn(section: Section) -> str:
        return f"result_{section.index}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = run_section_tasks(sections, section_fn, pool, doc_id="doc1")

    assert results == ["result_1", "result_2", "result_3"]


def test_one_section_fails_raises_partial_document_error() -> None:
    """run_section_tasks raises PartialDocumentError when one section raises."""
    sections = [_make_section(i) for i in range(1, 4)]

    def section_fn(section: Section) -> str:
        if section.index == 2:
            raise ValueError("section 2 failed")
        return f"result_{section.index}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(PartialDocumentError):
            run_section_tasks(sections, section_fn, pool, doc_id="doc1")


def test_partial_error_carries_successful_results() -> None:
    """PartialDocumentError.successful_results contains only the non-failed sections."""
    sections = [_make_section(i) for i in range(1, 4)]

    def section_fn(section: Section) -> str:
        if section.index == 2:
            raise ValueError("section 2 failed")
        return f"result_{section.index}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(PartialDocumentError) as exc_info:
            run_section_tasks(sections, section_fn, pool, doc_id="doc1")

    assert exc_info.value.successful_results == ["result_1", "result_3"]


def test_partial_error_carries_failed_sections_list() -> None:
    """PartialDocumentError.failed_sections has (index, label, exc) tuples."""
    sections = [_make_section(2, label="Medications")]

    boom = RuntimeError("boom")

    def section_fn(section: Section) -> str:
        raise boom

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(PartialDocumentError) as exc_info:
            run_section_tasks(sections, section_fn, pool, doc_id="doc_x")

    failed = exc_info.value.failed_sections
    assert len(failed) == 1
    idx, label, exc = failed[0]
    assert idx == 2
    assert label == "Medications"
    assert exc is boom


def test_all_sections_fail_raises_with_empty_successful_results() -> None:
    """PartialDocumentError.successful_results is empty when all sections fail."""
    sections = [_make_section(i) for i in range(1, 3)]

    def section_fn(section: Section) -> str:
        raise RuntimeError("always fails")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(PartialDocumentError) as exc_info:
            run_section_tasks(sections, section_fn, pool, doc_id="all_fail")

    assert exc_info.value.successful_results == []
    assert len(exc_info.value.failed_sections) == 2
