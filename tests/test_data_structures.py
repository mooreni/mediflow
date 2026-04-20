"""Checks that the pipeline's data containers (Section, SplitResult,
SectionTranslationResult) are truly immutable, and that cost arithmetic
(gemini_cost, sum_costs) produces correct USD totals.
"""

from __future__ import annotations

import pytest

from src.app.translation.cost import CostRecord, gemini_cost, sum_costs
from src.app.data.loader import DatasetDoc
from src.app.translation.base import SectionTranslationResult, TranslationResult
from src.app.translation.splitter import Section, SplitResult


# ---------------------------------------------------------------------------
# Section and SplitResult
# ---------------------------------------------------------------------------

def test_section_and_split_result_fields_accessible():
    section = Section(index=1, label="Diagnosis", hebrew_text="טקסט")
    result = SplitResult(
        context_header="summary",
        total_sections=1,
        sections=[section],
    )

    assert section.index == 1
    assert section.label == "Diagnosis"
    assert section.hebrew_text == "טקסט"
    assert result.context_header == "summary"
    assert result.total_sections == 1
    assert result.sections == [section]


def test_section_is_frozen():
    section = Section(index=1, label="A", hebrew_text="b")
    with pytest.raises(Exception):
        section.index = 2  # type: ignore[misc]


def test_split_result_is_frozen():
    result = SplitResult(context_header="h", total_sections=0, sections=[])
    with pytest.raises(Exception):
        result.total_sections = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SectionTranslationResult
# ---------------------------------------------------------------------------

def test_section_translation_result_fields_accessible():
    cost = CostRecord(input_tokens=10, output_tokens=5, cost_usd=0.001)
    result = SectionTranslationResult(
        section_index=2,
        section_label="Medications",
        hebrew_text="תרופות",
        russian_text="Лекарства",
        cost=cost,
        elapsed_sec=1.5,
        was_corrected=False,
        midway_score=0.9,
        midway_errors=[],
    )

    assert result.section_index == 2
    assert result.section_label == "Medications"
    assert result.hebrew_text == "תרופות"
    assert result.russian_text == "Лекарства"
    assert result.cost is cost
    assert result.elapsed_sec == 1.5


def test_section_translation_result_is_frozen():
    cost = CostRecord(input_tokens=1, output_tokens=1, cost_usd=0.0)
    result = SectionTranslationResult(
        section_index=1, section_label="x", hebrew_text="a",
        russian_text="b", cost=cost, elapsed_sec=0.0,
        was_corrected=False, midway_score=1.0, midway_errors=[],
    )
    with pytest.raises(Exception):
        result.section_index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# sum_costs
# ---------------------------------------------------------------------------

def test_sum_costs_two_token_records():
    a = gemini_cost(100, 50, model="flash")
    b = gemini_cost(200, 80, model="flash")
    total = sum_costs([a, b])

    assert total.input_tokens == 300
    assert total.output_tokens == 130
    assert abs(total.cost_usd - (a.cost_usd + b.cost_usd)) < 1e-9


def test_sum_costs_empty_list_returns_zero_record():
    total = sum_costs([])

    assert total.input_tokens is None
    assert total.output_tokens is None
    assert total.cost_usd == 0.0
