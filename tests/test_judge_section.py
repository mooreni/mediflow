"""Checks that score_section() formats its evaluation prompt correctly and returns
an EvaluationResult, and that score_document() sums penalties across all sections
rather than averaging per-section scores. All tests mock the Gemini client at the
boundary (get_eval_client) — no real API calls are made.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.evaluation.judge import EvaluationResult, score_document, score_section

_MOCK_PATH = "src.app.evaluation.judge.get_eval_client"


def _make_client(errors: list[dict]) -> MagicMock:
    """Build a mock GenAI client that returns a fixed errors payload."""
    response = MagicMock()
    response.text = json.dumps({"errors": errors})
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    client._api_client = MagicMock()
    return client


def test_prompt_format():
    """score_section() formats the prompt with section index, label, context header,
    and plain-text hypothesis block; must NOT contain 'JSON array'."""
    client = _make_client([])
    captured_prompts: list[str] = []

    async def capture_generate(model, contents, config):
        captured_prompts.append(contents)
        response = MagicMock()
        response.text = json.dumps({"errors": []})
        return response

    client.aio.models.generate_content = capture_generate

    with patch(_MOCK_PATH, return_value=client):
        score_section(
            section_hebrew="טקסט עברי",
            section_russian="Русский текст",
            context_header="Patient: male. Diagnosis: hypertension.",
            section_index=2,
            total_sections=5,
            section_label="Medications",
        )

    assert captured_prompts, "generate_content was never called"
    prompt = captured_prompts[0]

    assert "2 of 5" in prompt
    assert '"Medications"' in prompt
    assert "Patient: male. Diagnosis: hypertension." in prompt
    assert "HYPOTHESIS (Russian translation):" in prompt
    assert "Русский текст" in prompt
    assert "JSON array" not in prompt


def test_returns_evaluation_result():
    """score_section() returns EvaluationResult with correct score when errors is empty."""
    client = _make_client([])

    with patch(_MOCK_PATH, return_value=client):
        result = score_section(
            section_hebrew="Hebrew",
            section_russian="Russian",
            context_header="Context",
            section_index=1,
            total_sections=1,
            section_label="Summary",
        )

    assert isinstance(result, EvaluationResult)
    assert result.quality_score == 1.0
    assert result.errors == []


def test_score_document_sums_penalties():
    """score_document() uses summed MQM penalties, not an average of per-section scores.

    Section 1: 0 errors  → per-section score 1.00
    Section 2: 1 major   → per-section score 0.95
    Section 3: 2 major   → per-section score 0.90
    Average of section scores would be (1.00 + 0.95 + 0.90) / 3 ≈ 0.950.
    Summed penalties: 3 × 0.05 = 0.15 → correct doc score = 0.85.
    """
    payloads = [
        [],
        [{"span": "a", "category": "accuracy", "severity": "major", "justification": "x"}],
        [
            {"span": "b", "category": "accuracy", "severity": "major", "justification": "y"},
            {"span": "c", "category": "accuracy", "severity": "major", "justification": "z"},
        ],
    ]

    call_count = 0

    def make_client_cycled():
        nonlocal call_count
        client = _make_client(payloads[call_count % len(payloads)])
        call_count += 1
        return client

    with patch(_MOCK_PATH, side_effect=make_client_cycled):
        doc_score, results = score_document(
            sections=[
                ("heb1", "rus1", "Section 1"),
                ("heb2", "rus2", "Section 2"),
                ("heb3", "rus3", "Section 3"),
            ],
            context_header="Context",
        )

    assert len(results) == 3
    # 3 major errors combined → penalty 3 × 0.05 = 0.15 → score 0.85
    assert abs(doc_score - 0.85) < 1e-9


def test_score_document_excludes_failure():
    """score_document() excludes a section that raises and averages the remaining two."""
    call_count = 0

    def make_client_or_fail():
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("Simulated evaluation failure")
        return _make_client([])

    with patch(_MOCK_PATH, side_effect=make_client_or_fail):
        avg, results = score_document(
            sections=[
                ("heb1", "rus1", "Section 1"),
                ("heb2", "rus2", "Section 2"),  # this one fails
                ("heb3", "rus3", "Section 3"),
            ],
            context_header="Context",
        )

    assert len(results) == 2
    assert avg == 1.0  # both surviving sections have no errors


def test_score_document_all_fail():
    """score_document() returns (0.0, []) when every section fails."""
    with patch(_MOCK_PATH, side_effect=ValueError("fail")):
        avg, results = score_document(
            sections=[
                ("heb1", "rus1", "Section 1"),
                ("heb2", "rus2", "Section 2"),
            ],
            context_header="Context",
        )

    assert avg == 0.0
    assert results == []


