"""Checks that split_document() correctly parses the model's JSON response into
sections, validates that section indices are contiguous, rejects malformed or
invalid responses, and builds the context header from the parsed fields. All tests
mock the Gemini client at the boundary (get_translate_client) — no real API calls
are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.app.translation.splitter import Section, SplitResult, split_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_mock(response_text: str) -> MagicMock:
    """Return a mock get_translate_client() result that yields response_text.

    Args:
        response_text: The string the mock response's .text attribute returns.

    Returns:
        A MagicMock whose .models.generate_content() returns a mock response
        with .text set to response_text.
    """
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def _well_formed_payload(
    doc_desc: str = "Hospital discharge summary",
    patient_sex: str = "male",
    primary_diagnosis: str | None = "Type 2 diabetes",
    num_sections: int = 2,
) -> str:
    """Return a valid JSON payload string accepted by split_document().

    Args:
        doc_desc:          Human-readable document description for the context block.
        patient_sex:       Patient sex string (e.g. "male", "female", "unknown").
        primary_diagnosis: Diagnosis string, or None to omit from the context block.
        num_sections:      Number of sequential sections to include in the payload.

    Returns:
        A JSON-encoded string with the structure expected by split_document().
    """
    sections = [
        {"index": i + 1, "label": f"Section {i + 1}", "hebrew_text": f"טקסט {i + 1}"}
        for i in range(num_sections)
    ]
    return json.dumps(
        {
            "context": {
                "document_description": doc_desc,
                "patient_sex": patient_sex,
                "primary_diagnosis": primary_diagnosis,
            },
            "total_sections": num_sections,
            "sections": sections,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_split_document_well_formed_returns_split_result():
    """Well-formed JSON response produces a correctly populated SplitResult."""
    payload = _well_formed_payload(num_sections=2)
    mock_client = _make_client_mock(payload)

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        result = split_document("שלום עולם")

    assert isinstance(result, SplitResult)
    assert result.total_sections == 2
    assert len(result.sections) == 2
    assert result.sections[0].index == 1
    assert result.sections[1].index == 2
    assert result.sections[0].hebrew_text == "טקסט 1"
    assert isinstance(result.sections[0], Section)


def test_split_document_mismatched_total_sections_raises_value_error():
    """Declared total_sections != actual section count raises ValueError."""
    data = json.loads(_well_formed_payload(num_sections=2))
    # Claim 3 but only provide 2.
    data["total_sections"] = 3
    mock_client = _make_client_mock(json.dumps(data))

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        with pytest.raises(ValueError, match="total_sections=3"):
            split_document("שלום עולם")


def test_split_document_gap_in_indices_raises_value_error():
    """Non-contiguous section indices (e.g. 1, 3) raise ValueError."""
    data = {
        "context": {
            "document_description": "test",
            "patient_sex": "female",
            "primary_diagnosis": None,
        },
        "total_sections": 2,
        "sections": [
            {"index": 1, "label": "A", "hebrew_text": "טקסט א"},
            # Index 2 is missing; index 3 is present instead.
            {"index": 3, "label": "C", "hebrew_text": "טקסט ג"},
        ],
    }
    mock_client = _make_client_mock(json.dumps(data))

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        with pytest.raises(ValueError, match="not contiguous"):
            split_document("שלום עולם")


def test_split_document_empty_hebrew_text_raises_value_error():
    """A section with empty hebrew_text raises ValueError."""
    data = json.loads(_well_formed_payload(num_sections=2))
    # Blank out the second section's text.
    data["sections"][1]["hebrew_text"] = "   "
    mock_client = _make_client_mock(json.dumps(data))

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        with pytest.raises(ValueError, match="empty hebrew_text"):
            split_document("שלום עולם")


def test_split_document_invalid_json_raises_value_error():
    """A non-JSON response raises ValueError with helpful context."""
    mock_client = _make_client_mock("this is not json {{{")

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        with pytest.raises(ValueError, match="invalid JSON"):
            split_document("שלום עולם")


def test_split_document_context_header_includes_all_fields():
    """context_header contains document description, patient sex, and diagnosis."""
    payload = _well_formed_payload(
        doc_desc="Referral letter",
        patient_sex="female",
        primary_diagnosis="Hypertension",
        num_sections=1,
    )
    mock_client = _make_client_mock(payload)

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        result = split_document("שלום עולם")

    assert "Referral letter" in result.context_header
    assert "female" in result.context_header
    assert "Hypertension" in result.context_header


def test_split_document_context_header_omits_null_diagnosis():
    """When primary_diagnosis is null, the diagnosis segment is absent."""
    payload = _well_formed_payload(
        doc_desc="Lab results",
        patient_sex="male",
        primary_diagnosis=None,
        num_sections=1,
    )
    mock_client = _make_client_mock(payload)

    with patch("src.app.translation.splitter.get_translate_client", return_value=mock_client):
        result = split_document("שלום עולם")

    assert "Diagnosis" not in result.context_header
    assert "Lab results" in result.context_header
    assert "male" in result.context_header
