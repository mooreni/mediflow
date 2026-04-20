"""Checks that MedicalTranslator runs the 3-step pipeline (Flash translate →
midway judge → Pro correct), skips the correction step when the midway judge
finds no errors, propagates evaluation failures, and correctly aggregates costs
from all active steps. All tests inject mock clients and patch judge.score_section
— no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.translation.cost import CostRecord, gemini_cost, sum_costs
from src.app.translation.base import SectionTranslationResult
from src.app.translation.translator import MedicalTranslator
from src.app.evaluation.judge import EvaluationResult
from src.app.translation.splitter import Section

_PATCH_SCORE_SECTION = "src.app.evaluation.judge.score_section"

_ONE_ERROR = [
    {
        "span": "неверный термин",
        "category": "terminology",
        "severity": "minor",
        "justification": "wrong term",
    }
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(index: int = 1, label: str = "Medications") -> Section:
    """Return a minimal Section for use in tests.

    Args:
        index: 1-based section index.
        label: Human-readable section label.

    Returns:
        A frozen Section with a short Hebrew text stub.
    """
    return Section(index=index, label=label, hebrew_text='מינון: 500 מ"ג')


def _make_api_response(json_text: str, prompt_tokens: int, output_tokens: int) -> MagicMock:
    """Build a mock Gemini API response object.

    Args:
        json_text:     The JSON string to return as response.text.
        prompt_tokens: Value for usage_metadata.prompt_token_count.
        output_tokens: Value for usage_metadata.candidates_token_count.

    Returns:
        A MagicMock mimicking a Gemini GenerateContentResponse.
    """
    resp = MagicMock()
    resp.text = json_text
    resp.usage_metadata.prompt_token_count = prompt_tokens
    resp.usage_metadata.candidates_token_count = output_tokens
    return resp


def _make_translate_client(
    russian_text: str = "Дозировка: 500 мг",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Return a mock GenAI client configured for Flash translation responses.

    Args:
        russian_text:  The translation string to embed in the JSON response.
        prompt_tokens: Simulated prompt token count.
        output_tokens: Simulated completion token count.

    Returns:
        A MagicMock whose .aio.models.generate_content is an AsyncMock returning
        a response with {"russian_translation": russian_text}.
    """
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=_make_api_response(
            json.dumps({"russian_translation": russian_text}),
            prompt_tokens,
            output_tokens,
        )
    )
    return client


def _make_correct_client(
    corrected_text: str = "Дозировка: 500 мг (исправлено)",
    prompt_tokens: int = 200,
    output_tokens: int = 60,
) -> MagicMock:
    """Return a mock GenAI client configured for Pro correction responses.

    Args:
        corrected_text: The post-edited text to embed in the JSON response.
        prompt_tokens:  Simulated prompt token count.
        output_tokens:  Simulated completion token count.

    Returns:
        A MagicMock whose .aio.models.generate_content is an AsyncMock returning
        a response with {"corrected_translation": corrected_text}.
    """
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=_make_api_response(
            json.dumps({"corrected_translation": corrected_text}),
            prompt_tokens,
            output_tokens,
        )
    )
    return client


def _make_eval_result(
    errors: list[dict] | None = None,
    score: float = 0.99,
    prompt_tokens: int = 150,
    completion_tokens: int = 80,
) -> EvaluationResult:
    """Return an EvaluationResult for use as a score_section mock return value.

    Args:
        errors:            MQM error list; defaults to _ONE_ERROR if None.
        score:             quality_score value.
        prompt_tokens:     Simulated prompt token count from the eval call.
        completion_tokens: Simulated completion token count from the eval call.

    Returns:
        An EvaluationResult with the specified fields.
    """
    return EvaluationResult(
        quality_score=score,
        errors=errors if errors is not None else _ONE_ERROR,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# translate_section() — happy path (errors found → correction runs)
# ---------------------------------------------------------------------------


def test_translate_section_happy_path():
    """translate_section() returns corrected text when midway eval finds errors.

    Verifies that:
    - The returned russian_text is from the correction step, not the Flash step.
    - Both translate and correct API clients are called exactly once.
    - The result is a SectionTranslationResult with the correct section metadata.
    """
    flash_text = "Первый перевод"
    corrected_text = "Исправленный перевод"

    translate_client = _make_translate_client(russian_text=flash_text)
    correct_client = _make_correct_client(corrected_text=corrected_text)
    eval_result = _make_eval_result(errors=_ONE_ERROR)

    translator = MedicalTranslator(
        translate_client=translate_client,
        correct_client=correct_client,
    )
    section = _make_section(index=2, label="Diagnoses")

    with patch(_PATCH_SCORE_SECTION, return_value=eval_result):
        result = translator.translate_section("ctx_header", section, total_sections=5)

    assert isinstance(result, SectionTranslationResult)
    assert result.russian_text == corrected_text
    assert result.section_index == 2
    assert result.section_label == "Diagnoses"
    assert result.hebrew_text == section.hebrew_text
    assert result.elapsed_sec >= 0.0
    assert isinstance(result.cost, CostRecord)
    translate_client.aio.models.generate_content.assert_called_once()
    correct_client.aio.models.generate_content.assert_called_once()


# ---------------------------------------------------------------------------
# translate_section() — skip correction on perfect midway score
# ---------------------------------------------------------------------------


def test_translate_section_skip_correction_on_perfect_midway():
    """Correction step is skipped when midway eval returns zero errors.

    When EvaluationResult.errors is empty the translator must return the Flash
    translation unchanged and must NOT call the correction client.
    """
    flash_text = "Дозировка: 500 мг"
    translate_client = _make_translate_client(russian_text=flash_text)
    correct_client = MagicMock()
    eval_result = _make_eval_result(errors=[], score=1.0)

    translator = MedicalTranslator(
        translate_client=translate_client,
        correct_client=correct_client,
    )

    with patch(_PATCH_SCORE_SECTION, return_value=eval_result):
        result = translator.translate_section("ctx", _make_section(), total_sections=2)

    assert result.russian_text == flash_text
    correct_client.aio.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# translate_section() — midway eval failure propagates
# ---------------------------------------------------------------------------


def test_translate_section_midway_eval_failure_propagates():
    """An exception from score_section propagates so the runner can retry.

    The translator must not swallow evaluation errors — the caller (runner) is
    responsible for retrying the whole document.
    """
    translate_client = _make_translate_client()
    correct_client = MagicMock()

    translator = MedicalTranslator(
        translate_client=translate_client,
        correct_client=correct_client,
    )

    with patch(_PATCH_SCORE_SECTION, side_effect=ValueError("Eval failed")):
        with pytest.raises(ValueError, match="Eval failed"):
            translator.translate_section("ctx", _make_section(), total_sections=1)


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def test_cost_sums_all_three_steps():
    """CostRecord reflects the combined cost of Flash translate + Pro eval + Pro correct.

    Uses known token counts for each step and verifies the aggregated cost
    matches the expected sum of gemini_cost() calls.
    """
    flash_in, flash_out = 100, 50
    eval_in, eval_out = 150, 60
    correct_in, correct_out = 200, 80

    translate_client = _make_translate_client(prompt_tokens=flash_in, output_tokens=flash_out)
    correct_client = _make_correct_client(prompt_tokens=correct_in, output_tokens=correct_out)
    eval_result = _make_eval_result(
        errors=_ONE_ERROR, prompt_tokens=eval_in, completion_tokens=eval_out
    )

    translator = MedicalTranslator(
        translate_client=translate_client,
        correct_client=correct_client,
    )

    with patch(_PATCH_SCORE_SECTION, return_value=eval_result):
        result = translator.translate_section("ctx", _make_section(), total_sections=3)

    expected = sum_costs(
        [
            gemini_cost(flash_in, flash_out, model="flash"),
            gemini_cost(eval_in, eval_out, model="pro"),
            gemini_cost(correct_in, correct_out, model="pro"),
        ]
    )
    assert result.cost.cost_usd == pytest.approx(expected.cost_usd, rel=1e-9)
    assert result.cost.input_tokens == flash_in + eval_in + correct_in
    assert result.cost.output_tokens == flash_out + eval_out + correct_out


def test_cost_two_steps_when_no_errors():
    """When midway eval finds no errors only Flash + eval costs are included.

    The correction step is skipped so its cost must not appear in the result.
    """
    flash_in, flash_out = 100, 50
    eval_in, eval_out = 150, 60

    translate_client = _make_translate_client(prompt_tokens=flash_in, output_tokens=flash_out)
    correct_client = MagicMock()
    eval_result = _make_eval_result(
        errors=[], score=1.0, prompt_tokens=eval_in, completion_tokens=eval_out
    )

    translator = MedicalTranslator(
        translate_client=translate_client,
        correct_client=correct_client,
    )

    with patch(_PATCH_SCORE_SECTION, return_value=eval_result):
        result = translator.translate_section("ctx", _make_section(), total_sections=2)

    expected = sum_costs(
        [
            gemini_cost(flash_in, flash_out, model="flash"),
            gemini_cost(eval_in, eval_out, model="pro"),
        ]
    )
    assert result.cost.cost_usd == pytest.approx(expected.cost_usd, rel=1e-9)
    assert result.cost.input_tokens == flash_in + eval_in
    assert result.cost.output_tokens == flash_out + eval_out
