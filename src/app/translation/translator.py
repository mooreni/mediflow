"""Production medical translator: 3-step pipeline per document section.

For each section the pipeline runs:
  1. **Flash translate** — Gemini Flash produces a Hebrew→Russian draft (JSON response).
  2. **Midway judge** — Gemini Pro evaluates the draft via MQM scoring (score_section).
  3. **Pro correct** (conditional) — Gemini Pro post-edits the draft using the MQM errors.
     Skipped entirely when the midway evaluation finds zero errors.

``translate_section()`` is the public entry point. ``_translate_section_flash()`` and
``_correct_section_pro()`` are the private helpers for steps 1 and 3 respectively.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from google.genai import types
from google.genai.types import HttpOptions, HttpRetryOptions

from src.app.translation.cost import CostRecord, gemini_cost, sum_costs
from src.app.translation.base import SectionTranslationResult
from src.app.evaluation import judge
from src.app.clients.vertex import MODEL, get_eval_client, get_translate_client

if TYPE_CHECKING:
    from src.app.translation.splitter import Section

_FLASH_MODEL = "gemini-3-flash-preview"
_PRO_MODEL = MODEL  # "gemini-3.1-pro-preview" from clients/vertex.py

# Hard wall-clock timeouts (seconds) per API call.
# Slightly below the HTTP-level timeout (130 s) so asyncio cancels before
# the transport layer gives up, mirroring the pattern in judge.py.
_TRANSLATE_TIMEOUT_SEC = 120
_CORRECT_TIMEOUT_SEC = 120

_TRANSLATE_PROMPT = """\
You are a professional medical translator. Translate the Hebrew medical document section \
below into Russian.

## Clinical context
{context_header}

## Section metadata
- Label: {section_label}
- Position: {section_position}

## Source text (Hebrew)
{hebrew_text}

## Instructions
- Translate for a patient (non-medical reader). Use plain, accessible Russian.
- Preserve all medical terms, drug names, dosages, dates, diagnoses, and measurements exactly.
- Do not add commentary, footnotes, or markdown formatting.
- Return a JSON object with a single key "russian_translation" whose value is the plain \
Russian translation string.

Example output format:
{{"russian_translation": "..."}}
"""

_CORRECT_PROMPT = """\
You are a professional medical translator and post-editor. Your task is to correct a \
machine-produced Russian translation of a Hebrew medical document section.

## Clinical context
{context_header}

## Section metadata
- Label: {section_label}
- Position: {section_position}

## Hebrew source text
{hebrew_source}

## Machine translation (Russian — may contain errors)
{flash_translation}

## MQM errors identified by the quality evaluator
{midway_errors}

## Post-editing instructions
1. Fix every error listed above. Each error includes the erroneous span, its category, \
severity, and a clinical justification — use all of this to produce the correct wording.
2. Only make additional corrections beyond the listed errors if you are 100% certain \
they are genuine errors that the evaluator missed. When in doubt, leave the original \
wording unchanged.
3. Preserve all correct segments exactly as they appear in the machine translation — \
do not rephrase, reorder, or paraphrase segments that are already correct.
4. The output must be plain Russian text suitable for a patient (non-medical reader). \
Do not add commentary, footnotes, or markdown formatting.
5. Return a JSON object with a single key "corrected_translation" whose value is the \
plain corrected Russian text.

Example output format:
{{"corrected_translation": "..."}}
"""


class MedicalTranslator:
    """Production Hebrew→Russian medical document translator.

    Translates one document section at a time using a 3-step pipeline:
    Flash translate → midway MQM judge → Pro correct (conditional).
    Use ``translate_section()`` from a ``ThreadPoolExecutor`` worker; clients
    are resolved lazily per thread to avoid event-loop binding issues.
    """

    def __init__(
        self,
        translate_client: object | None = None,
        correct_client: object | None = None,
    ) -> None:
        """Initialise the translator, optionally injecting API clients for testing.

        Clients are stored only when explicitly provided (e.g. in unit tests).
        In production, they are resolved lazily inside each worker thread via
        get_translate_client() / get_eval_client() — both of which are thread-local
        factories. Resolving at construction time would bind the async HTTP session
        to the main thread's event loop, causing "bound to a different event loop"
        errors when ThreadPoolExecutor workers call asyncio.run() concurrently.

        Args:
            translate_client: An optional pre-built GenAI client for Flash
                              translation calls. If None, get_translate_client()
                              is called lazily inside each translate call.
            correct_client:   An optional pre-built GenAI client for Pro
                              correction calls. If None, get_eval_client()
                              is called lazily inside each correct call.
        """
        self._translate_client = translate_client  # None → resolve lazily per thread
        self._correct_client = correct_client       # None → resolve lazily per thread

    def _translate_section_flash(
        self,
        context_header: str,
        section: Section,
        total_sections: int,
    ) -> tuple[str, CostRecord, float]:
        """Translate a single document section using Gemini Flash (raw API).

        Calls the Gemini Flash model with a JSON-structured prompt and reads
        token counts from response.usage_metadata to compute cost. Uses
        asyncio.wait_for() with a hard timeout, matching the pattern in
        judge._score_section_async.

        Args:
            context_header: Compact clinical context string (document type,
                            patient sex, primary diagnosis).
            section:        The Section to translate (contains index, label,
                            hebrew_text).
            total_sections: Total number of sections in the document.

        Returns:
            A tuple of (russian_text, cost_record, elapsed_sec) where:
              - russian_text  is the plain Russian translation string,
              - cost_record   is a CostRecord computed with Flash pricing,
              - elapsed_sec   is wall-clock seconds for the API call.

        Raises:
            ValueError:           If the model returns an empty translation or
                                  the JSON response is missing the expected key.
            asyncio.TimeoutError: If the API call exceeds _TRANSLATE_TIMEOUT_SEC.
        """
        prompt = _TRANSLATE_PROMPT.format(
            context_header=context_header,
            section_label=section.label,
            section_position=f"{section.index} of {total_sections}",
            hebrew_text=section.hebrew_text,
        )
        client = self._translate_client if self._translate_client is not None else get_translate_client()

        async def _call() -> object:
            try:
                return await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=_FLASH_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                            # gemini-3-flash-preview enables thinking by default; disable it to
                            # preserve benchmark baseline latency and quality profile.
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            http_options=HttpOptions(
                                # Slightly above the asyncio timeout as defence-in-depth,
                                # same pattern as judge.py.
                                timeout=130_000,
                                retry_options=HttpRetryOptions(attempts=1),
                            ),
                        ),
                    ),
                    timeout=_TRANSLATE_TIMEOUT_SEC,
                )
            except (asyncio.TimeoutError, TimeoutError):
                # Close the aiohttp session so asyncio.run() can shut down the
                # event loop without hanging on dangling connections.
                await client._api_client.aclose()
                client._api_client._aiohttp_session = None
                raise

        start = time.monotonic()
        response = asyncio.run(_call())
        elapsed = time.monotonic() - start

        try:
            data = json.loads(response.text)
            russian_text: str = data["russian_translation"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"Flash translation response for section {section.index} "
                f"('{section.label}') could not be parsed — "
                f"expected JSON with key 'russian_translation', got: {response.text!r}"
            ) from exc

        if not russian_text:
            raise ValueError(
                f"Empty translation from Gemini Flash for section {section.index} "
                f"('{section.label}') — expected a non-empty Russian string."
            )

        input_tokens: int = response.usage_metadata.prompt_token_count
        output_tokens: int = response.usage_metadata.candidates_token_count
        cost = gemini_cost(input_tokens, output_tokens, model="flash")

        return russian_text, cost, elapsed

    def _correct_section_pro(
        self,
        context_header: str,
        section: "Section",
        total_sections: int,
        flash_translation: str,
        midway_errors: list[dict],
    ) -> tuple[str, CostRecord, float]:
        """Post-edit a Flash translation using Gemini Pro based on MQM errors (raw API).

        Calls the Gemini Pro model with a JSON-structured correction prompt and
        reads token counts from response.usage_metadata to compute cost. Uses
        asyncio.wait_for() with a hard timeout, matching the pattern in
        _translate_section_flash.

        Args:
            context_header:    Compact clinical context string (document type,
                               patient sex, primary diagnosis).
            section:           The Section being corrected (contains index, label,
                               hebrew_text).
            total_sections:    Total number of sections in the document.
            flash_translation: The raw Russian translation produced by Flash.
            midway_errors:     List of MQM error dicts from the midway evaluator,
                               each with keys: span, category, severity, justification.

        Returns:
            A tuple of (corrected_text, cost_record, elapsed_sec) where:
              - corrected_text is the post-edited plain Russian string,
              - cost_record    is a CostRecord computed with Pro pricing,
              - elapsed_sec    is wall-clock seconds for the API call.

        Raises:
            ValueError:           If the model returns an empty correction or the
                                  JSON response is missing the expected key.
            asyncio.TimeoutError: If the API call exceeds _CORRECT_TIMEOUT_SEC.
        """
        prompt = _CORRECT_PROMPT.format(
            context_header=context_header,
            section_label=section.label,
            section_position=f"{section.index} of {total_sections}",
            hebrew_source=section.hebrew_text,
            flash_translation=flash_translation,
            midway_errors=json.dumps(midway_errors, ensure_ascii=False, indent=2),
        )
        client = self._correct_client if self._correct_client is not None else get_eval_client()

        async def _call() -> object:
            try:
                return await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=_PRO_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                            http_options=HttpOptions(
                                # Slightly above the asyncio timeout as defence-in-depth,
                                # same pattern as judge.py.
                                timeout=130_000,
                                retry_options=HttpRetryOptions(attempts=1),
                            ),
                        ),
                    ),
                    timeout=_CORRECT_TIMEOUT_SEC,
                )
            except (asyncio.TimeoutError, TimeoutError):
                # Close the aiohttp session so asyncio.run() can shut down the
                # event loop without hanging on dangling connections.
                await client._api_client.aclose()
                client._api_client._aiohttp_session = None
                raise

        start = time.monotonic()
        response = asyncio.run(_call())
        elapsed = time.monotonic() - start

        try:
            data = json.loads(response.text)
            corrected_text: str = data["corrected_translation"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"Pro correction response for section {section.index} "
                f"('{section.label}') could not be parsed — "
                f"expected JSON with key 'corrected_translation', got: {response.text!r}"
            ) from exc

        if not corrected_text:
            raise ValueError(
                f"Empty correction from Gemini Pro for section {section.index} "
                f"('{section.label}') — expected a non-empty Russian string."
            )

        input_tokens: int = response.usage_metadata.prompt_token_count
        output_tokens: int = response.usage_metadata.candidates_token_count
        cost = gemini_cost(input_tokens, output_tokens, model="pro")

        return corrected_text, cost, elapsed

    def translate_section(
        self,
        context_header: str,
        section: "Section",
        total_sections: int,
    ) -> SectionTranslationResult:
        """Translate a single document section using the full translate→eval→correct pipeline.

        Orchestrates three steps:
          1. TRANSLATE: Gemini Flash translates the section to Russian.
          2. MIDWAY_EVAL: Gemini Pro evaluates the Flash translation for MQM errors.
          3. CORRECT (conditional): Gemini Pro post-edits the translation using the
             MQM errors. Skipped if the midway evaluation returns zero errors.

        Args:
            context_header: Compact clinical context string (document type,
                            patient sex, primary diagnosis).
            section:        The Section to translate (contains index, label,
                            hebrew_text).
            total_sections: Total number of sections in the document.

        Returns:
            A SectionTranslationResult with the corrected (or original, if no
            errors were found) Russian translation, combined cost, and combined
            elapsed time for all pipeline steps.

        Raises:
            ValueError:           If any model call returns an empty or
                                  unparseable response.
            asyncio.TimeoutError: If any API call exceeds its hard timeout.
        """
        # Step 1: TRANSLATE
        flash_russian, flash_cost, flash_elapsed = self._translate_section_flash(
            context_header=context_header,
            section=section,
            total_sections=total_sections,
        )

        # Step 2: MIDWAY_EVAL
        midway_result = judge.score_section(
            section_hebrew=section.hebrew_text,
            section_russian=flash_russian,
            context_header=context_header,
            section_index=section.index,
            total_sections=total_sections,
            section_label=section.label,
        )

        # Cost of the midway Pro evaluation call (token counts from EvaluationResult).
        midway_eval_cost = gemini_cost(
            midway_result.prompt_tokens,
            midway_result.completion_tokens,
            model="pro",
        )

        # Step 3: CORRECT — skip entirely when midway eval found no errors;
        # the Flash output is already acceptable and a Pro correction would be a no-op.
        if not midway_result.errors:
            return SectionTranslationResult(
                section_index=section.index,
                section_label=section.label,
                hebrew_text=section.hebrew_text,
                russian_text=flash_russian,
                cost=sum_costs([flash_cost, midway_eval_cost]),
                elapsed_sec=flash_elapsed + midway_result.elapsed_sec,
                was_corrected=False,
                midway_score=midway_result.quality_score,
                midway_errors=midway_result.errors,
            )

        corrected_russian, correct_cost, correct_elapsed = self._correct_section_pro(
            context_header=context_header,
            section=section,
            total_sections=total_sections,
            flash_translation=flash_russian,
            midway_errors=midway_result.errors,
        )

        total_cost = sum_costs([flash_cost, midway_eval_cost, correct_cost])
        total_elapsed = flash_elapsed + midway_result.elapsed_sec + correct_elapsed

        return SectionTranslationResult(
            section_index=section.index,
            section_label=section.label,
            hebrew_text=section.hebrew_text,
            russian_text=corrected_russian,
            cost=total_cost,
            elapsed_sec=total_elapsed,
            was_corrected=True,
            midway_score=midway_result.quality_score,
            midway_errors=midway_result.errors,
        )
