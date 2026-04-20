"""MQM/ISO 5060:2024 LLM-as-a-Judge evaluator for Hebrew→Russian medical translations.

Scoring model:
  - Five error categories: accuracy, terminology, audience, linguistic, locale
  - Three severity levels with penalty weights: critical=0.25, major=0.05, minor=0.01
  - Document score = max(0.0, 1.0 - sum_of_penalties) over all section errors combined
  - Incomplete-translation penalty is coverage-proportional:
      max(0.25, 1.0 - coverage_fraction) — ensures 1% coverage scores far lower than 70%

Public API:
  score_section()  — evaluate one section with clinical context
  score_document() — evaluate all sections and aggregate penalties
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx
from google.genai import errors, types
from google.genai.types import HttpOptions, HttpRetryOptions
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential_jitter

from src.app.clients.vertex import MODEL, get_eval_client

# MQM penalty deductions per error: critical errors can invalidate a translation outright,
# major errors mislead the reader, minor errors have low clinical impact.
_SEVERITY_PENALTY = {"critical": 0.25, "major": 0.05, "minor": 0.01}
_EVAL_TIMEOUT_SEC = 240  # hard wall-clock timeout per LLM evaluation call

_COVERAGE_RE = re.compile(r"only\s+(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def _calc_score(errs: list[dict]) -> float:
    """Sum MQM penalties and return a clamped quality score in [0.0, 1.0].

    For ``[INCOMPLETE TRANSLATION]`` errors the flat Critical penalty (0.25)
    is replaced by a coverage-proportional penalty:
    ``max(0.25, 1.0 - coverage_fraction)``.  This ensures that a document
    translated at 1% coverage scores far lower than one at 70% coverage,
    whereas the old flat penalty gave both the same 0.25 deduction.

    All other errors use the standard severity weights from ``_SEVERITY_PENALTY``.

    The LLM occasionally returns error entries wrapped in an extra
    ``{"errors": [...]}`` envelope.  These are flattened before scoring so
    that nested errors receive their real severity penalty instead of zero.

    Args:
        errs: List of MQM error dicts, each expected to have at minimum a
              ``severity`` key.  Nested ``{"errors": [...]}`` wrappers are
              unwrapped automatically.

    Returns:
        A float in ``[0.0, 1.0]``: ``max(0.0, 1.0 - total_penalty)``.
    """
    # Unwrap nested {"errors": [...]} entries the LLM sometimes produces.
    flat: list[dict] = []
    for e in errs:
        if "errors" in e and isinstance(e["errors"], list):
            flat.extend(e["errors"])
        else:
            flat.append(e)

    total = 0.0
    for e in flat:
        if e.get("span") == "[INCOMPLETE TRANSLATION]":
            # Coverage-proportional penalty: partial translations with very low
            # coverage (e.g. 10%) should score near 0, not just lose 0.25.
            m = _COVERAGE_RE.search(e.get("justification", ""))
            if m:
                coverage_pct = float(m.group(1))
                missing = 1.0 - coverage_pct / 100.0
                total += max(_SEVERITY_PENALTY["critical"], missing)
            else:
                total += _SEVERITY_PENALTY["critical"]
        else:
            total += _SEVERITY_PENALTY.get(e.get("severity", "").lower(), 0.0)
    return max(0.0, 1.0 - total)


_MQM_SECTION_PROMPT = """
{{
  "instruction_metadata": {{
    "role": "You are an expert Medical Translation Quality Evaluator and Clinical Linguist.",
    "task_overview": "Your task is to evaluate the accuracy and quality of a target translation against its source text. You will apply an analytical, segment-by-segment evaluation methodology based on the Multidimensional Quality Metrics (MQM) framework and ISO 5060:2024 standards, heavily customized for the rigorous and high-risk demands of the medical, pharmaceutical, and healthcare domains.",
    "context_and_stakes": "Medical translation errors carry exceptionally high risks, including severe patient harm, misdiagnosis, medication overdoses, and regulatory non-compliance (e.g., EMA, FDA, EU MDR/IVDR mandates). Your evaluation must strictly prioritize clinical accuracy, terminological precision, regulatory compliance, and audience appropriateness over mere stylistic fluency."
  }},
  "translation_parameters": {{
    "source_language": "Source language: Hebrew",
    "target_language": "Target language: Russian",
    "audience": "Audience: Patient (non-medical reader)"
  }},
  "evaluation_criteria": {{
    "error_categories": {{
      "accuracy": "The target text distorts, omits, or adds to the propositional content of the source. Look specifically for altered medication dosages, missed contraindications, misunderstood symptoms, or hallucinated clinical data.",
      "terminology": "The translation fails to use the correct, normative equivalent of a medical term (e.g., violating MedDRA guidelines), uses an incorrect medical abbreviation, or ignores standard regulatory phrasing (e.g., EMA QRD template mandates).",
      "audience_appropriateness": "The language level is inappropriate for the target audience. For instance, using dense medical jargon in patient-facing materials that should be readable at an 8th-grade level, or conversely, using overly colloquial language in clinical documents.",
      "linguistic_conventions": "Morphosyntactic, grammatical, spelling, or punctuation errors that violate the fundamental rules of the target language.",
      "locale_conventions": "Violations of target-region formatting for numbers, dates, or measurement units (e.g., failure to localize imperial to metric, or incorrect decimal separators in vital signs/lab results)."
    }},
    "severity_levels": {{
      "critical": "Critical (penalty: 0.25) — the error invalidates the translation and causes dangerous misunderstandings leading to potential patient harm or strict regulatory failure. Examples: incorrect dosage numbers, omitted allergy warnings, altered surgical instructions, mistranslated anatomical sites, or failure to translate critical safety warnings.",
      "major": "Major (penalty: 0.05) — the error disrupts reading and misleads the reader, but is not immediately life-threatening. Examples: mistranslation of a non-critical symptom, failure to follow stylistic glossaries, using a non-preferred but understandable medical term.",
      "minor": "Minor (penalty: 0.01) — the error does not seriously impede usability or clinical understanding. Examples: minor typos in non-critical text, slight formatting issues, double spaces."
    }}
  }},
  "operational_rules": {{
    "boilerplate_exclusion": "Do not flag errors in boilerplate legal or administrative footers (e.g., privacy protection law references, document disclaimer lines, digital signature blocks). These have no clinical impact on patient understanding and should be ignored.",
    "hypothesis_format": "The hypothesis is provided as plain Russian text — the direct translation of the Hebrew source section. Evaluate the hypothesis holistically against the source section.",
    "task_execution": "Compare the source and hypothesis segment by segment. For each error, extract the exact span from the hypothesis, classify it using the categories above, assign a severity level, and provide a brief clinical justification."
  }},
  "input_payload": {{
    "clinical_context": "CLINICAL CONTEXT: {context_header}",
    "source_text": "SOURCE (Hebrew - section {index} of {total}, \"{label}\"): {section_hebrew}",
    "hypothesis_text": "HYPOTHESIS (Russian translation): {section_russian}"
  }},
  "output_specification": {{
    "instruction": "If the translation is flawless, return an empty errors array. Respond in JSON only — no markdown, no explanation outside the JSON:",
    "format_template": "{{\\"errors\\": [{{\\"span\\": \\"<exact text from hypothesis>\\", \\"category\\": \\"accuracy|terminology|audience|linguistic|locale\\", \\"severity\\": \\"critical|major|minor\\", \\"justification\\": \\"<clinical justification>\\"}}]}}"
  }}
}}
"""


@dataclass
class EvaluationResult:
    """Result of a single MQM evaluation pass.

    Attributes:
        quality_score:     Penalty-adjusted score in the range [0.0, 1.0].
        errors:            List of MQM error dicts, each with keys: span, category,
                           severity, justification.
        prompt_tokens:     Input token count from the evaluation API call (0 if unavailable).
        completion_tokens: Output token count from the evaluation API call (0 if unavailable).
    """

    quality_score: float   # 0.0-1.0
    errors: list[dict]     # [{span, category, severity, justification}, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0


def _parse_response(text: str) -> dict:
    """Extract and parse JSON from the model response text.

    Handles two response shapes:
    - Expected: {"errors": [...]}
    - Fallback: a bare JSON array [...] returned by some model versions

    Args:
        text: Raw response text from Gemini.

    Returns:
        Parsed JSON dict with an "errors" key.

    Raises:
        ValueError: If no valid JSON can be extracted from the response.
    """
    stripped = text.strip()

    # Try bare json.loads first — handles both {"errors":[...]} and [...] shapes.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return {"errors": parsed}
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to regex extraction of a JSON object.
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fall back to regex extraction of a JSON array (some model versions omit the wrapper).
    match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return {"errors": parsed}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model response:\n{text}")


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry attempt.

    Args:
        exc: The exception raised by the API call.

    Returns:
        True if the exception is a 429 rate-limit error, an httpx read
        timeout, or an asyncio/OS timeout; False otherwise.
    """
    if isinstance(exc, errors.ClientError) and exc.code == 429:
        return True
    if isinstance(exc, httpx.ReadTimeout):
        return True
    # asyncio.TimeoutError is a subclass of TimeoutError in Python 3.11+;
    # in 3.10 it is a distinct subclass. Checking both covers all versions.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return False


async def _score_section_async(
    section_hebrew: str,
    section_russian: str,
    context_header: str,
    section_index: int,
    total_sections: int,
    section_label: str,
    client: object,
) -> EvaluationResult:
    """Evaluate a single section translation asynchronously with a hard wall-clock timeout.

    Uses the provided GenAI client's async interface wrapped in
    asyncio.wait_for() to guarantee cancellation after _EVAL_TIMEOUT_SEC
    seconds — even if the API streams keepalive bytes that prevent httpx's
    per-chunk timeout from firing.

    Args:
        section_hebrew:  The original Hebrew text for this section.
        section_russian: The translated Russian text for this section (plain text).
        context_header:  Compact clinical context string (description, patient sex,
                         primary diagnosis).
        section_index:   1-based index of this section within the document.
        total_sections:  Total number of sections in the document.
        section_label:   Human-readable label for this section (e.g. "Medications").
        client:          An initialised GenAI client instance; resolved by the caller
                         so that I/O stays out of this business-logic function.

    Returns:
        An EvaluationResult with quality_score (0.0-1.0) and a list of MQM errors.

    Raises:
        asyncio.TimeoutError: If the API call exceeds _EVAL_TIMEOUT_SEC seconds.
        ValueError: If the model response cannot be parsed as valid JSON.
    """
    prompt = _MQM_SECTION_PROMPT.format(
        context_header=context_header,
        index=section_index,
        total=total_sections,
        label=section_label,
        section_hebrew=section_hebrew,
        section_russian=section_russian,
    )

    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    http_options=HttpOptions(
                        timeout=250_000,  # ms; slightly above async timeout as defence-in-depth
                        retry_options=HttpRetryOptions(attempts=1),  # disable SDK retries; decorator handles backoff
                    ),
                ),
            ),
            timeout=_EVAL_TIMEOUT_SEC,
        )
    except (asyncio.TimeoutError, TimeoutError):
        await client._api_client.aclose()
        client._api_client._aiohttp_session = None
        raise
    elapsed = time.monotonic() - start

    data = _parse_response(response.text)
    errs = data.get("errors", [])
    prompt_tokens = response.usage_metadata.prompt_token_count or 0
    completion_tokens = response.usage_metadata.candidates_token_count or 0
    return EvaluationResult(
        quality_score=_calc_score(errs),
        errors=errs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_sec=elapsed,
    )


def _log_eval_retry(state: RetryCallState) -> None:
    """Log evaluation retry attempts so they are visible in pipeline output.

    Args:
        state: The tenacity RetryCallState object containing attempt number,
               the outcome exception, and the next scheduled wait duration.

    Returns:
        None
    """
    attempt = state.attempt_number
    exc = state.outcome.exception() if state.outcome else None
    wait = state.next_action.sleep if state.next_action else 0
    print(
        f"  [EVAL RETRY {attempt}] {type(exc).__name__}: {exc} "
        f"— waiting {wait:.0f}s before next attempt"
    )


# Retry up to 6 attempts OR 10 minutes total (whichever limit is reached first).
# Exponential backoff with jitter (10s–120s) avoids thundering-herd on 429s.
@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=10, max=120),
    stop=stop_after_attempt(6) | stop_after_delay(600),
    before_sleep=_log_eval_retry,
    reraise=True,
)
def score_section(
    section_hebrew: str,
    section_russian: str,
    context_header: str,
    section_index: int,
    total_sections: int,
    section_label: str,
) -> EvaluationResult:
    """Evaluate a single Hebrew→Russian section translation using MQM/ISO 5060 penalty scoring.

    Internally runs an async API call with a hard wall-clock timeout. Hung
    calls are cleanly cancelled and retried by the tenacity decorator. Total
    retry time is capped at 10 minutes.

    Args:
        section_hebrew:  The original Hebrew text for this section.
        section_russian: The translated Russian text for this section (plain text).
        context_header:  Compact clinical context string (description, patient sex,
                         primary diagnosis).
        section_index:   1-based index of this section within the document.
        total_sections:  Total number of sections in the document.
        section_label:   Human-readable label for this section (e.g. "Medications").

    Returns:
        An EvaluationResult with quality_score (0.0-1.0) and a list of MQM errors.

    Raises:
        ValueError: If the model response cannot be parsed as valid JSON.
        asyncio.TimeoutError: If all retry attempts exhaust the timeout budget.
        google.genai.errors.ClientError: If the API returns a non-retryable error.
    """
    client = get_eval_client()
    return asyncio.run(
        _score_section_async(
            section_hebrew=section_hebrew,
            section_russian=section_russian,
            context_header=context_header,
            section_index=section_index,
            total_sections=total_sections,
            section_label=section_label,
            client=client,
        )
    )


def score_document(
    sections: list[tuple[str, str, str]],
    context_header: str,
) -> tuple[float, list[EvaluationResult]]:
    """Evaluate all sections of a document and return a penalty-based quality score.

    Calls score_section() sequentially for each section tuple. If a section's
    evaluation fails after all retries, a warning is printed and that section
    is excluded from scoring.

    The document-level score is computed by collecting all errors from all
    successful sections and passing them to _calc_score() as a single list.
    This preserves MQM's additive penalty semantics: three major errors across
    three sections produce the same penalty deduction as three major errors in
    one section.  Averaging per-section scores would dilute penalties and
    inflate the document score.

    Args:
        sections:       List of (section_hebrew, section_russian, section_label)
                        tuples in document order.
        context_header: Compact clinical context string shared across all sections.

    Returns:
        A tuple of (doc_quality_score, list_of_per_section_results).
        doc_quality_score is the penalty-based score over all combined errors,
        or 0.0 if all sections fail.
        list_of_per_section_results contains only the successful EvaluationResults.
    """
    total_sections = len(sections)
    results: list[EvaluationResult] = []

    for section_index, (section_hebrew, section_russian, section_label) in enumerate(sections, start=1):
        try:
            result = score_section(
                section_hebrew=section_hebrew,
                section_russian=section_russian,
                context_header=context_header,
                section_index=section_index,
                total_sections=total_sections,
                section_label=section_label,
            )
            results.append(result)
        except (ValueError, asyncio.TimeoutError, TimeoutError, errors.ClientError) as exc:
            print(
                f"  [EVAL WARNING] Section {section_index}/{total_sections} "
                f'("{section_label}") failed after retries: {type(exc).__name__}: {exc} '
                f"— excluding from document score."
            )

    if not results:
        return 0.0, []

    all_errors = [e for r in results for e in r.errors]
    doc_score = _calc_score(all_errors)
    return doc_score, results
