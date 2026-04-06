# LLM-as-a-Judge evaluator for MediFlow translation quality assessment.
# Uses MQM/ISO 5060:2024 penalty-based scoring for Hebrew→Russian medical translations.
# Score = max(0.0, 1.0 - sum_of_penalties); severity weights: Critical=0.25, Major=0.05, Minor=0.01
# Incomplete-translation penalty is coverage-proportional: max(0.25, 1.0 - coverage_fraction)

import asyncio
import json
import re
from dataclasses import dataclass

import httpx
from google.genai import errors, types
from google.genai.types import HttpOptions, HttpRetryOptions
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential_jitter

from .vertex_client import MODEL, get_eval_client

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
    """
    total = 0.0
    for e in errs:
        if e.get("span") == "[INCOMPLETE TRANSLATION]":
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

_MQM_PROMPT = """\
You are an expert Medical Translation Quality Evaluator and Clinical Linguist. Your task is to evaluate the accuracy and quality of a target translation against its source text. You will apply an analytical, segment-by-segment evaluation methodology based on the Multidimensional Quality Metrics (MQM) framework and ISO 5060:2024 standards, heavily customized for the rigorous and high-risk demands of the medical, pharmaceutical, and healthcare domains.

Source language: Hebrew
Target language: Russian
Audience: Patient (non-medical reader)

## Context and Stakes
Medical translation errors carry exceptionally high risks, including severe patient harm, misdiagnosis, medication overdoses, and regulatory non-compliance (e.g., EMA, FDA, EU MDR/IVDR mandates). Your evaluation must strictly prioritize clinical accuracy, terminological precision, regulatory compliance, and audience appropriateness over mere stylistic fluency.

## Error Categories
Identify errors belonging only to the following categories:

1. **Accuracy** — the target text distorts, omits, or adds to the propositional content of the source. Look specifically for altered medication dosages, missed contraindications, misunderstood symptoms, or hallucinated clinical data.
2. **Terminology** — the translation fails to use the correct, normative equivalent of a medical term (e.g., violating MedDRA guidelines), uses an incorrect medical abbreviation, or ignores standard regulatory phrasing (e.g., EMA QRD template mandates).
3. **Audience Appropriateness** — the language level is inappropriate for the target audience. For instance, using dense medical jargon in patient-facing materials that should be readable at an 8th-grade level, or conversely, using overly colloquial language in clinical documents.
4. **Linguistic Conventions** — morphosyntactic, grammatical, spelling, or punctuation errors that violate the fundamental rules of the target language.
5. **Locale Conventions** — violations of target-region formatting for numbers, dates, or measurement units (e.g., failure to localize imperial to metric, or incorrect decimal separators in vital signs/lab results).

## Severity Levels
For every error identified, assign a severity level based on the potential clinical and regulatory impact:

- **Critical** (penalty: 0.25) — the error invalidates the translation and causes dangerous misunderstandings leading to potential patient harm or strict regulatory failure. Examples: incorrect dosage numbers, omitted allergy warnings, altered surgical instructions, mistranslated anatomical sites, or failure to translate critical safety warnings.
- **Major** (penalty: 0.05) — the error disrupts reading and misleads the reader, but is not immediately life-threatening. Examples: mistranslation of a non-critical symptom, failure to follow stylistic glossaries, using a non-preferred but understandable medical term.
- **Minor** (penalty: 0.01) — the error does not seriously impede usability or clinical understanding. Examples: minor typos in non-critical text, slight formatting issues, double spaces.

## Deduplication Rule
If the same **source concept** is mistranslated throughout the document, report it as a **single error** with the first occurrence as the span — even if each occurrence produces a different wrong target word. The test is whether the errors share the same root cause (e.g., not knowing the standard Russian term for "embryo transfer"), not whether the wrong words are identical.

Additionally, if two or more errors refer to the **same source sentence** (even if their target spans differ), they count as one error. Do not split a single source sentence's mistranslation into multiple entries — choose the span that best represents the error and assign the highest applicable severity. Errors in different sentences are always counted separately, even if they are adjacent.

## Boilerplate Exclusion
Do **not** flag errors in boilerplate legal or administrative footers (e.g., privacy protection law references, document disclaimer lines, digital signature blocks). These have no clinical impact on patient understanding and should be ignored.

## Hypothesis Format
The hypothesis is provided as a JSON array of aligned sentence pairs. Each element has the form:
{{"source_hebrew": "<original Hebrew sentence>", "translated_russian": "<Russian translation>"}}
Use the per-pair alignment to locate error spans precisely. Evaluate only the "translated_russian" values against their corresponding "source_hebrew" counterparts.

## Coverage Check
Before evaluating quality, count the number of sentence pairs present in the hypothesis JSON array and compare this to the number of sentences in the source document. If the hypothesis covers less than 95% of the source content, you MUST add one additional Critical accuracy error:
  span: "[INCOMPLETE TRANSLATION]"
  justification: "Hypothesis contains approximately N pairs but the source has approximately M sentences — only X% coverage."
Do not flag missing boilerplate footers, but do flag missing clinical content.

## Task
Compare the source and hypothesis segment by segment. For each error, extract the exact span from the hypothesis, classify it using the categories above, assign a severity level, and provide a brief clinical justification.

SOURCE (Hebrew):
{source}

HYPOTHESIS (Russian translation — JSON array of aligned sentence pairs, each with "source_hebrew" and "translated_russian" fields):
{hypothesis}

## Output
If the translation is flawless, return an empty errors array.

Respond in JSON only — no markdown, no explanation outside the JSON:
{{"errors": [{{"span": "<exact text from hypothesis>", "category": "accuracy|terminology|audience|linguistic|locale", "severity": "critical|major|minor", "justification": "<clinical justification>"}}]}}"""


@dataclass
class EvaluationResult:
    """Result of a single MQM evaluation pass.

    Attributes:
        quality_score: Penalty-adjusted score in the range [0.0, 1.0].
        errors: List of MQM error dicts, each with keys: span, category,
                severity, justification.
    """

    quality_score: float   # 0.0-1.0
    errors: list[dict]     # [{span, category, severity, justification}, ...]


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

    # Fall back to regex extraction of a JSON array.
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


async def _score_async(source: str, hypothesis: str) -> EvaluationResult:
    """Evaluate a translation asynchronously with a hard wall-clock timeout.

    Uses the thread-local GenAI client's async interface wrapped in
    asyncio.wait_for() to guarantee cancellation after _EVAL_TIMEOUT_SEC
    seconds — even if the API streams keepalive bytes that prevent httpx's
    per-chunk timeout from firing.

    Args:
        source: The original Hebrew document text.
        hypothesis: The translated Russian text (plain text or JSON sentence pairs).

    Returns:
        An EvaluationResult with quality_score (0.0-1.0) and a list of MQM errors.

    Raises:
        asyncio.TimeoutError: If the API call exceeds _EVAL_TIMEOUT_SEC seconds.
        ValueError: If the model response cannot be parsed as valid JSON.
    """
    prompt = _MQM_PROMPT.format(source=source, hypothesis=hypothesis)
    client = get_eval_client()

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
        # Close the aiohttp session so asyncio.run() can shut down the event
        # loop without hanging on dangling connections.
        await client._api_client.aclose()
        client._api_client._aiohttp_session = None
        raise

    data = _parse_response(response.text)
    errs = data.get("errors", [])
    return EvaluationResult(quality_score=_calc_score(errs), errors=errs)


def _log_eval_retry(state: RetryCallState) -> None:
    """Log evaluation retry attempts so they are visible in benchmark output.

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


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=10, max=120),
    stop=stop_after_attempt(6) | stop_after_delay(600),
    before_sleep=_log_eval_retry,
    reraise=True,
)
def score(source: str, hypothesis: str) -> EvaluationResult:
    """Evaluate a Hebrew→Russian translation using MQM/ISO 5060 penalty scoring.

    Internally runs an async API call with a hard {_EVAL_TIMEOUT_SEC}s
    wall-clock timeout. Hung calls are cleanly cancelled (connection closed,
    resources freed) and retried by the tenacity decorator. Total retry time
    is capped at 10 minutes.

    Args:
        source: The original Hebrew document text.
        hypothesis: The translated Russian text (plain text or JSON sentence pairs).

    Returns:
        An EvaluationResult with quality_score (0.0-1.0) and a list of MQM errors.

    Raises:
        ValueError: If the model response cannot be parsed as valid JSON.
        asyncio.TimeoutError: If all retry attempts exhaust the timeout budget.
        google.genai.errors.ClientError: If the API returns a non-retryable error.
    """
    return asyncio.run(_score_async(source, hypothesis))
