# LLM-as-a-Judge evaluator for MediFlow translation quality assessment.
# Uses Gemini 3.1 Pro via Vertex AI with a medical-specific rubric.
# Scores Hebrew→Russian translations on three dimensions (0–100 each).

import json
import re
from dataclasses import dataclass

import httpx
from google.genai import errors, types
from google.genai.types import HttpOptions, HttpRetryOptions
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from .vertex_client import MODEL, get_client

_WEIGHTS = {
    "critical_terms": 0.40,
    "completeness": 0.35,
    "semantic": 0.25,
}

_PROMPT_WITH_REFERENCE = """\
You are a medical translation quality evaluator. Assess a Hebrew→Russian translation.

SOURCE (Hebrew):
{source}

HYPOTHESIS (translated Russian):
{hypothesis}

REFERENCE (gold standard — may contain mixed Hebrew and Russian text extracted from a bilingual PDF):
{reference}

IMPORTANT: The reference file may contain Hebrew text interleaved with Russian. When evaluating, extract and use only the Russian portions of the reference as the gold standard. Ignore any Hebrew text in the reference.

Score on THREE dimensions (0–100 each), comparing the hypothesis to both the source and the Russian portions of the reference:
1. Critical Terms Accuracy: Are all dosages, dates, drug names, diagnoses, and medical procedures correctly translated? Any error here is severe (0 = completely wrong, 100 = all critical terms correct).
2. Completeness: Is all content from the source present in the hypothesis? Penalize omissions of any kind and hallucinated content not present in the source.
3. Semantic Similarity: Does the hypothesis preserve the full meaning and intent of the source?

Also provide:
- A verbal evaluation (2–4 sentences) summarising overall translation quality
- A list of specific problems found (empty list if none)

Respond in JSON only — no markdown, no explanation outside the JSON:
{{"critical_terms_score": <0-100>, "completeness_score": <0-100>, "semantic_score": <0-100>, "verbal_evaluation": "<text>", "problems": ["<issue1>", ...]}}"""

_PROMPT_WITHOUT_REFERENCE = """\
You are a medical translation quality evaluator. Assess a Hebrew→Russian translation. No reference translation is available.

SOURCE (Hebrew):
{source}

HYPOTHESIS (translated Russian):
{hypothesis}

Score on THREE dimensions (0–100 each) using your medical and linguistic knowledge:
1. Critical Terms Accuracy: Are all dosages, dates, drug names, diagnoses, and medical procedures correctly translated? Any error here is severe (0 = completely wrong, 100 = all critical terms correct).
2. Completeness: Is all content from the source reflected in the hypothesis? Penalize omissions and hallucinated content not present in the source.
3. Semantic Similarity: Does the hypothesis preserve the full meaning and intent of the source?

Also provide:
- A verbal evaluation (2–4 sentences) summarising overall translation quality
- A list of specific problems found (empty list if none)

Respond in JSON only — no markdown, no explanation outside the JSON:
{{"critical_terms_score": <0-100>, "completeness_score": <0-100>, "semantic_score": <0-100>, "verbal_evaluation": "<text>", "problems": ["<issue1>", ...]}}"""


@dataclass
class EvaluationResult:
    """Structured result from the LLM translation judge.

    Attributes:
        critical_terms_score: Accuracy of dosages, dates, medical terms (0–100).
        completeness_score: No dropped content, no hallucinations (0–100).
        semantic_score: Meaning and intent preserved (0–100).
        overall_score: Weighted average of the three dimensions (0–100).
        verbal_evaluation: Narrative summary of translation quality.
        problems: List of specific issues found in the translation.
    """

    critical_terms_score: float
    completeness_score: float
    semantic_score: float
    overall_score: float
    verbal_evaluation: str
    problems: list[str]


def _compute_overall(critical: float, completeness: float, semantic: float) -> float:
    """Compute weighted overall score from three dimension scores.

    Args:
        critical: Critical terms accuracy score (0–100).
        completeness: Completeness score (0–100).
        semantic: Semantic similarity score (0–100).

    Returns:
        Weighted overall score (0–100).
    """
    return (
        critical * _WEIGHTS["critical_terms"]
        + completeness * _WEIGHTS["completeness"]
        + semantic * _WEIGHTS["semantic"]
    )


def _parse_response(text: str) -> dict:
    """Extract and parse JSON from the model response text.

    Args:
        text: Raw response text from Gemini.

    Returns:
        Parsed JSON dict with evaluation fields.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{text}")
    return json.loads(match.group())


def _is_retryable(exc: BaseException) -> bool:
    """Return True for rate-limit errors and read timeouts."""
    if isinstance(exc, errors.ClientError) and exc.code == 429:
        return True
    if isinstance(exc, httpx.ReadTimeout):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=10, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
def score(source: str, hypothesis: str, reference: str | None = None) -> EvaluationResult:
    """Evaluate a Hebrew→Russian translation using Gemini 3.1 Pro.

    Passes the full document text to the model — no truncation needed given
    Gemini 3.1 Pro's large context window.

    Args:
        source: The original Hebrew document text.
        hypothesis: The translated Russian text to evaluate.
        reference: Optional gold-standard Russian reference translation.
                   When provided, the judge compares hypothesis to it directly.

    Returns:
        An EvaluationResult with dimension scores, overall score, verbal
        evaluation, and a list of specific problems found.

    Raises:
        ValueError: If the model response cannot be parsed as valid JSON.
    """
    if reference is not None:
        prompt = _PROMPT_WITH_REFERENCE.format(
            source=source, hypothesis=hypothesis, reference=reference
        )
    else:
        prompt = _PROMPT_WITHOUT_REFERENCE.format(
            source=source, hypothesis=hypothesis
        )

    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            http_options=HttpOptions(
                timeout=300_000,  # milliseconds (= 5 minutes)
                retry_options=HttpRetryOptions(attempts=1),  # disable SDK retries; our decorator handles backoff
            ),
        ),
    )

    data = _parse_response(response.text)

    critical = float(data["critical_terms_score"])
    completeness = float(data["completeness_score"])
    semantic = float(data["semantic_score"])

    return EvaluationResult(
        critical_terms_score=critical,
        completeness_score=completeness,
        semantic_score=semantic,
        overall_score=_compute_overall(critical, completeness, semantic),
        verbal_evaluation=data["verbal_evaluation"],
        problems=data.get("problems", []),
    )
