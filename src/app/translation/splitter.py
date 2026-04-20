"""Split-before-translate strategy for Hebrew medical documents.

Rather than translating a document as a monolithic block, MediFlow splits it
into labelled sections first. This approach has three benefits:

1. **Context preservation** — each section is translated with a shared clinical
   context header (document type, patient sex, primary diagnosis) that the
   translator can use to resolve ambiguous Hebrew phrasing.
2. **Section-level evaluation** — the LLM judge scores each section
   independently, producing granular quality data and allowing targeted
   correction only where needed.
3. **Parallelism** — once split, sections can be translated concurrently by a
   thread pool without any inter-section dependencies.

``split_document()`` is the sole public entry point.  It calls Gemini Flash
with a strict JSON response schema and validates the returned sections before
returning a ``SplitResult``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
from google.genai import errors, types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.app.clients.vertex import get_translate_client

_FLASH_MODEL = "gemini-3-flash-preview"

_SPLITTER_SYSTEM_INSTRUCTION = """{
  "role": "medical_document_analyst",
  "task": "Split the Hebrew medical document provided by the user into labelled sections suitable for sequential translation.",
  "rules": {
    "section_length": {
      "max_sentences": 20,
      "min_sentences": 2,
      "min_sentences_exception": "A section may contain fewer than 2 sentences only if it is a standalone heading or list that cannot be split further."
    },
    "long_topics": {
      "instruction": "Sub-split long topics across multiple sections.",
      "label_format": "Append a counter to the label.",
      "examples": ["Medications 1/2", "Medications 2/2"]
    },
    "structure_preservation": "Respect the document's natural structure: headings, paragraphs, and lists.",
    "content_preservation": "Preserve ALL content. Every Hebrew word must appear in exactly one section — no word may be omitted or duplicated."
  },
  "output": {
    "format": "Respond with a single JSON object and nothing else.",
    "schema": {
      "context": {
        "document_description": "<brief document type and purpose>",
        "patient_sex": "<male|female|unknown>",
        "primary_diagnosis": "<diagnosis string or null>"
      },
      "total_sections": "<integer>",
      "sections": [
        {
          "index": "<1-based integer>",
          "label": "<section label>",
          "hebrew_text": "<full Hebrew text of this section>"
        }
      ]
    }
  }
}"""


@dataclass(frozen=True)
class Section:
    """An immutable slice of a Hebrew medical document.

    Attributes:
        index:       1-based position of this section within the document.
        label:       Human-readable label, e.g. "Patient Demographics".
        hebrew_text: The raw Hebrew text content of this section.
    """

    index: int
    label: str
    hebrew_text: str


@dataclass(frozen=True)
class SplitResult:
    """Immutable result of splitting a single Hebrew medical document.

    Attributes:
        context_header:  Compact clinical context string built from document
                         description, patient sex, and primary diagnosis.
        total_sections:  Total number of sections the document was split into.
        sections:        Ordered list of Section objects (1-based indices).
    """

    context_header: str
    total_sections: int
    sections: list[Section]


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry on the splitter call.

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
    # asyncio.TimeoutError is a subclass of TimeoutError in Python 3.11+.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return False


def _log_split_retry(state) -> None:  # type: ignore[type-arg]
    """Log splitter retry attempts so they are visible in benchmark output.

    Args:
        state: The tenacity RetryCallState object.

    Returns:
        None
    """
    attempt = state.attempt_number
    exc = state.outcome.exception() if state.outcome else None
    wait = state.next_action.sleep if state.next_action else 0
    print(
        f"  [SPLIT RETRY {attempt}] {type(exc).__name__}: {exc} "
        f"— waiting {wait:.0f}s before next attempt"
    )


# Retry policy for split_document():
#   retry_if_exception(_is_retryable) — only retry transient errors (429
#       rate-limits, read timeouts); propagate hard failures immediately.
#   stop_after_attempt(3)             — at most 3 total attempts (2 retries).
#   wait_exponential_jitter(...)      — exponential back-off starting at 10 s,
#       capped at 60 s, with random jitter to avoid thundering-herd when
#       multiple workers retry simultaneously.
#   reraise=True                      — surface the original exception rather
#       than wrapping it in a tenacity RetryError after all attempts fail.
@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=10, max=60),
    before_sleep=_log_split_retry,
    reraise=True,
)
def split_document(hebrew_text: str, client=None) -> SplitResult:
    """Split a Hebrew medical document into labelled sections via Gemini Flash.

    Calls the Gemini Flash model with a structured JSON response schema to
    divide the document into sections of at most 20 sentences each, and
    extracts a compact clinical context header.

    Args:
        hebrew_text: The full Hebrew text of the medical document to split.
        client:      A Gemini API client exposing models.generate_content().
                     Defaults to get_translate_client() when None, so callers
                     can omit it in production and inject a mock in tests.

    Returns:
        A SplitResult containing the context header, section count, and an
        ordered list of Section objects with 1-based indices.

    Raises:
        ValueError: If the model returns an empty response, invalid JSON,
                    a mismatched total_sections count, non-contiguous section
                    indices, or any section with empty hebrew_text.
        google.genai.errors.ClientError: If the Vertex AI API call fails and
                                         retries are exhausted.
    """
    if client is None:
        client = get_translate_client()
    response = client.models.generate_content(
        model=_FLASH_MODEL,
        contents=hebrew_text,
        config=types.GenerateContentConfig(
            system_instruction=_SPLITTER_SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            # gemini-3-flash-preview enables thinking by default; disable it to
            # preserve benchmark baseline latency and quality profile.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=120_000),
        ),
    )

    raw = response.text
    if not raw:
        raise ValueError(
            "Splitter returned an empty response — expected a JSON split result."
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Splitter returned invalid JSON: {exc} — raw response: {raw[:200]!r}"
        ) from exc

    # Build compact context header from the clinical context fields.
    ctx = data.get("context", {})
    doc_desc = ctx.get("document_description", "")
    patient_sex = ctx.get("patient_sex", "")
    diagnosis = ctx.get("primary_diagnosis")  # nullable
    header_parts = [f"Document: {doc_desc}", f"Patient sex: {patient_sex}"]
    if diagnosis:
        header_parts.append(f"Diagnosis: {diagnosis}")
    context_header = " | ".join(header_parts)

    total = data.get("total_sections", 0)
    raw_sections = data.get("sections", [])

    # Validate section count matches declared total.
    if len(raw_sections) != total:
        raise ValueError(
            f"Splitter declared total_sections={total} but returned "
            f"{len(raw_sections)} sections — expected equal counts."
        )

    # Validate indices form a contiguous 1..N sequence.  A non-contiguous
    # sequence (e.g. [1, 2, 4]) would mean the downstream assembler silently
    # skips sections, producing an incomplete translation.  Using sorted list
    # comparison rather than a set check also catches duplicates (e.g. [1,1,3])
    # that a set would collapse.
    indices = sorted(s["index"] for s in raw_sections)
    expected = list(range(1, total + 1))
    if indices != expected:
        raise ValueError(
            f"Section indices are not contiguous 1..{total}: got {indices}, "
            f"expected {expected}."
        )

    # Validate no section has empty hebrew_text.
    for s in raw_sections:
        if not s.get("hebrew_text", "").strip():
            raise ValueError(
                f"Section {s['index']} ('{s.get('label', '')}') has empty "
                f"hebrew_text — expected non-empty Hebrew content."
            )

    # Build the sections list sorted by index.
    sections = [
        Section(index=s["index"], label=s["label"], hebrew_text=s["hebrew_text"])
        for s in sorted(raw_sections, key=lambda x: x["index"])
    ]

    return SplitResult(
        context_header=context_header,
        total_sections=total,
        sections=sections,
    )
