"""Scenario 2: Gemini Flash zero-shot Hebrew → Russian translation.

Uses Google GenAI (Vertex AI backend) with a direct generate_content call —
no DSPy. Serves as a low-cost LLM baseline for the benchmark.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from src.benchmark.cost import gemini_cost
from src.benchmark.dataset import DatasetDoc
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario
from src.benchmark.scenarios.glossary import MEDICAL_GLOSSARY_TEXT
from src.evaluation.vertex_client import get_translate_client

FLASH_MODEL = "gemini-3-flash-preview"

_SYSTEM_INSTRUCTION = (
    "You are a professional medical translator. "
    "Translate the Hebrew medical document below into Russian. "
    "Preserve all medical terminology, drug names, dosages, dates, diagnoses, "
    "and document structure exactly. "
    "Output a JSON array of aligned sentence pairs. Each element must have: "
    '{"source_hebrew": "<original sentence>", "translated_russian": "<translated sentence>"} '
    "Output only the JSON array — no commentary, no markdown.\n\n"
    + MEDICAL_GLOSSARY_TEXT
)


class GeminiFlashZeroShotScenario(TranslationScenario):
    """Translation scenario using Gemini Flash with a zero-shot prompt."""

    def __init__(self, client: genai.Client | None = None) -> None:
        """Initialise the scenario with an optional pre-built GenAI client.

        Args:
            client: A configured google.genai.Client instance. If None,
                    translate() resolves a thread-local client at call time via
                    get_translate_client() — one isolated client per worker
                    thread, avoiding httpx connection-pool contention.
        """
        self._client: genai.Client | None = client

    @property
    def name(self) -> str:
        """Unique identifier for this scenario.

        Returns:
            The scenario name string.
        """
        return "s2_gemini_flash_zeroshot"

    @property
    def description(self) -> str:
        """Human-readable description of this scenario.

        Returns:
            A plain-text description string.
        """
        return "Gemini Flash zero-shot — direct Hebrew→Russian, no DSPy"

    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document using Gemini Flash zero-shot.

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult containing the Russian translation, cost record
            (token-based, no char fields), and elapsed wall-clock time.

        Raises:
            ValueError: If the model response is empty, truncated (MAX_TOKENS),
                        or missing usage metadata.
            google.genai.errors.APIError: If the Vertex AI call fails.
        """
        client = self._client if self._client is not None else get_translate_client()
        start = time.monotonic()
        response = client.models.generate_content(
            model=FLASH_MODEL,
            contents=doc.hebrew_text,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=32768,
                http_options=types.HttpOptions(timeout=240_000),
            ),
        )
        elapsed = time.monotonic() - start

        # Retrieve text from candidates directly — response.text can be None
        # when finish_reason is MAX_TOKENS even if content was generated.
        text = response.text
        if not text and response.candidates:
            parts = response.candidates[0].content.parts if response.candidates[0].content else []
            text = "".join(p.text for p in parts if hasattr(p, "text") and p.text) if parts else ""

        if not text:
            raise ValueError(
                f"Empty response from Gemini Flash for doc '{doc.doc_id}'"
            )

        # Fail loudly on truncation so the runner retries rather than storing
        # a partial translation silently.
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS"):
                out_tokens = (
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata else "unknown"
                )
                raise ValueError(
                    f"Gemini Flash hit MAX_TOKENS for doc '{doc.doc_id}' "
                    f"({out_tokens} output tokens) — translation is truncated."
                )

        if not response.usage_metadata:
            raise ValueError(f"No usage metadata in response for doc '{doc.doc_id}'")
        input_tokens: int = response.usage_metadata.prompt_token_count
        output_tokens: int = response.usage_metadata.candidates_token_count
        cost = gemini_cost(input_tokens, output_tokens, model="flash")

        return TranslationResult(
            doc_id=doc.doc_id,
            translation=text,
            cost=cost,
            elapsed_sec=elapsed,
        )
