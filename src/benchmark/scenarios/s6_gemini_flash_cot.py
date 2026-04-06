"""Scenario 6: Gemini Flash DSPy ChainOfThought — Hebrew → Russian translation.

Zero-shot translation using dspy.ChainOfThought backed by Gemini Flash via the
Vertex AI / LiteLLM adapter. Reuses the shared MedicalTranslation signature
from Scenario 3. Outputs aligned Hebrew/Russian sentence pairs.
"""

from __future__ import annotations

import os
import time

import dspy

from src.benchmark.cost import gemini_cost
from src.benchmark.dataset import DatasetDoc
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario
from src.benchmark.scenarios.s3_gemini_pro_dspy_predict import MedicalTranslation
from src.evaluation.vertex_client import get_translate_client

FLASH_MODEL = "vertex_ai/gemini-3-flash-preview"
_NATIVE_MODEL = "gemini-3-flash-preview"


class GeminiFlashCoTScenario(TranslationScenario):
    """Translation scenario using Gemini Flash with dspy.ChainOfThought (zero-shot)."""

    def __init__(self) -> None:
        """Initialise the scenario and configure the DSPy language model.

        Reads GOOGLE_CLOUD_PROJECT (required) and GOOGLE_CLOUD_LOCATION
        (optional, defaults to "global") from the environment. The token-count
        client is created here so that I/O setup stays out of translate().

        Raises:
            KeyError: If GOOGLE_CLOUD_PROJECT is not set.
        """
        self._lm = dspy.LM(
            model=FLASH_MODEL,
            temperature=0.0,
            max_tokens=32768,
            vertex_project=os.environ["GOOGLE_CLOUD_PROJECT"],
            vertex_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
        self._predictor = dspy.ChainOfThought(MedicalTranslation)
        self._token_client = get_translate_client()

    @property
    def name(self) -> str:
        """Unique identifier for this scenario.

        Returns:
            The scenario name string.
        """
        return "s6_gemini_flash_cot"

    @property
    def description(self) -> str:
        """Human-readable description of this scenario.

        Returns:
            A plain-text description string.
        """
        return "Gemini Flash DSPy ChainOfThought — zero-shot Hebrew→Russian via dspy.ChainOfThought"

    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document using dspy.ChainOfThought with Gemini Flash.

        The model is instructed (via MedicalTranslation) to return a JSON array
        of aligned sentence pairs: [{"source_hebrew": "...", "translated_russian": "..."}, ...].

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult containing the Russian translation, token-based
            cost record (input_chars and output_chars are None), and elapsed
            wall-clock time.

        Raises:
            ValueError: If the model returns an empty translation.
        """
        start = time.monotonic()
        with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
            prediction = self._predictor(hebrew_text=doc.hebrew_text)
        elapsed = time.monotonic() - start

        translation: str = prediction.russian_translation
        if not translation:
            raise ValueError(
                f"Empty translation returned by Gemini Flash DSPy ChainOfThought for doc "
                f"'{doc.doc_id}'; expected a non-empty Russian string."
            )

        input_tokens: int = self._token_client.models.count_tokens(
            model=_NATIVE_MODEL, contents=doc.hebrew_text
        ).total_tokens
        output_tokens: int = self._token_client.models.count_tokens(
            model=_NATIVE_MODEL, contents=translation
        ).total_tokens
        cost = gemini_cost(input_tokens, output_tokens, model="flash")

        return TranslationResult(
            doc_id=doc.doc_id,
            translation=translation,
            cost=cost,
            elapsed_sec=elapsed,
        )
