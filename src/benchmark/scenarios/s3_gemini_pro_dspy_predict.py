"""Scenario 3: Gemini Pro DSPy Predict — Hebrew → Russian translation.

Defines the shared MedicalTranslation DSPy signature (re-exported for use by
Scenarios 4 and 5) and a dspy.Predict-based translation scenario backed by
Gemini Pro via the Vertex AI / LiteLLM adapter.
"""

from __future__ import annotations

import os
import time

import dspy

from src.benchmark.cost import gemini_cost
from src.benchmark.dataset import DatasetDoc
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario
from src.benchmark.scenarios.glossary import MEDICAL_GLOSSARY_TEXT
from src.evaluation.vertex_client import get_translate_client

PRO_MODEL = "vertex_ai/gemini-3.1-pro-preview"
_NATIVE_MODEL = "gemini-3.1-pro-preview"


class MedicalTranslation(dspy.Signature):
    """Translate a Hebrew medical document into Russian.
    Preserve all medical terminology, dosages, dates, diagnoses, and document structure exactly.
    Output a JSON array of aligned sentence pairs. Each element: {"source_hebrew": "...", "translated_russian": "..."}
    """

    hebrew_text: str = dspy.InputField()
    russian_translation: str = dspy.OutputField(desc="JSON array of aligned sentence pairs: [{source_hebrew, translated_russian}, ...]")


# Append the shared glossary so S4/S5 (which import this signature) also benefit.
MedicalTranslation.__doc__ += "\n\n" + MEDICAL_GLOSSARY_TEXT


class GeminiProDSPyPredictScenario(TranslationScenario):
    """Translation scenario using Gemini Pro with dspy.Predict (zero-shot)."""

    def __init__(self) -> None:
        """Initialise the scenario and configure the DSPy language model.

        Reads GOOGLE_CLOUD_PROJECT (required) and GOOGLE_CLOUD_LOCATION
        (optional, defaults to "global") from the environment.

        Raises:
            KeyError: If GOOGLE_CLOUD_PROJECT is not set.
        """
        self._lm = dspy.LM(
            model=PRO_MODEL,
            temperature=0.0,
            max_tokens=32768,
            vertex_project=os.environ["GOOGLE_CLOUD_PROJECT"],
            vertex_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
        self._predictor = dspy.Predict(MedicalTranslation)

    @property
    def name(self) -> str:
        """Unique identifier for this scenario.

        Returns:
            The scenario name string.
        """
        return "s3_gemini_pro_dspy_predict"

    @property
    def description(self) -> str:
        """Human-readable description of this scenario.

        Returns:
            A plain-text description string.
        """
        return "Gemini Pro DSPy Predict — zero-shot Hebrew→Russian via dspy.Predict"

    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document using dspy.Predict with Gemini Pro.

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult containing the Russian translation, token-based
            cost record (input_chars and output_chars are None), and elapsed
            wall-clock time.

        Raises:
            ValueError: If the model returns an empty translation or if the LM
                history is empty after the predict call (no usage data available).
        """
        start = time.monotonic()
        with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
            prediction = self._predictor(hebrew_text=doc.hebrew_text)
        elapsed = time.monotonic() - start

        translation: str = prediction.russian_translation
        if not translation:
            raise ValueError(
                f"Empty translation from Gemini Pro DSPy for doc '{doc.doc_id}'"
            )

        _client = get_translate_client()
        input_tokens: int = _client.models.count_tokens(
            model=_NATIVE_MODEL, contents=doc.hebrew_text
        ).total_tokens
        output_tokens: int = _client.models.count_tokens(
            model=_NATIVE_MODEL, contents=translation
        ).total_tokens
        cost = gemini_cost(input_tokens, output_tokens, model="pro")

        return TranslationResult(
            doc_id=doc.doc_id,
            translation=translation,
            cost=cost,
            elapsed_sec=elapsed,
        )
