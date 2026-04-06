"""Scenario 1: Google Translate Hebrew → Russian.

Uses the Google Cloud Translation API (v2) for direct translation without
any LLM involvement. Serves as the cost/quality baseline for the benchmark.
"""

from __future__ import annotations

import time

from google.cloud import translate_v2

from src.benchmark.cost import google_translate_cost
from src.benchmark.dataset import DatasetDoc
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario


class GoogleTranslateScenario(TranslationScenario):
    """Translation scenario using Google Cloud Translate v2."""

    def __init__(self, client: translate_v2.Client | None = None) -> None:
        """Initialise the scenario with an optional pre-built Translate client.

        Args:
            client: A Google Cloud Translate v2 client. If None, a default
                    client is constructed from the environment credentials.
        """
        self._client: translate_v2.Client = client if client is not None else translate_v2.Client()

    @property
    def name(self) -> str:
        """Unique identifier for this scenario.

        Returns:
            The scenario name string.
        """
        return "s1_google_translate"

    @property
    def description(self) -> str:
        """Human-readable description of this scenario.

        Returns:
            A plain-text description string.
        """
        return "Google Cloud Translate v2 — direct Hebrew→Russian, no LLM"

    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document using Google Cloud Translate.

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult containing the Russian translation, cost record
            (char-based, no token fields), and elapsed wall-clock time.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API call fails.
        """
        start = time.monotonic()
        response = self._client.translate(
            doc.hebrew_text,
            target_language="ru",
            source_language="iw",
        )
        elapsed = time.monotonic() - start

        translation: str = response["translatedText"]
        cost = google_translate_cost(
            input_chars=len(doc.hebrew_text),
            output_chars=len(translation),
        )

        return TranslationResult(
            doc_id=doc.doc_id,
            translation=translation,
            cost=cost,
            elapsed_sec=elapsed,
        )
