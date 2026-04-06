"""Base classes for translation benchmark scenarios.

Defines the TranslationResult dataclass and the TranslationScenario abstract
base class that all benchmark scenarios must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.benchmark.cost import CostRecord
from src.benchmark.dataset import DatasetDoc


@dataclass(frozen=True)
class TranslationResult:
    """Immutable result of translating a single document.

    Attributes:
        doc_id:      Identifier of the source document (e.g. "Form_039").
        translation: The translated text produced by the scenario.
        cost:        API usage and cost for this translation call.
        elapsed_sec: Wall-clock seconds taken to produce the translation.
    """

    doc_id: str
    translation: str
    cost: CostRecord
    elapsed_sec: float


class TranslationScenario(ABC):
    """Abstract base class for all translation benchmark scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier for this scenario (e.g. 's1_google_translate').

        Returns:
            A snake_case string uniquely identifying the scenario.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the scenario's approach.

        Returns:
            A plain-text string describing the scenario.
        """

    def train(self, train_docs: list[DatasetDoc]) -> None:
        """Optionally train or optimise on the training split.

        Args:
            train_docs: Training documents (Forms 051–080).

        Returns:
            None. No-op by default. Scenarios 4 and 5 override this to run
            DSPy optimisation before the test loop begins.
        """

    @abstractmethod
    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document and return the result.

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult with the translation, cost, and elapsed time.
        """
