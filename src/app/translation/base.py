"""Result dataclasses for the MediFlow translation pipeline.

Defines ``SectionTranslationResult`` and ``TranslationResult``: immutable
value objects used to carry translation output, cost, and timing data
between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app.translation.cost import CostRecord


@dataclass(frozen=True)
class SectionTranslationResult:
    """Immutable result of translating a single document section.

    Attributes:
        section_index:  1-based index of this section within the document.
        section_label:  Human-readable label for this section (e.g. "Medications").
        hebrew_text:    The original Hebrew text of the section.
        russian_text:   The translated Russian text of the section.
        cost:           API usage and cost for this translation call.
        elapsed_sec:    Wall-clock seconds taken to produce the translation.
        was_corrected:  True iff the Pro correction step ran for this section.
        midway_score:   Quality score from the midway judge (before correction).
        midway_errors:  MQM error list from the midway judge (before correction).
    """

    section_index: int
    section_label: str
    hebrew_text: str
    russian_text: str
    cost: CostRecord
    elapsed_sec: float
    was_corrected: bool
    midway_score: float
    midway_errors: list[dict]


@dataclass(frozen=True)
class TranslationResult:
    """Immutable result of translating a single document.

    Attributes:
        doc_id:      Identifier of the source document (e.g. "Form_039").
        translation: The translated text produced by the pipeline.
        cost:        API usage and cost for this translation call.
        elapsed_sec: Wall-clock seconds taken to produce the translation.
    """

    doc_id: str
    translation: str
    cost: CostRecord
    elapsed_sec: float
