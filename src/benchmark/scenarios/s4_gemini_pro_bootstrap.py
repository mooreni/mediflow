"""Scenario 4: Gemini Pro DSPy BootstrapFewShot — Hebrew → Russian translation.

Optimises a ChainOfThought translator using DSPy BootstrapFewShot on the
training split before running inference on the eval split.
"""

from __future__ import annotations

import os
import pathlib
import time

import dspy

from src.benchmark.cost import gemini_cost
from src.benchmark.dataset import DatasetDoc
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario
from src.benchmark.scenarios.s3_gemini_pro_dspy_predict import MedicalTranslation
from src.evaluation.llm_judge import score
from src.evaluation.vertex_client import get_translate_client

_NATIVE_MODEL = "gemini-3.1-pro-preview"
PRO_MODEL = "vertex_ai/gemini-3.1-pro-preview"
_COMPILED_MODULE_PATH = pathlib.Path("s4_compiled_module.json")


def _translation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object = None,
) -> float:
    """DSPy metric: score a predicted translation against the reference.

    Args:
        example:    DSPy example containing hebrew_text and russian_translation.
        prediction: DSPy prediction containing russian_translation.
        trace:      Unused; present to satisfy the DSPy metric signature.

    Returns:
        Normalised quality score in [0, 1].

    Raises:
        RuntimeError: If the underlying score() call fails, with context
            about the example that triggered the failure.
    """
    try:
        result = score(
            source=example.hebrew_text,
            hypothesis=prediction.russian_translation,
        )
        return result.quality_score
    except Exception as exc:
        raise RuntimeError(
            f"score() failed for example (hebrew_text length={len(example.hebrew_text)}): {exc}"
        ) from exc


class _ChainOfThoughtTranslatorModule(dspy.Module):
    """DSPy module wrapping a ChainOfThought predictor for medical translation."""

    def __init__(self) -> None:
        """Initialise the ChainOfThought predictor."""
        self.translator = dspy.ChainOfThought(MedicalTranslation)

    def forward(self, hebrew_text: str) -> dspy.Prediction:
        """Run the chain-of-thought translation.

        Args:
            hebrew_text: Source Hebrew document text.

        Returns:
            A DSPy Prediction with a russian_translation field.
        """
        return self.translator(hebrew_text=hebrew_text)


class GeminiProBootstrapScenario(TranslationScenario):
    """Translation scenario using Gemini Pro with DSPy BootstrapFewShot optimisation."""

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
        self._module: _ChainOfThoughtTranslatorModule | None = None

    @property
    def name(self) -> str:
        """Unique identifier for this scenario.

        Returns:
            The scenario name string.
        """
        return "s4_gemini_pro_bootstrap"

    @property
    def description(self) -> str:
        """Human-readable description of this scenario.

        Returns:
            A plain-text description string.
        """
        return "Gemini Pro DSPy BootstrapFewShot — few-shot optimised Hebrew→Russian via ChainOfThought"

    def train(self, train_docs: list[DatasetDoc]) -> None:
        """Optimise the translator using BootstrapFewShot on the training split.

        If a compiled module already exists at _COMPILED_MODULE_PATH, loads it
        directly and skips optimisation. Otherwise runs BootstrapFewShot and
        saves the result to disk for future runs.

        Args:
            train_docs: Training documents.
        """
        self._module = _ChainOfThoughtTranslatorModule()

        if _COMPILED_MODULE_PATH.exists():
            print(f"[s4] Loading compiled module from {_COMPILED_MODULE_PATH}")
            self._module.load(_COMPILED_MODULE_PATH)
            return

        examples = [
            dspy.Example(hebrew_text=doc.hebrew_text).with_inputs("hebrew_text")
            for doc in train_docs
        ]

        optimizer = dspy.BootstrapFewShot(
            metric=_translation_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=0,
            max_rounds=10,
        )

        with dspy.context(lm=self._lm):
            self._module = optimizer.compile(
                self._module,
                trainset=examples,
            )

        self._module.save(_COMPILED_MODULE_PATH)
        print(f"[s4] Compiled module saved to {_COMPILED_MODULE_PATH}")

    def translate(self, doc: DatasetDoc) -> TranslationResult:
        """Translate a single document using the compiled BootstrapFewShot module.

        Args:
            doc: The document to translate.

        Returns:
            A TranslationResult containing the Russian translation, token-based
            cost record (input_chars and output_chars are None), and elapsed
            wall-clock time.

        Raises:
            RuntimeError: If train() has not been called before translate().
            ValueError:   If the model returns an empty translation.
        """
        if self._module is None:
            raise RuntimeError("train() must be called before translate()")

        start = time.monotonic()
        with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
            prediction = self._module(hebrew_text=doc.hebrew_text)
        elapsed = time.monotonic() - start

        translation: str = prediction.russian_translation
        if not translation:
            raise ValueError(
                f"Empty translation from Gemini Pro BootstrapFewShot for doc '{doc.doc_id}'"
            )

        native_client = get_translate_client()
        input_tokens: int = native_client.models.count_tokens(
            model=_NATIVE_MODEL, contents=doc.hebrew_text
        ).total_tokens
        output_tokens: int = native_client.models.count_tokens(
            model=_NATIVE_MODEL, contents=translation
        ).total_tokens
        cost = gemini_cost(input_tokens, output_tokens, model="pro")

        return TranslationResult(
            doc_id=doc.doc_id,
            translation=translation,
            cost=cost,
            elapsed_sec=elapsed,
        )
