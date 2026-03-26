# Validation experiment for the MediFlow evaluation mechanism.
# Tests that the Gemini judge detects intentional translation errors:
#   - Dosage corruption: numbers near dose units inflated 10x
#   - Omission: every 3rd paragraph dropped
# Expected: perfect translation scores ~95-100; corrupted variants score lower
# on the relevant dimension.
#
# Usage: python scripts/test_evaluator.py

import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.llm_judge import score

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_HE = os.path.join(os.path.dirname(__file__), "..", "data", "informed_consent_forms", "text", "he")
DATA_RU = os.path.join(os.path.dirname(__file__), "..", "data", "informed_consent_forms", "text", "ru")

FORMS = ["Form_001", "Form_002", "Form_003"]


def load_pair(form_id: str) -> tuple[str, str]:
    """Load a Hebrew source and Russian reference text pair.

    Args:
        form_id: Form identifier, e.g. 'Form_001'.

    Returns:
        Tuple of (hebrew_text, russian_text).
    """
    he_path = os.path.join(DATA_HE, f"{form_id}_HE.txt")
    ru_path = os.path.join(DATA_RU, f"{form_id}_RU.txt")
    with open(he_path, encoding="utf-8") as f:
        he = f.read()
    with open(ru_path, encoding="utf-8") as f:
        ru = f.read()
    return he, ru


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

# Medical terminology swaps for Russian informed-consent documents.
# Maps correct term → wrong term to simulate critical mistranslations.
_TERM_SWAPS = [
    ("анестезия", "антибиотик"),
    ("анестезии", "антибиотики"),
    ("анестезиолог", "терапевт"),
    ("хирург", "терапевт"),
    ("операци", "процедур"),   # prefix covers операция/операции/операцию
    ("наркоз", "седация"),
    ("обрезани", "операци"),   # prefix covers обрезание/обрезания
]


def corrupt_terminology(text: str) -> int:
    """Return (corrupted_text, n_replacements).

    Swaps key medical terms with plausible but wrong alternatives to simulate
    a critical mistranslation. Only replaces Russian text; case-insensitive
    on the first character.

    This is a diagnostic wrapper used by the experiment runner — the public
    corruption function is ``corrupt_terminology_text``.

    Args:
        text: Russian translation text (may contain interleaved Hebrew).

    Returns:
        Tuple of (modified text, number of replacements made).
    """
    total = 0
    for correct, wrong in _TERM_SWAPS:
        new_text, n = re.subn(correct, wrong, text, flags=re.IGNORECASE)
        text = new_text
        total += n
    return text, total


def corrupt_omission(text: str) -> tuple[str, int]:
    """Return (corrupted_text, n_sentences_dropped).

    Removes every 3rd Russian sentence from the text. Splits on sentence-ending
    punctuation followed by whitespace so it works on documents that use single
    line-breaks rather than blank lines between paragraphs.

    Args:
        text: Russian translation text.

    Returns:
        Tuple of (modified text, number of sentences dropped).
    """
    # Split on sentence boundaries: period/!/? followed by space or newline.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    dropped = 0
    for i, sentence in enumerate(sentences):
        if (i + 1) % 3 == 0:
            dropped += 1
        else:
            kept.append(sentence)
    return " ".join(kept), dropped


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

VARIANTS = [
    ("perfect", lambda ru: (ru, 0)),
    ("terminology", corrupt_terminology),
    ("omission", corrupt_omission),
]

COL_W = {"form": 10, "variant": 16, "changes": 9, "critical": 10, "complete": 10, "semantic": 10, "overall": 10}


def _header() -> str:
    return (
        f"{'Form':<{COL_W['form']}} {'Variant':<{COL_W['variant']}} "
        f"{'Changes':>{COL_W['changes']}} "
        f"{'Critical':>{COL_W['critical']}} {'Complete':>{COL_W['complete']}} "
        f"{'Semantic':>{COL_W['semantic']}} {'Overall':>{COL_W['overall']}}"
    )


def _row(form_id: str, variant: str, n_changes: int, result) -> str:
    return (
        f"{form_id:<{COL_W['form']}} {variant:<{COL_W['variant']}} "
        f"{n_changes:>{COL_W['changes']}} "
        f"{result.critical_terms_score:>{COL_W['critical']}.1f} "
        f"{result.completeness_score:>{COL_W['complete']}.1f} "
        f"{result.semantic_score:>{COL_W['semantic']}.1f} "
        f"{result.overall_score:>{COL_W['overall']}.1f}"
    )


def _detail(result) -> str:
    problems_str = "; ".join(result.problems[:2]) if result.problems else "none"
    if len(result.problems) > 2:
        problems_str += f" (+{len(result.problems) - 2} more)"
    return f"  Judge: {result.verbal_evaluation}\n  Problems: {problems_str}"


def main() -> None:
    """Run the evaluation experiment and print a results table."""
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        print("ERROR: GOOGLE_CLOUD_PROJECT is not set.")
        sys.exit(1)

    print("MediFlow Evaluation Experiment")
    print("=" * 100)
    print(_header())
    print("-" * 100)

    for form_id in FORMS:
        print(f"Loading {form_id}...", flush=True)
        he, ru_ref = load_pair(form_id)

        for variant_name, corrupt_fn in VARIANTS:
            hypothesis, n_changes = corrupt_fn(ru_ref)
            print(f"  Evaluating {variant_name} (changes={n_changes})...", flush=True)
            result = score(source=he, hypothesis=hypothesis, reference=ru_ref)
            print(_row(form_id, variant_name, n_changes, result))
            print(_detail(result))

        print()

    print("=" * 100)
    print("Expected: perfect ≈ 95–100 | terminology: lower critical | omission: lower completeness")


if __name__ == "__main__":
    main()
