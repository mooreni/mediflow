# Public evaluation interface for MediFlow.
# Exposes a single evaluate() function compatible with DSPy's optimization API.
# Returns a normalized float in [0, 1] representing overall translation quality.

from .llm_judge import EvaluationResult, score


def evaluate(source: str, hypothesis: str, reference: str | None = None) -> float:
    """Evaluate a Hebrew→Russian translation and return a normalized quality score.

    This is the DSPy-compatible interface. For full structured results including
    per-dimension scores and verbal feedback, use llm_judge.score() directly.

    Args:
        source: The original Hebrew document text.
        hypothesis: The translated Russian text to evaluate.
        reference: Optional gold-standard Russian reference translation.

    Returns:
        Overall translation quality score normalized to [0, 1].
    """
    result = score(source, hypothesis, reference)
    return result.overall_score / 100.0


__all__ = ["evaluate", "EvaluationResult", "score"]
