# MediFlow evaluation package.
# Primary entry point: evaluate(source, hypothesis, reference) -> float
# Full results: score(source, hypothesis, reference) -> EvaluationResult

from .evaluator import evaluate
from .llm_judge import EvaluationResult, score

__all__ = ["evaluate", "score", "EvaluationResult"]
