"""Cost tracking for the MediFlow translation pipeline.

Provides cost calculation functions and the CostRecord dataclass for
Gemini (token-based) API calls. Only Gemini models are used in the
production pipeline; Google Translate cost tracking has been removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Pricing constants (USD per 1 000 tokens) ---
# Rates sourced from Google Cloud Vertex AI pricing page.
# Pro is used for evaluation and correction; Flash is used for initial translation.
GEMINI_PRO_INPUT_USD_PER_1K_TOKENS    = 0.002
GEMINI_PRO_OUTPUT_USD_PER_1K_TOKENS   = 0.012
GEMINI_FLASH_INPUT_USD_PER_1K_TOKENS  = 0.00025
GEMINI_FLASH_OUTPUT_USD_PER_1K_TOKENS = 0.0005


@dataclass(frozen=True)
class CostRecord:
    """Immutable record of API usage and associated cost for one model call.

    Attributes:
        input_tokens:  Prompt tokens consumed by the model.
        output_tokens: Completion tokens produced by the model.
        cost_usd:      Total cost in US dollars for this call.
    """

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float


def gemini_cost(
    input_tokens: int,
    output_tokens: int,
    model: Literal["pro", "flash"],
) -> CostRecord:
    """Calculate cost for a single Gemini API call.

    Cost = (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate,
    where rates differ between Pro and Flash variants.

    Args:
        input_tokens:  Number of prompt tokens sent to the model.
        output_tokens: Number of completion tokens returned by the model.
        model:         Which Gemini variant — "pro" or "flash".

    Returns:
        A CostRecord with token counts and the computed cost_usd.

    Raises:
        ValueError: If model is not "pro" or "flash".
    """
    if model == "pro":
        # Pro is the higher-capability, higher-cost model used for judge and correction.
        input_rate  = GEMINI_PRO_INPUT_USD_PER_1K_TOKENS
        output_rate = GEMINI_PRO_OUTPUT_USD_PER_1K_TOKENS
    elif model == "flash":
        # Flash is the faster, cheaper model used for the initial translation pass.
        input_rate  = GEMINI_FLASH_INPUT_USD_PER_1K_TOKENS
        output_rate = GEMINI_FLASH_OUTPUT_USD_PER_1K_TOKENS
    else:
        raise ValueError(f"Unknown model '{model}': expected 'pro' or 'flash'.")

    # Rates are per-1 000 tokens, so divide by 1 000 before multiplying.
    cost_usd = (
        input_tokens  / 1_000 * input_rate
        + output_tokens / 1_000 * output_rate
    )

    return CostRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def sum_costs(costs: list[CostRecord]) -> CostRecord:
    """Sum a list of CostRecords into a single aggregate record.

    Integer fields (input_tokens, output_tokens) are summed treating None as 0
    when at least one record has a non-None value. If every record has None for
    a given field, the result also has None for that field. cost_usd is always
    summed as a float.

    Args:
        costs: List of CostRecord objects to aggregate. May be empty.

    Returns:
        A single CostRecord with aggregated values. An empty list returns
        CostRecord(None, None, 0.0).
    """
    if not costs:
        return CostRecord(
            input_tokens=None,
            output_tokens=None,
            cost_usd=0.0,
        )

    def _sum_nullable(values: list[int | None]) -> int | None:
        # Keep None only when every value is None; otherwise treat None as 0.
        if all(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    return CostRecord(
        input_tokens=_sum_nullable([c.input_tokens for c in costs]),
        output_tokens=_sum_nullable([c.output_tokens for c in costs]),
        cost_usd=sum(c.cost_usd for c in costs),
    )
