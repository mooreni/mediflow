"""Cost tracking for translation benchmark scenarios.

Provides cost calculation functions and the CostRecord dataclass for
Gemini (token-based) and Google Translate (character-based) API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Pricing constants (USD) ---
GEMINI_PRO_INPUT_USD_PER_1K_TOKENS    = 0.002
GEMINI_PRO_OUTPUT_USD_PER_1K_TOKENS   = 0.012
GEMINI_FLASH_INPUT_USD_PER_1K_TOKENS  = 0.00025
GEMINI_FLASH_OUTPUT_USD_PER_1K_TOKENS = 0.0005
GOOGLE_TRANSLATE_USD_PER_1M_CHARS     = 20.0


@dataclass(frozen=True)
class CostRecord:
    """Immutable record of API usage and associated cost for one translation call.

    Attributes:
        input_tokens:  Prompt tokens consumed (None for Google Translate).
        output_tokens: Completion tokens produced (None for Google Translate).
        input_chars:   Source characters sent (None for Gemini calls).
        output_chars:  Translated characters received (None for Gemini calls).
        cost_usd:      Total cost in US dollars for this call.
    """

    input_tokens: int | None
    output_tokens: int | None
    input_chars: int | None
    output_chars: int | None
    cost_usd: float


def gemini_cost(
    input_tokens: int,
    output_tokens: int,
    model: Literal["pro", "flash"],
) -> CostRecord:
    """Calculate cost for a single Gemini API call.

    Args:
        input_tokens:  Number of prompt tokens.
        output_tokens: Number of completion tokens.
        model:         Which Gemini variant — "pro" or "flash".

    Returns:
        A CostRecord with token fields populated and char fields set to None.

    Raises:
        ValueError: If model is not "pro" or "flash".
    """
    if model == "pro":
        input_rate  = GEMINI_PRO_INPUT_USD_PER_1K_TOKENS
        output_rate = GEMINI_PRO_OUTPUT_USD_PER_1K_TOKENS
    elif model == "flash":
        input_rate  = GEMINI_FLASH_INPUT_USD_PER_1K_TOKENS
        output_rate = GEMINI_FLASH_OUTPUT_USD_PER_1K_TOKENS
    else:
        raise ValueError(f"Unknown model '{model}': expected 'pro' or 'flash'.")

    cost_usd = (
        input_tokens  / 1_000 * input_rate
        + output_tokens / 1_000 * output_rate
    )

    return CostRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_chars=None,
        output_chars=None,
        cost_usd=cost_usd,
    )


def google_translate_cost(input_chars: int, output_chars: int) -> CostRecord:
    """Calculate cost for a single Google Translate API call.

    Args:
        input_chars:  Number of characters in the source text.
        output_chars: Number of characters in the translated text.

    Returns:
        A CostRecord with char fields populated and token fields set to None.
    """
    total_chars = input_chars + output_chars
    cost_usd = total_chars / 1_000_000 * GOOGLE_TRANSLATE_USD_PER_1M_CHARS

    return CostRecord(
        input_tokens=None,
        output_tokens=None,
        input_chars=input_chars,
        output_chars=output_chars,
        cost_usd=cost_usd,
    )
