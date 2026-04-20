"""Diagnostic script for translation timeout investigation.

Runs a series of targeted tests to determine why section 3 of Form_001
consistently times out in the new MedicalTranslator pipeline.

Tests:
  T3a: Raw sync call (no asyncio) on section 3 — no timeout limit
  T3b: Raw async call (asyncio.run) on section 3 — no timeout limit
  T3c: Sequential async calls on sections 1→2→3 — isolates session state issue
  T4:  Minimal prompt on section 3 — rules out prompt complexity
  T5:  No response_mime_type on section 3 — rules out JSON mode overhead

Run:
  venv/bin/python3 scripts/diagnose_timeout.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT is not set.")
    sys.exit(1)

from google import genai
from google.genai import types
from google.genai.types import HttpOptions, HttpRetryOptions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FLASH_MODEL = "gemini-3-flash-preview"

_TRANSLATE_PROMPT = """\
You are a professional medical translator. Translate the Hebrew medical document section \
below into Russian.

## Clinical context
{context_header}

## Section metadata
- Label: {section_label}
- Position: {section_position}

## Source text (Hebrew)
{hebrew_text}

## Instructions
- Translate for a patient (non-medical reader). Use plain, accessible Russian.
- Preserve all medical terms, drug names, dosages, dates, diagnoses, and measurements exactly.
- Do not add commentary, footnotes, or markdown formatting.
- Return a JSON object with a single key "russian_translation" whose value is the plain \
Russian translation string.

Example output format:
{{"russian_translation": "..."}}
"""


def _load_sections(n: int) -> list[dict]:
    """Return first n sections from Form_001 split cache."""
    cache = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))
    return cache["Form_001"]["sections"][:n]


def _new_client() -> genai.Client:
    """Create a fresh genai.Client (not thread-local)."""
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    return genai.Client(vertexai=True, project=project, location=location)


def _fmt_prompt(section: dict, context_header: str, total: int) -> str:
    return _TRANSLATE_PROMPT.format(
        context_header=context_header,
        section_label=section["label"],
        section_position=f"{section['index']} of {total}",
        hebrew_text=section["hebrew_text"],
    )


# ---------------------------------------------------------------------------
# T3a: Raw SYNC call on section 3 — no timeout, no asyncio
# ---------------------------------------------------------------------------
def t3a_sync_section3():
    print("\n=== T3a: Sync call on section 3 (no timeout) ===")
    sections = _load_sections(3)
    sec = sections[2]
    context_header = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))["Form_001"]["context_header"]
    prompt = _fmt_prompt(sec, context_header, 13)
    client = _new_client()
    print(f"Section: {sec['label']!r} ({len(sec['hebrew_text'])} chars)")
    start = time.monotonic()
    response = client.models.generate_content(
        model=_FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            http_options=HttpOptions(timeout=300_000),
        ),
    )
    elapsed = time.monotonic() - start
    print(f"DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")
    print(f"Response (first 200 chars): {response.text[:200]!r}")


# ---------------------------------------------------------------------------
# T3b: Async call on section 3 via asyncio.run — no timeout
# ---------------------------------------------------------------------------
def t3b_async_section3():
    print("\n=== T3b: asyncio.run() on section 3 (no timeout) ===")
    sections = _load_sections(3)
    sec = sections[2]
    context_header = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))["Form_001"]["context_header"]
    prompt = _fmt_prompt(sec, context_header, 13)
    client = _new_client()
    print(f"Section: {sec['label']!r} ({len(sec['hebrew_text'])} chars)")

    async def _call():
        return await client.aio.models.generate_content(
            model=_FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                http_options=HttpOptions(timeout=300_000),
            ),
        )

    start = time.monotonic()
    response = asyncio.run(_call())
    elapsed = time.monotonic() - start
    print(f"DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")
    print(f"Response (first 200 chars): {response.text[:200]!r}")


# ---------------------------------------------------------------------------
# T3c: Sequential asyncio.run() on sections 1→2→3 with same client
#       Isolates whether the issue is client/session state after sec 1 and 2
# ---------------------------------------------------------------------------
def t3c_sequential_async_shared_client():
    print("\n=== T3c: Sequential asyncio.run() on sec 1→2→3, SHARED client ===")
    cache = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))
    sections = cache["Form_001"]["sections"][:3]
    context_header = cache["Form_001"]["context_header"]
    client = _new_client()  # one client, reused across all 3 calls

    for sec in sections:
        prompt = _fmt_prompt(sec, context_header, 13)
        print(f"\nSection {sec['index']}: {sec['label']!r} ({len(sec['hebrew_text'])} chars) ...")

        async def _call(p=prompt):
            return await client.aio.models.generate_content(
                model=_FLASH_MODEL,
                contents=p,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    http_options=HttpOptions(timeout=300_000),
                ),
            )

        start = time.monotonic()
        response = asyncio.run(_call())
        elapsed = time.monotonic() - start
        print(f"  DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")


# ---------------------------------------------------------------------------
# T3d: Sequential asyncio.run() on sections 1→2→3 with FRESH client per call
#       Rules out the shared-client/session-state hypothesis
# ---------------------------------------------------------------------------
def t3d_sequential_async_fresh_client():
    print("\n=== T3d: Sequential asyncio.run() on sec 1→2→3, FRESH client per call ===")
    cache = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))
    sections = cache["Form_001"]["sections"][:3]
    context_header = cache["Form_001"]["context_header"]

    for sec in sections:
        prompt = _fmt_prompt(sec, context_header, 13)
        client = _new_client()  # fresh client per section
        print(f"\nSection {sec['index']}: {sec['label']!r} ({len(sec['hebrew_text'])} chars) ...")

        async def _call(p=prompt, c=client):
            return await c.aio.models.generate_content(
                model=_FLASH_MODEL,
                contents=p,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    http_options=HttpOptions(timeout=300_000),
                ),
            )

        start = time.monotonic()
        response = asyncio.run(_call())
        elapsed = time.monotonic() - start
        print(f"  DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")


# ---------------------------------------------------------------------------
# T4: Minimal prompt on section 3 — rules out prompt complexity/length
# ---------------------------------------------------------------------------
def t4_minimal_prompt():
    print("\n=== T4: Minimal prompt on section 3 ===")
    sections = _load_sections(3)
    sec = sections[2]
    minimal_prompt = f'Translate this Hebrew medical text to Russian. Return JSON: {{"russian_translation": "..."}}.\n\nText:\n{sec["hebrew_text"]}'
    client = _new_client()
    print(f"Section 3: {len(sec['hebrew_text'])} chars Hebrew, minimal prompt")
    start = time.monotonic()
    response = client.models.generate_content(
        model=_FLASH_MODEL,
        contents=minimal_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            http_options=HttpOptions(timeout=300_000),
        ),
    )
    elapsed = time.monotonic() - start
    print(f"DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")


# ---------------------------------------------------------------------------
# T5b: Full prompt + thinking_budget=0 — does disabling thinking fix it?
# ---------------------------------------------------------------------------
def t5b_no_thinking():
    print("\n=== T5b: Full prompt + thinking_budget=0 on section 3 ===")
    sections = _load_sections(3)
    sec = sections[2]
    context_header = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))["Form_001"]["context_header"]
    prompt = _fmt_prompt(sec, context_header, 13)
    client = _new_client()
    print(f"Section 3: {len(sec['hebrew_text'])} chars Hebrew, thinking disabled")
    start = time.monotonic()
    response = client.models.generate_content(
        model=_FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=HttpOptions(timeout=300_000),
        ),
    )
    elapsed = time.monotonic() - start
    print(f"DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")
    print(f"Response (first 200 chars): {response.text[:200]!r}")


# ---------------------------------------------------------------------------
# T5: No response_mime_type on section 3 — rules out JSON mode overhead
# ---------------------------------------------------------------------------
def t5_no_json_mode():
    print("\n=== T5: No response_mime_type (plain text) on section 3 ===")
    sections = _load_sections(3)
    sec = sections[2]
    context_header = json.loads((_PROJECT_ROOT / "data" / "split_cache.json").read_text("utf-8"))["Form_001"]["context_header"]
    prompt = _fmt_prompt(sec, context_header, 13)
    client = _new_client()
    print(f"Section 3: {len(sec['hebrew_text'])} chars Hebrew, no JSON mode")
    start = time.monotonic()
    response = client.models.generate_content(
        model=_FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            http_options=HttpOptions(timeout=300_000),
        ),
    )
    elapsed = time.monotonic() - start
    print(f"DONE in {elapsed:.1f}s | tokens_in={response.usage_metadata.prompt_token_count} out={response.usage_metadata.candidates_token_count}")
    print(f"Response (first 200 chars): {response.text[:200]!r}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose translation timeout")
    parser.add_argument("--test", choices=["t3a", "t3b", "t3c", "t3d", "t4", "t5b", "t5", "all"], default="all")
    args = parser.parse_args()

    if args.test in ("t3a", "all"):
        t3a_sync_section3()
    if args.test in ("t3b", "all"):
        t3b_async_section3()
    if args.test in ("t3c", "all"):
        t3c_sequential_async_shared_client()
    if args.test in ("t3d", "all"):
        t3d_sequential_async_fresh_client()
    if args.test in ("t4", "all"):
        t4_minimal_prompt()
    if args.test in ("t5b", "all"):
        t5b_no_thinking()
    if args.test in ("t5", "all"):
        t5_no_json_mode()

    print("\n=== Done ===")
