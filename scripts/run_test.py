"""CLI entry point for the MediFlow test pipeline.

Translates Hebrew medical documents and runs a final MQM evaluation on each
section so quality scores can be stored in BenchmarkDB for analysis.

For sections that were NOT corrected by the Pro step, the midway evaluation
result is reused directly — no redundant API call. For sections that WERE
corrected, a final judge call scores the corrected output.

Usage:
  python scripts/run_test.py                         # all 120 documents
  python scripts/run_test.py --doc-type form         # one document type (30 docs)
  python scripts/run_test.py --doc-id Form_001       # single document

Environment variables (via .env or shell):
  GOOGLE_CLOUD_PROJECT  — required for all Vertex AI calls
  GOOGLE_CLOUD_LOCATION — optional, defaults to "us-central1"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT is not set. Add it to .env or your shell environment.")
    sys.exit(1)

from scripts.run import PartialDocumentError, run_section_tasks
from src.app.db import BenchmarkDB
from src.app.data.loader import load_documents
from src.app.evaluation.judge import EvaluationResult, score_section
from src.app.translation.translator import MedicalTranslator
from src.app.translation.splitter import Section, SplitResult, split_document
from src.app.translation.base import SectionTranslationResult
from src.app.translation.cost import gemini_cost, sum_costs


SECTION_WORKERS = 8
DB_PATH = Path("data/benchmark.db")
SPLIT_CACHE_PATH = Path("data/split_cache.json")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_translator: MedicalTranslator | None = None


def translate_and_evaluate_section(
    section: Section,
    context_header: str,
    total_sections: int,
) -> tuple[SectionTranslationResult, EvaluationResult]:
    """Translate a section and produce a final EvaluationResult.

    Reuses midway judge output for uncorrected sections to avoid a redundant
    API call. Only corrected sections incur an extra score_section() call.

    Args:
        section:         The document section to translate.
        context_header:  Compact clinical context string for this document.
        total_sections:  Total section count in the document.

    Returns:
        A (SectionTranslationResult, EvaluationResult) pair.
    """
    result = _translator.translate_section(  # type: ignore[union-attr]
        context_header=context_header,
        section=section,
        total_sections=total_sections,
    )

    if result.was_corrected:
        eval_result = score_section(
            section_hebrew=result.hebrew_text,
            section_russian=result.russian_text,
            context_header=context_header,
            section_index=result.section_index,
            total_sections=total_sections,
            section_label=result.section_label,
        )
    else:
        eval_result = EvaluationResult(
            quality_score=result.midway_score,
            errors=result.midway_errors,
        )

    return result, eval_result


def _split_result_from_cache(entry: dict) -> SplitResult:
    """Reconstruct a SplitResult from a cached JSON dict entry."""
    return SplitResult(
        context_header=entry["context_header"],
        total_sections=entry["total_sections"],
        sections=[Section(**s) for s in entry["sections"]],
    )


def translate_document(
    doc,
    pool: ThreadPoolExecutor,
    split_cache: dict,
    cache_lock: threading.Lock,
) -> list[tuple[SectionTranslationResult, EvaluationResult]]:
    """Split a document and translate+evaluate all sections concurrently.

    Reuses a cached SplitResult when available; otherwise splits the document
    and updates the cache under a lock.

    Args:
        doc:         The document to process.
        pool:        Shared ThreadPoolExecutor for section-level concurrency.
        split_cache: In-memory dict keyed by doc_id.
        cache_lock:  Lock protecting writes to split_cache.

    Returns:
        List of (SectionTranslationResult, EvaluationResult) pairs in section order.

    Raises:
        PartialDocumentError: If one or more sections fail, carrying successful_results.
    """
    if doc.doc_id in split_cache:
        split_result = _split_result_from_cache(split_cache[doc.doc_id])
    else:
        split_result = split_document(doc.hebrew_text)
        with cache_lock:
            split_cache[doc.doc_id] = {
                "context_header": split_result.context_header,
                "total_sections": split_result.total_sections,
                "sections": [dataclasses.asdict(s) for s in split_result.sections],
            }

    def _section_fn(section: Section) -> tuple[SectionTranslationResult, EvaluationResult]:
        return translate_and_evaluate_section(
            section=section,
            context_header=split_result.context_header,
            total_sections=split_result.total_sections,
        )

    return run_section_tasks(
        split_result.sections,
        _section_fn,
        pool,
        doc_id=doc.doc_id,
    )


def _ts() -> str:
    """Return a short HH:MM:SS timestamp string for log prefixes."""
    return datetime.now().strftime("%H:%M:%S")


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the MediFlow Hebrew→Russian test pipeline (translate + evaluate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_test.py\n"
            "  python scripts/run_test.py --doc-type form\n"
            "  python scripts/run_test.py --doc-id Form_001"
        ),
    )
    parser.add_argument(
        "--doc-type",
        choices=["form", "summary", "prescript", "referral"],
        help="Translate only documents of this type. Omit to translate all types.",
    )
    parser.add_argument(
        "--doc-id",
        metavar="DOC_ID",
        help="Translate a single document by its ID (e.g. Form_001). Omit for all.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments, load documents, and run the test translation+evaluation pipeline.

    Documents are processed sequentially. Sections within each document are
    dispatched to a single shared ThreadPoolExecutor. Failed sections are
    caught via PartialDocumentError; partial documents are still persisted.
    The split cache is persisted to disk after all documents are processed.
    """
    global _translator

    args = _parse_args()

    if args.doc_type is not None and args.doc_id is not None:
        print("ERROR: --doc-type and --doc-id are mutually exclusive.")
        sys.exit(1)

    split_cache: dict = {}
    if SPLIT_CACHE_PATH.exists():
        split_cache = json.loads(SPLIT_CACHE_PATH.read_text(encoding="utf-8"))
    cache_lock = threading.Lock()

    db = BenchmarkDB(DB_PATH)
    db.create_tables()

    docs = load_documents(_DATA_DIR, doc_type=args.doc_type)
    if args.doc_id is not None:
        docs = [d for d in docs if d.doc_id == args.doc_id]
        if not docs:
            print(
                f"ERROR: doc_id {args.doc_id!r} not found in dataset "
                f"(type filter: {args.doc_type!r})."
            )
            sys.exit(1)

    run_label = args.doc_id or args.doc_type or "all"
    run_name = f"mediflow_test_{run_label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_id = db.get_or_create_run(
        name=run_name,
        description=f"MediFlow test pipeline — filter: {run_label}",
        doc_count=len(docs),
    )
    print(f"Run: {run_name!r} (id={run_id}) | {len(docs)} docs | DB: {DB_PATH}")
    print()

    _translator = MedicalTranslator()

    with ThreadPoolExecutor(max_workers=SECTION_WORKERS) as pool:
        for doc in docs:
            if db.result_exists(run_id, doc.doc_id):
                print(f"[{_ts()}] SKIP   {doc.doc_id} (already in DB)")
                continue

            print(f"[{_ts()}] START  {doc.doc_id}")
            doc_start = time.monotonic()
            results: list[tuple[SectionTranslationResult, EvaluationResult]] = []
            try:
                results = translate_document(doc, pool, split_cache, cache_lock)
            except PartialDocumentError as e:
                results = e.successful_results
                for idx, label, exc in e.failed_sections:
                    print(
                        f"[{_ts()}] FAIL   {doc.doc_id} section {idx} "
                        f"({label!r}): {exc}"
                    )

            elapsed = time.monotonic() - doc_start

            if results:
                section_results = [r for r, _ in results]
                eval_results = [ev for _, ev in results]

                section_costs = [r.cost for r in section_results]
                final_eval_costs = [
                    gemini_cost(ev.prompt_tokens, ev.completion_tokens, "pro")
                    for r, ev in results
                    if r.was_corrected
                ]
                total_cost = sum_costs(section_costs + final_eval_costs)

                db.insert_result_with_sections(
                    run_id=run_id,
                    doc_id=doc.doc_id,
                    sections=section_results,
                    section_eval_results=eval_results,
                    cost=total_cost,
                    elapsed_sec=elapsed,
                )
                print(
                    f"[{_ts()}] DONE   {doc.doc_id} "
                    f"({len(section_results)} sections, ${total_cost.cost_usd:.4f})"
                )
            else:
                print(f"[{_ts()}] SKIP   {doc.doc_id} (all sections failed — not stored)")

    SPLIT_CACHE_PATH.write_text(
        json.dumps(split_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    db.mark_run_complete(run_id)
    print(f"\nRun {run_name!r} complete.")


if __name__ == "__main__":
    main()
