"""CLI entry point for the MediFlow production translation pipeline.

Translates Hebrew medical documents to Russian using a 3-step per-section pipeline:
  1. Flash translate  (Gemini Flash)            — draft Russian translation
  2. Midway MQM judge (Gemini Pro, conditional) — detects errors, drives correction
  3. Pro correct      (Gemini Pro, conditional) — post-edits draft when errors found

Sections within each document are translated concurrently via a single shared
ThreadPoolExecutor. Documents are processed sequentially by the main thread.

Usage:
  python scripts/run.py                          # all 120 documents
  python scripts/run.py --doc-type form          # one document type (30 docs)
  python scripts/run.py --doc-id Form_001        # single document

Environment variables (via .env or shell):
  GOOGLE_CLOUD_PROJECT  — required for all Vertex AI calls
  GOOGLE_CLOUD_LOCATION — optional, defaults to "us-central1"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Validate required environment variables before loading any heavy dependencies.
# Fail fast with a clear message rather than an obscure ImportError or auth error.
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT is not set. Add it to .env or your shell environment.")
    sys.exit(1)

from src.app.production_db import ProductionDB
from src.app.data.loader import DatasetDoc, load_documents
from src.app.translation.translator import MedicalTranslator
from src.app.translation.splitter import Section, split_document
from src.app.translation.base import SectionTranslationResult
from src.app.translation.cost import sum_costs


SECTION_WORKERS = 8
DB_PATH = Path("data/translations.db")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_translator: MedicalTranslator | None = None


class PartialDocumentError(Exception):
    """Raised by run_section_tasks when one or more sections fail after retries.

    Carries successful_results so callers can persist partial documents before
    handling the error.
    """

    def __init__(
        self,
        doc_id: str,
        failed_sections: list[tuple[int, str, BaseException]],
        successful_results: list,
    ) -> None:
        super().__init__(f"Doc {doc_id!r}: {len(failed_sections)} section(s) failed")
        self.doc_id = doc_id
        self.failed_sections = failed_sections
        self.successful_results = successful_results


def run_section_tasks(
    sections: list,
    section_fn: Callable,
    pool: ThreadPoolExecutor,
    *extra_args,
    doc_id: str = "",
) -> list:
    """Submit one task per section to pool and collect results in section order.

    Catches per-section exceptions without aborting in-flight tasks. Returns only
    successful results. If any section failed, raises PartialDocumentError after all
    futures resolve so the caller can persist the partial document first.

    Args:
        sections:    Ordered list of section objects to process.
        section_fn:  Callable invoked as section_fn(section, *extra_args).
        pool:        ThreadPoolExecutor to submit tasks into.
        *extra_args: Forwarded verbatim to section_fn after the section arg.
        doc_id:      Document identifier surfaced in PartialDocumentError.

    Returns:
        List of successful results in original section order.

    Raises:
        PartialDocumentError: If one or more sections fail, carrying successful_results.
    """
    futures = [pool.submit(section_fn, section, *extra_args) for section in sections]

    successful: list = []
    failures: list[tuple[int, str, BaseException]] = []

    for section, future in zip(sections, futures):
        try:
            successful.append(future.result())
        except Exception as exc:
            failures.append((section.index, section.label, exc))

    if failures:
        raise PartialDocumentError(
            doc_id=doc_id,
            failed_sections=failures,
            successful_results=successful,
        )

    return successful


def _ts() -> str:
    """Return a short HH:MM:SS timestamp string for log prefixes."""
    return datetime.now().strftime("%H:%M:%S")


def translate_document(
    doc: DatasetDoc,
    pool: ThreadPoolExecutor,
) -> list[SectionTranslationResult]:
    """Split a document and translate all sections concurrently.

    Args:
        doc:  The document to translate.
        pool: Shared ThreadPoolExecutor for section-level concurrency.

    Returns:
        List of SectionTranslationResult in section order.

    Raises:
        PartialDocumentError: If one or more sections fail, carrying successful_results.
    """
    split_result = split_document(doc.hebrew_text)

    def _section_fn(section: Section) -> SectionTranslationResult:
        return _translator.translate_section(  # type: ignore[union-attr]
            context_header=split_result.context_header,
            section=section,
            total_sections=split_result.total_sections,
        )

    return run_section_tasks(
        split_result.sections,
        _section_fn,
        pool,
        doc_id=doc.doc_id,
    )


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the MediFlow Hebrew→Russian medical translation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run.py\n"
            "  python scripts/run.py --doc-type form\n"
            "  python scripts/run.py --doc-id Form_001"
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
    """Parse arguments, load documents, and run the production translation pipeline.

    Documents are processed sequentially. Sections within each document are
    dispatched to a single shared ThreadPoolExecutor. Failed sections are
    caught via PartialDocumentError; partial documents are still persisted.
    """
    global _translator

    args = _parse_args()

    if args.doc_type is not None and args.doc_id is not None:
        print("ERROR: --doc-type and --doc-id are mutually exclusive.")
        sys.exit(1)

    db = ProductionDB(DB_PATH)

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
    run_name = f"mediflow_{run_label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_id = db.get_or_create_run(
        name=run_name,
        description=f"MedicalTranslator pipeline — filter: {run_label}",
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
            results: list[SectionTranslationResult] = []
            try:
                results = translate_document(doc, pool)
            except PartialDocumentError as e:
                results = e.successful_results
                for idx, label, exc in e.failed_sections:
                    print(
                        f"[{_ts()}] FAIL   {doc.doc_id} section {idx} "
                        f"({label!r}): {exc}"
                    )

            elapsed = time.monotonic() - doc_start

            if results:
                total_cost = sum_costs([sr.cost for sr in results])
                db.insert_result_with_sections(
                    run_id=run_id,
                    doc_id=doc.doc_id,
                    sections=results,
                    cost=total_cost,
                    elapsed_sec=elapsed,
                )
                print(
                    f"[{_ts()}] DONE   {doc.doc_id} "
                    f"({len(results)} sections, ${total_cost.cost_usd:.4f})"
                )
            else:
                print(f"[{_ts()}] SKIP   {doc.doc_id} (all sections failed — not stored)")

    db.mark_run_complete(run_id)
    print(f"\nRun {run_name!r} complete.")


if __name__ == "__main__":
    main()
