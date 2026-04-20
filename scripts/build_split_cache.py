"""Pre-split all Hebrew source documents and persist the results to disk.

Run this script once before the first execution of ``scripts/run.py``, or any
time the source documents in ``data/`` change.  The resulting
``data/split_cache.json`` is loaded by ``scripts/run.py`` at startup so
documents are not re-split on every run.

The script is incremental: it loads any existing cache and only calls the
Gemini Flash splitter for documents whose ``doc_id`` is not already present.
This means it is safe to re-run after a partial failure — only the missing
documents will be processed.

Usage:
  python scripts/build_split_cache.py

Environment variables (via .env or shell):
  GOOGLE_CLOUD_PROJECT  — required
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT is not set. Add it to .env or your shell environment.")
    sys.exit(1)

from google.genai import errors as genai_errors

from src.app.data.loader import load_documents
from src.app.translation.splitter import Section, SplitResult, split_document

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_CACHE_PATH = _DATA_DIR / "split_cache.json"


def _load_split_cache(path: Path) -> dict[str, SplitResult]:
    """Load persisted section splits from a JSON cache file.

    Args:
        path: Path to the JSON cache file produced by this script or
              accumulated on-demand by ``scripts/run.py``.

    Returns:
        Dict mapping doc_id → SplitResult. Empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cache: dict[str, SplitResult] = {}
    for doc_id, obj in raw.items():
        sections = [
            Section(index=s["index"], label=s["label"], hebrew_text=s["hebrew_text"])
            for s in obj["sections"]
        ]
        cache[doc_id] = SplitResult(
            context_header=obj["context_header"],
            total_sections=obj["total_sections"],
            sections=sections,
        )
    return cache


def _save_split_cache(cache: dict[str, SplitResult], path: Path) -> None:
    """Serialize split results to a JSON file atomically.

    Writes to a sibling ``.tmp`` file then renames so the target is never
    partially written.

    Args:
        cache: Mapping of doc_id → SplitResult to persist.
        path:  Target file path; parent directory must already exist.

    Returns:
        None
    """
    data = {
        doc_id: {
            "context_header": sr.context_header,
            "total_sections": sr.total_sections,
            "sections": [
                {"index": s.index, "label": s.label, "hebrew_text": s.hebrew_text}
                for s in sr.sections
            ],
        }
        for doc_id, sr in cache.items()
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Atomic rename: guarantees the target is never partially written.
    os.replace(tmp, path)


def main() -> None:
    """Build the split cache incrementally and save it to disk.

    Loads any existing cache at ``_CACHE_PATH`` and only splits documents
    whose ``doc_id`` is not already present, then merges and rewrites the
    file.  This avoids re-splitting docs that were successfully split on a
    prior run.
    """
    print("Building split cache...")
    all_docs = load_documents(_DATA_DIR)

    existing: dict[str, SplitResult] = {}
    if _CACHE_PATH.exists():
        existing = _load_split_cache(_CACHE_PATH)
        print(f"Loaded {len(existing)} existing split results from {_CACHE_PATH}")

    missing_docs = [d for d in all_docs if d.doc_id not in existing]
    if not missing_docs:
        print(f"All {len(all_docs)} documents already split — nothing to do.")
        return

    print(f"Splitting {len(missing_docs)} new/missing documents (of {len(all_docs)} total)...")
    new_cache: dict[str, SplitResult] = {}
    for doc in missing_docs:
        print(f"  Splitting {doc.doc_id}...")
        try:
            new_cache[doc.doc_id] = split_document(doc.hebrew_text)
        except (ValueError, genai_errors.ClientError) as exc:
            print(f"  WARNING: Failed to split {doc.doc_id}: {type(exc).__name__}: {exc}")

    merged = {**existing, **new_cache}
    _save_split_cache(merged, _CACHE_PATH)
    print(f"Saved {len(merged)}/{len(all_docs)} split results to {_CACHE_PATH}")
    if len(merged) < len(all_docs):
        failed = len(all_docs) - len(merged)
        print(f"WARNING: {failed} document(s) failed to split and were excluded.")


if __name__ == "__main__":
    main()
