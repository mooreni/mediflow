"""Benchmark runner: orchestrates scenario execution and result persistence.

Processes all eval documents concurrently within each scenario using a
ThreadPoolExecutor. DB writes are serialised with a threading.Lock.
Scenarios themselves run sequentially (called one at a time by the CLI).
"""

from __future__ import annotations

import datetime as _dt
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.benchmark.dataset import DatasetDoc
from src.benchmark.db import BenchmarkDB
from src.benchmark.scenarios.base import TranslationResult, TranslationScenario
from src.evaluation import llm_judge

DOC_WORKERS = 4  # concurrent docs per scenario; kept low to avoid Vertex AI 504 DEADLINE_EXCEEDED under quota pressure
MAX_RETRIES = 5   # per-doc retry attempts before skipping
_RETRY_BASE_DELAY = 30.0  # seconds; doubles on each subsequent attempt (503 needs ~30s to recover)


def _ts() -> str:
    """Return a short HH:MM:SS timestamp for log lines."""
    return _dt.datetime.now().strftime("%H:%M:%S")


def run_scenario(
    scenario: TranslationScenario,
    train_docs: list[DatasetDoc],
    eval_docs: list[DatasetDoc],
    db: BenchmarkDB,
    doc_workers: int = DOC_WORKERS,
) -> None:
    """Run a single translation scenario against all eval docs and persist results.

    Skips the scenario if it is already marked complete in the database.
    If the scenario was interrupted mid-run, resumes from where it left off —
    already-stored docs are skipped; only missing docs are re-processed.

    Calls scenario.train() before processing — this is a no-op for S1/S2/S3
    and runs DSPy optimisation for S4/S5. All eval docs are then processed
    concurrently; DB writes are serialised with a lock.

    Each failing doc is retried up to MAX_RETRIES times with exponential
    backoff before being logged as skipped.

    Args:
        scenario: The translation scenario to run.
        train_docs: Training documents passed to scenario.train() (S1/S2/S3 ignore these).
        eval_docs: Evaluation documents to translate and score.
        db: Open database instance for persisting results.
        doc_workers: Number of concurrent worker threads for document processing.

    Returns:
        None. Side effects: all successful translation results and evaluations
        are written to the database and committed immediately; scenario is
        marked complete in the database when finished; failed/skipped documents
        are logged to stdout only and not persisted.
    """
    if db.scenario_is_complete(scenario.name):
        print(f"[{_ts()}] Skipping {scenario.name} — already complete")
        return

    scenario_id = db.get_or_create_scenario(
        scenario.name,
        scenario.description,
        test_doc_count=len(eval_docs),
        train_doc_count=None,
    )

    done_count = db.get_result_count(scenario_id)
    if done_count == len(eval_docs):
        db.mark_scenario_complete(scenario_id)
        print(f"[{_ts()}] Skipping {scenario.name} — all {done_count} docs already done, marking complete")
        return
    remaining = len(eval_docs) - done_count
    if done_count > 0:
        print(f"[{_ts()}]   Resuming {scenario.name} — {done_count}/{len(eval_docs)} docs already done, {remaining} remaining")

    scenario.train(train_docs)

    db_lock = threading.Lock()
    completed_count = done_count
    completed_lock = threading.Lock()
    scenario_start = time.monotonic()

    def _translate_and_evaluate(doc: DatasetDoc) -> tuple:
        """Translate a document and score it with the LLM judge, with retries.

        Retries both translation and evaluation as a unit up to MAX_RETRIES
        times with exponential backoff. Evaluate is retried together with
        translate because a failed evaluation invalidates the translation result.

        Args:
            doc: The document to translate and evaluate.

        Returns:
            A tuple of (TranslationResult, EvalResult) on success.

        Raises:
            RuntimeError: Wraps the last exception if all retry attempts fail,
                          including doc id, scenario name, and attempt count.
        """
        last_exc: Exception | None = None
        result = None
        eval_result = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(f"[{_ts()}]   TRANSLATE {doc.doc_id} attempt {attempt + 1}/{MAX_RETRIES + 1}")
                result = scenario.translate(doc)
                print(f"[{_ts()}]   TRANSLATE {doc.doc_id} done ({result.elapsed_sec:.1f}s, cost=${result.cost.cost_usd:.4f})")

                print(f"[{_ts()}]   EVALUATE {doc.doc_id} ...")
                eval_result = llm_judge.score(
                    source=doc.hebrew_text,
                    hypothesis=result.translation,
                )
                print(f"[{_ts()}]   EVALUATE {doc.doc_id} done (overall={eval_result.quality_score:.3f})")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, 10)
                    delay = _RETRY_BASE_DELAY * (2 ** attempt) + jitter
                    print(
                        f"[{_ts()}]   [RETRY {attempt + 1}/{MAX_RETRIES}] {doc.doc_id}: {exc}"
                        f" — retrying in {delay:.0f}s"
                    )
                    time.sleep(delay)
                else:
                    print(f"[{_ts()}]   [EXHAUSTED] {doc.doc_id}: all {MAX_RETRIES} retries failed. Last error: {exc}")

        if last_exc is not None:
            raise RuntimeError(
                f"Failed processing doc={doc.doc_id!r} in scenario={scenario.name!r}"
                f" after {MAX_RETRIES} retries: {last_exc}"
            ) from last_exc

        return result, eval_result

    def _persist_result(result: TranslationResult, eval_result: llm_judge.EvaluationResult) -> None:
        """Write a completed translation result and its evaluation scores to the database.

        Acquires db_lock to serialise concurrent writes from worker threads.

        Args:
            result: TranslationResult returned by the scenario.
            eval_result: EvalResult returned by the LLM judge.

        Returns:
            None. Side effect: one row is inserted into the database and committed.
        """
        with db_lock:
            db.insert_result(
                scenario_id=scenario_id,
                doc_id=result.doc_id,
                translation=result.translation,
                quality_score=eval_result.quality_score,
                errors=eval_result.errors,
                cost=result.cost,
                elapsed_sec=result.elapsed_sec,
            )

    def process_doc(doc: DatasetDoc) -> None:
        """Coordinate translation, evaluation, and persistence for one document.

        Skips silently if a result for this doc already exists (resume path).
        Delegates translate+evaluate to _translate_and_evaluate and persistence
        to _persist_result.

        Args:
            doc: The document to translate and evaluate.

        Returns:
            None. Side effects: if result does not already exist, translates the
            document and evaluates it with the LLM judge, then writes both
            translation and evaluation scores to the database via an atomic
            locked critical section.

        Raises:
            RuntimeError: Re-raised from _translate_and_evaluate if all retry
                          attempts fail, including doc id, scenario name, and
                          attempt count.
        """
        if db.result_exists(scenario_id, doc.doc_id):
            return

        print(f"[{_ts()}]   START  {doc.doc_id} (workers={doc_workers})")
        doc_start = time.monotonic()

        result, eval_result = _translate_and_evaluate(doc)
        _persist_result(result, eval_result)

        doc_elapsed = time.monotonic() - doc_start
        with completed_lock:
            nonlocal completed_count
            completed_count += 1
            current = completed_count
        pct = current / len(eval_docs) * 100
        print(
            f"[{_ts()}]   DONE   {doc.doc_id} "
            f"[{current}/{len(eval_docs)} = {pct:.0f}%] "
            f"total_elapsed={doc_elapsed:.1f}s score={eval_result.quality_score:.3f}"
        )

    print(f"[{_ts()}] Running {scenario.name} — {remaining} docs, {doc_workers} workers")
    with ThreadPoolExecutor(max_workers=doc_workers) as pool:
        futures = {pool.submit(process_doc, doc): doc for doc in eval_docs}
        for fut in as_completed(futures):
            doc = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[{_ts()}]   [SKIP] {doc.doc_id}: {exc}")

    scenario_elapsed = time.monotonic() - scenario_start
    print(f"[{_ts()}] FINISHED {scenario.name} in {scenario_elapsed:.1f}s")
    db.mark_scenario_complete(scenario_id)
