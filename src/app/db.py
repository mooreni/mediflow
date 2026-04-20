"""SQLite persistence layer for the MediFlow translation pipeline.

Schema overview:
  runs            — one row per translation run (name, timestamps, doc count)
  results         — one row per translated document, FK → runs
  section_results — one row per document section, FK → results

Each insert commits immediately so partial runs survive crashes.
Section-based documents persist per-section rows in the child
``section_results`` table via :meth:`insert_result_with_sections`,
which wraps both the document row and all section rows in a single
atomic transaction.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.app.translation.cost import CostRecord
from src.app.evaluation.judge import _calc_score

if TYPE_CHECKING:
    from src.app.translation.base import SectionTranslationResult
    from src.app.evaluation.judge import EvaluationResult


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL UNIQUE,
    description    TEXT,
    run_at         TEXT NOT NULL,
    completed_at   TEXT,
    doc_count      INTEGER NOT NULL
);
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    doc_id        TEXT NOT NULL,
    translation   TEXT NOT NULL,
    quality_score REAL NOT NULL,
    errors        TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL NOT NULL,
    elapsed_sec   REAL NOT NULL
);
"""

_CREATE_SECTION_RESULTS = """
CREATE TABLE IF NOT EXISTS section_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES results(id),
    section_index   INTEGER NOT NULL,
    section_label   TEXT NOT NULL,
    section_hebrew  TEXT NOT NULL,
    section_russian TEXT NOT NULL,
    quality_score   REAL NOT NULL,
    errors          TEXT
);
"""


class BenchmarkDB:
    """SQLite persistence layer for MediFlow translation runs and results."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database at the given path.

        Args:
            db_path: Filesystem path to the SQLite database file.
                     Created automatically if it does not exist.
        """
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Enforce referential integrity between runs → results → section_results.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def create_tables(self) -> None:
        """Create all tables if they do not already exist.

        Returns:
            None
        """
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_RESULTS)
        self._conn.execute(_CREATE_SECTION_RESULTS)
        self._conn.commit()

    def run_exists(self, name: str) -> bool:
        """Return True if a run with this name is already in the DB.

        Args:
            name: The unique run name to look up.

        Returns:
            True if a matching row exists, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id FROM runs WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def run_is_complete(self, name: str) -> bool:
        """Return True only if the run exists and has been fully completed.

        A run that was interrupted mid-flight will have a row in the DB but
        a NULL completed_at, and is therefore resumable (returns False).

        Args:
            name: The unique run name to look up.

        Returns:
            True if the run row exists and completed_at is non-null,
            False otherwise.
        """
        row = self._conn.execute(
            "SELECT completed_at FROM runs WHERE name = ?", (name,)
        ).fetchone()
        return row is not None and row["completed_at"] is not None

    def result_exists(self, run_id: int, doc_id: str) -> bool:
        """Return True if a result for this (run, doc) pair is already stored.

        Used during resume to skip docs that were successfully processed in a
        previous interrupted run.

        Args:
            run_id: Primary key of the parent run row.
            doc_id: Document identifier to look up.

        Returns:
            True if a matching result row exists, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id FROM results WHERE run_id = ? AND doc_id = ?",
            (run_id, doc_id),
        ).fetchone()
        return row is not None

    def get_result_count(self, run_id: int) -> int:
        """Return the number of result rows stored for the given run.

        Used to determine resume state: if the count equals the number of
        documents in the run, the run can be marked complete without
        re-processing anything.

        Args:
            run_id: Primary key of the run row to count results for.

        Returns:
            Integer count of result rows for this run (0 if none).
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row[0]

    def mark_run_complete(self, run_id: int) -> None:
        """Stamp the run row with the current UTC time as completed_at.

        Called after all documents in a run have been successfully processed.
        Until this is called the run is considered in-progress and will be
        resumed (not skipped) on re-run.

        Args:
            run_id: Primary key of the run row to update.

        Returns:
            None
        """
        completed_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET completed_at = ? WHERE id = ?",
                (completed_at, run_id),
            )
            self._conn.commit()

    def get_or_create_run(
        self,
        name: str,
        description: str,
        doc_count: int,
    ) -> int:
        """Return the id of an existing run row, or insert and return a new one.

        Used on resume: if the run was interrupted previously its row already
        exists and we reuse it so that existing result rows remain correctly
        linked.

        Args:
            name: Unique run name.
            description: Human-readable description of the run.
            doc_count: Number of documents to be translated in this run.

        Returns:
            The integer primary key of the run row (existing or newly created).
        """
        row = self._conn.execute(
            "SELECT id FROM runs WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return row["id"]
        return self.insert_run(name, description, doc_count)

    def insert_run(
        self,
        name: str,
        description: str,
        doc_count: int,
    ) -> int:
        """Insert a run row and return its auto-assigned id.

        Args:
            name: Unique run name (must not already exist in the DB).
            description: Human-readable description of the run.
            doc_count: Number of documents to be translated in this run.

        Returns:
            The integer primary key of the newly inserted run row.

        Raises:
            sqlite3.IntegrityError: If the name already exists.
            RuntimeError: If the INSERT did not return a valid row id.
        """
        run_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO runs (name, description, run_at, doc_count)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, run_at, doc_count),
            )
            self._conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError(f"INSERT into runs returned no lastrowid for name={name!r}")
        return cursor.lastrowid

    def insert_result(
        self,
        run_id: int,
        doc_id: str,
        translation: str,
        quality_score: float,
        errors: list[dict],
        cost: CostRecord,
        elapsed_sec: float,
    ) -> None:
        """Insert a single translation result row and commit immediately.

        Args:
            run_id: Foreign key referencing the parent run row.
            doc_id: Identifier of the source document being translated.
            translation: The translated text produced by the pipeline.
            quality_score: MQM penalty-based quality score (0.0–1.0).
            errors: List of MQM error dicts (span, category, severity, justification).
            cost: Token and cost accounting for this translation call.
            elapsed_sec: Wall-clock seconds taken to produce the translation.

        Returns:
            None
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO results (
                    run_id, doc_id, translation,
                    quality_score, errors,
                    input_tokens, output_tokens,
                    cost_usd, elapsed_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    translation,
                    quality_score,
                    json.dumps(errors),  # serialised as JSON; deserialised by load_all_results()
                    cost.input_tokens,
                    cost.output_tokens,
                    cost.cost_usd,
                    elapsed_sec,
                ),
            )
            self._conn.commit()

    def load_all_results(self) -> list[dict[str, Any]]:
        """Return all result rows joined with their run name.

        The ``errors`` field is deserialised from its JSON-encoded form back
        into a Python list for each row.

        Returns:
            List of dicts, one per result row, ordered by run name then
            doc_id. Each dict includes all result columns plus ``run_name``.
        """
        rows = self._conn.execute(
            """
            SELECT r.*, s.name AS run_name
            FROM results r
            JOIN runs s ON r.run_id = s.id
            ORDER BY s.name, r.doc_id
            """
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            raw_errors = d.get("errors") or "[]"
            try:
                d["errors"] = json.loads(raw_errors)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"load_all_results: failed to deserialise errors JSON for "
                    f"result id={d.get('id')!r}; got {raw_errors!r}; "
                    f"expected a valid JSON array."
                ) from exc
            result.append(d)
        return result

    def insert_section_result(
        self,
        result_id: int,
        section_index: int,
        section_label: str,
        section_hebrew: str,
        section_russian: str,
        quality_score: float,
        errors: list[dict],
    ) -> None:
        """Insert a single section result row without committing.

        This method is an internal helper intended to be called only from
        within the transaction managed by :meth:`insert_result_with_sections`.
        It does not commit so the parent method can wrap the whole document
        (document row + all section rows) in one atomic transaction.

        Args:
            result_id:       Foreign key referencing the parent ``results`` row.
            section_index:   1-based position of this section in the document.
            section_label:   Human-readable label for this section.
            section_hebrew:  Original Hebrew text of the section.
            section_russian: Translated Russian text of the section.
            quality_score:   MQM quality score for this section (0.0–1.0).
            errors:          List of MQM error dicts for this section.

        Returns:
            None
        """
        self._conn.execute(
            """
            INSERT INTO section_results (
                result_id, section_index, section_label,
                section_hebrew, section_russian, quality_score, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                section_index,
                section_label,
                section_hebrew,
                section_russian,
                quality_score,
                json.dumps(errors),  # serialised as JSON; deserialised by load_section_results()
            ),
        )

    def insert_result_with_sections(
        self,
        run_id: int,
        doc_id: str,
        sections: list[SectionTranslationResult],
        section_eval_results: list[EvaluationResult],
        cost: CostRecord,
        elapsed_sec: float,
    ) -> None:
        """Insert a document-level result and all per-section rows atomically.

        The document-level ``results`` row is built by aggregating section data:
        Russian texts are joined with ``"\\n\\n"``, the quality score is computed
        via penalty-based MQM scoring over all combined section errors (not an
        average of per-section scores), and errors are the concatenated lists
        from every section. Both the ``results`` insert and all
        ``section_results`` inserts happen within a single SQLite transaction
        so a crash leaves neither partial documents nor orphaned section rows.

        ``sections`` and ``section_eval_results`` must have the same length and
        be ordered so that ``sections[i]`` corresponds to
        ``section_eval_results[i]``.

        Args:
            run_id:               Foreign key referencing the parent run.
            doc_id:               Document identifier (e.g. ``"Form_039"``).
            sections:             Per-section translation results, in order.
            section_eval_results: Per-section evaluation results, in order.
            cost:                 Aggregated API usage and cost (all sections).
            elapsed_sec:          Total wall-clock seconds for the document.

        Returns:
            None

        Raises:
            ValueError: If ``sections`` and ``section_eval_results`` have
                        different lengths or are empty.
        """
        if len(sections) != len(section_eval_results):
            raise ValueError(
                f"sections and section_eval_results must have equal length; "
                f"got {len(sections)} sections and {len(section_eval_results)} "
                f"eval results for doc_id={doc_id!r}."
            )
        if not sections:
            raise ValueError(
                f"sections must not be empty for doc_id={doc_id!r}."
            )

        translation = "\n\n".join(s.russian_text for s in sections)
        all_errors: list[dict] = []
        for e in section_eval_results:
            all_errors.extend(e.errors)
        # Use summed-penalty scoring (MQM-correct) rather than averaging per-section
        # scores, which would dilute penalties and inflate the document-level score.
        quality_score = _calc_score(all_errors)

        # Wrap both the document-level insert and all section inserts in one
        # transaction so a mid-write crash never leaves orphaned section rows.
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO results (
                    run_id, doc_id, translation,
                    quality_score, errors,
                    input_tokens, output_tokens,
                    cost_usd, elapsed_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    translation,
                    quality_score,
                    json.dumps(all_errors),
                    cost.input_tokens,
                    cost.output_tokens,
                    cost.cost_usd,
                    elapsed_sec,
                ),
            )
            result_id = cursor.lastrowid
            if result_id is None:
                raise RuntimeError(
                    f"INSERT into results returned no lastrowid for doc_id={doc_id!r}."
                )
            for section, eval_result in zip(sections, section_eval_results):
                self.insert_section_result(
                    result_id=result_id,
                    section_index=section.section_index,
                    section_label=section.section_label,
                    section_hebrew=section.hebrew_text,
                    section_russian=section.russian_text,
                    quality_score=eval_result.quality_score,
                    errors=eval_result.errors,
                )

    def load_section_results(self, result_id: int) -> list[dict[str, Any]]:
        """Return all section result rows for the given parent result.

        The ``errors`` field is deserialised from its JSON-encoded form back
        into a Python list for each row.

        Args:
            result_id: Primary key of the parent ``results`` row.

        Returns:
            List of dicts, one per section row, ordered by ``section_index``.
            Each dict includes all columns from the ``section_results`` table.
        """
        rows = self._conn.execute(
            """
            SELECT *
            FROM section_results
            WHERE result_id = ?
            ORDER BY section_index
            """,
            (result_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            raw_errors = d.get("errors") or "[]"
            try:
                d["errors"] = json.loads(raw_errors)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"load_section_results: failed to deserialise errors JSON for "
                    f"section_result id={d.get('id')!r}; got {raw_errors!r}; "
                    f"expected a valid JSON array."
                ) from exc
            result.append(d)
        return result

    def close(self) -> None:
        """Close the database connection and release all resources.

        Returns:
            None
        """
        self._conn.close()
