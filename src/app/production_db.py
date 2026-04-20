"""SQLite persistence layer for the MediFlow production translation pipeline.

Schema overview:
  runs            — one row per translation run (name, timestamps, doc count)
  results         — one row per translated document, FK → runs
  section_results — one row per document section, FK → results

No quality score or error columns are stored; this DB is for production
output only. Use BenchmarkDB (src/app/db.py) for evaluation runs.

All writes are serialised through a threading.Lock so a single connection
can be shared across a ThreadPoolExecutor without data races.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.app.translation.base import SectionTranslationResult
from src.app.translation.cost import CostRecord

_log = logging.getLogger(__name__)

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
    cost_usd      REAL NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
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
    elapsed_sec     REAL NOT NULL
);
"""


class ProductionDB:
    """SQLite persistence layer for MediFlow production translation runs."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database at the given path.

        Args:
            db_path: Filesystem path to the SQLite database file.
                     Created automatically if it does not exist.
        """
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.create_tables()

    def create_tables(self) -> None:
        """Create all tables if they do not already exist.

        Returns:
            None
        """
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_RESULTS)
        self._conn.execute(_CREATE_SECTION_RESULTS)
        self._conn.commit()

    def insert_run(self, name: str, description: str, doc_count: int) -> int:
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
            raise RuntimeError(
                f"INSERT into runs returned no lastrowid for name={name!r}"
            )
        return cursor.lastrowid

    def get_or_create_run(self, name: str, description: str, doc_count: int) -> int:
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
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM runs WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                return row["id"]
            run_at = datetime.now(timezone.utc).isoformat()
            cursor = self._conn.execute(
                """
                INSERT INTO runs (name, description, run_at, doc_count)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, run_at, doc_count),
            )
            self._conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError(
                    f"INSERT into runs returned no lastrowid for name={name!r}"
                )
            return cursor.lastrowid

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

    def insert_result_with_sections(
        self,
        run_id: int,
        doc_id: str,
        sections: list[SectionTranslationResult],
        cost: CostRecord,
        elapsed_sec: float,
    ) -> None:
        """Insert a document-level result and all per-section rows atomically.

        The full translation text is derived by joining each section's
        ``russian_text`` with ``"\\n\\n"``. Both the ``results`` insert and all
        ``section_results`` inserts happen within a single SQLite transaction
        so a crash leaves neither partial documents nor orphaned section rows.

        Args:
            run_id:      Foreign key referencing the parent run.
            doc_id:      Document identifier (e.g. ``"Form_039"``).
            sections:    Per-section translation results, in order.
            cost:        Aggregated API usage and cost (all sections).
            elapsed_sec: Total wall-clock seconds for the document.

        Returns:
            None

        Raises:
            ValueError: If sections is empty.
            RuntimeError: If the INSERT into results returned no lastrowid.
        """
        if not sections:
            raise ValueError(
                f"sections must not be empty for doc_id={doc_id!r}."
            )

        translation = "\n\n".join(s.russian_text for s in sections)

        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO results (
                        run_id, doc_id, translation,
                        cost_usd, input_tokens, output_tokens, elapsed_sec
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        doc_id,
                        translation,
                        cost.cost_usd,
                        cost.input_tokens,
                        cost.output_tokens,
                        elapsed_sec,
                    ),
                )
                result_id = cursor.lastrowid
                if result_id is None:
                    raise RuntimeError(
                        f"INSERT into results returned no lastrowid for doc_id={doc_id!r}."
                    )
                for section in sections:
                    self._conn.execute(
                        """
                        INSERT INTO section_results (
                            result_id, section_index, section_label,
                            section_hebrew, section_russian, elapsed_sec
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result_id,
                            section.section_index,
                            section.section_label,
                            section.hebrew_text,
                            section.russian_text,
                            section.elapsed_sec,
                        ),
                    )

    def close(self) -> None:
        """Close the database connection and release all resources.

        Returns:
            None
        """
        self._conn.close()
