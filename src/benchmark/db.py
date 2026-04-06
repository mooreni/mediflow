"""SQLite persistence layer for the translation benchmark.

Wraps a SQLite connection with typed methods for creating tables,
inserting scenarios/results, and querying benchmark data. Each insert
commits immediately so partial runs survive crashes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark.cost import CostRecord


_CREATE_SCENARIOS = """
CREATE TABLE IF NOT EXISTS scenarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    run_at          TEXT NOT NULL,
    train_doc_count INTEGER,
    test_doc_count  INTEGER NOT NULL
);
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id   INTEGER NOT NULL REFERENCES scenarios(id),
    doc_id        TEXT NOT NULL,
    translation   TEXT NOT NULL,
    quality_score REAL NOT NULL,
    errors        TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    input_chars   INTEGER,
    output_chars  INTEGER,
    cost_usd      REAL NOT NULL,
    elapsed_sec   REAL NOT NULL
);
"""

_CREATE_TRAINING_LOG = """
CREATE TABLE IF NOT EXISTS training_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id   INTEGER NOT NULL REFERENCES scenarios(id),
    iteration     INTEGER NOT NULL,
    metric_score  REAL NOT NULL,
    examples_used INTEGER NOT NULL
);
"""


class BenchmarkDB:
    """Wraps a SQLite connection for storing benchmark scenarios and results."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database at the given path.

        Args:
            db_path: Filesystem path to the SQLite database file.
                     Created automatically if it does not exist.
        """
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = sqlite3.Row

    def create_tables(self) -> None:
        """Create all benchmark tables if they do not already exist.

        Also applies any schema migrations needed for existing databases
        (e.g. adding columns introduced after the initial schema).

        Returns:
            None
        """
        self._conn.execute(_CREATE_SCENARIOS)
        self._conn.execute(_CREATE_RESULTS)
        self._conn.execute(_CREATE_TRAINING_LOG)
        self._conn.commit()
        # Migration: add completed_at if this is an existing DB that pre-dates it.
        # OperationalError is raised by SQLite when the column already exists.
        try:
            self._conn.execute("ALTER TABLE scenarios ADD COLUMN completed_at TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already present — nothing to do

    def scenario_exists(self, name: str) -> bool:
        """Return True if a scenario with this name is already in the DB.

        Args:
            name: The unique scenario name to look up.

        Returns:
            True if a matching row exists, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id FROM scenarios WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def scenario_is_complete(self, name: str) -> bool:
        """Return True only if the scenario exists and has been fully completed.

        A scenario that was interrupted mid-run will have a row in the DB but
        a NULL completed_at, and is therefore resumable (returns False).

        Args:
            name: The unique scenario name to look up.

        Returns:
            True if the scenario row exists and completed_at is non-null,
            False otherwise.
        """
        row = self._conn.execute(
            "SELECT completed_at FROM scenarios WHERE name = ?", (name,)
        ).fetchone()
        return row is not None and row["completed_at"] is not None

    def result_exists(self, scenario_id: int, doc_id: str) -> bool:
        """Return True if a result for this (scenario, doc) pair is already stored.

        Used during resume to skip docs that were successfully processed in a
        previous interrupted run.

        Args:
            scenario_id: Primary key of the parent scenario row.
            doc_id: Document identifier to look up.

        Returns:
            True if a matching result row exists, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id FROM results WHERE scenario_id = ? AND doc_id = ?",
            (scenario_id, doc_id),
        ).fetchone()
        return row is not None

    def get_result_count(self, scenario_id: int) -> int:
        """Return the number of result rows stored for the given scenario.

        Used to determine resume state: if the count equals the number of eval
        docs, the scenario can be marked complete without re-processing anything.

        Args:
            scenario_id: Primary key of the scenario row to count results for.

        Returns:
            Integer count of result rows for this scenario (0 if none).
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
        return row[0]

    def mark_scenario_complete(self, scenario_id: int) -> None:
        """Stamp the scenario row with the current UTC time as completed_at.

        Called after all documents in a scenario have been successfully
        processed. Until this is called the scenario is considered in-progress
        and will be resumed (not skipped) on re-run.

        Args:
            scenario_id: Primary key of the scenario row to update.

        Returns:
            None
        """
        completed_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE scenarios SET completed_at = ? WHERE id = ?",
            (completed_at, scenario_id),
        )
        self._conn.commit()

    def get_or_create_scenario(
        self,
        name: str,
        description: str,
        test_doc_count: int,
        train_doc_count: int | None = None,
    ) -> int:
        """Return the id of an existing scenario row, or insert and return a new one.

        Used on resume: if the scenario was interrupted previously its row
        already exists and we reuse it so that existing result rows remain
        correctly linked.

        Args:
            name: Unique scenario name.
            description: Human-readable description of the scenario.
            test_doc_count: Number of test documents evaluated in this run.
            train_doc_count: Number of training documents used, or None.

        Returns:
            The integer primary key of the scenario row (existing or newly created).
        """
        row = self._conn.execute(
            "SELECT id FROM scenarios WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return row["id"]
        return self.insert_scenario(name, description, test_doc_count, train_doc_count)

    def insert_scenario(
        self,
        name: str,
        description: str,
        test_doc_count: int,
        train_doc_count: int | None = None,
    ) -> int:
        """Insert a scenario row and return its auto-assigned id.

        Args:
            name: Unique scenario name (must not already exist in the DB).
            description: Human-readable description of the scenario.
            test_doc_count: Number of test documents evaluated in this run.
            train_doc_count: Number of training documents used, or None if
                             no training phase applies.

        Returns:
            The integer primary key of the newly inserted scenario row.

        Raises:
            sqlite3.IntegrityError: If the name already exists.
            RuntimeError: If the INSERT did not return a valid row id.
        """
        run_at = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO scenarios (name, description, run_at, train_doc_count, test_doc_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, description, run_at, train_doc_count, test_doc_count),
        )
        self._conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError(f"INSERT into scenarios returned no lastrowid for name={name!r}")
        return cursor.lastrowid

    def insert_result(
        self,
        scenario_id: int,
        doc_id: str,
        translation: str,
        quality_score: float,
        errors: list[dict],
        cost: CostRecord,
        elapsed_sec: float,
    ) -> None:
        """Insert a single translation result row and commit immediately.

        Args:
            scenario_id: Foreign key referencing the parent scenario row.
            doc_id: Identifier of the source document being evaluated.
            translation: The translated text produced by the scenario.
            quality_score: MQM penalty-based quality score (0.0–1.0).
            errors: List of MQM error dicts (span, category, severity, justification).
            cost: Token and cost accounting for this translation call.
            elapsed_sec: Wall-clock seconds taken to produce the translation.

        Returns:
            None
        """
        self._conn.execute(
            """
            INSERT INTO results (
                scenario_id, doc_id, translation,
                quality_score, errors,
                input_tokens, output_tokens, input_chars, output_chars,
                cost_usd, elapsed_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                doc_id,
                translation,
                quality_score,
                json.dumps(errors),
                cost.input_tokens,
                cost.output_tokens,
                cost.input_chars,
                cost.output_chars,
                cost.cost_usd,
                elapsed_sec,
            ),
        )
        self._conn.commit()

    def load_all_results(self) -> list[dict[str, Any]]:
        """Return all result rows joined with their scenario name.

        The `errors` field is deserialised from its JSON-encoded form back
        into a Python list for each row.

        Returns:
            List of dicts, one per result row, ordered by scenario name then
            doc_id. Each dict includes all result columns plus `scenario_name`.
        """
        rows = self._conn.execute(
            """
            SELECT r.*, s.name AS scenario_name
            FROM results r
            JOIN scenarios s ON r.scenario_id = s.id
            ORDER BY s.name, r.doc_id
            """
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            raw_errors = d.get("errors") or "[]"
            try:
                d["errors"] = json.loads(raw_errors)
            except json.JSONDecodeError:
                d["errors"] = []
            result.append(d)
        return result

    def close(self) -> None:
        """Close the database connection and release all resources.

        Calls the underlying SQLite connection's close method to release the
        file descriptor and finalize any remaining prepared statements.

        Returns:
            None
        """
        self._conn.close()
