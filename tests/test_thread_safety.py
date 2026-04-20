"""Tests verifying thread-safety properties of BenchmarkDB and ProductionDB."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.app.db import BenchmarkDB
from src.app.production_db import ProductionDB


def _make_benchmark_db() -> BenchmarkDB:
    """Return an in-memory BenchmarkDB with tables created.

    Returns:
        A fresh BenchmarkDB backed by an in-memory SQLite database.
    """
    db = BenchmarkDB(Path(":memory:"))
    db.create_tables()
    return db


def _make_production_db() -> ProductionDB:
    """Return an in-memory ProductionDB (tables created automatically).

    Returns:
        A fresh ProductionDB backed by an in-memory SQLite database.
    """
    return ProductionDB(Path(":memory:"))


def test_benchmark_db_has_write_lock() -> None:
    """BenchmarkDB exposes a _lock attribute that supports acquire/release."""
    db = _make_benchmark_db()
    assert hasattr(db, "_lock"), "_lock attribute missing from BenchmarkDB"
    assert callable(getattr(db._lock, "acquire", None)), "_lock must have acquire()"
    assert callable(getattr(db._lock, "release", None)), "_lock must have release()"


def test_production_db_has_write_lock() -> None:
    """ProductionDB exposes a _lock attribute that supports acquire/release."""
    db = _make_production_db()
    assert hasattr(db, "_lock"), "_lock attribute missing from ProductionDB"
    assert callable(getattr(db._lock, "acquire", None)), "_lock must have acquire()"
    assert callable(getattr(db._lock, "release", None)), "_lock must have release()"


def test_benchmark_db_accessible_from_non_creating_thread() -> None:
    """BenchmarkDB connection is accessible from threads other than the creating thread.

    Verifies that check_same_thread=False is in effect: if it were False
    (the default) a ProgrammingError would be raised when a worker thread
    accesses the connection.
    """
    db = _make_benchmark_db()
    errors: list[Exception] = []

    def _read() -> None:
        try:
            db._conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    assert not errors, f"Cross-thread access raised: {errors[0]}"


def test_production_db_accessible_from_non_creating_thread() -> None:
    """ProductionDB connection is accessible from threads other than the creating thread."""
    db = _make_production_db()
    errors: list[Exception] = []

    def _read() -> None:
        try:
            db._conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    assert not errors, f"Cross-thread access raised: {errors[0]}"


def test_production_db_get_or_create_run_is_idempotent_under_concurrent_calls() -> None:
    """Concurrent get_or_create_run calls with the same name return one consistent run_id.

    Four threads race to call get_or_create_run with the same run name. The lock
    inside get_or_create_run must serialise the SELECT+INSERT so exactly one row
    is created and all callers return the same id.
    """
    db = _make_production_db()
    results: list[int] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def _call() -> None:
        barrier.wait()  # start all threads simultaneously to maximise contention
        try:
            run_id = db.get_or_create_run("concurrent_run", "desc", 5)
            results.append(run_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Unexpected error during concurrent get_or_create_run: {errors[0]}"
    assert len(results) == 4, "Expected all 4 threads to return a run_id"
    assert len(set(results)) == 1, f"All threads must return the same run_id; got {set(results)}"

    row = db._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    assert row[0] == 1, f"Expected exactly 1 run row; found {row[0]}"
