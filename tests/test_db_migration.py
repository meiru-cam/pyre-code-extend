"""Backward-compatible contract-version migration."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import grading_service.main as gs


def _make_legacy_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_token TEXT UNIQUE NOT NULL, created_at TEXT)"
    )
    connection.execute(
        "CREATE TABLE progress (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_id INTEGER NOT NULL, task_id TEXT NOT NULL, status TEXT NOT NULL, "
        "best_time_ms REAL, attempts INTEGER DEFAULT 0, solved_at TEXT, "
        "UNIQUE(user_id, task_id))"
    )
    connection.execute(
        "CREATE TABLE submissions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_id INTEGER NOT NULL, task_id TEXT NOT NULL, code TEXT NOT NULL, "
        "passed INTEGER NOT NULL, exec_time_ms REAL, submitted_at TEXT)"
    )
    connection.execute("INSERT INTO users (session_token) VALUES ('tok')")
    connection.execute(
        "INSERT INTO progress (user_id, task_id, status, attempts) "
        "VALUES (1, 'relu', 'solved', 3)"
    )
    connection.execute(
        "INSERT INTO submissions (user_id, task_id, code, passed) "
        "VALUES (1, 'relu', 'def relu(x): ...', 1)"
    )
    connection.commit()
    connection.close()


def test_legacy_rows_become_version_one(tmp_path, monkeypatch):
    database = tmp_path / "pyre.db"
    _make_legacy_db(database)
    monkeypatch.setattr(gs, "_DB_PATH", str(database))
    with gs._get_db() as connection:
        assert connection.execute(
            "SELECT contract_version FROM progress WHERE task_id='relu'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT contract_version FROM submissions WHERE task_id='relu'"
        ).fetchone()[0] == 1
        revision = connection.execute(
            "SELECT contract_version, status, attempts FROM progress_revisions "
            "WHERE task_id='relu'"
        ).fetchone()
        assert revision == (1, "solved", 3)


def test_migration_is_idempotent(tmp_path, monkeypatch):
    database = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(database))
    gs._get_db().close()
    gs._get_db().close()


def _install_task(monkeypatch, version: int) -> dict:
    task = {
        "title": "V",
        "difficulty": "Easy",
        "function_name": "f",
        "description_en": "x",
        "hint": "x",
        "version": version,
        "solution": "def f(): pass",
        "tests": [{"name": "t", "code": "assert True"}],
    }
    monkeypatch.setitem(gs.get_task.__globals__["TASKS"], "_versioned", task)
    return task


def test_new_revision_does_not_overwrite_solved_history(tmp_path, monkeypatch):
    database = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(database))
    task = _install_task(monkeypatch, version=1)
    gs.get_or_create_user(gs.UserRequest(sessionToken="tok"))
    gs.save_progress(gs.SaveProgressRequest(
        sessionToken="tok", taskId="_versioned", status="solved", execTimeMs=1.0,
    ))

    task["version"] = 2
    gs.save_progress(gs.SaveProgressRequest(
        sessionToken="tok", taskId="_versioned", status="attempted", execTimeMs=2.0,
        code="def f(): pass", allPassed=False,
    ))

    with gs._get_db() as connection:
        rows = connection.execute(
            "SELECT contract_version, status FROM progress_revisions "
            "WHERE task_id='_versioned' ORDER BY contract_version"
        ).fetchall()
        assert rows == [(1, "solved"), (2, "attempted")]

    progress = gs.get_progress(1)
    assert progress["_versioned"].contractVersion == 2
    assert progress["_versioned"].status == "attempted"
    assert progress["_versioned"].completedVersions == [1]
    assert gs.get_submissions(1, "_versioned")[0]["contractVersion"] == 2


def test_version_bump_without_attempt_is_todo_with_old_completion(tmp_path, monkeypatch):
    database = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(database))
    task = _install_task(monkeypatch, version=1)
    gs.get_or_create_user(gs.UserRequest(sessionToken="tok"))
    gs.save_progress(gs.SaveProgressRequest(
        sessionToken="tok", taskId="_versioned", status="solved", execTimeMs=1.0,
    ))
    task["version"] = 2

    progress = gs.get_progress(1)["_versioned"]
    assert progress.contractVersion == 2
    assert progress.status == "todo"
    assert progress.attempts == 0
    assert progress.completedVersions == [1]
