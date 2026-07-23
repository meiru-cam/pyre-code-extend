"""Structured design-note persistence and version/session boundaries."""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import grading_service.main as gs


NOTE_FIELDS = {
    "api_boundaries": "Gateway owns the external API.",
    "state_ownership": "The scheduler owns mutable run state.",
    "failure_recovery": "Retry transient reads and checkpoint before effects.",
    "backpressure_concurrency": "Bound the ready queue and worker pool.",
    "durability_idempotency": "Persist idempotency keys with effect results.",
    "observability": "Trace every run and record terminal failure classes.",
    "security": "Use least-privilege capabilities and redact before egress.",
    "tradeoffs": "Durability costs latency but permits safe recovery.",
}


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(path))
    return path


def _request(
    *,
    session_token: str = "session-a",
    contract_version: int | None = None,
    fields: dict[str, str] | None = None,
):
    version = (
        contract_version
        if contract_version is not None
        else gs.get_task("tool_registry")["version"]
    )
    return gs.DesignNoteUpsertRequest(
        session_token=session_token,
        contract_version=version,
        fields=fields if fields is not None else NOTE_FIELDS,
    )


def test_routes_expose_put_and_get_contracts():
    methods_by_path: dict[str, set[str]] = {}
    for route in gs.app.routes:
        if hasattr(route, "methods"):
            methods_by_path.setdefault(route.path, set()).update(route.methods)
    assert "PUT" in methods_by_path["/design-notes/{task_id}"]
    assert "GET" in methods_by_path["/design-notes/{task_id}"]


def test_put_then_get_round_trips_all_structured_fields(database: Path):
    saved = gs.put_design_note("tool_registry", _request())

    assert saved.task_id == "tool_registry"
    assert saved.contract_version == gs.get_task("tool_registry")["version"]
    assert saved.fields.model_dump() == NOTE_FIELDS
    assert saved.updated_at

    loaded = gs.get_design_note(
        "tool_registry",
        session_token="session-a",
        contract_version=saved.contract_version,
    )
    assert loaded == saved


def test_upsert_is_idempotent_and_does_not_create_duplicate_rows(database: Path):
    first = gs.put_design_note("tool_registry", _request())
    changed = {**NOTE_FIELDS, "tradeoffs": "Prefer bounded latency over throughput."}
    second = gs.put_design_note("tool_registry", _request(fields=changed))

    assert second.fields.tradeoffs == changed["tradeoffs"]
    with gs._get_db() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM design_notes WHERE task_id = ?",
            ("tool_registry",),
        ).fetchone()[0]
    assert rows == 1
    assert first.task_id == second.task_id


def test_notes_are_isolated_by_session(database: Path):
    gs.put_design_note("tool_registry", _request(session_token="session-a"))

    with pytest.raises(HTTPException) as missing:
        gs.get_design_note(
            "tool_registry",
            session_token="session-b",
            contract_version=gs.get_task("tool_registry")["version"],
        )
    assert missing.value.status_code == 404

    other = {**NOTE_FIELDS, "state_ownership": "Session B owns its own state."}
    gs.put_design_note(
        "tool_registry",
        _request(session_token="session-b", fields=other),
    )
    loaded_a = gs.get_design_note(
        "tool_registry",
        session_token="session-a",
        contract_version=gs.get_task("tool_registry")["version"],
    )
    assert loaded_a.fields.state_ownership == NOTE_FIELDS["state_ownership"]


def test_unknown_tasks_and_stale_writes_are_rejected(database: Path):
    with pytest.raises(HTTPException) as unknown:
        gs.put_design_note("not-a-task", _request())
    assert unknown.value.status_code == 404

    current = gs.get_task("tool_registry")["version"]
    with pytest.raises(HTTPException) as stale:
        gs.put_design_note(
            "tool_registry",
            _request(contract_version=current + 1),
        )
    assert stale.value.status_code == 409


def test_request_rejects_missing_extra_invalid_and_oversized_fields(database: Path):
    missing = dict(NOTE_FIELDS)
    missing.pop("security")
    with pytest.raises(ValidationError):
        _request(fields=missing)

    with pytest.raises(ValidationError):
        _request(fields={**NOTE_FIELDS, "unknown": "not allowed"})

    with pytest.raises(ValidationError):
        _request(contract_version=0)

    with pytest.raises(ValidationError):
        _request(fields={**NOTE_FIELDS, "tradeoffs": "x" * 4001})

    individually_valid_but_too_large = {
        key: "x" * 2500 for key in NOTE_FIELDS
    }
    with pytest.raises(ValidationError):
        _request(fields=individually_valid_but_too_large)


def test_contract_versions_are_preserved_separately(database: Path, monkeypatch):
    task = gs.get_task("tool_registry")
    original_version = task["version"]
    monkeypatch.setitem(task, "version", 1)
    gs.put_design_note("tool_registry", _request(contract_version=1))

    monkeypatch.setitem(task, "version", 2)
    revised = {**NOTE_FIELDS, "api_boundaries": "Version two boundary."}
    gs.put_design_note(
        "tool_registry",
        _request(contract_version=2, fields=revised),
    )

    old = gs.get_design_note(
        "tool_registry", session_token="session-a", contract_version=1
    )
    new = gs.get_design_note(
        "tool_registry", session_token="session-a", contract_version=2
    )
    assert old.fields.api_boundaries == NOTE_FIELDS["api_boundaries"]
    assert new.fields.api_boundaries == revised["api_boundaries"]

    monkeypatch.setitem(task, "version", original_version)


def test_submission_history_records_note_presence_without_solving(database: Path):
    gs.put_design_note("tool_registry", _request())

    gs.save_progress(
        gs.SaveProgressRequest(
            sessionToken="session-a",
            taskId="tool_registry",
            status="attempted",
            code="class ToolRegistry: pass",
            allPassed=False,
        )
    )

    progress = gs.get_progress(1)["tool_registry"]
    history = gs.get_submissions(1, "tool_registry")
    assert progress.status == "attempted"
    assert history[0]["designNotePresent"] is True


def test_blank_note_is_not_reported_as_present_on_submission(database: Path):
    blank = {key: "   " for key in NOTE_FIELDS}
    gs.put_design_note("tool_registry", _request(fields=blank))
    gs.save_progress(
        gs.SaveProgressRequest(
            sessionToken="session-a",
            taskId="tool_registry",
            status="attempted",
            code="class ToolRegistry: pass",
            allPassed=False,
        )
    )
    assert gs.get_submissions(1, "tool_registry")[0]["designNotePresent"] is False


def test_additive_migration_preserves_legacy_submission(database: Path):
    connection = sqlite3.connect(database)
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
    connection.execute("INSERT INTO users (session_token) VALUES ('legacy')")
    connection.execute(
        "INSERT INTO submissions (user_id, task_id, code, passed) "
        "VALUES (1, 'tool_registry', 'legacy code', 0)"
    )
    connection.commit()
    connection.close()

    with gs._get_db() as migrated:
        legacy = migrated.execute(
            "SELECT code, design_note_present FROM submissions"
        ).fetchone()
        table = migrated.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='design_notes'"
        ).fetchone()

    assert legacy == ("legacy code", 0)
    assert table == ("design_notes",)


def test_concurrent_first_open_creates_design_note_schema_once(
    database: Path,
):
    def open_database() -> tuple[str, ...]:
        with gs._get_db() as connection:
            return tuple(
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(design_notes)"
                ).fetchall()
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: open_database(), range(8)))

    assert all(columns == results[0] for columns in results)
    assert set(NOTE_FIELDS).issubset(results[0])
