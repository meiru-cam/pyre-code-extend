"""FastAPI grading service for torch_judge tasks."""

import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path for torch_judge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import os
import threading
import time
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torch_judge.harness import HarnessFailure
from torch_judge.tasks import get_task

app = FastAPI(title="Grading Service")

# Lock to serialize sys.stdout redirects across concurrent request threads
_stdout_lock = threading.Lock()

# Schema initialization runs lazily from request handlers. Serialize it within
# the process; BEGIN IMMEDIATE provides the equivalent lock across processes.
_db_init_lock = threading.Lock()

# ---------------------------------------------------------------------------
# SQLite DB (user sessions + progress)
# ---------------------------------------------------------------------------

_DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent.parent / "data" / "pyre.db"))

DESIGN_NOTE_FIELD_MAX_CHARS = 4_000
DESIGN_NOTE_TOTAL_MAX_CHARS = 16_000
DESIGN_NOTE_COLUMNS = (
    "api_boundaries",
    "state_ownership",
    "failure_recovery",
    "backpressure_concurrency",
    "durability_idempotency",
    "observability",
    "security",
    "tradeoffs",
)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _get_db() -> sqlite3.Connection:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    with _db_init_lock:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('todo', 'attempted', 'solved')),
                    best_time_ms REAL,
                    attempts INTEGER DEFAULT 0,
                    solved_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, task_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    exec_time_ms REAL,
                    submitted_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            _ensure_column(conn, "progress", "contract_version", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(conn, "submissions", "contract_version", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(
                conn,
                "submissions",
                "design_note_present",
                "INTEGER NOT NULL DEFAULT 0",
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    contract_version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('todo', 'attempted', 'solved')),
                    best_time_ms REAL,
                    attempts INTEGER DEFAULT 0,
                    solved_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, task_id, contract_version)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS design_notes (
                    user_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    contract_version INTEGER NOT NULL,
                    api_boundaries TEXT NOT NULL,
                    state_ownership TEXT NOT NULL,
                    failure_recovery TEXT NOT NULL,
                    backpressure_concurrency TEXT NOT NULL,
                    durability_idempotency TEXT NOT NULL,
                    observability TEXT NOT NULL,
                    security TEXT NOT NULL,
                    tradeoffs TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    PRIMARY KEY (user_id, task_id, contract_version)
                )
            """)
            conn.execute("""
                INSERT OR IGNORE INTO progress_revisions
                    (user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at)
                SELECT user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at
                FROM progress
            """)
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise
    return conn


class SubmitRequest(BaseModel):
    taskId: str
    code: str


class RunRequest(BaseModel):
    taskId: str
    code: str
    testIndices: list[int] | None = None


class TestResult(BaseModel):
    name: str
    passed: bool
    execTimeMs: float
    error: str | None = None
    output: str | None = None
    behavior: str | None = None
    visibility: str = "visible"
    testIndex: int


class GradeResponse(BaseModel):
    passed: int
    total: int
    allPassed: bool
    results: list[TestResult]
    totalTimeMs: float
    error: str | None = None


def _validate_code(code: str) -> str | None:
    """Return an error message if code contains disallowed top-level statements."""
    import ast
    allowed = (
        ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import, ast.ImportFrom,
        ast.Assign, ast.AnnAssign, ast.AugAssign,
        ast.Expr,  # top-level expressions / docstrings
    )
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    for node in tree.body:
        if not isinstance(node, allowed):
            return f"Only definitions and assignments are allowed at the top level (found: {type(node).__name__})"
    return None


def _finalize_result(result: TestResult, test: dict, test_index: int) -> TestResult:
    """Attach behavior metadata and mask unshown-case details."""
    result.behavior = test.get("behavior")
    result.visibility = test.get("visibility", "visible")
    result.testIndex = test_index
    if result.visibility == "unshown":
        result.output = None
        if not result.passed:
            result.error = test.get("failure_message") or (
                f"Behavior check failed: {result.behavior}"
                if result.behavior
                else "Evaluator case failed."
            )
    return result


def _execute_tests(code: str, task: dict, test_indices: list[int] | None = None, capture_output: bool = True) -> GradeResponse:
    import torch, math
    err = _validate_code(code)
    if err:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=err)
    user_ns: dict[str, Any] = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "np": __import__("numpy"),
        "math": math,
    }
    try:
        exec(code, user_ns)
    except SyntaxError as e:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=f"Syntax error: {e}")

    fn_name = task.get("function_name")
    if fn_name is None:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error="Task has no function_name defined")

    if fn_name not in user_ns:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=f"Function '{fn_name}' not found in submitted code")

    all_tests = task.get("tests", [])
    if test_indices is not None and len(test_indices) == 0:
        return GradeResponse(
            passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0,
            error="No visible test cases are available to run.",
        )
    indexed_tests = (
        [(index, all_tests[index]) for index in test_indices if 0 <= index < len(all_tests)]
        if test_indices is not None
        else list(enumerate(all_tests))
    )

    if test_indices is not None and len(test_indices) > 0 and len(indexed_tests) == 0:
        return GradeResponse(
            passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0,
            error=f"All provided test indices are out of range (valid range: 0..{len(all_tests) - 1})",
        )

    results: list[TestResult] = []
    passed = 0
    total_time_ms = 0.0

    for test_index, test in indexed_tests:
        _torch = __import__("torch")
        test_ns: dict[str, Any] = {
            "torch": _torch,
            "Tensor": _torch.Tensor,
            "nn": _torch.nn,
            "F": _torch.nn.functional,
            "np": __import__("numpy"),
            "math": math,
            fn_name: user_ns[fn_name],
        }
        test_code = test["code"].replace("{fn}", fn_name)

        # Capture stdout for print output
        output = None
        if capture_output:
            with _stdout_lock:
                old_stdout = sys.stdout
                sys.stdout = captured = io.StringIO()
                try:
                    start = time.perf_counter()
                    exec(test_code, test_ns)
                    exec_time_ms = (time.perf_counter() - start) * 1000
                    output = captured.getvalue() or None
                    results.append(_finalize_result(
                        TestResult(
                            name=test["name"], passed=True, execTimeMs=exec_time_ms,
                            output=output, testIndex=test_index,
                        ),
                        test,
                        test_index,
                    ))
                    passed += 1
                except HarnessFailure:
                    raise
                except AssertionError as e:
                    exec_time_ms = (time.perf_counter() - start) * 1000
                    output = captured.getvalue() or None
                    results.append(_finalize_result(
                        TestResult(
                            name=test["name"], passed=False, execTimeMs=exec_time_ms,
                            error=str(e), output=output, testIndex=test_index,
                        ),
                        test,
                        test_index,
                    ))
                except Exception as e:
                    exec_time_ms = (time.perf_counter() - start) * 1000
                    output = captured.getvalue() or None
                    results.append(_finalize_result(
                        TestResult(
                            name=test["name"], passed=False, execTimeMs=exec_time_ms,
                            error=f"{type(e).__name__}: {e}", output=output,
                            testIndex=test_index,
                        ),
                        test,
                        test_index,
                    ))
                finally:
                    sys.stdout = old_stdout
        else:
            start = time.perf_counter()
            try:
                exec(test_code, test_ns)
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(_finalize_result(
                    TestResult(
                        name=test["name"], passed=True, execTimeMs=exec_time_ms,
                        testIndex=test_index,
                    ),
                    test,
                    test_index,
                ))
                passed += 1
            except HarnessFailure:
                raise
            except AssertionError as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(_finalize_result(
                    TestResult(
                        name=test["name"], passed=False, execTimeMs=exec_time_ms,
                        error=str(e), testIndex=test_index,
                    ),
                    test,
                    test_index,
                ))
            except Exception as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(_finalize_result(
                    TestResult(
                        name=test["name"], passed=False, execTimeMs=exec_time_ms,
                        error=f"{type(e).__name__}: {e}", testIndex=test_index,
                    ),
                    test,
                    test_index,
                ))
        total_time_ms += exec_time_ms

    return GradeResponse(passed=passed, total=len(results), allPassed=passed == len(results), results=results, totalTimeMs=total_time_ms)


@app.post("/grade", response_model=GradeResponse)
def grade(request: SubmitRequest) -> GradeResponse:
    task = get_task(request.taskId)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{request.taskId}' not found")
    try:
        return _execute_tests(request.code, task)
    except HarnessFailure as error:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluator harness failure: {error}",
        ) from error


@app.post("/run", response_model=GradeResponse)
def run(request: RunRequest) -> GradeResponse:
    task = get_task(request.taskId)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{request.taskId}' not found")
    try:
        return _execute_tests(request.code, task, request.testIndices)
    except HarnessFailure as error:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluator harness failure: {error}",
        ) from error



@app.get("/tasks/{task_id}/notebook")
def get_notebook(task_id: str) -> dict:
    task = get_task(task_id)
    if task is None or not task.get("solution"):
        raise HTTPException(status_code=404, detail=f"Notebook for '{task_id}' not found")
    cells = [{"type": "code", "source": task["solution"].strip(), "role": "solution"}]
    if "explanation" in task:
        cells.append({"type": "markdown", "source": task["explanation"].strip(), "role": "explanation"})
    if "demo" in task:
        cells.append({"type": "code", "source": task["demo"].strip(), "role": "demo"})
    return {"cells": cells}


@app.get("/tasks/{task_id}/solution")
def get_solution(task_id: str) -> dict[str, str]:
    task = get_task(task_id)
    if task is None or not task.get("solution"):
        raise HTTPException(status_code=404, detail=f"Solution for '{task_id}' not found")
    return {"solution": task["solution"]}


class UserRequest(BaseModel):
    sessionToken: str


class DesignNoteFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_boundaries: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    state_ownership: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    failure_recovery: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    backpressure_concurrency: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    durability_idempotency: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    observability: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    security: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)
    tradeoffs: str = Field(max_length=DESIGN_NOTE_FIELD_MAX_CHARS)

    @model_validator(mode="after")
    def check_total_size(self) -> "DesignNoteFields":
        if sum(len(getattr(self, column)) for column in DESIGN_NOTE_COLUMNS) > DESIGN_NOTE_TOTAL_MAX_CHARS:
            raise ValueError(
                f"design note exceeds {DESIGN_NOTE_TOTAL_MAX_CHARS} characters"
            )
        return self


class DesignNoteUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_token: str = Field(min_length=1, max_length=200)
    contract_version: int = Field(ge=1, strict=True)
    fields: DesignNoteFields


class DesignNoteResponse(BaseModel):
    task_id: str
    contract_version: int
    fields: DesignNoteFields
    updated_at: str


class ProgressEntry(BaseModel):
    status: str
    bestTimeMs: float | None = None
    attempts: int
    solvedAt: str | None = None
    contractVersion: int = 1
    completedVersions: list[int] = Field(default_factory=list)


class SaveProgressRequest(BaseModel):
    sessionToken: str
    taskId: str
    status: str
    execTimeMs: float | None = None
    code: str | None = None
    allPassed: bool | None = None


@app.post("/users")
def get_or_create_user(request: UserRequest) -> dict[str, int]:
    with _get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE session_token = ?", (request.sessionToken,)).fetchone()
        if row:
            return {"userId": row[0]}
        cur = conn.execute("INSERT INTO users (session_token) VALUES (?)", (request.sessionToken,))
        return {"userId": cur.lastrowid}


def _validate_design_note_version(
    task_id: str,
    contract_version: int,
    *,
    writing: bool,
) -> dict:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version < 1
    ):
        raise HTTPException(status_code=422, detail="Invalid contract version")
    current_version = task.get("version", 1)
    if contract_version > current_version or (
        writing and contract_version != current_version
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Contract version {contract_version} does not match "
                f"current version {current_version}"
            ),
        )
    return task


def _design_note_from_row(task_id: str, row: sqlite3.Row | tuple) -> DesignNoteResponse:
    return DesignNoteResponse(
        task_id=task_id,
        contract_version=row[0],
        fields=DesignNoteFields(
            **dict(zip(DESIGN_NOTE_COLUMNS, row[1:9], strict=True))
        ),
        updated_at=row[9],
    )


@app.put("/design-notes/{task_id}", response_model=DesignNoteResponse)
def put_design_note(
    task_id: str,
    request: DesignNoteUpsertRequest,
) -> DesignNoteResponse:
    _validate_design_note_version(
        task_id,
        request.contract_version,
        writing=True,
    )
    values = request.fields.model_dump()
    with _get_db() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE session_token = ?",
            (request.session_token,),
        ).fetchone()
        if user is None:
            cursor = conn.execute(
                "INSERT INTO users (session_token) VALUES (?)",
                (request.session_token,),
            )
            user_id = cursor.lastrowid
        else:
            user_id = user[0]
        columns = ", ".join(DESIGN_NOTE_COLUMNS)
        placeholders = ", ".join("?" for _ in DESIGN_NOTE_COLUMNS)
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in DESIGN_NOTE_COLUMNS
        )
        conn.execute(
            f"INSERT INTO design_notes "
            f"(user_id, task_id, contract_version, {columns}) "
            f"VALUES (?, ?, ?, {placeholders}) "
            f"ON CONFLICT(user_id, task_id, contract_version) DO UPDATE SET "
            f"{updates}, updated_at = datetime('now')",
            (
                user_id,
                task_id,
                request.contract_version,
                *(values[column] for column in DESIGN_NOTE_COLUMNS),
            ),
        )
        row = conn.execute(
            f"SELECT contract_version, {columns}, updated_at "
            "FROM design_notes "
            "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
            (user_id, task_id, request.contract_version),
        ).fetchone()
    return _design_note_from_row(task_id, row)


@app.get("/design-notes/{task_id}", response_model=DesignNoteResponse)
def get_design_note(
    task_id: str,
    contract_version: int,
    session_token: Annotated[str, Header(alias="X-Session-Token")],
) -> DesignNoteResponse:
    _validate_design_note_version(task_id, contract_version, writing=False)
    if not isinstance(session_token, str) or not session_token:
        raise HTTPException(status_code=422, detail="Invalid session token")
    columns = ", ".join(DESIGN_NOTE_COLUMNS)
    with _get_db() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE session_token = ?",
            (session_token,),
        ).fetchone()
        if user is None:
            raise HTTPException(status_code=404, detail="Design note not found")
        row = conn.execute(
            f"SELECT contract_version, {columns}, updated_at "
            "FROM design_notes "
            "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
            (user[0], task_id, contract_version),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Design note not found")
    return _design_note_from_row(task_id, row)


@app.get("/progress/{user_id}")
def get_progress(user_id: int) -> dict[str, ProgressEntry]:
    with _get_db() as conn:
        task_ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT task_id FROM progress_revisions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        result: dict[str, ProgressEntry] = {}
        for task_id in task_ids:
            contract_version = (get_task(task_id) or {}).get("version", 1)
            current = conn.execute(
                "SELECT status, best_time_ms, attempts, solved_at FROM progress_revisions "
                "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
                (user_id, task_id, contract_version),
            ).fetchone()
            completed = [
                row[0]
                for row in conn.execute(
                    "SELECT contract_version FROM progress_revisions "
                    "WHERE user_id = ? AND task_id = ? AND status = 'solved' "
                    "ORDER BY contract_version",
                    (user_id, task_id),
                ).fetchall()
            ]
            if current is None:
                result[task_id] = ProgressEntry(
                    status="todo",
                    attempts=0,
                    contractVersion=contract_version,
                    completedVersions=completed,
                )
            else:
                result[task_id] = ProgressEntry(
                    status=current[0],
                    bestTimeMs=current[1],
                    attempts=current[2],
                    solvedAt=current[3],
                    contractVersion=contract_version,
                    completedVersions=completed,
                )
    return result


def _save_revision_progress(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    task_id: str,
    contract_version: int,
    status: str,
    exec_time_ms: float | None,
) -> None:
    existing = conn.execute(
        "SELECT status, best_time_ms FROM progress_revisions "
        "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
        (user_id, task_id, contract_version),
    ).fetchone()
    if existing:
        existing_status, existing_best = existing
        next_status = "solved" if status == "solved" else (
            "solved" if existing_status == "solved" else status
        )
        best = existing_best
        if status == "solved" and exec_time_ms is not None:
            best = min(existing_best, exec_time_ms) if existing_best is not None else exec_time_ms
        conn.execute(
            "UPDATE progress_revisions SET status = ?, best_time_ms = ?, "
            "attempts = attempts + 1, "
            "solved_at = CASE WHEN ? = 'solved' "
            "THEN COALESCE(solved_at, datetime('now')) ELSE solved_at END "
            "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
            (next_status, best, next_status, user_id, task_id, contract_version),
        )
        return
    conn.execute(
        "INSERT INTO progress_revisions "
        "(user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at) "
        "VALUES (?, ?, ?, ?, ?, 1, "
        "CASE WHEN ? = 'solved' THEN datetime('now') ELSE NULL END)",
        (
            user_id,
            task_id,
            contract_version,
            status,
            exec_time_ms if status == "solved" else None,
            status,
        ),
    )


@app.post("/progress")
def save_progress(request: SaveProgressRequest) -> dict[str, str]:
    with _get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE session_token = ?", (request.sessionToken,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = row[0]
        contract_version = (get_task(request.taskId) or {}).get("version", 1)
        _save_revision_progress(
            conn,
            user_id=user_id,
            task_id=request.taskId,
            contract_version=contract_version,
            status=request.status,
            exec_time_ms=request.execTimeMs,
        )
        # Compatibility table: status remains lifetime-monotonic (once solved,
        # always solved), while contract_version records the latest revision
        # touched. Current-version truth lives in progress_revisions.
        existing = conn.execute(
            "SELECT status, best_time_ms FROM progress WHERE user_id = ? AND task_id = ?",
            (user_id, request.taskId)
        ).fetchone()
        if existing:
            existing_status, existing_best_time = existing
            if request.status == "solved":
                if existing_best_time is not None and request.execTimeMs is not None:
                    best = min(existing_best_time, request.execTimeMs)
                else:
                    best = existing_best_time if existing_best_time is not None else request.execTimeMs
                conn.execute(
                    "UPDATE progress SET status = ?, best_time_ms = ?, "
                    "attempts = attempts + 1, "
                    "solved_at = COALESCE(solved_at, datetime('now')), "
                    "contract_version = ? WHERE user_id = ? AND task_id = ?",
                    ("solved", best, contract_version, user_id, request.taskId)
                )
            else:
                next_status = existing_status if existing_status == "solved" and request.status in ("todo", "attempted") else request.status
                conn.execute(
                    "UPDATE progress SET status = ?, attempts = attempts + 1, "
                    "contract_version = ? WHERE user_id = ? AND task_id = ?",
                    (next_status, contract_version, user_id, request.taskId)
                )
        else:
            if request.status == "solved":
                conn.execute(
                    "INSERT INTO progress "
                    "(user_id, task_id, status, best_time_ms, attempts, solved_at, contract_version) "
                    "VALUES (?, ?, ?, ?, 1, datetime('now'), ?)",
                    (user_id, request.taskId, request.status, request.execTimeMs, contract_version)
                )
            else:
                conn.execute(
                    "INSERT INTO progress "
                    "(user_id, task_id, status, best_time_ms, attempts, solved_at, contract_version) "
                    "VALUES (?, ?, ?, ?, 1, NULL, ?)",
                    (user_id, request.taskId, request.status, None, contract_version)
                )
        if request.code is not None:
            note_predicate = " OR ".join(
                f"trim({column}) <> ''" for column in DESIGN_NOTE_COLUMNS
            )
            note_present = conn.execute(
                "SELECT 1 FROM design_notes "
                "WHERE user_id = ? AND task_id = ? AND contract_version = ? "
                f"AND ({note_predicate})",
                (user_id, request.taskId, contract_version),
            ).fetchone() is not None
            conn.execute(
                "INSERT INTO submissions "
                "(user_id, task_id, code, passed, exec_time_ms, "
                "contract_version, design_note_present) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, request.taskId, request.code,
                    1 if request.allPassed else 0, request.execTimeMs, contract_version,
                    1 if note_present else 0,
                )
            )
    return {"ok": "true"}


@app.get("/submissions/{user_id}/{task_id}")
def get_submissions(user_id: int, task_id: str) -> list[dict]:
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT id, passed, exec_time_ms, submitted_at, code, "
            "contract_version, design_note_present "
            "FROM submissions "
            "WHERE user_id = ? AND task_id = ? ORDER BY submitted_at DESC LIMIT 50",
            (user_id, task_id)
        ).fetchall()
    return [
        {
            "id": r[0], "passed": bool(r[1]), "execTimeMs": r[2],
            "submittedAt": r[3], "code": r[4], "contractVersion": r[5],
            "designNotePresent": bool(r[6]),
        }
        for r in rows
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
