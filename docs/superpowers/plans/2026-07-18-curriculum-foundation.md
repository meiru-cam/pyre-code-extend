# Curriculum Foundation Implementation Plan (Plan 1 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the metadata, hint, evaluator-result, and persistence foundation from the spec `docs/superpowers/specs/2026-07-18-advanced-ai-curriculum-design.md` (Delivery Sequence items 1–4), so later plans can add harnesses and the 12 vertical-slice exercises.

**Architecture:** Tasks stay plain `TASK` dicts in `torch_judge/tasks/*.py` discovered by `_registry.py`. A new `_schema.py` validates legacy + new fields (versions, two-level hints, behavior categories, provenance) and is wired into `scripts/export_problems.py` so invalid metadata fails the build with a precise field error. The FastAPI grading service (`grading_service/main.py`) gains behavior-aware, visibility-aware test results and a backward-compatible `contract_version` SQLite migration. The Next.js frontend (`web/`) gains additive types, independent Level 1/Level 2 hint controls, and unshown-case handling.

**Tech Stack:** Python 3.10+ (FastAPI, pydantic, pytest, sqlite3), Next.js/TypeScript, vitest (new, node-env only), existing npm build scripts (`npm run build:problems`).

## Plan Series (spec → plans)

This plan is the first of a series; each later plan produces working software on its own:

1. **This plan** — repo remotes, task schema + validation + versioning, two-level hint API/UI, behavior-aware evaluator results, export compatibility, DB migration. (Spec: Exercise Contract, Evaluation Model, Compatibility and Migration, Git workflow, Error Handling for metadata.)
2. **Tensor harness + Advanced Attention slice** — `torch_judge/harness/tensor/`, author-quality-gate tooling (mutation runner, differential/gradient helpers), exercises `qk_norm`, `hybrid_attention_schedule`, `frontier_attention_block`, new path entry in `web/src/lib/paths.json`.
3. **MoE slice** — exercises `moe_topk_router`, `moe_capacity_dispatch`, `tiny_moe_train_step`, MoE-specific evaluator cases + mutation suite, path entry.
4. **Agent harness + Agent Runtime slice** — `torch_judge/harness/agents/` (protocol, scripted_model, fake_tools, virtual_clock, scenarios, trace, assertions), exercises `tool_registry`, `budgeted_agent_loop`, `supervisor_orchestration`, path entry.
5. **Guardrails slice** — exercises `policy_engine`, `approval_gate`, `guarded_runtime`, adversarial fixtures, path entry.
6. **Design notes + AI review** — design-note editor/persistence, DeepSeek V4 Pro review flow, AI data boundary + secret-pattern check, frontend interaction tests (jsdom), full vertical-slice verification pass.

## Global Constraints

- Existing task IDs, path IDs, and progress records must remain valid; all new metadata is additive.
- Legacy `hint` / `hint_zh` fields remain accepted and readable; new tasks may be English-only (`*_zh` fields optional — export falls back to the English value).
- Difficulty values keep the repo convention: `"Easy" | "Medium" | "Hard"` (the spec's lowercase `"medium"` example is normalized to this).
- Existing DB records are treated as contract version 1. Preserve the existing aggregate `progress` table for compatibility and add version-keyed history; updates within one revision never discard or overwrite another revision's solved state.
- Deterministic local grading is the only source of Solved status; nothing in this plan calls a network service during grading.
- Unshown evaluator cases are never described as "secret/hidden"; failures report a behavior category + authored `failure_message`, never the case's code or full input.
- No commit message includes `Co-Authored-By` lines.
- `web/.env` and other local env files are never staged; keep the pre-existing dirty files (`.gitignore`, `.githooks/`, `ecosystem.config.cjs`) out of curriculum commits.
- Python commands run with the project venv: `.venv/bin/python`. Web commands run from `web/`.

---

### Task 1: Repository remotes and feature branch

**Files:** none (git configuration and isolated-worktree setup only)

**Interfaces:**
- Produces: remote `upstream` = `https://github.com/whwangovo/pyre-code.git`, remote `origin` = `https://github.com/meiru-cam/pyre-code-extend.git`, and an isolated worktree on branch `feature/curriculum-foundation` off the current local `main`. All later tasks run and commit inside that worktree.

- [ ] **Step 1: Rename current origin to upstream and add the new origin**

```bash
cd /Users/zhangmeiru/O_Documents/pyre-code
git remote rename origin upstream
git remote add origin https://github.com/meiru-cam/pyre-code-extend.git
git remote -v
```

Expected: `origin` → `meiru-cam/pyre-code-extend.git`, `upstream` → `whwangovo/pyre-code.git` (fetch+push each).

- [ ] **Step 2: Verify the supplied origin exists without mutating GitHub**

```bash
git ls-remote --exit-code origin HEAD
```

Expected: one SHA line for `HEAD`. If the command fails or returns no SHA, stop and report that the supplied repository is unavailable; do not create a repository or change its visibility.

- [ ] **Step 3: Fetch both remotes and push main to origin**

```bash
git fetch upstream && git fetch origin
git push -u origin main
```

Expected: `main` pushed to origin (this makes `main` a valid PR base on the user's repo).

- [ ] **Step 4: Create an isolated worktree for the feature branch**

Invoke `superpowers:using-git-worktrees`. It must first detect whether execution is already isolated and prefer a native worktree tool when available. When the git fallback is needed, verify the existing `.worktrees/` directory is ignored, then run:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/curriculum-foundation -b feature/curriculum-foundation main
cd .worktrees/curriculum-foundation
git status --short
```

Expected: the worktree is on `feature/curriculum-foundation` with a clean status. The primary workspace still contains the pre-existing ` M .gitignore`, `?? .githooks/`, and `?? ecosystem.config.cjs`; those files do not appear in the isolated worktree.

- [ ] **Step 5: Set up the worktree and verify the clean baseline**

From `.worktrees/curriculum-foundation`:

```bash
./setup.sh
.venv/bin/python -m pytest scripts/test_export_problems.py -v
cd web
npx tsc --noEmit
npm run build
```

Expected: setup completes; exporter tests pass; TypeScript reports no errors; the baseline Next.js build succeeds. If any baseline command fails, stop and report the existing failure before implementing Task 2.

---

### Task 2: Task schema validation module

**Files:**
- Create: `torch_judge/tasks/_schema.py`
- Create: `tests/__init__.py` (empty file)
- Test: `tests/test_task_schema.py`

**Interfaces:**
- Produces: `validate_task(task_id: str, task: dict, known_ids: set[str] | None = None) -> None` raising `TaskValidationError` (attrs: `task_id`, `field`; message format `Task '<id>', field '<field>': <message>`); constants `BEHAVIOR_CATEGORIES: frozenset[str]`, `HINT_KINDS = {"questions","analysis"}`, `VISIBILITIES = {"visible","unshown"}`, `DIFFICULTIES = {"Easy","Medium","Hard"}`. Tasks 3 and 5 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty). Create `tests/test_task_schema.py`:

```python
"""Tests for torch_judge.tasks._schema.validate_task."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.tasks._schema import (
    BEHAVIOR_CATEGORIES,
    TaskValidationError,
    validate_task,
)


def make_legacy_task() -> dict:
    return {
        "title": "ReLU",
        "title_zh": "ReLU",
        "difficulty": "Easy",
        "function_name": "relu",
        "hint": "clamp at zero",
        "hint_zh": "在零处截断",
        "description_en": "Implement ReLU.",
        "description_zh": "实现 ReLU。",
        "tests": [{"name": "basic", "code": "assert {fn} is not None"}],
        "solution": "def relu(x):\n    return x.clamp(min=0)",
    }


def make_new_task() -> dict:
    return {
        "title": "Top-k Expert Router",
        "difficulty": "Medium",
        "function_name": "topk_route",
        "version": 1,
        "description_en": "Implement a top-k router.",
        "advisory_prerequisites": ["softmax", "moe"],
        "hints": [
            {"level": 1, "kind": "questions", "content": "Which dim is experts?"},
            {"level": 2, "kind": "analysis", "content": "Normalize logits per token."},
        ],
        "model_connections": ["DeepSeek-V3 uses fine-grained experts."],
        "pro_con_analysis": {"pros": ["sparse compute"], "cons": ["routing instability"]},
        "sources": [
            {
                "kind": "code",
                "url": "https://github.com/deepseek-ai/DeepSeek-V3",
                "commit": "a" * 40,
                "path": "inference/model.py",
                "symbol": "DeepseekV3ForCausalLM.forward",
                "license": "MIT",
                "adapted": "Router input/output contract and top-k selection semantics.",
                "simplifications": "Single-process execution with tiny tensors.",
            }
        ],
        "tests": [
            {"name": "shape", "code": "assert True", "behavior": "tensor.shape"},
            {
                "name": "normalization",
                "code": "assert True",
                "behavior": "routing.normalization",
                "visibility": "unshown",
                "failure_message": "Expert probabilities must sum to one per token.",
            },
        ],
        "solution": "def topk_route(logits, k):\n    ...",
    }


def test_legacy_task_validates():
    validate_task("relu", make_legacy_task())


def test_new_task_validates():
    validate_task("moe_topk_router", make_new_task(), known_ids={"softmax", "moe", "moe_topk_router"})


@pytest.mark.parametrize("key", ["title", "difficulty", "function_name", "description_en", "solution"])
def test_missing_required_field(key):
    task = make_legacy_task()
    del task[key]
    with pytest.raises(TaskValidationError, match=key):
        validate_task("relu", task)


def test_bad_difficulty():
    task = make_legacy_task()
    task["difficulty"] = "medium"
    with pytest.raises(TaskValidationError, match="difficulty"):
        validate_task("relu", task)


def test_needs_hint_or_hints():
    task = make_legacy_task()
    del task["hint"]
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("relu", task)


def test_duplicate_hint_levels_rejected():
    task = make_new_task()
    task["hints"].append({"level": 1, "kind": "questions", "content": "dup"})
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_hint_level_out_of_range_rejected():
    task = make_new_task()
    task["hints"][0]["level"] = 3
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_bad_hint_kind_rejected():
    task = make_new_task()
    task["hints"][0]["kind"] = "spoiler"
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_both_hint_levels_are_required():
    task = make_new_task()
    task["hints"] = task["hints"][:1]
    with pytest.raises(TaskValidationError, match="levels 1 and 2"):
        validate_task("t", task)


def test_hint_kind_is_fixed_by_level():
    task = make_new_task()
    task["hints"][0]["kind"] = "analysis"
    with pytest.raises(TaskValidationError, match="level 1 must use kind 'questions'"):
        validate_task("t", task)


def test_version_must_be_positive_int():
    task = make_new_task()
    task["version"] = 0
    with pytest.raises(TaskValidationError, match="version"):
        validate_task("t", task)


def test_unknown_behavior_category_rejected():
    task = make_new_task()
    task["tests"][0]["behavior"] = "routing.made_up"
    with pytest.raises(TaskValidationError, match="tests"):
        validate_task("t", task)


def test_unshown_requires_behavior_and_failure_message():
    task = make_new_task()
    del task["tests"][1]["failure_message"]
    with pytest.raises(TaskValidationError, match="failure_message"):
        validate_task("t", task)


def test_unknown_advisory_prerequisite_rejected():
    task = make_new_task()
    with pytest.raises(TaskValidationError, match="advisory_prerequisites"):
        validate_task("t", task, known_ids={"softmax"})


def test_source_with_path_requires_commit():
    task = make_new_task()
    del task["sources"][0]["commit"]
    with pytest.raises(TaskValidationError, match="sources"):
        validate_task("t", task)


def test_code_source_requires_full_provenance():
    task = make_new_task()
    task["sources"][0]["commit"] = "abc1234"
    with pytest.raises(TaskValidationError, match="40-character commit"):
        validate_task("t", task)


def test_paper_source_requires_precise_locator():
    task = make_new_task()
    task["sources"] = [{"kind": "paper", "url": "https://arxiv.org/abs/2412.19437"}]
    with pytest.raises(TaskValidationError, match="precise locator"):
        validate_task("t", task)


def test_behavior_categories_match_spec_count():
    assert len(BEHAVIOR_CATEGORIES) == 27
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_schema.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'torch_judge.tasks._schema'`

- [ ] **Step 3: Write the implementation**

Create `torch_judge/tasks/_schema.py`:

```python
"""Validation for task metadata — legacy fields plus advanced-curriculum extensions."""

from __future__ import annotations

import re
from typing import Any

DIFFICULTIES = frozenset({"Easy", "Medium", "Hard"})
HINT_KINDS = frozenset({"questions", "analysis"})
VISIBILITIES = frozenset({"visible", "unshown"})

BEHAVIOR_CATEGORIES = frozenset({
    "contract.signature",
    "tensor.shape",
    "tensor.dtype_device",
    "numerics.stability",
    "edge.empty_or_boundary",
    "state.invariant",
    "gradient.flow",
    "attention.masking",
    "attention.cache",
    "routing.selection",
    "routing.normalization",
    "dispatch.token_conservation",
    "capacity.overflow",
    "experts.sparsity",
    "training.router_gradient",
    "training.load_balance",
    "protocol.validation",
    "events.ordering",
    "retry.classification",
    "retry.backoff",
    "effects.idempotency",
    "scheduler.concurrency",
    "budget.enforcement",
    "checkpoint.recovery",
    "security.permission",
    "security.injection",
    "security.redaction",
})


class TaskValidationError(ValueError):
    def __init__(self, task_id: str, field: str, message: str):
        self.task_id = task_id
        self.field = field
        super().__init__(f"Task '{task_id}', field '{field}': {message}")


def _require_str(task_id: str, task: dict, key: str) -> None:
    value = task.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(task_id, key, "required non-empty string")


def _validate_hints(task_id: str, hints: Any) -> None:
    if not isinstance(hints, list) or not hints:
        raise TaskValidationError(task_id, "hints", "must be a non-empty list")
    seen_levels: set[int] = set()
    for i, h in enumerate(hints):
        if not isinstance(h, dict):
            raise TaskValidationError(task_id, "hints", f"entry {i} must be a dict")
        level = h.get("level")
        if level not in (1, 2):
            raise TaskValidationError(task_id, "hints", f"entry {i}: level must be 1 or 2")
        if level in seen_levels:
            raise TaskValidationError(task_id, "hints", f"duplicate level {level}")
        seen_levels.add(level)
        if h.get("kind") not in HINT_KINDS:
            raise TaskValidationError(
                task_id, "hints", f"entry {i}: kind must be one of {sorted(HINT_KINDS)}"
            )
        if not isinstance(h.get("content"), str) or not h["content"].strip():
            raise TaskValidationError(task_id, "hints", f"entry {i}: content required")
        expected_kind = {1: "questions", 2: "analysis"}[level]
        if h["kind"] != expected_kind:
            raise TaskValidationError(
                task_id, "hints", f"level {level} must use kind '{expected_kind}'"
            )
    if seen_levels != {1, 2}:
        raise TaskValidationError(task_id, "hints", "new-style hints require levels 1 and 2")


def _validate_test(task_id: str, index: int, test: Any) -> None:
    field = f"tests[{index}]"
    if not isinstance(test, dict):
        raise TaskValidationError(task_id, field, "must be a dict")
    if not isinstance(test.get("name"), str) or not test["name"].strip():
        raise TaskValidationError(task_id, field, "name required")
    if not isinstance(test.get("code"), str) or not test["code"].strip():
        raise TaskValidationError(task_id, field, "code required")
    visibility = test.get("visibility", "visible")
    if visibility not in VISIBILITIES:
        raise TaskValidationError(task_id, field, f"visibility must be one of {sorted(VISIBILITIES)}")
    behavior = test.get("behavior")
    if behavior is not None and behavior not in BEHAVIOR_CATEGORIES:
        raise TaskValidationError(task_id, field, f"unknown behavior category '{behavior}'")
    if visibility == "unshown":
        if behavior is None:
            raise TaskValidationError(task_id, field, "unshown case requires a behavior category")
        if not isinstance(test.get("failure_message"), str) or not test["failure_message"].strip():
            raise TaskValidationError(task_id, field, "unshown case requires failure_message")


def _validate_sources(task_id: str, sources: Any) -> None:
    if not isinstance(sources, list):
        raise TaskValidationError(task_id, "sources", "must be a list")
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            raise TaskValidationError(task_id, "sources", f"entry {i} must be a dict")
        kind = src.get("kind")
        if kind not in {"code", "paper"}:
            raise TaskValidationError(task_id, "sources", f"entry {i}: kind must be code or paper")
        if not isinstance(src.get("url"), str) or not src["url"].strip():
            raise TaskValidationError(task_id, "sources", f"entry {i}: url required")
        if kind == "code":
            commit = src.get("commit")
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
                raise TaskValidationError(
                    task_id, "sources", f"entry {i}: code source requires a 40-character commit hash"
                )
            for key in ("path", "symbol", "license", "adapted", "simplifications"):
                if not isinstance(src.get(key), str) or not src[key].strip():
                    raise TaskValidationError(task_id, "sources", f"entry {i}: {key} required")
        else:
            locators = ("section", "equation", "figure", "pages")
            if not any(isinstance(src.get(key), str) and src[key].strip() for key in locators):
                raise TaskValidationError(
                    task_id, "sources", f"entry {i}: paper source requires a precise locator"
                )


def validate_task(task_id: str, task: dict, known_ids: set[str] | None = None) -> None:
    """Validate one TASK dict. Raises TaskValidationError with a precise field."""
    if not isinstance(task, dict):
        raise TaskValidationError(task_id, "<root>", "TASK must be a dict")
    for key in ("title", "difficulty", "function_name", "description_en", "solution"):
        _require_str(task_id, task, key)
    if task["difficulty"] not in DIFFICULTIES:
        raise TaskValidationError(
            task_id, "difficulty", f"must be one of {sorted(DIFFICULTIES)}"
        )

    version = task.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TaskValidationError(task_id, "version", "must be an integer >= 1")

    if "hint" not in task and "hints" not in task:
        raise TaskValidationError(task_id, "hints", "task needs legacy 'hint' or new 'hints'")
    if "hint" in task:
        _require_str(task_id, task, "hint")
    if "hints" in task:
        _validate_hints(task_id, task["hints"])

    tests = task.get("tests")
    if not isinstance(tests, list) or not tests:
        raise TaskValidationError(task_id, "tests", "required non-empty list")
    for i, test in enumerate(tests):
        _validate_test(task_id, i, test)

    if "advisory_prerequisites" in task:
        prereqs = task["advisory_prerequisites"]
        if not isinstance(prereqs, list) or not all(isinstance(p, str) for p in prereqs):
            raise TaskValidationError(task_id, "advisory_prerequisites", "must be a list of task ids")
        if known_ids is not None:
            unknown = sorted(set(prereqs) - known_ids)
            if unknown:
                raise TaskValidationError(
                    task_id, "advisory_prerequisites", f"unknown task ids: {unknown}"
                )

    if "pro_con_analysis" in task:
        pca = task["pro_con_analysis"]
        ok = (
            isinstance(pca, dict)
            and isinstance(pca.get("pros"), list)
            and isinstance(pca.get("cons"), list)
            and all(isinstance(x, str) for x in pca["pros"] + pca["cons"])
        )
        if not ok:
            raise TaskValidationError(
                task_id, "pro_con_analysis", "must be {'pros': [str], 'cons': [str]}"
            )

    if "model_connections" in task:
        mc = task["model_connections"]
        if not isinstance(mc, list) or not all(isinstance(x, str) for x in mc):
            raise TaskValidationError(task_id, "model_connections", "must be a list of strings")

    if "sources" in task:
        _validate_sources(task_id, task["sources"])

    if "solution" in task:
        _require_str(task_id, task, "solution")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_schema.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_judge/tasks/_schema.py tests/__init__.py tests/test_task_schema.py
git commit -m "feat: add task metadata schema validation with behavior categories"
```

---

### Task 3: Wire validation into export and export new fields

**Files:**
- Modify: `scripts/export_problems.py`
- Modify: `scripts/test_export_problems.py`
- Test: `tests/test_all_tasks_validate.py` (new)
- Regenerate: `web/src/lib/problems.json`

**Interfaces:**
- Consumes: `validate_task`, `TaskValidationError` from Task 2.
- Produces: `problems.json` entries gain `version` (always, default 1) and, when present on the task: `hints`, `advisoryPrerequisites`, `modelConnections`, `proConAnalysis`, `sources`. Exported test entries are `{name, code, behavior?}` for visible cases and `{name, visibility: "unshown", behavior}` (no `code`, no `failure_message`) for unshown cases. Task 7's TypeScript types mirror exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_all_tasks_validate.py`:

```python
"""Every registered task must satisfy the schema (spec: invalid metadata fails the build)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.tasks import TASKS
from torch_judge.tasks._schema import validate_task


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_task_validates(task_id):
    validate_task(task_id, TASKS[task_id], known_ids=set(TASKS))
```

In `scripts/test_export_problems.py`, replace the exact-key-set assertion in `test_exported_problem_shape` with a superset check and add two new tests at the end of the file:

```python
def test_exported_problem_shape():
    problem = build_problem_catalog()["problems"][0]
    required = {
        "id", "title", "titleZh", "difficulty", "functionName",
        "hint", "hintZh", "descriptionEn", "descriptionZh", "tests", "version",
    }
    assert required <= set(problem)
    assert isinstance(problem["version"], int) and problem["version"] >= 1


def test_visible_test_entries_keep_code():
    for problem in build_problem_catalog()["problems"]:
        for test in problem["tests"]:
            if test.get("visibility") != "unshown":
                assert isinstance(test["code"], str) and test["code"].strip()


def test_unshown_test_entries_have_no_code():
    for problem in build_problem_catalog()["problems"]:
        for test in problem["tests"]:
            if test.get("visibility") == "unshown":
                assert "code" not in test
                assert "failure_message" not in test
                assert test["behavior"]
```

(Keep the rest of `test_export_problems.py` unchanged.)

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/python -m pytest tests/test_all_tasks_validate.py scripts/test_export_problems.py -v`
Expected: every `test_all_tasks_validate` parameter passes because the existing 76 tasks have the required legacy metadata and reference solutions; `test_exported_problem_shape` FAILS because exported problems do not yet contain `version`.

- [ ] **Step 3: Update export_problems.py**

In `scripts/export_problems.py`:

1. Replace the `REQUIRED_TASK_KEYS` tuple and `_validate_task` function with an import and thin wrapper:

```python
from torch_judge.tasks._schema import validate_task
```

2. Replace `_problem_entry` with:

```python
_OPTIONAL_FIELDS = (
    ("hints", "hints"),
    ("advisory_prerequisites", "advisoryPrerequisites"),
    ("model_connections", "modelConnections"),
    ("pro_con_analysis", "proConAnalysis"),
    ("sources", "sources"),
)


def _test_entry(test: dict[str, Any]) -> dict[str, Any]:
    if test.get("visibility") == "unshown":
        return {"name": test["name"], "visibility": "unshown", "behavior": test["behavior"]}
    entry: dict[str, Any] = {"name": test["name"], "code": test["code"]}
    if "behavior" in test:
        entry["behavior"] = test["behavior"]
    return entry


def _problem_entry(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    validate_task(task_id, task, known_ids=set(TASKS))
    entry = {
        "id": task_id,
        "title": task["title"],
        "titleZh": task.get("title_zh", task["title"]),
        "difficulty": task["difficulty"],
        "functionName": task["function_name"],
        "hint": task.get("hint", ""),
        "hintZh": task.get("hint_zh", task.get("hint", "")),
        "descriptionEn": task["description_en"],
        "descriptionZh": task.get("description_zh", task["description_en"]),
        "version": task.get("version", 1),
        "tests": [_test_entry(t) for t in task["tests"]],
    }
    for task_key, out_key in _OPTIONAL_FIELDS:
        if task_key in task:
            entry[out_key] = task[task_key]
    return entry
```

- [ ] **Step 4: Regenerate problems.json and run all Python tests**

```bash
npm run build:problems
npm run build:solutions
.venv/bin/python -m pytest tests/ scripts/ -v
```

Expected: export prints the problem count; `git diff --stat web/src/lib/solutions.json` shows no change; all tests PASS (including `test_build_problem_catalog_matches_committed_json` against the regenerated file). The `problems.json` diff should only add `"version": 1` per problem — inspect with `git diff web/src/lib/problems.json | head -40`.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_problems.py scripts/test_export_problems.py tests/test_all_tasks_validate.py web/src/lib/problems.json
git commit -m "feat: validate task metadata on export and emit versioned, visibility-aware problem data"
```

---

### Task 4: Two-level hint API in torch_judge

**Files:**
- Modify: `torch_judge/engine.py:104-111` (the `hint` function)
- Test: `tests/test_hint_api.py`

**Interfaces:**
- Consumes: `hints` list format validated in Task 2.
- Produces: `hint(task_id: str, level: int | None = None) -> None` — legacy tasks print `task["hint"]` unchanged; tasks with `hints` print exactly one level (default 1) and never both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hint_api.py`:

```python
"""Notebook hint() must support explicit levels while keeping legacy behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge import engine
from torch_judge.tasks import TASKS

NEW_TASK = {
    "title": "Top-k Router",
    "difficulty": "Medium",
    "function_name": "topk_route",
    "description_en": "x",
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which dim is experts?"},
        {"level": 2, "kind": "analysis", "content": "Normalize logits per token."},
    ],
    "tests": [{"name": "t", "code": "assert True"}],
}


def test_legacy_hint_unchanged(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_legacy", {"title": "L", "hint": "old style hint"})
    engine.hint("_legacy")
    assert "old style hint" in capsys.readouterr().out


def test_default_level_is_one(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new")
    out = capsys.readouterr().out
    assert "Which dim is experts?" in out
    assert "Normalize logits" not in out


def test_explicit_level_two(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new", level=2)
    out = capsys.readouterr().out
    assert "Normalize logits per token." in out
    assert "Which dim is experts?" not in out


def test_missing_level_reports_available(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new", level=3)
    out = capsys.readouterr().out
    assert "3" in out and "1, 2" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hint_api.py -v`
Expected: FAIL — `hint() got an unexpected keyword argument 'level'` (and default-level test fails).

- [ ] **Step 3: Replace the hint function in torch_judge/engine.py**

Replace the existing `hint` function (lines 104–111) with:

```python
def hint(task_id: str, level: int | None = None) -> None:
    """Show a hint. New-style tasks have independent levels: hint(id, level=1|2)."""
    task = get_task(task_id)
    if task is None:
        print(f"{_RED}Unknown task '{task_id}'.{_RESET}")
        return
    hints = task.get("hints")
    if hints:
        chosen = 1 if level is None else level
        match = next((h for h in hints if h["level"] == chosen), None)
        if match is None:
            available = ", ".join(str(h["level"]) for h in sorted(hints, key=lambda h: h["level"]))
            print(f"{_YELLOW}No level {chosen} hint for '{task_id}'. Available levels: {available}{_RESET}")
            return
        label = "Guiding questions" if match["kind"] == "questions" else "Analysis"
        print(f"\n{_YELLOW}💡 Hint (level {match['level']} — {label}) for {task['title']}:{_RESET}")
        print(f"   {match['content']}\n")
        return
    print(f"\n{_YELLOW}💡 Hint for {task['title']}:{_RESET}")
    print(f"   {task['hint']}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hint_api.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_judge/engine.py tests/test_hint_api.py
git commit -m "feat: two-level hint API with legacy fallback"
```

---

### Task 5: Behavior-aware, visibility-aware grading results

**Files:**
- Modify: `grading_service/main.py` (`TestResult` model ~line 85; `_execute_tests` ~lines 122-211)
- Test: `tests/test_grading_service.py`

**Interfaces:**
- Consumes: per-test `behavior` / `visibility` / `failure_message` keys (Task 2 format).
- Produces: `TestResult` gains `behavior: str | None = None`, `visibility: str = "visible"`, and `testIndex: int` containing the original task-test index. For a failing unshown case, `error` is exactly the authored `failure_message` (fallback `"Behavior check failed: <behavior>"`), and `output` is always `None` for unshown cases. Task 7 uses `testIndex` to align selected visible runs with the original test metadata.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grading_service.py`:

```python
"""Unit tests for behavior-aware grading results (no HTTP server needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grading_service.main import _execute_tests

TASK = {
    "title": "Add",
    "difficulty": "Easy",
    "function_name": "add",
    "description_en": "x",
    "hint": "x",
    "tests": [
        {"name": "visible pass", "code": "assert {fn}(1, 2) == 3", "behavior": "contract.signature"},
        {
            "name": "unshown case",
            "code": "print('leak'); assert {fn}(2, 2) == 4, 'raw assertion detail'",
            "behavior": "routing.normalization",
            "visibility": "unshown",
            "failure_message": "Routing normalization failed: probabilities must sum to one.",
        },
    ],
}

GOOD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"


def test_behavior_and_visibility_on_results():
    resp = _execute_tests(GOOD, TASK)
    assert resp.allPassed
    assert resp.results[0].behavior == "contract.signature"
    assert resp.results[0].visibility == "visible"
    assert resp.results[0].testIndex == 0
    assert resp.results[1].visibility == "unshown"
    assert resp.results[1].testIndex == 1


def test_selected_run_preserves_original_test_index():
    resp = _execute_tests(GOOD, TASK, test_indices=[1])
    assert len(resp.results) == 1
    assert resp.results[0].testIndex == 1


def test_unshown_failure_masks_detail():
    resp = _execute_tests(BAD, TASK)
    failing = resp.results[1]
    assert not failing.passed
    assert failing.error == "Routing normalization failed: probabilities must sum to one."
    assert "raw assertion detail" not in (failing.error or "")
    assert failing.output is None


def test_unshown_pass_hides_output():
    resp = _execute_tests(GOOD, TASK)
    assert resp.results[1].output is None


def test_unshown_failure_without_message_falls_back_to_behavior():
    task = {**TASK, "tests": [dict(TASK["tests"][1])]}
    del task["tests"][0]["failure_message"]
    resp = _execute_tests(BAD, task)
    assert resp.results[0].error == "Behavior check failed: routing.normalization"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grading_service.py -v`
Expected: FAIL — `TestResult` has no field `behavior` / assertions on masking fail.

- [ ] **Step 3: Implement in grading_service/main.py**

1. Extend the `TestResult` model:

```python
class TestResult(BaseModel):
    name: str
    passed: bool
    execTimeMs: float
    error: str | None = None
    output: str | None = None
    behavior: str | None = None
    visibility: str = "visible"
    testIndex: int
```

2. Add a helper directly above `_execute_tests`:

```python
def _finalize_result(result: TestResult, test: dict, test_index: int) -> TestResult:
    """Attach behavior metadata and mask unshown-case details (spec: Evaluation Model)."""
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
```

3. In `_execute_tests`, preserve original indices before executing:

```python
    indexed_tests = (
        [(i, all_tests[i]) for i in test_indices if 0 <= i < len(all_tests)]
        if test_indices is not None
        else list(enumerate(all_tests))
    )
```

Replace later uses of `tests` for the out-of-range check and result count with `indexed_tests`, then iterate with `for test_index, test in indexed_tests:`. Wrap every one of the six `results.append(TestResult(...))` call sites as:

```python
results.append(_finalize_result(TestResult(..., testIndex=test_index), test, test_index))
```

Use `total=len(results)` as today. `testIndex` is required on every result, including visible legacy cases.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_grading_service.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add grading_service/main.py tests/test_grading_service.py
git commit -m "feat: behavior categories and unshown-case masking in grading results"
```

---

### Task 6: Contract-version database migration

**Files:**
- Modify: `grading_service/main.py` (`_get_db` ~lines 33-71; `ProgressEntry`, `save_progress`, `get_progress`, `get_submissions`)
- Test: `tests/test_db_migration.py`

**Interfaces:**
- Consumes: `task.get("version", 1)` (Task 2 semantics).
- Produces: existing `progress` and `submissions` tables gain `contract_version INTEGER NOT NULL DEFAULT 1`; new version-keyed `progress_revisions` is unique on `(user_id, task_id, contract_version)`; `ProgressEntry` gains `contractVersion: int = 1` and `completedVersions: list[int]`; submissions API returns `contractVersion`. Existing rows are copied into revision 1 without rewriting the legacy aggregate row.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_migration.py`:

```python
"""Backward-compatible contract_version migration (spec: Compatibility and Migration)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import grading_service.main as gs


def _make_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, session_token TEXT UNIQUE NOT NULL, created_at TEXT)")
    conn.execute("CREATE TABLE progress (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, task_id TEXT NOT NULL, status TEXT NOT NULL, best_time_ms REAL, attempts INTEGER DEFAULT 0, solved_at TEXT, UNIQUE(user_id, task_id))")
    conn.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, task_id TEXT NOT NULL, code TEXT NOT NULL, passed INTEGER NOT NULL, exec_time_ms REAL, submitted_at TEXT)")
    conn.execute("INSERT INTO users (session_token) VALUES ('tok')")
    conn.execute("INSERT INTO progress (user_id, task_id, status, attempts) VALUES (1, 'relu', 'solved', 3)")
    conn.execute("INSERT INTO submissions (user_id, task_id, code, passed) VALUES (1, 'relu', 'def relu(x): ...', 1)")
    conn.commit()
    conn.close()


def test_legacy_rows_become_version_one(tmp_path, monkeypatch):
    db = tmp_path / "pyre.db"
    _make_legacy_db(db)
    monkeypatch.setattr(gs, "_DB_PATH", str(db))
    with gs._get_db() as conn:
        assert conn.execute("SELECT contract_version FROM progress WHERE task_id='relu'").fetchone()[0] == 1
        assert conn.execute("SELECT contract_version FROM submissions WHERE task_id='relu'").fetchone()[0] == 1
        revision = conn.execute(
            "SELECT contract_version, status, attempts FROM progress_revisions WHERE task_id='relu'"
        ).fetchone()
        assert revision == (1, "solved", 3)


def test_migration_is_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(db))
    gs._get_db().close()
    gs._get_db().close()  # second open must not fail on existing column


def _install_task(monkeypatch, version: int) -> dict:
    task = {
        "title": "V", "difficulty": "Easy", "function_name": "f",
        "description_en": "x", "hint": "x", "version": version,
        "solution": "def f(): pass", "tests": [{"name": "t", "code": "assert True"}],
    }
    monkeypatch.setitem(gs.get_task.__globals__["TASKS"], "_versioned", task)
    return task


def test_new_revision_does_not_overwrite_solved_history(tmp_path, monkeypatch):
    db = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(db))
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

    with gs._get_db() as conn:
        rows = conn.execute(
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
    db = tmp_path / "pyre.db"
    monkeypatch.setattr(gs, "_DB_PATH", str(db))
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_db_migration.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such column: contract_version` or `no such table: progress_revisions`.

- [ ] **Step 3: Implement in grading_service/main.py**

1. Add above `_get_db`:

```python
def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
```

2. In `_get_db`, after the existing three `CREATE TABLE IF NOT EXISTS` statements and before `conn.commit()`, add columns, create revision history, and copy legacy aggregates as version 1:

```python
    _ensure_column(conn, "progress", "contract_version", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "submissions", "contract_version", "INTEGER NOT NULL DEFAULT 1")
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
        INSERT OR IGNORE INTO progress_revisions
            (user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at)
        SELECT user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at
        FROM progress
    """)
```

3. Add `Field` to the existing pydantic import (`from pydantic import BaseModel, Field`) and replace `ProgressEntry` with:

```python
class ProgressEntry(BaseModel):
    status: str
    bestTimeMs: float | None = None
    attempts: int
    solvedAt: str | None = None
    contractVersion: int = 1
    completedVersions: list[int] = Field(default_factory=list)
```

4. Add the per-version upsert helper above `save_progress`:

```python
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
            "solved_at = CASE WHEN ? = 'solved' THEN COALESCE(solved_at, datetime('now')) ELSE solved_at END "
            "WHERE user_id = ? AND task_id = ? AND contract_version = ?",
            (next_status, best, next_status, user_id, task_id, contract_version),
        )
        return
    conn.execute(
        "INSERT INTO progress_revisions "
        "(user_id, task_id, contract_version, status, best_time_ms, attempts, solved_at) "
        "VALUES (?, ?, ?, ?, ?, 1, CASE WHEN ? = 'solved' THEN datetime('now') ELSE NULL END)",
        (
            user_id, task_id, contract_version, status,
            exec_time_ms if status == "solved" else None, status,
        ),
    )
```

5. Replace `get_progress` with a revision-aware implementation. It reports current-version progress while retaining solved older versions:

```python
@app.get("/progress/{user_id}")
def get_progress(user_id: int) -> dict[str, ProgressEntry]:
    with _get_db() as conn:
        task_ids = [
            row[0] for row in conn.execute(
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
                row[0] for row in conn.execute(
                    "SELECT contract_version FROM progress_revisions "
                    "WHERE user_id = ? AND task_id = ? AND status = 'solved' "
                    "ORDER BY contract_version",
                    (user_id, task_id),
                ).fetchall()
            ]
            if current is None:
                result[task_id] = ProgressEntry(
                    status="todo", attempts=0, contractVersion=contract_version,
                    completedVersions=completed,
                )
            else:
                result[task_id] = ProgressEntry(
                    status=current[0], bestTimeMs=current[1], attempts=current[2],
                    solvedAt=current[3], contractVersion=contract_version,
                    completedVersions=completed,
                )
    return result
```

6. Keep the existing aggregate `progress` update logic for backward compatibility, but calculate `contract_version` at the top of `save_progress`, stamp it on every aggregate update/insert, and call `_save_revision_progress` after obtaining `user_id`:

```python
    contract_version = (get_task(request.taskId) or {}).get("version", 1)
    _save_revision_progress(
        conn,
        user_id=user_id,
        task_id=request.taskId,
        contract_version=contract_version,
        status=request.status,
        exec_time_ms=request.execTimeMs,
    )
```

Use these exact aggregate changes:

- solved UPDATE gains `contract_version = ?` with params `("solved", best, contract_version, user_id, request.taskId)`
- non-solved UPDATE gains `contract_version = ?` with params `(next_status, contract_version, user_id, request.taskId)`
- both progress INSERTs include `contract_version`
- submissions INSERT becomes `INSERT INTO submissions (user_id, task_id, code, passed, exec_time_ms, contract_version) VALUES (?, ?, ?, ?, ?, ?)`

7. In `get_submissions`, select `contract_version` after `code` and add `"contractVersion": r[5]` to each returned dictionary.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_db_migration.py tests/test_grading_service.py -v`
Expected: all database migration and grading tests PASS, including coexistence of solved v1 and attempted v2 rows.

- [ ] **Step 5: Commit**

```bash
git add grading_service/main.py tests/test_db_migration.py
git commit -m "feat: backward-compatible contract_version migration for progress and submissions"
```

---

### Task 7: Frontend types, two-level hint UI, unshown-case handling

**Files:**
- Modify: `web/src/lib/types.ts`
- Create: `web/src/lib/hints.ts`
- Modify: `web/src/components/workspace/DescriptionTab.tsx` (hint section, lines 60-109)
- Modify: `web/src/components/workspace/TestCasesView.tsx:19`
- Modify: `web/src/components/workspace/TestResultsView.tsx` (guards + behavior chip)
- Modify: `web/src/app/problems/[id]/page.tsx:145`
- Modify: `web/src/lib/problemContext.ts`
- Create: `web/vitest.config.ts`, `web/src/lib/hints.test.ts`, `web/src/lib/problemContext.test.ts`
- Modify: `web/package.json` (add `"test": "vitest run"` script; dev-dep `vitest`)

**Interfaces:**
- Consumes: JSON field names from Task 3 (`version`, `hints[{level,kind,content}]`, `tests[{name,code?,behavior?,visibility?}]`) and result fields from Task 5 (`behavior`, `visibility`, `testIndex`).
- Produces: `getHintLevels(problem): HintLevel[]`, `visibleTestIndices(tests): number[]`, and `resultTestIndex(result, fallback): number` in `web/src/lib/hints.ts`; updated `Problem`/`Test`/`TestResult` types used by later plans.

- [ ] **Step 1: Install vitest and add the test script**

```bash
cd web && npm install -D vitest
```

In `web/package.json` scripts add: `"test": "vitest run"`.

- [ ] **Step 2: Write the failing lib tests**

Create `web/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
});
```

Create `web/src/lib/hints.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { getHintLevels, resultTestIndex, visibleTestIndices } from '@/lib/hints';
import type { Problem, Test, TestResult } from '@/lib/types';

const base: Problem = {
  id: 'x', title: 'X', titleZh: 'X', difficulty: 'Easy', functionName: 'f',
  hint: 'legacy', hintZh: '', descriptionEn: '', descriptionZh: '', tests: [],
};

describe('getHintLevels', () => {
  it('returns empty for legacy problems', () => {
    expect(getHintLevels(base)).toEqual([]);
  });

  it('returns hints sorted by level', () => {
    const p: Problem = {
      ...base,
      hints: [
        { level: 2, kind: 'analysis', content: 'b' },
        { level: 1, kind: 'questions', content: 'a' },
      ],
    };
    expect(getHintLevels(p).map(h => h.level)).toEqual([1, 2]);
  });
});

describe('visibleTestIndices', () => {
  it('keeps original indices and drops unshown cases', () => {
    const tests: Test[] = [
      { name: 'a', code: 'x' },
      { name: 'b', visibility: 'unshown', behavior: 'tensor.shape' },
      { name: 'c', code: 'y' },
    ];
    expect(visibleTestIndices(tests)).toEqual([0, 2]);
  });
});

describe('resultTestIndex', () => {
  it('uses the original task-test index returned by the grader', () => {
    const result = { name: 'visible two', passed: true, execTimeMs: 1, testIndex: 2 } satisfies TestResult;
    expect(resultTestIndex(result, 0)).toBe(2);
  });

  it('falls back for responses produced before testIndex was added', () => {
    const legacy = { name: 'legacy', passed: true, execTimeMs: 1 } satisfies TestResult;
    expect(resultTestIndex(legacy, 1)).toBe(1);
  });
});
```

Create `web/src/lib/problemContext.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { getSampleTests } from '@/lib/problemContext';
import type { Problem } from '@/lib/types';

const problem: Problem = {
  id: 'x', title: 'X', titleZh: 'X', difficulty: 'Easy', functionName: 'f',
  hint: '', hintZh: '', descriptionEn: '', descriptionZh: '',
  tests: [
    { name: 'unshown', visibility: 'unshown', behavior: 'tensor.shape' },
    { name: 'visible one', code: 'assert {fn}(1) == 1' },
    { name: 'visible two', code: 'assert {fn}(2) == 2' },
  ],
};

describe('getSampleTests', () => {
  it('filters unshown cases before applying the sample limit', () => {
    expect(getSampleTests(problem, 1)).toEqual([
      { name: 'visible one', code: 'assert f(1) == 1' },
    ]);
  });
});
```

Run: `cd web && npm test`
Expected: FAIL — cannot resolve `@/lib/hints`.

- [ ] **Step 3: Add types and the hints lib**

In `web/src/lib/types.ts`, replace the `Test`, `Problem`, and `TestResult` interfaces with:

```ts
export interface Test {
  name: string;
  code?: string;
  behavior?: string;
  visibility?: 'visible' | 'unshown';
}

export interface HintLevel {
  level: number;
  kind: 'questions' | 'analysis';
  content: string;
}

export interface SourceRef {
  kind: 'code' | 'paper';
  url: string;
  commit?: string;
  path?: string;
  symbol?: string;
  line?: number;
  section?: string;
  equation?: string;
  figure?: string;
  pages?: string;
  license?: string;
  adapted?: string;
  simplifications?: string;
}

export interface Problem {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  functionName: string;
  hint: string;
  hintZh: string;
  descriptionEn: string;
  descriptionZh: string;
  tests: Test[];
  version?: number;
  hints?: HintLevel[];
  advisoryPrerequisites?: string[];
  modelConnections?: string[];
  proConAnalysis?: { pros: string[]; cons: string[] };
  sources?: SourceRef[];
}

export interface TestResult {
  name: string;
  passed: boolean;
  execTimeMs: number;
  error?: string;
  output?: string;
  behavior?: string;
  visibility?: string;
  testIndex?: number;
}
```

Also add these fields to the existing interfaces:

```ts
// ProblemProgress
contractVersion?: number;
completedVersions?: number[];

// SubmissionHistory
contractVersion?: number;
```

Create `web/src/lib/hints.ts`:

```ts
import type { HintLevel, Problem, Test, TestResult } from '@/lib/types';

export function getHintLevels(problem: Problem): HintLevel[] {
  if (!problem.hints || problem.hints.length === 0) return [];
  return [...problem.hints].sort((a, b) => a.level - b.level);
}

export function visibleTestIndices(tests: Test[]): number[] {
  return tests
    .map((t, i) => (t.visibility === 'unshown' ? -1 : i))
    .filter(i => i >= 0);
}

export function resultTestIndex(result: TestResult, fallback: number): number {
  return result.testIndex ?? fallback;
}
```

In `web/src/lib/problemContext.ts`, replace `getSampleTests` with:

```ts
export function getSampleTests(problem: Problem, limit = 2): Array<{ name: string; code: string }> {
  return problem.tests
    .filter((test): test is Test & { code: string } => (
      test.visibility !== 'unshown' && typeof test.code === 'string'
    ))
    .slice(0, limit)
    .map((test) => ({
      name: test.name,
      code: formatTestCode(test.code, problem.functionName),
    }));
}
```

Add `Test` to that file's type import: `import type { Problem, Test } from '@/lib/types';`.

Run: `cd web && npm test`
Expected: PASS (4 tests).

- [ ] **Step 4: Independent hint controls in DescriptionTab**

In `web/src/components/workspace/DescriptionTab.tsx`:

1. Add imports: `import { getHintLevels } from '@/lib/hints';`
2. In the component body replace `const [hintOpen, setHintOpen] = useState(false);` with:

```tsx
  const [hintOpen, setHintOpen] = useState(false);
  const [openLevels, setOpenLevels] = useState<Record<number, boolean>>({});
  const hintLevels = getHintLevels(problem);
```

3. Replace the entire `{hint && ( ... )}` block (lines 83-109) with a leveled render that falls back to the legacy block:

```tsx
      {hintLevels.length > 0 ? (
        <div className="space-y-3">
          {hintLevels.map((h) => {
            const label = h.kind === 'questions' ? 'Guiding questions' : 'Analysis';
            const open = !!openLevels[h.level];
            return (
              <div key={h.level}>
                <button
                  onClick={() => setOpenLevels((prev) => ({ ...prev, [h.level]: !prev[h.level] }))}
                  className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
                >
                  <Lightbulb className="w-4 h-4" />
                  <span>{`${t('hint')} ${h.level} · ${label}`}</span>
                  {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                </button>
                {open && (
                  <div
                    className="mt-2 p-3 px-3.5 rounded-[9px] text-sm text-text-2 leading-relaxed space-y-1"
                    style={{
                      background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                      border: '1px solid var(--accent-line)',
                      borderLeft: '3px solid var(--accent)',
                    }}
                  >
                    <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">
                      ⚑ HINT · LEVEL {h.level}
                    </span>
                    {h.content.split('\n').map((line, i) => (
                      <p key={i}>{parseInline(line)}</p>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : hint ? (
        <div>
          <button
            onClick={() => setHintOpen(!hintOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <Lightbulb className="w-4 h-4" />
            <span>{t('hint')}</span>
            {hintOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {hintOpen && (
            <div
              className="mt-2 p-3 px-3.5 rounded-[9px] text-sm text-text-2 leading-relaxed space-y-1"
              style={{
                background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                border: '1px solid var(--accent-line)',
                borderLeft: '3px solid var(--accent)',
              }}
            >
              <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">⚑ HINT</span>
              {hint.split('\n').map((line, i) => (
                <p key={i}>{parseInline(line)}</p>
              ))}
            </div>
          )}
        </div>
      ) : null}
```

(The legacy branch is the current hint JSX from today's lines 84-108, unchanged, so legacy exercises render exactly as before.)

- [ ] **Step 5: Unshown-aware test selection and results**

1. `web/src/app/problems/[id]/page.tsx:145` — replace:

```ts
      const testIndices = problem.tests.slice(0, 2).map((_, i) => i);
```

with:

```ts
      const testIndices = visibleTestIndices(problem.tests).slice(0, 2);
```

and add `import { visibleTestIndices } from '@/lib/hints';` at the top.

2. `web/src/components/workspace/TestCasesView.tsx:19` — replace:

```ts
  const sampleTests = tests.slice(0, 2);
```

with:

```ts
  const sampleTests = tests
    .filter((test): test is Test & { code: string } => (
      test.visibility !== 'unshown' && typeof test.code === 'string'
    ))
    .slice(0, 2);
```

3. `web/src/components/workspace/TestResultsView.tsx`:
   - Add `import { resultTestIndex } from '@/lib/hints';`. Immediately after the existing `activeResult` assignment, align the task metadata with the grader's original test index:

```ts
  const activeResult = result.results[activeIndex];
  const metadataIndex = activeResult ? resultTestIndex(activeResult, activeIndex) : -1;
  const activeTest = metadataIndex >= 0 && metadataIndex < tests.length ? tests[metadataIndex] : null;
```

     This replaces the current `activeTest` assignment that indexes `tests` with `activeIndex`. It preserves correct names/code when selected runs skip interleaved unshown cases.

   - Guard the test-code block against unshown cases (no `code`). Replace the current block:

```tsx
            {activeTest && (
              <div>
                <h4 className="eyebrow mb-1.5">{t('testCasesTab')}</h4>
                <pre className="p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed" style={{ background: 'var(--bg-sunken)' }}>
                  <PythonCode code={formatTestCode(activeTest.code, functionName)} />
                </pre>
              </div>
            )}
```

with:

```tsx
            {activeTest?.code && (
              <div>
                <h4 className="eyebrow mb-1.5">{t('testCasesTab')}</h4>
                <pre className="p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed" style={{ background: 'var(--bg-sunken)' }}>
                  <PythonCode code={formatTestCode(activeTest.code, functionName)} />
                </pre>
              </div>
            )}
```

   - Below the case-header `<div className="flex items-center gap-3">`, after the runtime span, add a behavior chip:

```tsx
              {activeResult.behavior && (
                <span
                  className="mono text-[11px] px-1.5 py-0.5 rounded"
                  style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
                >
                  {activeResult.behavior}
                </span>
              )}
```

   - Where the test-code block is skipped because `activeTest?.code` is missing, render instead:

```tsx
            {activeTest && !activeTest.code && (
              <p className="text-xs text-text-3">
                Unshown evaluator case — the check runs during grading, but its inputs are not displayed.
              </p>
            )}
```

- [ ] **Step 6: Typecheck, build, and lib tests**

```bash
cd web && npx tsc --noEmit && npm test && npm run build
```

Expected: no type errors, 6 vitest tests PASS, and `next build` succeeds.

- [ ] **Step 7: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/hints.ts web/src/lib/hints.test.ts web/src/lib/problemContext.ts web/src/lib/problemContext.test.ts web/vitest.config.ts web/package.json web/package-lock.json web/src/components/workspace/DescriptionTab.tsx web/src/components/workspace/TestCasesView.tsx web/src/components/workspace/TestResultsView.tsx "web/src/app/problems/[id]/page.tsx"
git commit -m "feat: independent hint levels, behavior chips, and unshown-case handling in web UI"
```

---

### Task 8: End-to-end verification, push, and PR

**Files:** none new (verification and git only)

**Interfaces:**
- Consumes: everything above.
- Produces: green full test suite, smoke-tested app, branch pushed to `origin`, PR open against `meiru-cam/pyre-code-extend:main`.

- [ ] **Step 1: Full Python + web verification**

```bash
.venv/bin/python -m pytest tests/ scripts/ -v
npm run build:problems && git diff --exit-code web/src/lib/problems.json
cd web && npx tsc --noEmit && npm test && npm run build
```

Expected: all pytest tests PASS; `problems.json` regeneration produces no diff (export is deterministic); web typecheck/tests/build succeed.

- [ ] **Step 2: Manual smoke test without disturbing the running app on 4001/8000**

In terminal A, from the isolated worktree root:

```bash
.venv/bin/python -m uvicorn grading_service.main:app --host 127.0.0.1 --port 8101
```

In terminal B, from the isolated worktree root:

```bash
cd web
GRADING_SERVICE_URL=http://127.0.0.1:8101 npm run dev -- --port 4101
```

Open `http://localhost:4101/problems/gqa`: the legacy single hint still renders and toggles; run + submit a solution; results view remains correct for the legacy task. Stop both isolated smoke-test processes. The existing app on frontend port 4001 and grading service port 8000 remain untouched. New-format hint independence and selected-index alignment are covered by the six deterministic TypeScript tests above; browser interaction coverage remains explicitly deferred to Plan 6. Report any breakage before proceeding.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feature/curriculum-foundation
gh pr create --repo meiru-cam/pyre-code-extend --base main --head feature/curriculum-foundation \
  --title "Curriculum foundation: task schema, two-level hints, behavior-aware grading, contract versions" \
  --body "$(cat <<'EOF'
Implements Delivery Sequence items 1-4 of docs/superpowers/specs/2026-07-18-advanced-ai-curriculum-design.md:

- Task metadata schema validation (versions, two-level hints, behavior categories, pinned sources) failing export with precise field errors
- Two-level hint API (Python `hint(id, level=...)`) and independent hint controls in the web UI, legacy hints unchanged
- Behavior-aware, visibility-aware grading results with unshown-case masking
- Backward-compatible `contract_version` SQLite migration (existing records = version 1)

All new behavior is additive; existing task IDs, paths, progress, and legacy hint fields are preserved.
EOF
)"
```

Expected: PR URL printed.

---

## Deviations from the spec (agreed at planning time)

- Difficulty values keep the repo's `"Easy"/"Medium"/"Hard"` casing instead of the spec example's lowercase `"medium"`.
- The spec's `TestCase(...)` object with a `run=` callable maps onto the repo's existing string-code test dicts via the added `behavior`/`visibility`/`failure_message` keys; a richer object model arrives with the harnesses in Plans 2 and 4.
- Frontend interaction tests for hint disclosure (jsdom + provider mocks) are deferred to Plan 6; Plan 1 covers the pure hint/visibility logic with node-env vitest plus typecheck/build and a manual smoke test.
- Harness-vs-learner failure distinction (spec: Error Handling) requires the harness layer and lands with Plans 2/4.
