"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK

# Interview hints live beside the tasks, keyed by task id and then by the exact question
# text, so editing a question without updating its hint shows up as a missing hint.
INTERVIEW_HINTS_DIR = Path(__file__).parent / "_interview_hints"


def load_interview_hints() -> dict[str, dict[str, str]]:
    hints: dict[str, dict[str, str]] = {}
    for path in sorted(INTERVIEW_HINTS_DIR.glob("*.json")):
        for task_id, by_question in json.loads(path.read_text(encoding="utf-8")).items():
            hints.setdefault(task_id, {}).update(by_question)
    return hints


for _task_id, _by_question in load_interview_hints().items():
    for _question in TASKS.get(_task_id, {}).get("interview_questions", []):
        if _question["question"] in _by_question:
            _question["hint"] = _by_question[_question["question"]]


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        TASKS.items(),
        key=lambda t: DIFFICULTY_ORDER.get(t[1]["difficulty"], 9),
    )
