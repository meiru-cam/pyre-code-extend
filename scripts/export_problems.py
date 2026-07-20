#!/usr/bin/env python3
"""Export problem metadata from torch_judge task definitions for the frontend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "web" / "src" / "lib" / "problems.json"

# Make torch_judge importable when the script is run directly.
sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS, list_tasks
from torch_judge.tasks._schema import validate_task


_OPTIONAL_FIELDS = (
    ("hints", "hints"),
    ("advisory_prerequisites", "advisoryPrerequisites"),
    ("model_connections", "modelConnections"),
    ("pro_con_analysis", "proConAnalysis"),
    ("sources", "sources"),
    ("design_note_rubric", "designNoteRubric"),
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
        "tests": [_test_entry(test) for test in task["tests"]],
    }
    for task_key, output_key in _OPTIONAL_FIELDS:
        if task_key in task:
            entry[output_key] = task[task_key]
    return entry


def _load_existing_order(output_path: Path) -> list[str]:
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return [problem["id"] for problem in data.get("problems", [])]
    except (json.JSONDecodeError, KeyError):
        return []


def _ordered_task_ids(existing_order: list[str]) -> list[str]:
    ordered = [task_id for task_id in existing_order if task_id in TASKS]
    seen = set(ordered)
    ordered.extend(task_id for task_id, _ in list_tasks() if task_id not in seen)
    return ordered


def build_problem_catalog(output_path: Path = OUTPUT) -> dict[str, list[dict[str, Any]]]:
    task_ids = _ordered_task_ids(_load_existing_order(output_path))
    return {"problems": [_problem_entry(task_id, TASKS[task_id]) for task_id in task_ids]}


def export_problem_catalog(output_path: Path = OUTPUT) -> dict[str, list[dict[str, Any]]]:
    data = build_problem_catalog(output_path)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data


def main() -> None:
    data = export_problem_catalog()
    print(f"Written {len(data['problems'])} problems to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
