"""Every interview question carries a hint, and every stored hint still matches a question."""

from __future__ import annotations

import json

import pytest

from torch_judge.tasks import TASKS
from torch_judge.tasks._registry import INTERVIEW_HINTS_DIR, load_interview_hints

MAX_HINT_CHARS = 260


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_interview_question_has_a_hint(task_id):
    for question in TASKS[task_id]["interview_questions"]:
        hint = question.get("hint", "")
        assert hint.strip(), f"{task_id}: no hint for {question['question']!r}"
        assert len(hint) <= MAX_HINT_CHARS, f"{task_id}: hint over {MAX_HINT_CHARS} chars"
        assert hint.strip() != question["question"].strip()


def test_no_orphan_hints():
    orphans = []
    for task_id, by_question in load_interview_hints().items():
        questions = {q["question"] for q in TASKS.get(task_id, {}).get("interview_questions", [])}
        orphans += [(task_id, text) for text in by_question if text not in questions]
    assert not orphans, f"hints whose question was edited or removed: {orphans[:5]}"


def test_each_task_lives_in_one_hint_file():
    seen: dict[str, str] = {}
    for path in sorted(INTERVIEW_HINTS_DIR.glob("*.json")):
        for task_id in json.loads(path.read_text(encoding="utf-8")):
            assert task_id not in seen, f"{task_id} in both {seen[task_id]} and {path.name}"
            seen[task_id] = path.name
