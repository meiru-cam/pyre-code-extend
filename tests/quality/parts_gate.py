"""Shared mutation-gate helpers for multi-part exercises.

A multi-part exercise grades parts 1..k together, so every mutation must not only be
rejected but first fail in the part it targets, passing every earlier part.
"""

from __future__ import annotations

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

# (name, first part that must fail, [(text in the reference, replacement), ...]).
PartMutation = tuple[str, int, list[tuple[str, str]]]


def mutant(original: str, edits: list[tuple[str, str]], suffix: str = "") -> str:
    code = original
    for old, new in edits:
        assert old in code, f"mutation target drifted: {old[:60]!r}"
        code = code.replace(old, new, 1)
    return code + suffix


def first_failing_part(task: dict, code: str) -> int | None:
    response = _execute_tests(code, task, capture_output=False)
    assert response.error is None, response.error
    parts = [task["tests"][r.testIndex]["part"] for r in response.results if not r.passed]
    return min(parts) if parts else None


def assert_part_mutations_rejected(task_id: str, mutations: list[PartMutation], suffix: str = "") -> None:
    original = get_task(task_id)["solution"]
    rejected = assert_mutations_rejected(
        task_id, [Mutation(name, mutant(original, edits, suffix)) for name, _, edits in mutations],
    )
    assert set(rejected) == {name for name, _, _ in mutations}


def assert_first_fails_in_part(task_id: str, part: int, edits: list[tuple[str, str]], suffix: str = "") -> None:
    task = get_task(task_id)
    assert first_failing_part(task, mutant(task["solution"], edits, suffix)) == part
