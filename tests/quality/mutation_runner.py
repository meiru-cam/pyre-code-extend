"""Author-quality gate for deliberately broken exercise implementations."""

from __future__ import annotations

from dataclasses import dataclass

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task


@dataclass(frozen=True)
class Mutation:
    name: str
    code: str


def _failed_behaviors(response) -> set[str]:
    if response.error:
        return {"contract.signature"}
    return {
        result.behavior or f"uncategorized:{result.name}"
        for result in response.results
        if not result.passed
    }


def assert_mutations_rejected(task_id: str, mutations: list[Mutation]) -> dict[str, set[str]]:
    """Require the task reference to pass and every named mutation to fail."""
    task = get_task(task_id)
    if task is None:
        raise AssertionError(f"unknown task {task_id!r}")
    reference = _execute_tests(task["solution"], task, capture_output=False)
    assert reference.allPassed, (
        f"{task_id} reference solution failed: "
        f"{[(result.name, result.error) for result in reference.results if not result.passed]}"
    )

    rejected: dict[str, set[str]] = {}
    for mutation in mutations:
        response = _execute_tests(mutation.code, task, capture_output=False)
        failed = _failed_behaviors(response)
        assert failed, f"mutation survived for {task_id}: {mutation.name}"
        rejected[mutation.name] = failed
    return rejected
