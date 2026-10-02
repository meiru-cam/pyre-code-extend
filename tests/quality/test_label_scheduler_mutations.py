"""Mutation gate for the multi-part labeling task scheduler exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "label_scheduler"

_KEY = "key=lambda i: (task_count[task][i], human_count[u][i], i)"

MUTATIONS = [
    ("k above t accepted", 1, [("        if k > t:  # a human can see each task once, so at most t assignments each\n            return None\n", "")]),
    ("k == 0 gives None", 1, [("        if k == 0:\n            return []\n", "        if k == 0:\n            return None\n")]),
    ("m not validated", 1, [("if t <= 0 or m <= 0 or h <= 0:", "if t <= 0 or h <= 0:")]),
    ("tasks repeat", 1, [("((u + r) % t, 0, u)", "((u * r) % t, 0, u)")]),
    ("one round short", 1, [("for r in range(k) for u in range(h)]", "for r in range(k - 1) for u in range(h)] + [(0, 0, u) for u in range(h)]")]),
    ("model never rotates", 2, [("schedule.append((task, seen[task] % m, u))", "schedule.append((task, 0, u))")]),
    ("rotation by human", 2, [("schedule.append((task, seen[task] % m, u))", "schedule.append((task, u % m, u))")]),
    ("occurrences not counted", 2, [("            seen[task] += 1\n", "")]),
    ("human tie-break skipped", 3, [(_KEY, "key=lambda i: (task_count[task][i], i)")]),
    ("human count first", 3, [(_KEY, "key=lambda i: (human_count[u][i], task_count[task][i], i)")]),
    ("human counts never updated", 3, [("            human_count[u][model] += 1\n", "")]),
    ("prefix length off by one", 3, [("enumerate(schedule, start=1)", "enumerate(schedule)")]),
    ("difference of 1 flagged", 3, [("if max(row) - min(row) > 1:", "if max(row) - min(row) >= 1:")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
