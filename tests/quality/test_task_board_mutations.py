"""Mutation gate for the multi-part task board exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "task_board"

MUTATIONS = [
    ("owner not checked", 1, [("if task is None or task.user != user or not task.active(timestamp):", "if task is None or not task.active(timestamp):")]),
    ("finished stays active", 1, [("        if self.finished is not None and self.finished <= at:\n            return False\n", "")]),
    ("finish counts from the next instant", 1, [("self.finished is not None and self.finished <= at", "self.finished is not None and self.finished < at")]),
    ("edit replaces created_at", 1, [("return (task_id, title, priority, task.created)", "return (task_id, title, priority, task.times[-1])")]),
    ("same-time edit ignored", 1, [("bisect.bisect_right(self.times, at) - 1", "bisect.bisect_left(self.times, at) - 1")]),
    ("ascending priority", 2, [("rows.sort(key=lambda row: -row[2])", "rows.sort(key=lambda row: row[2])")]),
    ("ties by title", 2, [("rows.sort(key=lambda row: -row[2])", "rows.sort(key=lambda row: (-row[2], row[1]))")]),
    ("zero minimum ignored", 2, [("if min_priority is None or row[2] >= min_priority:", "if not min_priority or row[2] >= min_priority:")]),
    ("strict minimum", 2, [("if min_priority is None or row[2] >= min_priority:", "if min_priority is None or row[2] > min_priority:")]),
    ("expiry inclusive", 3, [("return self.ttl is None or at < self.created + self.ttl", "return self.ttl is None or at <= self.created + self.ttl")]),
    ("ttl ignored", 3, [("return self.ttl is None or at < self.created + self.ttl", "return True")]),
    ("edit moves the deadline", 3, [("        task.times.append(timestamp)\n", "        task.times.append(timestamp)\n        if task.ttl is not None:\n            task.ttl += timestamp - task.created\n")]),
    ("latest version always", 4, [("return self.versions[bisect.bisect_right(self.times, at) - 1]", "return self.versions[-1]")]),
    ("later tasks visible", 4, [("        if at < self.created:\n            return False\n", "")]),
    ("versions scanned", 4, [("return self.versions[bisect.bisect_right(self.times, at) - 1]", "return [v for t, v in zip(self.times, self.versions) if t <= at][-1]")]),
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
