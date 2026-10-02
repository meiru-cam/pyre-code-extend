"""Mutation gate for the multi-part delayed-answer search exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "delayed_search"

MUTATIONS = [
    ("call made when n is 1", 1, [("        w = _Window(n)\n        if w.solved():\n            return w.lo\n", "        w = _Window(n)\n")]),
    ("probe at the low end", 1, [("x = self.near(self.lo + i * (self.hi - self.lo) // (count + 1), picked)", "x = self.near(self.lo, picked)")]),
    ("used values reused", 1, [("if self.lo <= x <= self.hi and x not in self.used and x not in avoid:", "if self.lo <= x <= self.hi and x not in avoid:")]),
    ("spare answers ignored", 1, [("                for x, answer in zip(sent, answers):\n                    w.learn(x, answer)\n", "                if not real:\n                    for x, answer in zip(sent, answers):\n                        w.learn(x, answer)\n")]),
    ("one probe per batch", 2, [("return self._search(n, check_batch, 2)", "return self._search(n, check_batch, 1)")]),
    ("three probes per batch", 2, [("return self._search(n, check_batch, 2)", "return self._search(n, check_batch, 3)")]),
    ("two probes on two candidates", 2, [("        if self.hi - self.lo + 1 <= 2:\n            count = 1\n", "")]),
    ("games one at a time", 3, [("{g: (w.lo + w.hi) // 2 for g, w in windows.items() if not w.solved()}", "dict([(g, (w.lo + w.hi) // 2) for g, w in windows.items() if not w.solved()][:1])")]),
    ("solved games guessed", 3, [("{g: (w.lo + w.hi) // 2 for g, w in windows.items() if not w.solved()}", "{g: (w.lo + w.hi) // 2 for g, w in windows.items()}")]),
    ("no empty round", 3, [("guesses = {} if sent else {g:", "guesses = {g:")]),
    ("batch answers paired in reverse", 2, [("for x, answer in zip(sent, answers):", "for x, answer in zip(reversed(sent), answers):")]),
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
