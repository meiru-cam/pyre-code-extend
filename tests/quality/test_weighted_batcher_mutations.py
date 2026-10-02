"""Mutation gate for the multi-part weighted data batcher exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "weighted_batcher"

_OPEN = "            it = self._iters[name] = self._registry.get_iterator(name, self._offsets[name])\n"
_PICK = "            if self._offsets[name] * self._weights[best] < self._offsets[best] * self._weights[name]:\n"

MUTATIONS = [
    ("insertion order", 1, [("self._names = sorted(weights)", "self._names = list(weights)")]),
    ("one example dropped per batch", 1, [("        return [next(it) for _ in range(count)]\n",
                                           "        return [next(it) for _ in range(count + 1)][:count]\n")]),
    ("weights not validated", 1, [("        if not weights or any(w <= 0 for w in weights.values()):\n            raise ValueError(\"weights must be a non-empty dict of positive integers\")\n", "")]),
    ("batch_size not validated", 1, [("        if batch_size <= 0:\n            raise ValueError(\"batch_size must be positive\")\n", "")]),
    ("state is the live offsets", 2, [('state = {"offsets": dict(self._offsets)}', 'state = {"offsets": self._offsets}')]),
    ("old iterators kept on load", 2, [("        self._iters = {}  # reopened lazily at the restored offsets, never replayed\n", "")]),
    ("reopened at 0 and skipped", 2, [(_OPEN, "            it = self._iters[name] = self._registry.get_iterator(name, 0)\n"
                                               "            for _ in range(self._offsets[name]):\n                next(it)\n")]),
    ("tie goes to the later name", 3, [(_PICK, _PICK.replace(" < ", " <= "))]),
    ("smallest count, weights ignored", 3, [(_PICK, "            if self._offsets[name] < self._offsets[best]:\n")]),
    ("generator state not restored", 3, [("            self._rng.setstate((version, tuple(internal), gauss))\n", "            pass\n")]),
    ("bisect_left", 3, [("bisect_right(self._prefix,", "bisect_left(self._prefix,"), ("from bisect import bisect_right", "from bisect import bisect_left")]),
    ("float draw", 3, [("self._rng.randrange(self._total)", "int(self._rng.random() * self._total)")]),
    ("stochastic skips the exact rule", 3, [("        if self._batch_size % self._total == 0:\n",
                                             "        if self._batch_size % self._total == 0 and self._allocation == \"deterministic\":\n")]),
    ("unknown allocation accepted", 3, [("        if allocation not in (\"deterministic\", \"stochastic\"):\n            raise ValueError(f\"unknown allocation {allocation!r}\")\n", "")]),
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
