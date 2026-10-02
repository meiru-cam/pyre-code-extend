"""Mutation gate for the multi-part resumable iterator exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "resumable_iterator"

MUTATIONS = [
    ("bool accepted", 1, [("isinstance(i, int) and not isinstance(i, bool) for i in state", "isinstance(i, int) for i in state")]),
    ("only tuples accepted", 1, [("if (not isinstance(state, (tuple, list)) or len(state)", "if (not isinstance(state, tuple) or len(state)")]),
    ("length not checked", 1, [("if (not isinstance(state, (tuple, list)) or len(state) != self._depth\n                or", "if (not isinstance(state, (tuple, list))\n                or")]),
    ("negative index allowed", 1, [("            if not 0 <= i < len(path[k]):", "            if not i < len(path[k]):")]),
    ("state returned as a list", 1, [("        return tuple(self._pos)", "        return list(self._pos)")]),
    ("rejected state half applied", 1, [("        for k, i in enumerate(state):\n            if not 0", "        self._pos = state\n        for k, i in enumerate(state):\n            if not 0")]),
    ("end state rejected", 1, [("        if state[0] == len(self._items) and not any(state[1:]):", "        if False:")]),
    ("state shared with the caller", 1, [("        return tuple(self._pos)", "        return self._pos")]),
    ("range checked before type", 1, [("    def set_state(self, state):\n        if (not isinstance", "    def set_state(self, state):\n        if isinstance(state, (tuple, list)) and state and isinstance(state[0], int) and not 0 <= state[0] <= len(self._items):\n            raise ValueError(state)\n        if (not isinstance")]),
    ("end state accepts a nonzero inner index", 2, [("        if state[0] == len(self._items) and not any(state[1:]):", "        if state[0] == len(self._items):")]),
    ("empty rows not skipped up front", 2, [("        self._path = [items]  # the list at each level along the current position\n        self._settle()", "        self._path = [items]  # the list at each level along the current position\n        self._settle() if depth == 1 else None")]),
    ("end state keeps inner index", 2, [("            pos[k] = 0\n            pos[k - 1] += 1", "            pos[k - 1] += 1")]),
    ("end of row accepted", 2, [("            if not 0 <= i < len(path[k]):", "            if not 0 <= i <= len(path[k]) - (k == 0):")]),
    ("rescan from the first row", 2, [("        self._pos[-1] += 1\n        self._settle()", "        self._pos[-1] += 1\n        self._rescan()"),
                                      ("    def __iter__(self):", "    def _rescan(self):\n        target = tuple(self._pos)\n        self._pos, self._path = [0] * self._depth, [self._items]\n        self._settle()\n        while tuple(self._pos) < target and self._pos[0] < len(self._items):\n            self._pos[-1] += 1\n            self._settle()\n\n    def __iter__(self):")]),
    ("two levels at most", 3, [("                path.append(path[k][pos[k]])  # go one level down\n", "                path.append(path[k][pos[k]])  # go one level down\n                if k >= 2:\n                    return\n")]),
    ("recursive descent", 3, [("    def __iter__(self):", "    def _down(self, k):\n        if k == self._depth:\n            return\n        self._down(k + 1)\n\n    def __iter__(self):"),
                              ("        self._settle()\n\n    def _settle", "        self._settle()\n        self._down(0)\n\n    def _settle")]),
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
