"""Mutation gate for the multi-part mode lock exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "mode_lock"

MUTATIONS = [
    ("last release wakes one", 1, [("                self._mode = None\n                self._cond.notify_all()", "                self._mode = None\n                self._cond.notify()")]),
    ("no recheck after waking", 1, [("granted = self._cond.wait_for(lambda: self._mode in (None, mode), timeout)", "granted = self._mode in (None, mode) or self._cond.wait(timeout)")]),
    ("release ignores mode", 1, [("            if self._holders == 0 or self._mode != mode:", "            if self._holders == 0:")]),
    ("waiting never decremented", 1, [("            finally:\n                self._waiting -= 1\n", "            finally:\n                pass\n")]),
    ("fair flag ignored", 2, [("            if self._fair:\n                return self._acquire_fair(mode, timeout)\n", "")]),
    ("join any same-mode batch", 2, [("        if self._queue and self._queue[-1].mode == mode:\n            batch = self._queue[-1]",
                                       "        same = [b for b in self._queue if b.mode == mode]\n        if same:\n            batch = same[0]")]),
    ("next batch wakes one", 2, [("                    self._queue.popleft()\n                    self._cond.notify_all()", "                    self._queue.popleft()\n                    self._cond.notify()")]),
    ("fair release ignores mode", 2, [("if batch is None or batch.mode != mode or batch.active == 0:", "if batch is None or batch.active == 0:")]),
    ("ghost batch stays", 3, [("        if batch.waiting == 0 and batch.active == 0:\n            self._drop(batch)\n", "")]),
    ("neighbours never merge", 3, [("        if 0 < i < len(self._queue) and self._queue[i - 1].mode == self._queue[i].mode:", "        if False:")]),
    ("merge wakes nobody", 3, [("            del self._queue[i]\n        self._cond.notify_all()", "            del self._queue[i]")]),
    ("granted returns None", 3, [("            self._holders += 1\n            return True", "            self._holders += 1\n            return None")]),
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
