"""Mutation gate for the multi-part timesheet exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "timesheet"

MUTATIONS = [
    ("re-register overwrites", 1, [("        if worker_id in self._workers:\n            return False\n        self._workers[worker_id] = _Worker", "        self._workers[worker_id] = _Worker")]),
    ("second clock_in allowed", 1, [("        if worker is None or worker.open is not None:\n", "        if worker is None:\n")]),
    ("clock_out without a session", 1, [("        if worker is None or worker.open is None:\n            return False\n        start, rate = worker.open\n",
                                         "        if worker is None:\n            return False\n        start, rate = worker.open or (timestamp, worker.rate)\n")]),
    ("open session counted", 2, [("return None if worker is None else worker.total\n", "return None if worker is None else worker.total + (worker.open is not None)\n")]),
    ("least time first", 2, [("key=lambda wid: (-self._workers[wid].total, wid)", "key=lambda wid: (self._workers[wid].total, wid)")]),
    ("ties ignore case", 2, [("key=lambda wid: (-self._workers[wid].total, wid)", "key=lambda wid: (-self._workers[wid].total, wid.lower())")]),
    ("k ignored", 2, [("        return ranked[:k]\n", "        return ranked\n")]),
    ("promotion at once", 3, [("worker.pending = (new_position, new_hourly_rate)  # replaces any earlier pending promotion",
                               "worker.position, worker.rate = new_position, new_hourly_rate")]),
    ("first promotion kept", 3, [("worker.pending = (new_position, new_hourly_rate)  # replaces any earlier pending promotion",
                                  "worker.pending = worker.pending or (new_position, new_hourly_rate)")]),
    ("current rate for past sessions", 3, [("pay += (hi - lo) * rate + doubled * rate", "pay += (hi - lo) * worker.rate + doubled * worker.rate")]),
    ("open session unpaid", 3, [("            sessions.append((worker.open[0], end, worker.open[1]))  # still running through the window\n", "            pass\n")]),
    ("window end included", 3, [("lo, hi = max(a, start), min(b, end)", "lo, hi = max(a, start), min(b, end + 1)")]),
    ("overlaps doubled twice", 4, [("for p, q in self._union)", "for p, q in self._periods)")]),
    ("double pay tripled", 4, [("pay += (hi - lo) * rate + doubled * rate", "pay += (hi - lo) * rate + doubled * rate * 2")]),
    ("only the last period", 4, [("        self._periods.append((start, end))\n", "        self._periods = [(start, end)]\n")]),
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
