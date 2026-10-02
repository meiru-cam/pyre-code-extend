"""Mutation gate for the multi-part fault-tolerant job queue exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "job_queue"

MUTATIONS = [
    ("duplicate submit allowed", 1, [("        if task_id in self._tasks:\n            raise ValueError(f\"task {task_id!r} was already submitted\")\n", "")]),
    ("newest ready first", 1, [("task_id = self._ready.popleft()", "task_id = self._ready.pop()")]),
    ("token is task id", 1, [("task.state, task.token = \"reserved\", next(self._tokens)", "task.state, task.token = \"reserved\", task_id")]),
    ("token not checked", 1, [("if task.state != \"reserved\" or task.token != token:", "if task.state != \"reserved\":")]),
    ("state not checked", 1, [("if task.state != \"reserved\" or task.token != token:", "if task.token is not None and task.token != token:")]),
    ("unknown id is invalid reservation", 1, [("            raise UnknownTaskError(task_id)\n        if task.state != \"reserved\"", "            raise InvalidReservationError(task_id)\n        if task.state != \"reserved\"")]),
    ("failed task to the front", 1, [("            task.state = \"ready\"\n            self._ready.append(task_id)", "            task.state = \"ready\"\n            self._ready.appendleft(task_id)")]),
    ("lease ends before deadline", 2, [("self._leases[0][0] < now:", "self._leases[0][0] <= now:")]),
    ("submit does not reclaim", 2, [("        self._reclaim()  # reclaimed tasks queue ahead of this one\n", "")]),
    ("check before reclaim", 2, [("        self._reclaim()  # before the check: this very lease may have just expired\n        task = self._tasks.get(task_id)\n        if task is None:\n            raise UnknownTaskError(task_id)\n        if task.state != \"reserved\" or task.token != token:\n            raise InvalidReservationError(task_id)\n",
                                  "        task = self._tasks.get(task_id)\n        if task is None:\n            raise UnknownTaskError(task_id)\n        if task.state != \"reserved\" or task.token != token:\n            raise InvalidReservationError(task_id)\n        self._reclaim()\n")]),
    ("stale lease entry released", 2, [("            if task.state == \"reserved\" and task.token == token:\n", "            if task.state == \"reserved\":\n")]),
    ("deadline order ignores id", 2, [("heapq.heappush(self._leases, (self._clock.now() + self._lease, task_id, task.token))", "heapq.heappush(self._leases, (self._clock.now() + self._lease, -next(self._tokens), task_id, task.token))"),
                                      ("            _, task_id, token = heapq.heappop(self._leases)", "            _, _, task_id, token = heapq.heappop(self._leases)")]),
    ("one more attempt", 3, [("task.attempts >= self._max:", "task.attempts > self._max:")]),
    ("dead letters newest first", 3, [("        return list(self._dead)\n", "        return list(self._dead)[::-1]\n")]),
    ("requeue keeps attempts", 3, [("task.state, task.attempts = \"ready\", 0", "task.state = \"ready\"")]),
    ("requeue to the front", 3, [("        task.state, task.attempts = \"ready\", 0\n        self._ready.append(task_id)", "        task.state, task.attempts = \"ready\", 0\n        self._ready.appendleft(task_id)")]),
    ("requeue live task", 3, [("        if task.state != \"dead\":\n            raise ValueError(f\"task {task_id!r} is not dead\")\n        del self._dead[task_id]\n", "        if task.state == \"completed\":\n            raise ValueError(f\"task {task_id!r} is not dead\")\n        self._dead.pop(task_id, None)\n")]),
    ("dead letters skip reclaim", 3, [("        self._reclaim()\n        return list(self._dead)\n", "        return list(self._dead)\n")]),
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
