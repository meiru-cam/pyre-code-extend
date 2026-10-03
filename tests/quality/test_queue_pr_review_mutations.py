"""Mutation gate for the multi-part job queue pull request review exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "queue_pr_review"

MUTATIONS = [
    ("batch added item by item", 1, [("            return [self._add(job_id, payload, priority) for job_id, payload, priority in items]\n", "            pass\n        return [self.submit(job_id, payload, priority) for job_id, payload, priority in items]\n")]),
    ("duplicates inside a batch allowed", 1, [("if job_id in self._jobs or job_id in seen:", "if job_id in self._jobs:")]),
    ("batch checks only the queue", 1, [("if job_id in self._jobs or job_id in seen:", "if job_id in seen:")]),
    ("generator consumed twice", 1, [("        items = list(items)\n", "")]),
    ("pop order reversed for ties", 1, [("heapq.heappush(self._heap, (-priority, next(self._counter), job_id))", "heapq.heappush(self._heap, (-priority, -next(self._counter), job_id))")]),
    ("check outside the lock", 2, [("        with self._lock:  # check and insert together, or two callers can both see the id as free\n            if job_id in self._jobs:\n                raise ValueError(f\"duplicate job id {job_id!r}\")\n            return self._add(",
                                    "        if job_id in self._jobs:\n            raise ValueError(f\"duplicate job id {job_id!r}\")\n        with self._lock:\n            return self._add(")]),
    ("batch checked before taking the lock", 1, [("        items = list(items)\n        with self._lock:  # once per batch\n            seen = set()\n            for job_id, _, _ in items:\n                if job_id in self._jobs or job_id in seen:\n                    raise ValueError(f\"duplicate job id {job_id!r}\")\n                seen.add(job_id)\n            return",
                                          "        items = list(items)\n        seen = set()\n        for job_id, _, _ in items:\n            if job_id in self._jobs or job_id in seen:\n                raise ValueError(f\"duplicate job id {job_id!r}\")\n            seen.add(job_id)\n        with self._lock:\n            return")]),
    ("load trusts the file", 3, [("            rows = json.loads(f.read().decode(\"utf-8\"))  # data only: nothing in the file can run code", "            rows = __import__('pickle').loads(f.read()) if f.peek(1)[:1] == b'\\x80' else json.loads(f.read().decode(\"utf-8\"))")]),
    ("failed save truncates the file", 3, [("        text = json.dumps(rows)  # a payload JSON cannot hold raises here, before the file is touched\n        tmp = f\"{path}.tmp\"\n        with open(tmp, \"w\") as f:\n            f.write(text)\n        os.replace(tmp, path)\n",
                                            "        with open(path, \"w\") as f:\n            json.dump(rows, f)\n")]),
    ("unknown fields accepted", 3, [("not _FIELDS <= set(row) <= _FIELDS | {\"metadata\"}", "not _FIELDS <= set(row)")]),
    ("unknown status accepted", 3, [("            if row[\"status\"] not in _STATUSES:\n                raise ValueError(f\"bad status {row['status']!r}\")\n", "")]),
    ("non-list file accepted", 3, [("        if not isinstance(rows, list):\n            raise ValueError(\"a saved queue is a list of jobs\")\n", "")]),
    ("shared metadata default", 4, [("        self.metadata = {} if metadata is None else metadata  # never one dict shared by every job\n", "        self.metadata = metadata if metadata is not None else _SHARED\n"), ("_FIELDS = {", "_SHARED = {}\n_FIELDS = {")]),
    ("metadata not copied", 4, [("dict(metadata or {})", "metadata")]),
    ("metadata not saved", 4, [(",\n                    metadata=self.metadata)", ")")]),
    ("unknown policy is KeyError", 4, [("    if name not in _POLICIES:\n        raise ValueError(f\"unknown policy {name!r}\")\n", "")]),
    ("policy outside the lock", 4, [("        with self._lock:\n            job = self._policy.select(self._heap, self._jobs)\n", "        job = self._policy.select(self._heap, self._jobs)\n        with self._lock:\n")]),
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
