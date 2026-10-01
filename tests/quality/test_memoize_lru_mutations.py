"""Mutation gate for the multi-part LRU memoizer exercise.

Each mutation must be rejected, and must first fail in the part it targets while passing
every earlier part, so that grading parts 1..k together never fails a learner on part k+1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "memoize_lru"

_CALL_BODY = "        key = self.generate_key(*args, **kwargs)\n        with self._lock:\n            if key in self.cache:"

MUTATIONS = [
    ("keyword order matters", 1, [("tuple(sorted(kwargs.items()))", "tuple(kwargs.items())")]),
    ("positional and keyword flattened", 1, [(
        "        return (args, tuple(sorted(kwargs.items())))",
        "        return args + tuple(sorted(kwargs.items()))",
    )]),
    ("the starter's key", 1, [(
        "        return (args, tuple(sorted(kwargs.items())))",
        "        key = args + tuple(sorted(kwargs.items()))\n        if len(key) == 1:\n            return key[0]\n        return key",
    )]),
    ("a hit does not refresh recency", 1, [(
        "                self.hits += 1\n                self.cache.move_to_end(key)\n",
        "                self.hits += 1\n",
    )]),
    ("no recovery", 2, [("            self._recover()\n", "")]),
    ("no checksum", 2, [("if len(payload) < length or zlib.crc32(payload) != crc:", "if len(payload) < length:")]),
    ("recovery ignores uses", 2, [(
        "            elif key in self.cache:\n                self.cache.move_to_end(key)\n",
        "",
    )]),
    ("appends after a torn record", 2, [(
        "        if offset < len(data):\n            with open(self.path, \"r+b\") as fh:\n                fh.truncate(offset)\n",
        "",
    )]),
    ("tuples stored as lists", 2, [('return {"t": [_encode(v) for v in value]}', 'return {"l": [_encode(v) for v in value]}')]),
    ("rewrites the whole cache per call", 2, [(
        "        self._log.write(self._frame(record))\n",
        "        self._log.close()\n        with open(self.path, \"wb\") as out:\n            for k, v in self.cache.items():\n                out.write(self._frame({\"op\": \"put\", \"key\": _encode(k), \"value\": _encode(v)}))\n        self._log = open(self.path, \"ab\")\n",
    )]),
    ("compact does nothing", 2, [("        if self.path is None:\n            return\n        with self._lock:\n            tmp", "        return\n        with self._lock:\n            tmp")]),
    ("func runs under the lock", 3, [(
        "    def __call__(self, *args, **kwargs):\n" + _CALL_BODY,
        "    def __call__(self, *args, **kwargs):\n        with self._outer:\n            return self._call(*args, **kwargs)\n\n    def _call(self, *args, **kwargs):\n" + _CALL_BODY,
    ), ("        self._flights = {}\n", "        self._flights = {}\n        self._outer = threading.Lock()\n")]),
    ("every caller computes", 3, [
        ("            leader = flight is None\n", "            leader = True\n"),
        ("                del self._flights[key]\n            flight.error", "                self._flights.pop(key, None)\n            flight.error"),
        ("            del self._flights[key]\n        flight.result", "            self._flights.pop(key, None)\n        flight.result"),
    ]),
    ("waiters miss the exception", 3, [("            flight.error = error\n", "")]),
    ("no lock", 3, [("        self._lock = threading.Lock()\n", "        self._lock = _NoLock()\n")]),
]

_NO_LOCK = """


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
"""


@pytest.mark.parametrize("repeat", range(3))
def test_mutations_rejected(repeat):
    assert_part_mutations_rejected(TASK_ID, MUTATIONS, _NO_LOCK)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits, _NO_LOCK)


def test_starter_is_the_buggy_skeleton_and_fails_part_1():
    task = get_task(TASK_ID)
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    starter = starters[TASK_ID]
    assert "popitem(last=False)" in starter and "len(key) == 1" in starter
    assert first_failing_part(task, starter) == 1


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
