"""Mutation gate for the multi-part TimeMap exercise.

Beyond rejecting every mutation, each one must first fail in the part it targets and pass
every earlier part. That is what lets the workspace grade parts 1..k together: a learner
who has only reached part k is never failed by a later part's requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "time_map"

_NO_LOCK = '''

class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
'''

# (name, first part that must fail, [(text in the reference, replacement), ...]).
MUTATIONS = [
    ("linear scan", 1, [(
        "            index = bisect.bisect_right(stamps, timestamp)\n",
        "            index = 0\n            while index < len(stamps) and stamps[index] <= timestamp:\n                index += 1\n",
    )]),
    ("excludes an exact match", 1, [("bisect.bisect_right(", "bisect.bisect_left(")]),
    ("returns the first value", 1, [("return self._values[key][index - 1] if index else None",
                                     "return self._values[key][0] if index else None")]),
    ("one history for all keys", 1, [
        ("stamps = self._stamps.setdefault(key, [])", "stamps = self._stamps.setdefault('*', [])"),
        ("self._values.setdefault(key, []).append(value)", "self._values.setdefault('*', []).append(value)"),
        ("stamps = self._stamps.get(key)", "stamps = self._stamps.get('*')"),
        ("return self._values[key][index - 1]", "return self._values['*'][index - 1]"),
    ]),
    ("explicit timestamp loses to an injected clock", 2, [(
        "        with self._lock:\n            if timestamp is None:\n                timestamp = self._clock.now()\n            stamps = self._stamps.setdefault",
        "        with self._lock:\n            if timestamp is None or not isinstance(self._clock, _RealClock):\n                timestamp = self._clock.now()\n            stamps = self._stamps.setdefault",
    )]),
    ("clock read once at construction", 2, [
        ("        self._lock = threading.Lock()\n", "        self._lock = threading.Lock()\n        self._start = self._clock.now()\n"),
        ("                timestamp = self._clock.now()\n            stamps = self._stamps.setdefault",
         "                timestamp = self._start\n            stamps = self._stamps.setdefault"),
    ]),
    ("no default clock", 2, [("self._clock = clock if clock is not None else _RealClock()", "self._clock = clock")]),
    ("one last timestamp for all keys", 2, [
        ("        self._lock = threading.Lock()\n", "        self._lock = threading.Lock()\n        self._last = None\n"),
        ("            if stamps and timestamp <= stamps[-1]:\n                timestamp = stamps[-1] + 1\n",
         "            if self._last is not None and timestamp <= self._last:\n                timestamp = self._last + 1\n            self._last = timestamp\n"),
    ]),
    ("overwrites instead of bumping", 3, [(
        "            if stamps and timestamp <= stamps[-1]:\n                timestamp = stamps[-1] + 1\n            stamps.append(timestamp)\n            self._values.setdefault(key, []).append(value)\n",
        "            values = self._values.setdefault(key, [])\n            if stamps and timestamp <= stamps[-1]:\n                index = bisect.bisect_left(stamps, timestamp)\n                if stamps[index] == timestamp:\n                    values[index] = value\n                    return timestamp\n                stamps.insert(index, timestamp)\n                values.insert(index, value)\n                return timestamp\n            stamps.append(timestamp)\n            values.append(value)\n",
    )]),
    ("returns the requested timestamp", 3, [
        ("            stamps = self._stamps.setdefault(key, [])\n",
         "            requested = timestamp\n            stamps = self._stamps.setdefault(key, [])\n"),
        ("            return timestamp\n", "            return requested\n"),
    ]),
    ("bumps from the clock", 3, [("timestamp = stamps[-1] + 1", "timestamp = self._clock.now() + 1")]),
    ("no lock", 4, [("self._lock = threading.Lock()", "self._lock = _NoLock()")]),
]


def _mutant(original: str, edits: list[tuple[str, str]]) -> str:
    code = original
    for old, new in edits:
        assert old in code, f"mutation target drifted: {old[:60]!r}"
        code = code.replace(old, new, 1)
    return code + _NO_LOCK


def _first_failing_part(task: dict, code: str) -> int | None:
    response = _execute_tests(code, task, capture_output=False)
    assert response.error is None, response.error
    parts = [task["tests"][r.testIndex]["part"] for r in response.results if not r.passed]
    return min(parts) if parts else None


# Concurrency tests race real threads, so repeat to show the gate does not pass by luck.
@pytest.mark.parametrize("repeat", range(3))
def test_mutations_rejected(repeat):
    original = get_task(TASK_ID)["solution"]
    mutations = [Mutation(name, _mutant(original, edits)) for name, _, edits in MUTATIONS]
    rejected = assert_mutations_rejected(TASK_ID, mutations, require_unshown=False)
    assert set(rejected) == {name for name, _, _ in MUTATIONS}


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    task = get_task(TASK_ID)
    assert _first_failing_part(task, _mutant(task["solution"], edits)) == part


def test_parts_are_cumulative_for_the_reference():
    task = get_task(TASK_ID)
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed


def test_task_metadata_is_valid_and_listed():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert [t["part"] for t in task["tests"]] == sorted(t["part"] for t in task["tests"])
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert TASK_ID in starters
