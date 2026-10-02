"""Mutation gate for the multi-part IPv4 address iterator exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "ipv4_iterator"

_BOUND = "if not self._lo <= self._cursor <= self._hi:"
_DELTA = "self._delta = -step if reverse else step"
_BATCH_TRY = '''            try:
                out.append(next(self))
            except StopIteration:
                break
'''

MUTATIONS = [
    ("segments in the wrong order", 1, [("for shift in (24, 16, 8, 0)", "for shift in (0, 8, 16, 24)")]),
    ("last address excluded", 1, [(_BOUND, "if not self._lo <= self._cursor < self._hi:")]),
    ("leading zeros accepted", 1, [('_SEGMENT = re.compile(r"\\A(0|[1-9][0-9]{0,2})\\Z")', '_SEGMENT = re.compile(r"\\A([0-9]{1,3})\\Z")')]),
    ("segments above 255 accepted", 1, [(" or int(segment) > 255", "")]),
    ("int() decides", 1, [("if not _SEGMENT.match(segment) or int(segment) > 255:", "if int(segment) > 255 or int(segment) < 0:")]),
    ("dollar anchor", 1, [('[0-9]{0,2})\\Z")', '[0-9]{0,2})$")')]),
    ("reverse ignored", 2, [(_DELTA, "self._delta = step")]),
    ("block start not masked", 3, [("first = start >> host_bits << host_bits", "first = start")]),
    ("block one address short", 3, [("first + (1 << host_bits) - 1", "first + (1 << host_bits) - 2")]),
    ("/0 shifts like a 32-bit int", 3, [("host_bits = 32 - int(prefix_part)", "host_bits = (32 - int(prefix_part)) % 32")]),
    ("prefix leading zeros accepted", 3, [('_PREFIX = re.compile(r"\\A(0|[1-9][0-9]?)\\Z")', '_PREFIX = re.compile(r"\\A([0-9]{1,2})\\Z")')]),
    ("prefix above 32 clamped", 3, [(" or int(prefix_part) > 32", ""),
                                    ("host_bits = 32 - int(prefix_part)", "host_bits = max(32 - int(prefix_part), 0)")]),
    ("step ignored", 4, [(_DELTA, "self._delta = -1 if reverse else 1")]),
    ("step 0 accepted", 4, [("if step <= 0:", "if step < 0:")]),
    ("stops only on landing next to the end", 4, [(_BOUND, "if self._cursor in (self._lo - 1, self._hi + 1) or not 0 <= self._cursor <= MAX_IP:")]),
    ("negative size accepted", 5, [("        if size < 0:\n", "        if False:\n")]),
    ("batch raises at the end", 5, [(_BATCH_TRY, "            out.append(next(self))\n")]),
    ("size 0 reads one", 5, [("for _ in range(size):", "for _ in range(max(size, 1)):")]),
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
