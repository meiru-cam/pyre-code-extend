"""Mutation gate for the multi-part profiler stacks exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "profiler_stacks"

_ENDS = '''            for depth in range(len(prev) - 1, p - 1, -1):
                out.append(("end", ts, prev[depth], depth))
'''
_STARTS = '''            for depth in range(p, len(stack)):
                out.append(("start", ts, stack[depth], depth))
'''
_FINAL = '''        for depth in range(len(prev) - 1, -1, -1):
            out.append(("end", end_time, prev[depth], depth))
'''
_STREAM_LOOP = '''        closing = {}           # depth -> [name, samples since a confirmed frame there left]
        for ts, stack in samples:
'''
_SCAN = '''    def match_suffix(self, frames):
        n = len(frames)
        found = sorted(tid for tid, fs in self._traces if len(fs) >= n and list(fs[-n:]) == list(frames))
        return found or "UNKNOWN"

    def _trie_match_unused(self, frames):
'''

MUTATIONS = [
    ("ends shallowest first", 1, [(_ENDS, _ENDS.replace("range(len(prev) - 1, p - 1, -1)", "range(p, len(prev))"))]),
    ("starts before ends", 1, [(_ENDS + _STARTS, _STARTS + _ENDS)]),
    ("whole stacks compared", 1, [("            p = _common_prefix(prev, stack)\n            for depth in range(len(prev) - 1, p - 1, -1):\n                out",
                                   "            p = len(stack) if prev == stack else 0\n            for depth in range(len(prev) - 1, p - 1, -1):\n                out")]),
    ("no closing at end_time", 1, [(_FINAL, "")]),
    ("closing shallowest first", 1, [(_FINAL, _FINAL.replace("range(len(prev) - 1, -1, -1)", "range(len(prev))"))]),
    ("k ignored", 2, [("streak[confirmed] >= k", "streak[confirmed] >= 1")]),
    ("end one sample late", 2, [("if gone >= k), reverse=True)", "if gone > k), reverse=True)")]),
    ("unconfirmed frames end", 2, [("                if depth < confirmed:\n", "                if True:\n")]),
    ("ends shallowest first when streaming", 2, [("if gone >= k), reverse=True)", "if gone >= k))")]),
    ("keeps every sample", 2, [(_STREAM_LOOP, _STREAM_LOOP.replace("        for ts, stack", "        history = []\n        for ts, stack") + "            history.append(stack)\n")]),
    ("reads all input first", 2, [(_STREAM_LOOP, _STREAM_LOOP.replace("in samples:", "in list(samples):"))]),
    ("empty list for no match", 3, [('                return "UNKNOWN"\n', "                return []\n")]),
    ("ids unsorted", 3, [("                    child.sort()\n", "                    pass\n")]),
    ("matches prefixes", 3, [("for frame in reversed(frames):", "for frame in frames:"),
                             ("for frame in reversed(frames):", "for frame in frames:")]),
    ("scans every trace", 3, [("        self._trie = {}\n", "        self._trie = {}\n        self._traces = list(known_traces)\n"),
                              ("    def match_suffix(self, frames):\n", _SCAN)]),
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
