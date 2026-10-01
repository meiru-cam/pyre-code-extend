"""Mutation gate for the multi-part shard rebalancing exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "shard_rebalance"

_HEAP = '''        free, kept = [], []
        for sid, start, end in sorted(shards, key=lambda s: (s[1], s[2], s[0])):
            lane = heapq.heappop(free) if len(free) == limit else start  # an unused lane is free now
            new_start = max(start, lane)
            if new_start > end:
                heapq.heappush(free, lane)  # hand the lane back unused
                continue
            kept.append([sid, new_start, end])
            heapq.heappush(free, end + 1)
'''
# Correct, but compares every shard against every kept shard.
_SCAN = '''        kept = []
        for sid, start, end in sorted(shards, key=lambda s: (s[1], s[2], s[0])):
            candidates = sorted({start} | {k[2] + 1 for k in kept if start < k[2] + 1 <= end})
            new_start = next((c for c in candidates if sum(1 for k in kept if k[1] <= c <= k[2]) < limit), None)
            if new_start is not None:
                kept.append([sid, new_start, end])
'''
_FINAL_SORT = '''        kept.sort(key=lambda s: (s[1], s[2], s[0]))
        return [tuple(s) for s in kept]
'''

MUTATIONS = [
    ("lane not handed back", 1, [("                heapq.heappush(free, lane)  # hand the lane back unused\n", "")]),
    ("ties in input order", 1, [("sorted(shards, key=lambda s: (s[1], s[2], s[0]))", "sorted(shards, key=lambda s: (s[1], s[2]))")]),
    ("longer shards first", 1, [("sorted(shards, key=lambda s: (s[1], s[2], s[0]))", "sorted(shards, key=lambda s: (s[1], -s[2], s[0]))")]),
    ("start can move back", 1, [("new_start = max(start, lane)", "new_start = lane")]),
    ("one-key remainder dropped", 1, [("if new_start > end:", "if new_start >= end:")]),
    ("holes left open", 2, [("                owner[2] = shard[1] - 1\n", "                pass\n")]),
    ("hole goes to the later tie", 2, [("if owner is None or shard[2] > reach:", "if owner is None or shard[2] >= reach:")]),
    ("hole goes to the next shard", 2, [("                owner[2] = shard[1] - 1\n", "                shard[1] = reach + 1\n")]),
    ("not re-sorted after extending", 2, [(_FINAL_SORT, "        return [tuple(s) for s in kept]\n")]),
    ("scans every kept shard", 3, [(_HEAP, _SCAN)]),
    ("one point per shard", 4, [("VIRTUAL_NODES = 160", "VIRTUAL_NODES = 1")]),
    ("built-in hash", 4, [('    return int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "big")', "    return hash(text) % (1 << 64)")]),
    ("key modulo shard count", 4, [("        i = bisect.bisect_left(self._ring, (_point(str(key)),))  # the first point clockwise\n        return self._ring[i % len(self._ring)][1]",
                                    "        ids = sorted(self._shards)\n        return ids[_point(str(key)) % len(ids)]")]),
    ("duplicate add accepted", 4, [("        if shard_id in self._shards:\n            raise ValueError", "        if False:\n            raise ValueError")]),
    ("missing remove ignored", 4, [("        if shard_id not in self._shards:\n            raise ValueError(f\"no such shard: {shard_id!r}\")\n        self._shards.remove(shard_id)",
                                    "        self._shards.discard(shard_id)")]),
    ("no shards not a LookupError", 4, [('        if not self._ring:\n            raise LookupError("no shards")\n', "")]),
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
