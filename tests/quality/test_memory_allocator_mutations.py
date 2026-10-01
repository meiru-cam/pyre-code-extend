"""Mutation gate for the multi-part memory allocator exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "memory_allocator"

# The usual first answer: a sorted list of free blocks, walked on every call.
_LIST = '''class MemoryAllocator:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.gaps = [[0, capacity]]
        self.allocated = {}

    def allocate(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        for gap in self.gaps:
            if gap[1] >= size:
                start = gap[0]
                if gap[1] == size:
                    self.gaps.remove(gap)
                else:
                    gap[0] += size
                    gap[1] -= size
                self.allocated[start] = size
                return start
        raise MemoryError(size)

    def free(self, address, size):
        if size <= 0 or self.allocated.get(address) != size:
            raise ValueError((address, size))
        del self.allocated[address]
        i = 0
        while i < len(self.gaps) and self.gaps[i][0] < address:
            i += 1
        left = i > 0 and self.gaps[i - 1][0] + self.gaps[i - 1][1] == address
        right = i < len(self.gaps) and address + size == self.gaps[i][0]
        if left and right:
            self.gaps[i - 1][1] += size + self.gaps[i][1]
            del self.gaps[i]
        elif left:
            self.gaps[i - 1][1] += size
        elif right:
            self.gaps[i][0] = address
            self.gaps[i][1] += size
        else:
            self.gaps.insert(i, [address, size])

    def free_blocks(self):
        return [tuple(gap) for gap in self.gaps]
'''

_LINEAR_NEIGHBOURS = '''def _neighbours(node, address):
    before = after = None
    stack = []
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        if node.start < address:
            before = node
        elif after is None:
            after = node
        node = node.right
    return before, after


def _unused_neighbours(node, address):
'''

MUTATIONS = [
    ("capacity unchecked", 1, [('        if capacity <= 0:\n            raise ValueError', '        if False:\n            raise ValueError')]),
    ("MemoryError reported as ValueError", 1, [('raise MemoryError(f"no free block', 'raise ValueError(f"no free block')]),
    ("wrong size accepted", 1, [("if size <= 0 or self._allocated.get(address) != size:", "if size <= 0 or address not in self._allocated:")]),
    ("remainder lost", 1, [("        if node.size > size:\n", "        if False:\n")]),
    ("no merge on the left", 1, [("if before is not None and before.start + before.size == address:", "if False:")]),
    ("no merge on the right", 1, [("if after is not None and address + size == after.start:", "if False:")]),
    ("best fit", 1, [("        node = _first_fit(self._free, size)\n",
                      "        fits = [b for b in self.free_blocks() if b[1] >= size]\n"
                      "        node = _Node(*min(fits, key=lambda b: (b[1], b[0]))) if fits else None\n")]),
    ("first fit by walking", 2, [("        node = _first_fit(self._free, size)\n",
                                  "        node = next((_Node(s, z) for s, z in self.free_blocks() if z >= size), None)\n")]),
    ("neighbours by walking", 2, [("def _neighbours(node, address):\n", _LINEAR_NEIGHBOURS)]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_a_sorted_list_passes_part_1_only():
    assert first_failing_part(get_task(TASK_ID), _LIST) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
