"""Mutation gate for the multi-part GPU credit ledger exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "gpu_credits"

# The natural part 1 answer: apply each call the moment it arrives.
_IN_ARRIVAL_ORDER = '''import heapq


class GPUCreditLedger:
    def __init__(self):
        self.grants = []
        self.debt = 0

    def _expire(self, now):
        while self.grants and self.grants[0][0] < now:
            heapq.heappop(self.grants)

    def add_credit(self, credit_id, amount, timestamp, expiration):
        self._expire(timestamp)
        paid = min(self.debt, amount)
        self.debt -= paid
        if amount > paid:
            heapq.heappush(self.grants, [timestamp + expiration, amount - paid])

    def subtract(self, amount, timestamp):
        self._expire(timestamp)
        while amount and self.grants:
            take = min(self.grants[0][1], amount)
            self.grants[0][1] -= take
            amount -= take
            if self.grants[0][1] == 0:
                heapq.heappop(self.grants)
        self.debt += amount

    def get_balance(self, timestamp):
        self._expire(timestamp)
        value = sum(g[1] for g in self.grants) - self.debt
        return None if value < 0 else value
'''

MUTATIONS = [
    ("last usable time excluded", 1, [("while self._grants and self._grants[0][0] < before:", "while self._grants and self._grants[0][0] <= before:")]),
    ("expired grants still drained", 1, [("    def _apply(self, timestamp, amount, end):\n        self._expire(timestamp)\n",
                                          "    def _apply(self, timestamp, amount, end):\n")]),
    ("debt never repaid", 1, [("paid = min(self._debt, amount)", "paid = 0")]),
    ("shortfall forgotten", 1, [("        self._debt += need\n", "        pass\n")]),
    ("negative balance returned", 1, [("return None if value < 0 else value", "return value")]),
    ("late calls applied out of order", 2, [("        if call[0] <= self._clock:\n            self._reset()\n        else:\n            heapq.heappush",
                                             "        if False:\n            self._reset()\n        else:\n            heapq.heappush")]),
    ("earlier queries see later calls", 2, [("        if timestamp < self._clock:\n            self._reset()\n", "")]),
    ("replays on every query", 3, [("        if timestamp < self._clock:\n            self._reset()\n", "        self._reset()\n")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_applying_calls_on_arrival_passes_part_1_only():
    assert first_failing_part(get_task(TASK_ID), _IN_ARRIVAL_ORDER) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
