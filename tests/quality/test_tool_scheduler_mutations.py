"""Mutation gate for the multi-part tool-call scheduler exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "tool_scheduler"

MUTATIONS = [
    ("one release per step", 1, [("            while running and running[0][0] == now:", "            if running and running[0][0] == now:")]),
    ("ready order ignored", 1, [("                tool = heapq.heappop(ready[a])\n", "                tool = ready[a].pop()\n")]),
    ("cycle returns a partial plan", 1, [('                raise CycleError("some calls wait on a dependency cycle")\n', "                break\n")]),
    ("agents never rejoin", 1, [("            for a in touched:\n                if a not in queued", "            for a in ():\n                if a not in queued")]),
    ("freed agents wait until idle", 1, [("if a not in queued and ready[a] and holding[a] < caps[a]:", "if a not in queued and ready[a] and holding[a] == 0:")]),
    ("agent caps ignored", 2, [("                if ready[a] and holding[a] < caps[a]:\n                    heapq.heappush(eligible, a)\n", "                if ready[a]:\n                    heapq.heappush(eligible, a)\n")]),
    ("each agent gets the largest cap", 2, [("return self.run_capped(agents, capacities, sum(capacities))", "return self.run_capped(agents, capacities, max(capacities, default=0))")]),
    ("shared cap per agent", 3, [("return self.run_capped(agents, [capacity] * len(agents), capacity)", "return self.run_capped(agents, [capacity] * len(agents), capacity * len(agents))")]),
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
