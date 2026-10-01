"""Mutation gate for the multi-part infection spread exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "infection_spread"

# Correct, but rewrites the whole grid once per day.
_RESCAN = '''    def days_until_all_infected(self):
        state, days = [row[:] for row in self.grid], 0
        while any(cell == HEALTHY for row in state for cell in row):
            new = [row[:] for row in state]
            for r in range(self.rows):
                for c in range(self.cols):
                    if state[r][c] == HEALTHY and any(state[nr][nc] == INFECTED for nr, nc in self._neighbours(r, c)):
                        new[r][c] = INFECTED
            if new == state:
                return -1
            state, days = new, days + 1
        return days

    def _bfs_unused(self):
'''

_RECOVERY = '''            for cell in [cell for cell, since in infected_on.items() if day - since >= recover_after]:
                del infected_on[cell]
                state[cell[0]][cell[1]] = DEAD if cell in doomed else IMMUNE
                deaths += cell in doomed
'''

_SPREAD_APPLY = '''            for (r, c), count in pressure.items():
                if count >= spread_threshold:
                    state[r][c] = INFECTED
                    infected_on[(r, c)] = day
                    if death_threshold is not None and count >= death_threshold:
                        doomed.add((r, c))
'''

MUTATIONS = [
    ("one day too many", 1, [("return days if healthy == 0 else -1", "return days + 1 if healthy == 0 else -1")]),
    ("diagonal neighbours", 1, [("_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))",
                                 "_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))")]),
    ("marks the input grid", 1, [
        ("self.grid = [list(row) for row in grid]", "self.grid = grid"),
        ("reached = [[cell != HEALTHY for cell in row] for row in self.grid]", "reached = self.grid"),
    ]),
    ("cut-off cells ignored", 1, [("return days if healthy == 0 else -1", "return days")]),
    ("immune cells are entered", 2, [("reached = [[cell != HEALTHY", "reached = [[cell == INFECTED")]),
    ("rescans the grid every day", 2, [("    def days_until_all_infected(self):\n", _RESCAN)]),
    ("spread before recovery", 3, [(_RECOVERY, ""), (_SPREAD_APPLY, _SPREAD_APPLY + _RECOVERY)]),
    ("recovers a day late", 3, [("day - since >= recover_after", "day - since > recover_after")]),
    ("input cells start a day late", 3, [("infected_on = {(r, c): 0 for", "infected_on = {(r, c): -1 for")]),
    ("spread threshold off by one", 4, [("if count >= spread_threshold:", "if count >= spread_threshold + (spread_threshold > 1):")]),
    ("death threshold strict", 4, [("count >= death_threshold:", "count > death_threshold:")]),
    ("deaths on by default", 4, [("def simulate(self, recover_after, spread_threshold=1, death_threshold=None):",
                                   "def simulate(self, recover_after, spread_threshold=1, death_threshold=1):")]),
    ("doing nothing not considered", 5, [("        best = self.simulate(recover_after, spread_threshold, death_threshold)[1]\n",
                                           "        best = float('inf') if lines else 0\n")]),
    ("burnt line is free", 5, [("best = min(best, len(line) + deaths)", "best = min(best, deaths)")]),
    ("rows only", 5, [("        lines += [[(r, c) for r in range(self.rows)] for c in range(self.cols)]\n", "")]),
    ("burns accumulate", 5, [("            state = [row[:] for row in self.grid]\n            for r, c in line:",
                              "            state = self.grid\n            for r, c in line:")]),
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
