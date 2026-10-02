"""Mutation gate for the multi-part jump grid exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "jump_grid"

MUTATIONS = [
    ("no diagonal steps", 1, [("            for c in (j - 1, j, j + 1):", "            for c in (j,):")]),
    ("jump needs a budget of two", 1, [("        if jumps and i + 2 < self._n:", "        if jumps > 1 and i + 2 < self._n:")]),
    ("jump lands past the edge", 1, [("        if jumps and i + 2 < self._n:", "        if jumps and i + 1 < self._n:")]),
    ("last row adds nothing", 1, [("                        best[i][j][b] = board[i][j]\n", "                        best[i][j][b] = 0\n")]),
    ("start cell not counted", 1, [("        return self._table(k)[0][p][k]", "        return self._table(k)[0][p][k] - self._board[0][p]")]),
    ("largest path on ties", 2, [("next(move for move in self._moves(i, j, b) if best[move[0]][move[1]][move[2]] == want)", "[move for move in self._moves(i, j, b) if best[move[0]][move[1]][move[2]] == want][-1]")]),
    ("path skips the start", 2, [("        path = [(0, p)]\n", "        path = []\n")]),
    ("no modulus", 3, [("                                         if best[r][c][left] == want) % MOD", "                                         if best[r][c][left] == want)")]),
    ("count any move", 3, [("                    count[i][j][b] = sum(count[r][c][left] for r, c, left in self._moves(i, j, b)\n                                         if best[r][c][left] == want) % MOD",
                            "                    count[i][j][b] = sum(count[r][c][left] for r, c, left in self._moves(i, j, b)) % MOD")]),
    ("equal bonus never paid", 4, [("bonus = (x if here == nxt else 0)", "bonus = (0 if here == nxt else 0)")]),
    ("rise bonus on non-strict rise", 4, [("before < here < nxt", "before <= here < nxt")]),
    ("rise bonus ignores the earlier value", 4, [("before is not None and before < here < nxt", "here < nxt")]),
    ("count ignores the jump budget", 3, [("sum(count[r][c][left] for r, c, left in self._moves(i, j, b)", "sum(count[r][c][left] for r, c, left in self._moves(i, j, k)")]),
    ("jump ends not neighbours", 4, [("bonus = (x if here == nxt else 0)", "bonus = (x if here == nxt and r == i + 1 else 0)")]),
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
