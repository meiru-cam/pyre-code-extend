"""Mutation gate for the multi-part vector machine kernel scheduling exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "vector_kernel"

MUTATIONS = [
    ("tail drops an element", 1, [("    for _ in range(tail):", "    for _ in range(max(0, tail - 1)):")]),
    ("x scaled by b", 1, [("(\"vmadd\", vx, 0, vx, vy)", "(\"vmadd\", vx, 1, vx, vy)")]),
    ("tail swaps a and b", 1, [("(\"mul\", T, A, T), (\"mul\", U, B, U)", "(\"mul\", T, B, T), (\"mul\", U, A, U)")]),
    ("write before an earlier read", 1, [(" + [last_read[w] for w in writes]", "")]),
    ("unit limits ignored", 1, [("        while used[t, unit] == UNIT_WIDTH[unit]:\n            t += 1\n", "")]),
    ("z written over y", 1, [("(\"li\", PZ, 2 * n + 2)", "(\"li\", PZ, n)")]),
    ("scalar only", 2, [("chunks, tail = divmod(n, VLEN)", "chunks, tail = 0, n")]),
    ("one chunk in flight", 3, [("vx, vy = 2 + 2 * (k % 3), 3 + 2 * (k % 3)", "vx, vy = 2, 3")]),
    ("two chunks in flight", 3, [("vx, vy = 2 + 2 * (k % 3), 3 + 2 * (k % 3)", "vx, vy = 2 + 2 * (k % 2), 3 + 2 * (k % 2)")]),
    ("one instruction per bundle", 3, [("    return [bundles[t] for t in sorted(bundles)]", "    return [[op] for op in ops]")]),
    ("load latency ignored", 3, [("ready[w] = t + (LOAD_LATENCY if unit == \"load\" else 1)", "ready[w] = t + 1")]),
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
