"""Mutation gate for the multi-part in-memory database exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "in_memory_database"

MUTATIONS = [
    ("duplicate table allowed", 1, [("        if table in self._tables:\n            raise KeyError(table)\n", "")]),
    ("missing column checked first", 1, [(
        "        for key in row:\n            t.check(key)\n        missing = [c for c in t.columns if c not in row]\n        if missing:\n            raise ValueError(f\"missing columns: {missing}\")\n",
        "        missing = [c for c in t.columns if c not in row]\n        if missing:\n            raise ValueError(f\"missing columns: {missing}\")\n        for key in row:\n            t.check(key)\n",
    )]),
    ("insert keeps the caller's dict", 1, [("        t.rows.append({c: row[c] for c in t.columns})", "        t.rows.append(row)")]),
    ("select hands out stored rows", 1, [("        return [{c: r[c] for c in columns} for r in kept]", "        return kept if columns == t.columns else [{c: r[c] for c in columns} for r in kept]")]),
    ("columns checked only when rows exist", 1, [("        for column in columns:\n            t.check(column)\n", "        for column in columns if t.rows else []:\n            t.check(column)\n")]),
    ("None matches !=", 2, [("    return a is not None and b is not None and _OPS[op](_rank(a), _rank(b))", "    return (op == \"!=\" and (a is None) != (b is None)) or (a is not None and b is not None and _OPS[op](_rank(a), _rank(b)))")]),
    ("strs sort below ints", 2, [("    return (0, value) if isinstance(value, int) else (1, value)", "    return (1, value) if isinstance(value, int) else (0, value)")]),
    ("descending reverses ties", 3, [(
        "            kept.sort(key=lambda r: (0,) if r[column] is None else (1,) + _rank(r[column]), reverse=not ascending)",
        "            kept = sorted(kept, key=lambda r: (0,) if r[column] is None else (1,) + _rank(r[column]))[:: 1 if ascending else -1]",
    )]),
    ("None sorts last", 3, [("(0,) if r[column] is None else (1,) + _rank(r[column])", "(3,) if r[column] is None else (1,) + _rank(r[column])")]),
    ("least significant column first", 3, [("        for column, ascending in reversed(order_by):", "        for column, ascending in order_by:")]),
    ("index never used", 4, [("        ids = t.candidates(where)\n", "        ids = None\n")]),
    ("insert skips the indexes", 4, [("        t.index_row(len(t.rows) - 1)\n", "")]),
    ("range hits in value order", 4, [("                return sorted(row_id for _, row_id in chosen)", "                return [row_id for _, row_id in chosen]")]),
    ("<= misses equal values", 4, [('"<=": entries[:high]', '"<=": entries[:low]')]),
    ("rebuilding duplicates entries", 4, [
        ("            self.hash_indexes[column] = {}\n", "            self.hash_indexes.setdefault(column, {})\n"),
        ("            self.sorted_indexes[column] = []\n", "            self.sorted_indexes.setdefault(column, [])\n"),
    ]),
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
