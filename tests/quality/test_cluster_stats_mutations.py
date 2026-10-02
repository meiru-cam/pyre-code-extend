"""Mutation gate for the multi-part distributed mode, median and sort exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "cluster_stats"

MUTATIONS = [
    ("all counts to rank 0", 1, [("buckets[value % P][value] = count", "buckets[0][value] = count")]),
    ("ties to the larger value", 1, [("return min(candidates, key=lambda kv: (-kv[1], kv[0]))[0]", "return max(candidates, key=lambda kv: (kv[1], kv[0]))[0]")]),
    ("counts not summed", 1, [("            totals.update(recv(src))\n", "            totals |= Counter(recv(src))\n")]),
    ("upper median", 2, [("            k = (n - 1) // 2\n", "            k = n // 2\n")]),
    ("elements shipped", 2, [("send(0, (len(local), local[0] if local else None, local[-1] if local else None))",
                              "send(0, (len(local), local[0] if local else None, local[-1] if local else None, local))")]),
    ("count compared strictly", 2, [("if below >= k + 1:", "if below > k + 1:")]),
    ("buckets not merged in order", 3, [("        return sorted(out)\n", "        return out\n")]),
    ("everything to rank 0", 3, [("            send(dst, local[start:end])\n", "            send(dst, local if dst == 0 else [])\n")]),
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
