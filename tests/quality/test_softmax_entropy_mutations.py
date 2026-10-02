"""Mutation gate for the multi-part streaming softmax entropy exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "softmax_entropy"

MUTATIONS = [
    ("bits instead of nats", 1, [("return float(max(0.0, -(np.exp(log_p) * log_p).sum()))", "return float(max(0.0, -(np.exp(log_p) * log_p).sum() / math.log(2)))")]),
    ("not normalised", 1, [("log_p = shifted - log_z  #", "log_p = shifted  #")]),
    ("sign dropped", 1, [("return float(max(0.0, -(np.exp(log_p) * log_p).sum()))", "return float(max(0.0, (np.exp(log_p) * log_p).sum()))")]),
    ("no max shift", 2, [("shifted = x - x.max()  #", "shifted = x  #")]),
    ("blocks are the whole array", 3, [("        x = np.asarray(logits, dtype=float)\n        state = (-math.inf, 0.0, 0.0)\n",
                                        "        return self.entropy(logits)\n        x = np.asarray(logits, dtype=float)\n        state = (-math.inf, 0.0, 0.0)\n")]),
    ("last partial block dropped", 3, [("for start in range(0, len(x), block_size):", "for start in range(0, len(x) - block_size + 1, block_size):")]),
    ("old sums not rescaled", 3, [("            s, v = s * scale, (v + delta * s) * scale\n", "            pass\n")]),
    ("shift missing from the weighted sum", 3, [("s, v = s * scale, (v + delta * s) * scale", "s, v = s * scale, v * scale")]),
    ("chunks stored", 4, [("        state = (-math.inf, 0.0, 0.0)\n        for block in blocks:\n",
                           "        return self.entropy(np.concatenate([np.asarray(b, dtype=float) for b in blocks]))\n        state = (-math.inf, 0.0, 0.0)\n        for block in blocks:\n")]),
    ("empty chunk breaks", 4, [("        if block.size == 0:\n            return state\n", "")]),
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
