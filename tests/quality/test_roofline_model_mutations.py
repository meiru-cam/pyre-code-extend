"""Mutation gate for the multi-part roofline performance model exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "roofline_model"

MUTATIONS = [
    ("one FLOP per multiply-add", 1, [("flops = 2 * m * k * n", "flops = m * k * n")]),
    ("output not written", 1, [("moved = self.s * (m * k + k * n + m * n)", "moved = self.s * (m * k + k * n)")]),
    ("element size ignored", 1, [("moved = self.s * (m * k + k * n + m * n)", "moved = 2 * (m * k + k * n + m * n)")]),
    ("times added", 1, [("\"time\": max(compute, memory),", "\"time\": compute + memory,")]),
    ("tie goes to memory", 1, [("\"bound\": \"compute\" if compute >= memory else \"memory\"", "\"bound\": \"compute\" if compute > memory else \"memory\"")]),
    ("ridge inverted", 1, [("return self.peak / self.bw", "return self.bw / self.peak")]),
    ("layers never alternate", 3, [("return (T, d, f) if i % 2 == 1 else (T, f, d)", "return (T, d, f)")]),
    ("activations not resident", 2, [("resident = weights + self.s * T * (d + f)  # every weight", "resident = weights  # every weight")]),
    ("all activations resident", 2, [("resident = weights + self.s * T * (d + f)  # every weight", "resident = weights + self.s * T * (d + f) * layers  # every weight")]),
    ("headroom sign", 2, [("\"headroom\": self.capacity - resident}", "\"headroom\": resident - self.capacity}")]),
    ("handoff width fixed", 3, [("width = f if half % 2 == 1 else d", "width = d")]),
    ("handoff free", 3, [("\"latency\": first + handoff / self.link + second,", "\"latency\": first + second,")]),
    ("pipeline peak from the first device", 3, [("self.s * d * f * (layers - half) + self.s * T * (d + f)", "self.s * d * f * half + self.s * T * (d + f)")]),
    ("tensor shard not halved", 3, [("shard = self.matmul(T, k, n // 2)", "shard = self.matmul(T, k, n)")]),
    ("tensor sends the full output", 3, [("out = self.s * T * (n // 2)  # sent", "out = self.s * T * n  # sent")]),
    ("tensor exchange overlapped", 3, [("latency += shard[\"time\"] + out / self.link", "latency += max(shard[\"time\"], out / self.link)")]),
    ("all-reduce sends one copy", 4, [("comm_bytes = 2 * (p - 1) * T * d * self.s / p", "comm_bytes = (p - 1) * T * d * self.s / p")]),
    ("token weights sharded", 4, [("            weight_bytes = 2 * self.s * d * f\n", "            weight_bytes = 2 * self.s * d * f // p\n")]),
    ("token rows not split", 4, [("mms = [self.matmul(T // p, d, f), self.matmul(T // p, f, d)]", "mms = [self.matmul(T, d, f), self.matmul(T, f, d)]")]),
    ("link override ignored", 4, [("link = self.link if link_bandwidth is None else link_bandwidth", "link = self.link")]),
    ("all-reduce overlapped", 4, [("\"latency\": sum(x[\"time\"] for x in mms) + comm,", "\"latency\": max(sum(x[\"time\"] for x in mms), comm),")]),
    ("bound ignores communication", 4, [("times = {\"compute\": compute, \"memory\": memory, \"communication\": comm}", "times = {\"compute\": compute, \"memory\": memory}")]),
    ("unknown scheme treated as token", 4, [("        elif scheme == \"token\":", "        else:"), ("        else:\n            raise ValueError(f\"unknown scheme {scheme!r}\")\n", "")]),
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
