"""Mutation gate for the recurrent-depth unroll exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "recurrent_depth_unroll"

NO_GRAD_LOOP = (
    "    with torch.no_grad():\n"
    "        for _ in range(n_no_grad):\n"
    "            s = core(adapter(torch.cat([s, e], dim=-1)))\n"
)
GRAD_LOOP = (
    "    for _ in range(k_grad):\n"
    "        s = core(adapter(torch.cat([s, e], dim=-1)))\n"
)


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    _mutation(
        "adds the state and e, then projects",
        ("torch.cat([s, e], dim=-1)", "torch.cat([s + e, torch.zeros_like(e)], dim=-1)"),
        ("torch.cat([s, e], dim=-1)", "torch.cat([s + e, torch.zeros_like(e)], dim=-1)"),
    ),
    _mutation(
        "concatenates e before the state",
        ("torch.cat([s, e], dim=-1)", "torch.cat([e, s], dim=-1)"),
        ("torch.cat([s, e], dim=-1)", "torch.cat([e, s], dim=-1)"),
    ),
    _mutation(
        "no_grad on the last k iterations instead of the first n",
        (NO_GRAD_LOOP + GRAD_LOOP,
         "    for _ in range(n_no_grad):\n"
         "        s = core(adapter(torch.cat([s, e], dim=-1)))\n"
         "    with torch.no_grad():\n"
         "        for _ in range(k_grad):\n"
         "            s = core(adapter(torch.cat([s, e], dim=-1)))\n"),
    ),
    _mutation(
        "injects e only at the first iteration",
        (NO_GRAD_LOOP + GRAD_LOOP,
         "    inject = e\n"
         "    with torch.no_grad():\n"
         "        for _ in range(n_no_grad):\n"
         "            s = core(adapter(torch.cat([s, inject], dim=-1)))\n"
         "            inject = torch.zeros_like(e)\n"
         "    for _ in range(k_grad):\n"
         "        s = core(adapter(torch.cat([s, inject], dim=-1)))\n"
         "        inject = torch.zeros_like(e)\n"),
    ),
    _mutation(
        "detaches the state before every gradient step",
        (GRAD_LOOP,
         "    for _ in range(k_grad):\n"
         "        s = core(adapter(torch.cat([s.detach(), e], dim=-1)))\n"),
    ),
    _mutation(
        "backpropagates through every iteration",
        ("    with torch.no_grad():\n        for _ in range(n_no_grad):\n            s = core",
         "    if True:\n        for _ in range(n_no_grad):\n            s = core"),
    ),
    _mutation(
        "runs every iteration with autograd, then detaches at the boundary",
        ("    with torch.no_grad():\n        for _ in range(n_no_grad):\n            s = core(adapter(torch.cat([s, e], dim=-1)))\n",
         "    for _ in range(n_no_grad):\n        s = core(adapter(torch.cat([s, e], dim=-1)))\n"
         "    if n_no_grad:\n        s = s.detach()\n"),
    ),
    _mutation(
        "moves one iteration from the gradient phase to the no-grad phase",
        ("range(n_no_grad)", "range(n_no_grad + 1 if k_grad else n_no_grad)"),
        ("range(k_grad)", "range(max(k_grad - 1, 0))"),
    ),
    _mutation(
        "detaches e in the gradient phase",
        (GRAD_LOOP,
         "    for _ in range(k_grad):\n"
         "        s = core(adapter(torch.cat([s, e.detach()], dim=-1)))\n"),
    ),
    _mutation(
        "detaches s0",
        ("    s = s0\n", "    s = s0.detach()\n"),
    ),
    _mutation(
        "applies a final layer norm",
        ("    return s\n", "    return torch.nn.functional.layer_norm(s, s.shape[-1:]) if n_no_grad + k_grad else s\n"),
    ),
    _mutation(
        "always backpropagates through mean_backprop_depth steps",
        ("k_grad = min(mean_backprop_depth, p)", "k_grad = mean_backprop_depth"),
    ),
    _mutation(
        "does not clamp the no-grad count",
        ("n_no_grad = max(p - mean_backprop_depth, 0)", "n_no_grad = p - mean_backprop_depth"),
    ),
    _mutation(
        "returns the split in the wrong order",
        ("    return n_no_grad, k_grad\n", "    return k_grad, n_no_grad\n"),
    ),
    _mutation(
        "returns tensors from the split",
        ("    return n_no_grad, k_grad\n", "    return torch.tensor(n_no_grad), torch.tensor(k_grad)\n"),
    ),
    _mutation(
        "accepts mismatched state shapes",
        ('        raise ValueError("s0 must have the same shape as e")', "        pass"),
    ),
    _mutation(
        "accepts negative counts",
        ("value < 0", "value < -10"),
    ),
]


def test_recurrent_depth_unroll_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_recurrent_depth_unroll_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
