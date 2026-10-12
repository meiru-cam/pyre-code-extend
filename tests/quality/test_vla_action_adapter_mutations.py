"""Mutation gate for the VLA action adapter exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "vla_action_adapter"

UNNORM_LINE = "    x = (actions[..., :D] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01\n"
OFFSET_LINE = "    x = x + offset.unsqueeze(-2)\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    # From the research sketch: unnormalize with stats of the padded width and keep the wrong slice.
    _mutation(
        "keeps the last D model dims instead of the first",
        (UNNORM_LINE, "    x = (actions[..., -D:] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01\n"),
    ),
    # From the research sketch: delta added on unmasked dims.
    _mutation(
        "adds the state on every dim",
        ("offset = torch.where(delta_mask, state, torch.zeros_like(state))", "offset = state"),
    ),
    _mutation(
        "drops the 1e-6 in the unnormalize",
        (UNNORM_LINE, "    x = (actions[..., :D] + 1.0) / 2.0 * (q99 - q01) + q01\n"),
    ),
    _mutation(
        "adds the state before unnormalizing",
        (UNNORM_LINE,
         "    a = actions[..., :D] + torch.where(delta_mask, state, torch.zeros_like(state)).unsqueeze(-2)\n"
         "    x = (a + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01\n"),
        (OFFSET_LINE, ""),
    ),
    _mutation(
        "clips the model output to [-1, 1]",
        (UNNORM_LINE, "    x = (actions[..., :D].clamp(-1, 1) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01\n"),
    ),
    _mutation(
        "binarizes 0.5 to +1 instead of 0",
        ("        g = torch.sign(g)\n", "        g = torch.where(g >= 0, torch.ones_like(g), -torch.ones_like(g))\n"),
    ),
    _mutation(
        "takes the sign of the raw [0, 1] value",
        ("    g = 2.0 * x[..., -1:] - 1.0\n    if binarize_gripper:\n        g = torch.sign(g)\n",
         "    g = x[..., -1:]\n    if binarize_gripper:\n        g = torch.sign(g)\n    g = 2.0 * g - 1.0\n"),
    ),
    _mutation(
        "skips the [0, 1] to [-1, 1] mapping",
        ("    g = 2.0 * x[..., -1:] - 1.0\n", "    g = x[..., -1:]\n"),
    ),
    _mutation(
        "gripper read from the raw model output",
        ("    g = 2.0 * x[..., -1:] - 1.0\n", "    g = 2.0 * actions[..., D - 1:D] - 1.0\n"),
    ),
    _mutation(
        "never inverts the gripper",
        ("    if invert_gripper:\n        g = -g\n", ""),
    ),
    _mutation(
        "binarizes even when the flag is off",
        ("    if binarize_gripper:\n        g = torch.sign(g)\n", "    g = torch.sign(g)\n"),
    ),
    _mutation(
        "broadcasts the state over the batch instead of the steps",
        (OFFSET_LINE, "    x = x + (offset if offset.ndim == 1 else offset.unsqueeze(0))\n"),
    ),
    _mutation(
        "writes the result into the actions tensor",
        ("    return torch.cat([x[..., :-1], g], dim=-1).to(actions.dtype)\n",
         "    out = torch.cat([x[..., :-1], g], dim=-1).to(actions.dtype)\n"
         "    actions[..., :D] = out\n"
         "    return out\n"),
    ),
    _mutation(
        "accepts a state without the chunk's leading dims",
        ('        raise ValueError("state must have shape actions.shape[:-2] + (D,)")\n', "        pass\n"),
    ),
]


def test_vla_action_adapter_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_vla_action_adapter_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
