"""Mutation gate for the Dreamer 4 shortcut forcing exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "shortcut_forcing_loss"

B1 = "            b1 = (model(z_tilde, tau, half) - z_tilde) / (1 - t)\n"
B2 = "            b2 = (model(z_mid, tau + half, half) - z_mid) / (1 - (t + h))\n"
TARGET = "            v_target = (b1 + b2) / 2\n"
BOOT = "        bootstrap = (1 - tau) ** 2 * _sum_sq(v_hat - v_target)\n"
RETURN = "    return (0.9 * tau + 0.1) * loss  # ramp weight on both branches\n"
TAU_GRID = "    index = torch.floor(torch.rand(shape, generator=generator, dtype=torch.float64) * steps).long()\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    # The loss.
    _mutation(
        "no stop gradient on the bootstrap target",
        ("        with torch.no_grad():\n", "        if True:\n"),
    ),
    _mutation(
        "ramp weight only on the flow branch",
        (BOOT, BOOT + "        bootstrap = bootstrap / (0.9 * tau + 0.1)\n"),
    ),
    _mutation(
        "ramp weight only on the bootstrap branch",
        (BOOT, BOOT + "        bootstrap = bootstrap * (0.9 * tau + 0.1)\n"),
        (RETURN, "    return loss\n"),
    ),
    _mutation(
        "ramp weight reversed",
        (RETURN, "    return (0.9 * (1 - tau) + 0.1) * loss\n"),
    ),
    _mutation(
        "no ramp weight",
        (RETURN, "    return loss\n"),
    ),
    _mutation(
        "bootstrap without the (1 - tau)^2 factor",
        (BOOT, "        bootstrap = _sum_sq(v_hat - v_target)\n"),
    ),
    _mutation(
        "flow term in velocity space",
        ("    loss = _sum_sq(z1_hat - z1)  # x-space flow loss at the finest step\n",
         "    loss = _sum_sq((z1_hat - z1) / (1 - t))\n"),
    ),
    _mutation(
        "second half step divided by 1 - tau",
        (B2, "            b2 = (model(z_mid, tau + half, half) - z_mid) / (1 - t)\n"),
    ),
    _mutation(
        "second half step queried at tau",
        (B2, "            b2 = (model(z_mid, tau, half) - z_mid) / (1 - (t + h))\n"),
    ),
    _mutation(
        "second half step from z_tilde",
        (B2, "            b2 = (model(z_tilde, tau + half, half) - z_tilde) / (1 - (t + h))\n"),
    ),
    _mutation(
        "half steps conditioned on the full step size",
        (B1, "            b1 = (model(z_tilde, tau, d) - z_tilde) / (1 - t)\n"),
        (B2, "            b2 = (model(z_mid, tau + half, d) - z_mid) / (1 - (t + h))\n"),
    ),
    _mutation(
        "target is the first half step only",
        (TARGET, "            v_target = b1\n"),
    ),
    _mutation(
        "target is the empirical velocity",
        (TARGET, "            v_target = (b1 + b2) * 0 + (z1 - z0)\n"),
    ),
    _mutation(
        "mean over features instead of sum",
        ("    return (x * x).sum(dim=-1)\n", "    return (x * x).mean(dim=-1)\n"),
    ),
    _mutation(
        "every frame bootstrapped",
        ("    flow = d == d_min\n", "    flow = torch.zeros_like(d, dtype=torch.bool)\n"),
    ),
    _mutation(
        "midpoint kept at z_tilde on flow frames",
        ("            z_mid = z_tilde + b1 * h\n",
         "            z_mid = torch.where(flow[..., None], z_tilde, z_tilde + b1 * h)\n"),
    ),
    _mutation(
        "half-step calls made even when every frame is a flow frame",
        ("    if not bool(flow.all()):\n", "    if True:\n"),
    ),
    _mutation(
        "mean over frames returned",
        (RETURN, "    return ((0.9 * tau + 0.1) * loss).mean()\n"),
    ),
    _mutation(
        "accepts levels past the grid",
        ("or bool((tau + d > 1).any())", "or bool((tau > 1).any())"),
    ),
    # The sampler.
    _mutation(
        "level grid includes tau = 1",
        (TAU_GRID, "    index = torch.floor(torch.rand(shape, generator=generator, dtype=torch.float64) * (steps + 1)).long()\n"),
    ),
    _mutation(
        "level always on the finest grid",
        (TAU_GRID, "    index = torch.floor(torch.rand(shape, generator=generator, dtype=torch.float64) * k_max).long()\n"),
        ("    tau = index.double() * d\n", "    tau = torch.minimum(index.double() / k_max, 1 - d)\n"),
    ),
    _mutation(
        "step count uniform over 1..k_max",
        ("    exponent = torch.randint(0, num_sizes, shape, generator=generator)\n    steps = 2 ** exponent  # K = 1 / d for each element\n",
         "    steps = 2 ** torch.floor(torch.log2(torch.randint(1, k_max + 1, shape, generator=generator).double())).long()\n"),
    ),
    _mutation(
        "never samples d = 1",
        ("    exponent = torch.randint(0, num_sizes, shape, generator=generator)\n",
         "    exponent = torch.randint(min(1, num_sizes - 1), num_sizes, shape, generator=generator)\n"),
    ),
    _mutation(
        "level drawn from a continuous range",
        ("    tau = index.double() * d\n", "    tau = torch.rand(shape, generator=generator, dtype=torch.float64) * (1 - d)\n"),
    ),
    _mutation(
        "accepts k_max that is not a power of two",
        (" or k_max & (k_max - 1):", ":"),
    ),
]


def test_shortcut_forcing_loss_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert 1 <= sum(t.get("visibility", "visible") == "visible" for t in task["tests"]) <= 3


@pytest.mark.parametrize("_repeat", range(3))
def test_shortcut_forcing_loss_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
