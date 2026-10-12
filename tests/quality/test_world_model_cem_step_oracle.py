"""world_model_cem_step's reference solution, checked against its upstream counterparts.

V-JEPA 2's notebook CEM is the contract for mode="vjepa2" and LeRobot's TDMPCPolicy.plan
for mode="tdmpc". The deliberate divergences (tie order, topk = 1, gripper momenta,
TD-MPC's std clamp and NaN handling) are pinned below.
"""

from __future__ import annotations

import math

import pytest
import torch

from torch_judge.harness.embodied.vendored.vjepa2_tdmpc_cem import l1, tdmpc_cem_refit, vjepa2_cem_refit
from torch_judge.tasks import get_task

TASK_ID = "world_model_cem_step"
ATOL = 1e-10


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def to_vjepa2_layout(samples: torch.Tensor) -> torch.Tensor:
    """(S, H, 4) xyz + gripper -> upstream (S, H, 7) with the rotation dims fixed at zero."""
    zeros = torch.zeros(*samples.shape[:2], 3, dtype=samples.dtype)
    return torch.cat([samples[..., :3], zeros, samples[..., 3:]], dim=-1)


def draw(seed: int, S: int, H: int, A: int):
    g = torch.Generator().manual_seed(seed)
    samples = torch.randn(S, H, A, generator=g, dtype=torch.float64)
    scores = torch.randn(S, generator=g, dtype=torch.float64) * 2  # continuous, so no ties
    mean = torch.randn(H, A, generator=g, dtype=torch.float64)
    std = torch.rand(H, A, generator=g, dtype=torch.float64) + 0.1
    return samples, scores, mean, std


@pytest.mark.parametrize(
    "seed,S,H,topk,mm,ms",
    [(0, 100, 1, 10, 0.25, 0.95), (1, 32, 3, 5, 0.25, 0.95), (2, 50, 4, 2, 0.0, 0.0), (3, 20, 2, 20, 0.6, 0.3), (4, 64, 5, 8, 1.0, 0.5), (5, 9, 2, 3, 0.15, 0.15)],
)
def test_vjepa2_mode_matches_upstream_cem(seed, S, H, topk, mm, ms):
    samples, energies, mean, std = draw(seed, S, H, 4)
    got_mean, got_std = reference()(samples, energies, mean, std, topk, mm, ms, "vjepa2")
    # One momentum pair for every dim: pass the same values as the gripper momenta.
    want_mean, want_std = vjepa2_cem_refit(to_vjepa2_layout(samples), energies, mean.clone(), std.clone(), topk, mm, ms, mm, ms)
    assert torch.allclose(got_mean, want_mean, atol=ATOL, rtol=0)
    assert torch.allclose(got_std, want_std, atol=ATOL, rtol=0)


def test_vjepa2_mode_with_energies_from_the_upstream_l1_objective():
    g = torch.Generator().manual_seed(11)
    final_states = torch.randn(40, 16, generator=g, dtype=torch.float64)
    goal = torch.randn(1, 16, generator=g, dtype=torch.float64).repeat(40, 1)
    energies = l1(final_states, goal)
    samples, _, mean, std = draw(12, 40, 3, 4)
    got = reference()(samples, energies, mean, std, 6, 0.25, 0.95, "vjepa2")
    want = vjepa2_cem_refit(to_vjepa2_layout(samples), energies, mean, std, 6, 0.25, 0.95, 0.25, 0.95)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=ATOL, rtol=0)


def tdmpc_layout(samples: torch.Tensor, values: torch.Tensor, mean: torch.Tensor):
    """(S, H, A), (S,), (H, A) -> upstream (H, S, 1, A), (S, 1), (H, 1, A)."""
    return samples.permute(1, 0, 2).unsqueeze(2), values.unsqueeze(1), mean.unsqueeze(1)


@pytest.mark.parametrize(
    "seed,S,H,A,topk,temp,mm",
    [(0, 563, 5, 4, 50, 0.5, 0.1), (1, 40, 3, 2, 8, 0.5, 0.1), (2, 20, 2, 3, 1, 1.0, 0.0), (3, 30, 4, 1, 30, 2.0, 0.5), (4, 64, 1, 6, 10, 0.05, 0.9), (5, 12, 3, 3, 4, 10.0, 0.3)],
)
def test_tdmpc_mode_matches_lerobot_plan(seed, S, H, A, topk, temp, mm):
    samples, values, mean, std = draw(seed, S, H, A)
    got_mean, got_std = reference()(samples, values, mean, std, topk, mm, 0.0, "tdmpc", temperature=temp)
    # momentum_std = 0 is upstream's direct replacement; an unbounded clamp is no clamp.
    want_mean, want_std = tdmpc_cem_refit(*tdmpc_layout(samples, values, mean), topk, temp, mm, 0.0, math.inf)
    assert torch.allclose(got_mean, want_mean.squeeze(1), atol=ATOL, rtol=0)
    assert torch.allclose(got_std, want_std.squeeze(1), atol=ATOL, rtol=0)


def test_tdmpc_mode_matches_lerobot_on_large_values():
    samples, values, mean, std = draw(21, 30, 2, 2)
    values = values * 1000 + 5000
    got = reference()(samples, values, mean, std, 6, 0.1, 0.0, "tdmpc", temperature=0.5)
    want = tdmpc_cem_refit(*tdmpc_layout(samples, values, mean), 6, 0.5, 0.1, 0.0, math.inf)
    assert torch.isfinite(got[0]).all() and torch.isfinite(got[1]).all()
    assert torch.allclose(got[0], want[0].squeeze(1), atol=ATOL, rtol=0)
    assert torch.allclose(got[1], want[1].squeeze(1), atol=ATOL, rtol=0)


def test_pins_tdmpc_std_clamp_divergence():
    """Upstream clamps the refit std to [min_std, max_std]; the exercise does not."""
    samples, values, mean, std = draw(31, 24, 2, 2)
    samples = samples * 10  # elite spread far above max_std = 2
    ours = reference()(samples, values, mean, std, 6, 0.1, 0.0, "tdmpc", temperature=0.5)
    clamped = tdmpc_cem_refit(*tdmpc_layout(samples, values, mean), 6, 0.5, 0.1, 0.05, 2.0)
    assert torch.allclose(ours[0], clamped[0].squeeze(1), atol=ATOL, rtol=0)  # means agree
    assert (ours[1] > 2.0).any() and torch.allclose(ours[1].clamp(0.05, 2.0), clamped[1].squeeze(1), atol=ATOL, rtol=0)


def test_pins_tdmpc_nan_divergence():
    """Upstream maps a NaN value to 0 and keeps planning; the exercise rejects it."""
    samples, values, mean, std = draw(41, 10, 2, 2)
    values[3] = float("nan")
    upstream_mean, _ = tdmpc_cem_refit(*tdmpc_layout(samples, values, mean), 3, 0.5, 0.1, 0.0, math.inf)
    assert torch.isfinite(upstream_mean).all()
    with pytest.raises(ValueError):
        reference()(samples, values, mean, std, 3, 0.1, 0.0, "tdmpc")


def test_pins_vjepa2_topk_one_divergence():
    """Upstream's unbiased std of one elite is NaN; the exercise raises instead."""
    samples, energies, mean, std = draw(51, 10, 2, 4)
    with pytest.warns(UserWarning):
        _, upstream_std = vjepa2_cem_refit(to_vjepa2_layout(samples), energies, mean, std, 1, 0.25, 0.95, 0.25, 0.95)
    assert torch.isnan(upstream_std).all()
    with pytest.raises(ValueError):
        reference()(samples, energies, mean, std, 1, 0.25, 0.95, "vjepa2")


def test_pins_vjepa2_gripper_momentum_divergence():
    """Upstream gives the gripper dim its own momenta; with different values only that dim differs."""
    samples, energies, mean, std = draw(61, 30, 3, 4)
    ours = reference()(samples, energies, mean, std, 5, 0.25, 0.95, "vjepa2")
    upstream = vjepa2_cem_refit(to_vjepa2_layout(samples), energies, mean, std, 5, 0.25, 0.95, 0.15, 0.15)
    for a, b in zip(ours, upstream):
        assert torch.allclose(a[..., :3], b[..., :3], atol=ATOL, rtol=0)
        assert not torch.allclose(a[..., 3:], b[..., 3:])


@pytest.mark.parametrize("mode", ["vjepa2", "tdmpc"])
def test_pins_tie_order(mode):
    """With ties at the cut, the exercise keeps the lower indices; upstream topk picks some valid tie set.

    Both agree whenever the reference's choice is fed to the oracle as a tie-free ranking.
    """
    samples, _, mean, std = draw(71, 8, 2, 4)
    scores = torch.tensor([3.0, 1.0, 1.0, 5.0, 1.0, 0.0, 1.0, 9.0], dtype=torch.float64)
    if mode == "tdmpc":
        scores = -scores
    ours = reference()(samples, scores, mean, std, 3, 0.2, 0.0, mode, temperature=0.0)
    # The lower-index rule keeps samples 5, 1, 2. Break the tie explicitly for the oracle.
    tie_free = scores.clone()
    tie_free[[4, 6]] += 0.5 if mode == "vjepa2" else -0.5
    if mode == "vjepa2":
        want = vjepa2_cem_refit(to_vjepa2_layout(samples), tie_free, mean, std, 3, 0.2, 0.0, 0.2, 0.0)
        assert torch.allclose(ours[1], want[1], atol=ATOL, rtol=0)
    else:
        want = tdmpc_cem_refit(*tdmpc_layout(samples, tie_free, mean), 3, 0.0, 0.2, 0.0, math.inf)
        want = (want[0].squeeze(1), want[1].squeeze(1))
        assert torch.allclose(ours[1], want[1], atol=ATOL, rtol=0)
    assert torch.allclose(ours[0], want[0], atol=ATOL, rtol=0)
    expected = samples[[5, 1, 2]].mean(0) * 0.8 + mean * 0.2
    assert torch.allclose(ours[0], expected, atol=ATOL, rtol=0)
