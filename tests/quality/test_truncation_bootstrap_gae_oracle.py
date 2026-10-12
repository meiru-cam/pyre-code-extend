"""truncation_bootstrap_gae's reference solution, checked against its upstream counterparts.

ManiSkill's PPO baseline is the step-level contract once its ``final_values`` is
filled only where an episode was truncated (and not terminated). At the pinned
commit ManiSkill fills it for every finished env; that divergence is pinned below.
RLinf's chunk-level reshaping plus its single-flag GAE is the chunk-level contract
when there are no truncations; RLinf relies on the environment to zero rewards
after an episode ends, where the exercise uses an explicit valid mask, also pinned.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.maniskill_ppo_gae import (
    maniskill_gae,
    rlinf_gae,
    rlinf_preprocess_embodied_advantages_inputs,
)
from torch_judge.tasks import get_task

TASK_ID = "truncation_bootstrap_gae"
ATOL = 1e-10
# RLinf multiplies a Python float by a bool tensor (``gamma * gae_lambda * (~dones)``),
# which yields a float32 factor, so its float64 results carry float32 rounding.
RLINF_ATOL = 1e-6


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def rollout(seed: int, T: int, B: int, p_term: float = 0.15, p_trunc: float = 0.15):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(T, B, generator=g, dtype=torch.float64)
    values = torch.randn(T, B, generator=g, dtype=torch.float64)
    next_value = torch.randn(B, generator=g, dtype=torch.float64)
    terminated = torch.rand(T, B, generator=g) < p_term
    truncated = torch.rand(T, B, generator=g) < p_trunc
    true_final = torch.randn(T, B, generator=g, dtype=torch.float64)
    return rewards, values, next_value, terminated, truncated, true_final


def to_maniskill(terminated, truncated, true_final, *, bootstrap_terminations=False):
    """ManiSkill layout: dones[t] is the flag returned by step t - 1; final_values is
    written for each env in the bootstrap mask and is zero elsewhere."""
    done = (terminated | truncated).to(torch.float64)
    dones = torch.cat([torch.zeros_like(done[:1]), done[:-1]], dim=0)
    next_done = done[-1]
    mask = (terminated | truncated) if bootstrap_terminations else (truncated & ~terminated)
    final_values = torch.where(mask, true_final, torch.zeros_like(true_final))
    return dones, next_done, final_values


@pytest.mark.parametrize(
    "seed,T,B,gamma,lam",
    [
        (0, 10, 4, 0.99, 0.95),
        (1, 16, 3, 0.8, 0.9),
        (2, 1, 5, 0.9, 0.5),
        (3, 25, 2, 1.0, 1.0),
        (4, 8, 6, 0.5, 0.0),
        (5, 12, 8, 0.95, 0.7),
    ],
)
def test_step_level_matches_maniskill(seed, T, B, gamma, lam):
    rewards, values, next_value, terminated, truncated, true_final = rollout(seed, T, B)
    # The exercise must ignore final_values outside truncations, so feed junk there.
    junk = torch.where(truncated & ~terminated, true_final, torch.full_like(true_final, 1e3))
    got_adv, got_ret = reference()(rewards, values, next_value, terminated, truncated, junk, gamma=gamma, lam=lam)
    dones, next_done, final_values = to_maniskill(terminated, truncated, true_final)
    want_adv, want_ret = maniskill_gae(rewards, values, dones, next_done, next_value, final_values, gamma, lam)
    assert torch.allclose(got_adv, want_adv, atol=ATOL, rtol=0)
    assert torch.allclose(got_ret, want_ret, atol=ATOL, rtol=0)


def test_edge_every_step_ends():
    T, B = 5, 4
    rewards, values, next_value, _, _, true_final = rollout(9, T, B)
    terminated = torch.zeros(T, B, dtype=torch.bool)
    terminated[:, :2] = True
    truncated = torch.zeros(T, B, dtype=torch.bool)
    truncated[:, 1:] = True  # column 1 has both flags
    got, _ = reference()(rewards, values, next_value, terminated, truncated, true_final, gamma=0.9, lam=0.9)
    dones, next_done, final_values = to_maniskill(terminated, truncated, true_final)
    want, _ = maniskill_gae(rewards, values, dones, next_done, next_value, final_values, 0.9, 0.9)
    assert torch.allclose(got, want, atol=ATOL, rtol=0)


def test_edge_no_boundaries_reduces_to_plain_gae():
    T, B = 9, 3
    rewards, values, next_value, _, _, true_final = rollout(10, T, B)
    none = torch.zeros(T, B, dtype=torch.bool)
    got, _ = reference()(rewards, values, next_value, none, none, true_final, gamma=0.97, lam=0.9)
    dones, next_done, final_values = to_maniskill(none, none, true_final)
    want, _ = maniskill_gae(rewards, values, dones, next_done, next_value, final_values, 0.97, 0.9)
    assert torch.allclose(got, want, atol=ATOL, rtol=0)


def test_divergence_maniskill_also_bootstraps_terminations():
    """Pinned divergence: upstream fills final_values for every finished env."""
    T, B = 12, 4
    rewards, values, next_value, terminated, truncated, true_final = rollout(3, T, B, p_term=0.3)
    assert bool(terminated.any())
    got, _ = reference()(rewards, values, next_value, terminated, truncated, true_final, gamma=0.9, lam=0.9)
    dones, next_done, upstream_final = to_maniskill(terminated, truncated, true_final, bootstrap_terminations=True)
    upstream, _ = maniskill_gae(rewards, values, dones, next_done, next_value, upstream_final, 0.9, 0.9)
    assert not torch.allclose(got, upstream, atol=1e-6)
    # The rows that differ are exactly those whose episode ended by termination,
    # plus earlier rows of the same episode that the trace reaches.
    only_terminations = terminated & ~truncated
    first_diff = (got - upstream).abs() > 1e-9
    assert bool((first_diff & only_terminations).any())


def chunk_rollout(seed: int, T: int, B: int, H: int):
    g = torch.Generator().manual_seed(seed)
    step_rewards = torch.randn(T, B, H, generator=g, dtype=torch.float64)
    counts = torch.randint(1, H + 1, (T, B), generator=g)
    valid = torch.arange(H).view(1, 1, H) < counts.unsqueeze(-1)
    terminated = (counts < H) | (torch.rand(T, B, generator=g) < 0.1)
    # Ending mid-chunk means the episode ended; a full chunk may also end it.
    values = torch.randn(T + 1, B, generator=g, dtype=torch.float64)
    return step_rewards, valid, terminated, values


@pytest.mark.parametrize("seed,T,B,H,gamma,lam", [(0, 6, 3, 4, 0.9, 0.95), (1, 10, 2, 5, 0.5, 0.8), (2, 3, 5, 1, 0.99, 1.0), (3, 7, 4, 8, 1.0, 0.0), (4, 9, 3, 2, 0.8, 0.6)])
def test_chunk_level_matches_rlinf(seed, T, B, H, gamma, lam):
    step_rewards, valid, terminated, values_full = chunk_rollout(seed, T, B, H)
    zeroed = torch.where(valid, step_rewards, torch.zeros_like(step_rewards))
    # RLinf: dones carry one leading row, broadcast across the chunk axis; values carry one trailing row.
    dones = torch.cat([torch.zeros(1, B, dtype=torch.bool), terminated], dim=0).unsqueeze(-1).expand(T + 1, B, H)
    prepped = rlinf_preprocess_embodied_advantages_inputs(
        zeroed, dones.contiguous(), values_full.unsqueeze(-1), reward_type="chunk_level", adv_type="gae"
    )
    want_adv, want_ret = rlinf_gae(prepped["rewards"], gamma, lam, prepped["values"], prepped["dones"])
    none = torch.zeros(T, B, dtype=torch.bool)
    junk_rewards = torch.where(valid, step_rewards, torch.full_like(step_rewards, 50.0))
    got_adv, got_ret = reference()(
        junk_rewards, values_full[:T], values_full[T], terminated, none, torch.zeros(T, B, dtype=torch.float64),
        gamma=gamma, lam=lam, valid_mask=valid,
    )
    assert torch.allclose(got_adv, want_adv, atol=RLINF_ATOL, rtol=0)
    assert torch.allclose(got_ret, want_ret, atol=RLINF_ATOL, rtol=0)


def test_chunk_level_with_truncations_matches_maniskill_on_rlinf_rewards():
    T, B, H = 8, 4, 3
    step_rewards, valid, _, values_full = chunk_rollout(6, T, B, H)
    _, _, _, terminated, truncated, true_final = rollout(6, T, B)
    zeroed = torch.where(valid, step_rewards, torch.zeros_like(step_rewards))
    chunk_rewards = rlinf_preprocess_embodied_advantages_inputs(
        zeroed, torch.zeros(T + 1, B, H, dtype=torch.bool), values_full.unsqueeze(-1),
        reward_type="chunk_level", adv_type="gae",
    )["rewards"]
    dones, next_done, final_values = to_maniskill(terminated, truncated, true_final)
    want, _ = maniskill_gae(chunk_rewards, values_full[:T], dones, next_done, values_full[T], final_values, 0.9, 0.8)
    got, _ = reference()(
        step_rewards, values_full[:T], values_full[T], terminated, truncated, true_final,
        gamma=0.9, lam=0.8, valid_mask=valid,
    )
    assert torch.allclose(got, want, atol=ATOL, rtol=0)


def test_divergence_rlinf_sums_rewards_past_the_end():
    """Pinned divergence: RLinf sums every slot of the chunk; the exercise masks."""
    T, B, H = 4, 2, 3
    step_rewards, valid, terminated, values_full = chunk_rollout(7, T, B, H)
    assert not bool(valid.all())
    prepped = rlinf_preprocess_embodied_advantages_inputs(
        step_rewards, torch.zeros(T + 1, B, H, dtype=torch.bool), values_full.unsqueeze(-1),
        reward_type="chunk_level", adv_type="gae",
    )
    masked_sum = (step_rewards * valid).sum(-1)
    assert not torch.allclose(prepped["rewards"], masked_sum)
