"""Vendored CEM refit oracles: V-JEPA 2-AC's goal-energy CEM and LeRobot TD-MPC's MPPI/CEM.

Upstream:   https://github.com/facebookresearch/vjepa2
Commit:     204698b45b3712590f06245fbfba32d3be539812
Files:      notebooks/utils/mpc_utils.py
Symbols:    l1 (line 17); cem, inner select_topk_action_traj (line 120) and the
            mean/std refit inside the CEM loop (lines 127-150)
License:    MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

Upstream:   https://github.com/huggingface/lerobot
Commit:     75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee
Files:      src/lerobot/policies/tdmpc/modeling_tdmpc.py
Symbols:    TDMPCPolicy.plan, elite selection and gaussian refit (lines 225-251)
License:    Apache License 2.0
            Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su, and The
            HuggingFace Inc. team. Licensed under the Apache License, Version 2.0;
            you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for world_model_cem_step, so that exercise is not checked only
against an oracle written by the same author as its reference solution. Only the
refit arithmetic is vendored: how the top-k samples are chosen and how they turn
into the next sampling mean and standard deviation. Sampling, the world model and
the value estimate are infrastructure and stay upstream.

WHAT WAS CHANGED, AND WHY
-------------------------
  * V-JEPA 2: ``cem`` is one function with closures. The refit lines are lifted
    into ``vjepa2_cem_refit``, which receives the sampled ``action_traj`` and the
    per-sample objective values ``sims`` that the loop would have computed. The
    selection is the body of ``select_topk_action_traj`` after ``objective`` has
    run. Arithmetic, the 7-dim action layout (xyz, three zero rotation dims,
    gripper), the 4-dim mean/std layout and the separate gripper momenta are kept
    as upstream. ``logger.info`` and ``tqdm`` are dropped.
  * TD-MPC: the refit lines of ``plan`` are lifted into ``tdmpc_cem_refit``, which
    receives ``actions`` (horizon, n, batch, action_dim), ``value`` (n, batch)
    and the current ``mean``. ``self.config.*`` fields become arguments. The
    ``einops.rearrange`` calls are replaced with the equivalent ``unsqueeze``
    calls, because einops is not a dependency here; the arithmetic is unchanged.
    The in-place ``value.nan_to_num_(0)`` is kept, applied to a copy.

UPSTREAM BEHAVIOUR THE EXERCISE DOES NOT COPY (pinned in the oracle test)
-------------------------------------------------------------------------
  * ``torch.topk`` does not specify which of several tied samples it returns.
    The exercise breaks ties by lower sample index.
  * V-JEPA 2 with ``topk = 1``: ``Tensor.std`` is unbiased, so one elite gives a
    NaN std that poisons the next sampling round. The exercise raises instead.
  * V-JEPA 2 uses separate momenta for the gripper dimension; the exercise uses
    one pair of momenta for every action dimension.
  * TD-MPC replaces the std outright and clamps it to [min_std, max_std]; the
    exercise blends it with ``momentum_std`` (0 reproduces the replacement) and
    does not clamp.
  * TD-MPC maps NaN values to 0; the exercise rejects non-finite energies.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["l1", "vjepa2_cem_refit", "tdmpc_cem_refit"]


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def vjepa2_cem_refit(
    action_traj: Tensor,
    sims: Tensor,
    mean: Tensor,
    std: Tensor,
    topk: int,
    momentum_mean: float = 0.25,
    momentum_std: float = 0.95,
    momentum_mean_gripper: float = 0.15,
    momentum_std_gripper: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """One refit step of mpc_utils.py::cem.

    ``action_traj`` is ``(S, rollout, 7)``, ``sims`` is ``(S,)`` (lower is closer to
    the goal), ``mean`` and ``std`` are ``(rollout, 4)``.
    """
    # select_topk_action_traj, after sims = objective(final_state, goal_state)
    indices = sims.topk(topk, largest=False).indices
    selected_actions = action_traj[indices]

    mean_selected_actions = selected_actions.mean(dim=0)
    std_selected_actions = selected_actions.std(dim=0)

    # -- Update new sampling mean and std based on the top-k samples
    mean = torch.cat(
        [
            mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
            mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
            + mean[..., -1:] * momentum_mean_gripper,
        ],
        dim=-1,
    )
    std = torch.cat(
        [
            std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
            std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
        ],
        dim=-1,
    )
    return mean, std


def tdmpc_cem_refit(
    actions: Tensor,
    value: Tensor,
    mean: Tensor,
    n_elites: int,
    elite_weighting_temperature: float = 0.5,
    gaussian_mean_momentum: float = 0.1,
    min_std: float = 0.05,
    max_std: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """One refit step of TDMPCPolicy.plan.

    ``actions`` is ``(horizon, n, batch, action_dim)``, ``value`` is ``(n, batch)``
    (higher is better), ``mean`` is ``(horizon, batch, action_dim)``.
    """
    value = value.clone().nan_to_num_(0)
    elite_idxs = torch.topk(value, n_elites, dim=0).indices  # (n_elites, batch)
    elite_value = value.take_along_dim(elite_idxs, dim=0)  # (n_elites, batch)
    # (horizon, n_elites, batch, action_dim)
    elite_actions = actions.take_along_dim(elite_idxs.unsqueeze(0).unsqueeze(-1), dim=1)

    # Update gaussian PDF parameters to be the (weighted) mean and standard deviation of the elites.
    max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
    # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
    # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
    # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
    score = torch.exp(elite_weighting_temperature * (elite_value - max_value))
    score /= score.sum(axis=0, keepdim=True)
    # (horizon, batch, action_dim)
    _mean = torch.sum(score.unsqueeze(-1) * elite_actions, dim=1)
    _std = torch.sqrt(
        torch.sum(
            score.unsqueeze(-1)
            * (elite_actions - _mean.unsqueeze(1)) ** 2,
            dim=1,
        )
    )
    # Update mean with an exponential moving average, and std with a direct replacement.
    mean = gaussian_mean_momentum * mean + (1 - gaussian_mean_momentum) * _mean
    std = _std.clamp_(min_std, max_std)
    return mean, std
