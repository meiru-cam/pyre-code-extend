"""Vendored GAE with final-observation bootstrapping (ManiSkill) and chunk-level reshaping (RLinf).

Upstream:   https://github.com/haosulab/ManiSkill
Commit:     62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
Files:      examples/baselines/ppo/ppo.py
Symbols:    the GAE block of the training loop (final_values, real_next_values, lastgaelam)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

Upstream:   https://github.com/RLinf/RLinf
Commit:     0e9438dee0088be7673a1fb4789018aea168db1a
Files:      rlinf/algorithms/utils.py, rlinf/algorithms/advantages.py
Symbols:    preprocess_embodied_advantages_inputs, compute_gae_advantages_and_returns
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2025 The RLinf Authors. Licensed under the Apache License,
            Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for truncation_bootstrap_gae, so that exercise is not checked only
against an oracle written by the same author as its reference solution.
ManiSkill's PPO baseline is the step-level form that separates the bootstrap
value of a finished episode (``final_values``) from the value of the reset
observation that follows it. RLinf supplies the chunk-level reshaping (sum the
rewards of a chunk, max the done flags) and a GAE that takes a single done flag.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ManiSkill: the GAE block lives inline in a training script. It is lifted into
    ``maniskill_gae`` with the script's tensors (``rewards``, ``values``,
    ``dones``, ``next_done``, ``next_value``, ``final_values``) and
    ``args.gamma``/``args.gae_lambda`` as arguments. Only the default branch
    (``finite_horizon_gae=False``) is kept. Filling ``final_values`` is the
    caller's job, as in the script, where it is written for every env in
    ``infos["_final_info"]`` from ``agent.get_value(infos["final_observation"])``.
    ``next_value`` is passed with shape ``(num_envs,)`` instead of ``(1, num_envs)``;
    broadcasting is unchanged. ``torch.no_grad`` and ``.to(device)`` are dropped.
  * RLinf: ``preprocess_embodied_advantages_inputs`` keeps the ``chunk_level`` and
    ``gae`` paths and returns the reshaped tensors instead of updating a kwargs
    dict. ``loss_mask_sum`` is dropped. ``compute_gae_advantages_and_returns``
    keeps the critic path with ``normalize_advantages=False`` and
    ``normalize_returns=False`` (``safe_normalize`` and ``masked_mean`` are not
    needed); the registry decorator is dropped.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["maniskill_gae", "rlinf_gae", "rlinf_preprocess_embodied_advantages_inputs"]


def maniskill_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    next_value: torch.Tensor,
    final_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribed from ppo.py (default branch). ``dones[t]`` is the done flag observed
    *before* step t, i.e. the flag returned by step t - 1; ``next_done`` is the flag
    returned by the last step. Returns ``(advantages, returns)``."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        real_next_values = next_not_done * nextvalues + final_values[t]  # t instead of t+1
        delta = rewards[t] + gamma * real_next_values - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
    returns = advantages + values
    return advantages, returns


def rlinf_preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    reward_type: str = "chunk_level",
    adv_type: str = "gae",
) -> dict:
    """Transcribed from utils.py::preprocess_embodied_advantages_inputs.

    rewards: [n_chunk_steps, bsz, num_action_chunks]; dones and values carry one
    extra leading row (``n_chunk_steps + 1``)."""
    if reward_type == "chunk_level":
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size

    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    flattened_dones_full = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
    dones = flattened_dones_full[-(n_steps + 1) :]

    if adv_type == "gae":
        flattened_values_full = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
        values = flattened_values_full[: n_steps + 1]

    return {"rewards": rewards, "dones": dones, "values": values, "loss_mask": loss_mask}


def rlinf_gae(
    rewards: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    values: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribed from advantages.py::compute_gae_advantages_and_returns, critic path,
    no normalization. ``values`` and ``dones`` have ``seq_len + 1`` rows; ``dones``
    must be boolean (upstream negates it with ``~``)."""
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    gae = 0
    for step in reversed(range(T)):
        delta = rewards[step] + gamma * values[step + 1] * (~dones[step + 1]) - values[step]
        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae + values[step]
    advantages = returns - values[:-1]
    return advantages, returns
