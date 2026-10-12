"""Vendored action-chunk execution oracles from RLinf and ManiSkill.

Upstream:   https://github.com/RLinf/RLinf
Commit:     0e9438dee0088be7673a1fb4789018aea168db1a
Files:      rlinf/envs/sim/libero/libero_env.py, rlinf/envs/utils.py
Symbols:    LiberoEnv.chunk_step, valid_action_mask_from_counts
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2025 The RLinf Authors. Licensed under the Apache License,
            Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

Upstream:   https://github.com/haosulab/ManiSkill
Commit:     62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
Files:      mani_skill/utils/wrappers/action_repeat.py
Symbols:    ActionRepeatWrapper.step, ActionRepeatWrapper._update_dict_values
Retrieved:  2026-10-12
License:    Apache License 2.0 (repository LICENSE file). Licensed under the
            Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for chunk_step_executor, so its reference solution is not checked
only against an evaluator written by the same author. RLinf supplies the
executed-action count (including the ending action) and the prefix valid mask;
ManiSkill supplies the not-done update rule for the observation, reward and
flags and the early break once every env is done.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``LiberoEnv.chunk_step`` calls ``self.step(actions, auto_reset=False)``,
    which steps the LIBERO workers, counts elapsed steps, derives truncation
    from ``max_episode_steps`` and records metrics. Here a plain
    ``step_fn(actions) -> (obs, reward, terminations, truncations)`` stands in
    for it; the time limit lives inside ``step_fn``.
  * Camera-render skipping, ``infos`` and the eval-only ``eval_count_mask`` are
    dropped. The function follows the ``self.auto_reset`` branch (flags folded
    onto the last index) but does not call ``_handle_auto_reset``; resetting is
    out of scope. ``valid_action_mask`` is returned directly instead of being
    stored in ``infos_list[-1]``.
  * ``ActionRepeatWrapper.step`` calls ``super().step(action)``; here
    ``step_fn(action)`` stands in for it. Infos are dropped. ``Array`` is
    ``torch.Tensor``.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["maniskill_action_repeat_step", "rlinf_libero_chunk_step", "valid_action_mask_from_counts"]


def valid_action_mask_from_counts(counts: Any, chunk_size: int) -> torch.Tensor:
    """Transcribed from rlinf/envs/utils.py::valid_action_mask_from_counts.

    Build a ``[B, C]`` prefix mask from per-environment executed action counts.
    The count includes the episode-ending action. Later padded slots are false.
    """
    if isinstance(counts, torch.Tensor):
        count_tensor = counts.detach().to(dtype=torch.long)
    else:
        count_tensor = torch.as_tensor(np.asarray(counts), dtype=torch.long)
    if count_tensor.ndim == 0:
        count_tensor = count_tensor.unsqueeze(0)
    action_indices = torch.arange(chunk_size, device=count_tensor.device)
    return action_indices.unsqueeze(0) < count_tensor.unsqueeze(1)


def rlinf_libero_chunk_step(step_fn, chunk_actions):
    """Transcribed from libero_env.py::LiberoEnv.chunk_step (auto_reset branch, no reset).

    Returns ``(obs_list, chunk_rewards, chunk_terminations, chunk_truncations,
    valid_action_mask)``; rewards and flags are ``[num_envs, chunk_steps]``.
    """
    # chunk_actions: [num_envs, chunk_step, action_dim]
    chunk_size = chunk_actions.shape[1]
    obs_list = []

    chunk_rewards = []

    raw_chunk_terminations = []
    raw_chunk_truncations = []
    for i in range(chunk_size):
        actions = chunk_actions[:, i]
        extracted_obs, step_reward, terminations, truncations = step_fn(actions)
        obs_list.append(extracted_obs)

        chunk_rewards.append(step_reward)
        raw_chunk_terminations.append(terminations)
        raw_chunk_truncations.append(truncations)

    chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
    raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)  # [num_envs, chunk_steps]
    raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)  # [num_envs, chunk_steps]

    past_terminations = raw_chunk_terminations.any(dim=1)
    past_truncations = raw_chunk_truncations.any(dim=1)
    past_dones = torch.logical_or(past_terminations, past_truncations)
    raw_chunk_dones = torch.logical_or(raw_chunk_terminations, raw_chunk_truncations)
    first_done_steps = raw_chunk_dones.to(torch.int64).argmax(dim=1) + 1
    valid_action_counts = torch.where(
        past_dones,
        first_done_steps,
        torch.full_like(first_done_steps, chunk_size),
    )
    # Include the ending action and exclude later slots. Terminations below
    # are folded onto the last chunk index, so this mask is the prefix.
    valid_action_mask = valid_action_mask_from_counts(valid_action_counts, chunk_size)

    chunk_terminations = torch.zeros_like(raw_chunk_terminations)
    chunk_terminations[:, -1] = past_terminations

    chunk_truncations = torch.zeros_like(raw_chunk_truncations)
    chunk_truncations[:, -1] = past_truncations
    return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, valid_action_mask


def _update_dict_values(from_dict: dict, to_dict: dict, not_dones):
    """Transcribed from action_repeat.py::ActionRepeatWrapper._update_dict_values."""
    for k, v in from_dict.items():
        if isinstance(v, dict):
            _update_dict_values(from_dict=v, to_dict=to_dict[k], not_dones=not_dones)
        elif isinstance(v, torch.Tensor):
            to_dict[k][not_dones] = v[not_dones]
        else:
            to_dict[k] = v


def maniskill_action_repeat_step(step_fn, action, repeat: int):
    """Transcribed from action_repeat.py::ActionRepeatWrapper.step.

    Returns ``(final_obs, final_rew, final_terminations, final_truncations)``.
    """
    final_obs, final_rew, final_terminations, final_truncations = step_fn(action)

    is_obs_dict = isinstance(final_obs, dict)

    dones = torch.logical_or(final_terminations, final_truncations)
    not_dones = ~dones

    if not_dones.any():
        for _ in range(repeat - 1):
            new_obs, new_rew, new_terminations, new_truncations = step_fn(action)

            if is_obs_dict:
                _update_dict_values(from_dict=new_obs, to_dict=final_obs, not_dones=not_dones)
            else:
                final_obs[not_dones] = new_obs[not_dones]

            final_rew[not_dones] += new_rew[not_dones]
            final_terminations[not_dones] = torch.logical_or(final_terminations, new_terminations)[not_dones]
            final_truncations[not_dones] = torch.logical_or(final_truncations, new_truncations)[not_dones]

            dones = torch.logical_or(final_terminations, final_truncations)
            not_dones = ~dones

            if dones.all():
                break

    return final_obs, final_rew, final_terminations, final_truncations
