"""Vendored GRPO outcome advantage, chunk token mask and accuracy filter from SimpleVLA-RL.

Upstream:   https://github.com/PRIME-RL/SimpleVLA-RL
Commit:     7c51662df27b586f9e8a1ab35fcf849f2b8852f9
Files:      verl/trainer/ppo/core_algos.py, verl/trainer/ppo/ray_trainer.py
Symbols:    compute_grpo_outcome_advantage (core_algos.py),
            compute_advantage response mask, RayTrainer.filter accuracy branch (ray_trainer.py)
Retrieved:  2026-10-12
License:    MIT License, Copyright (c) 2025 PRIME-RL (repository LICENSE).
            The two files also carry Apache License 2.0 headers inherited from verl
            (core_algos.py: Copyright 2022 The HuggingFace Team; ray_trainer.py:
            Copyright 2024 Bytedance Ltd. and/or its affiliates). A copy of that
            license is at http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for init_state_group_advantage, so the exercise is not checked only
against an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``DataProto``, ``config`` and the trainer object are replaced by plain
    arguments: ``finish_step``, ``action_token_len``, ``traj_len`` and
    ``chunk_token_len`` for the mask (``responses.size(1)`` and
    ``responses.size(2)`` upstream), and ``n_samples``, ``lower`` and ``upper``
    for the filter (``config.data.*`` upstream).
  * ``RayTrainer.filter`` keeps only the accuracy branch and returns the per-rollout
    keep mask instead of slicing the batch; the truncation branch and the logging
    are dropped. The filter, like upstream, assumes ``n_samples`` consecutive
    rollouts per prompt row.
  * ``compute_grpo_outcome_advantage`` returns only the advantages (upstream returns
    the same tensor twice, as advantages and returns).

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from collections import defaultdict

import torch

__all__ = ["svr_accuracy_filter", "svr_grpo_outcome_advantage", "svr_response_mask"]


def svr_grpo_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index,
                               epsilon: float = 1e-6) -> torch.Tensor:
    """Transcribed from core_algos.py::compute_grpo_outcome_advantage; returns the advantages."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores


def svr_response_mask(finish_step: torch.Tensor, action_token_len: int, traj_len: int,
                      chunk_token_len: int) -> torch.Tensor:
    """Transcribed from ray_trainer.py::compute_advantage, the mask built before every estimator."""
    response_length = traj_len * chunk_token_len
    finish_step = finish_step * action_token_len
    steps = torch.arange(response_length, device=finish_step.device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(finish_step.size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
    return response_mask


def svr_accuracy_filter(reward_tensor: torch.Tensor, n_samples: int, lower: float, upper: float) -> torch.Tensor:
    """Transcribed from ray_trainer.py::RayTrainer.filter, accuracy branch; returns the per-rollout keep mask."""
    reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
    acc_tensor = torch.mean(reward_matrix, dim=-1)
    acc_mask = (acc_tensor >= lower) & (acc_tensor <= upper)
    combined_mask = acc_mask
    final_mask = combined_mask.repeat_interleave(n_samples)
    return final_mask
