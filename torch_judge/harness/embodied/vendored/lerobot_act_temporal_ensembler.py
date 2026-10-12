"""Vendored ACT temporal-ensembling oracles: LeRobot's online class and the original offline loop.

Upstream:   https://github.com/huggingface/lerobot
Commit:     75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee
Files:      src/lerobot/policies/act/modeling_act.py
Symbols:    ACTTemporalEnsembler (line 170)
License:    Apache License 2.0
            Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. Licensed
            under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

Upstream:   https://github.com/tonyzhaozh/act
Commit:     742c753c0d4a5d87076c8f69e5628c79a8cc5488
Files:      imitate_episodes.py
Symbols:    eval_bc, temporal_agg branch (lines 250-259)
License:    MIT License, Copyright (c) 2023 Tony Z. Zhao

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for act_temporal_ensemble, so that exercise is not checked only
against an oracle written by the same author as its reference solution.
LeRobot's class is the contract the exercise follows. The original ACT loop is
kept to pin one deliberate divergence (see below).

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``ACTTemporalEnsembler`` keeps its body as upstream. The docstring is
    shortened, and the ``.to(device=...)`` calls are kept so the class runs on
    any device.
  * The original ``imitate_episodes.py`` code is an inline block inside the
    evaluation loop of ``eval_bc``. It is lifted into ``act_original_temporal_agg``,
    which receives the per-step chunks the policy would have produced. The
    environment, rendering, ``pre_process``/``post_process`` and ``.cuda()`` are
    dropped. ``query_frequency`` is 1, as it is upstream whenever ``temporal_agg``
    is on. The hard-coded ``k = 0.01`` becomes the ``k`` argument.

UPSTREAM QUIRK THE EXERCISE DOES NOT COPY
-----------------------------------------
The original loop finds the predictions that cover step ``t`` with
``torch.all(actions_for_curr_step != 0, axis=1)``, because unfilled rows of its
zero-initialised buffer are zero. A real prediction whose action vector is all
zeros, or has any zero entry, is therefore dropped from the average. LeRobot's
online class has no such filter, and the exercise follows LeRobot.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = ["ACTTemporalEnsembler", "act_original_temporal_agg"]


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as w_i = exp(-temporal_ensemble_coeff * i) where w_0 is the oldest
        action. They are then normalized to sum to 1 by dividing by sum(w_i).
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


def act_original_temporal_agg(all_chunks: list[Tensor], num_queries: int, k: float = 0.01) -> list[Tensor]:
    """Transcribed from imitate_episodes.py::eval_bc (temporal_agg branch, query_frequency = 1).

    ``all_chunks[t]`` is the ``(1, num_queries, state_dim)`` chunk the policy predicts at step ``t``.
    Returns the ``(1, state_dim)`` raw action executed at each step.
    """
    max_timesteps = len(all_chunks)
    state_dim = all_chunks[0].shape[-1]
    all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim])
    raw_actions = []
    with torch.inference_mode():
        for t in range(max_timesteps):
            all_actions = all_chunks[t]
            all_time_actions[[t], t:t + num_queries] = all_actions
            actions_for_curr_step = all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            raw_actions.append(raw_action)
    return raw_actions
