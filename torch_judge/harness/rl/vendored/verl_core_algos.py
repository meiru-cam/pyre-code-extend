"""Vendored advantage, loss-aggregation and importance-weight oracles from verl.

Upstream:   https://github.com/volcengine/verl
Commit:     00094bd9cd3fef9cf8903daf60ea4a2bcf832efc
Files:      verl/trainer/ppo/core_algos.py, verl/trainer/ppo/rollout_corr_helper.py
Symbols:    compute_rloo_outcome_advantage, agg_loss, compute_rollout_correction_weights
Retrieved:  2026-09-28
License:    Apache License 2.0
            Copyright 2024 Bytedance Ltd. and/or its affiliates. Licensed under the
            Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for rloo_advantage, loss_aggregation_modes and
truncated_importance_sampling, so those exercises are not checked only against
oracles written by the same author as their reference solutions.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``AlgoConfig``, ``epsilon`` and the advantage-estimator registry are dropped;
    they do not affect the RLOO arithmetic.
  * ``agg_loss`` keeps only the single-process path: ``dp_size`` is 1 and
    ``batch_num_tokens``, ``global_batch_size`` and ``loss_scale_factor`` are
    ``None``. ``verl_F.masked_sum`` is inlined as ``(x * mask).sum()``.
  * ``compute_rollout_correction_weights`` keeps the TIS branch (a single upper
    threshold). IcePop thresholds, batch normalization and metrics are dropped.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from collections import defaultdict

import torch

__all__ = ["VERL_IS_SAFETY_BOUND", "verl_agg_loss", "verl_rloo_outcome_advantage", "verl_rollout_is_weights"]

VERL_IS_SAFETY_BOUND = 20.0


def verl_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index) -> torch.Tensor:
    """Transcribed from core_algos.py::compute_rloo_outcome_advantage; returns the advantages."""
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores


def verl_agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Transcribed from core_algos.py::agg_loss, single-process path."""
    if loss_agg_mode == "token-mean":
        batch_num_tokens = loss_mask.sum()
        loss = (loss_mat * loss_mask).sum() / batch_num_tokens
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        global_batch_size = seq_mask.sum()
        loss = (seq_losses * seq_mask).sum() / global_batch_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss /= loss_mask.shape[-1]
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)
        seq_mask = (seq_mask > 0).float()
        global_batch_size = seq_mask.sum()
        loss = (seq_losses * seq_mask).sum() / global_batch_size
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss


def verl_rollout_is_weights(log_ratio: torch.Tensor, response_mask: torch.Tensor, rollout_is: str,
                            threshold: float) -> torch.Tensor:
    """Transcribed from rollout_corr_helper.py::compute_rollout_correction_weights, TIS branch."""
    if rollout_is == "token":
        log_ratio_safe = torch.clamp(log_ratio, min=-VERL_IS_SAFETY_BOUND, max=VERL_IS_SAFETY_BOUND)
        raw_rollout_is_weights = torch.exp(log_ratio_safe)
    elif rollout_is == "sequence":
        log_ratio_sum = (log_ratio * response_mask).sum(dim=-1).unsqueeze(-1)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-VERL_IS_SAFETY_BOUND, max=VERL_IS_SAFETY_BOUND)
        raw_rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)
    else:
        raise ValueError(f"Unsupported rollout_is: {rollout_is}")
    raw_rollout_is_weights = raw_rollout_is_weights * response_mask
    return raw_rollout_is_weights.clamp(max=threshold)
