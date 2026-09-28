"""Vendored SimPO and IPO loss oracles from TRL.

Upstream:   https://github.com/huggingface/trl
Commit:     d947c4f5098c8d6ca30ea7ff58b98e6570fd6a01
Files:      trl/experimental/cpo/cpo_trainer.py, trl/trainer/dpo_trainer.py
Symbols:    CPOTrainer.cpo_loss (loss_type="simpo"), DPOTrainer loss_type="ipo" branch
Retrieved:  2026-09-28
License:    Apache License 2.0
            Copyright the HuggingFace Team. Licensed under the Apache License,
            Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for simpo_ipo_loss. It also documents one real divergence: TRL's IPO
divides each completion's log-ratio by its token count before forming the gap,
while the exercise follows the summed form of the IPO paper. The comparison test
feeds length-averaged log-probabilities to the exercise to show the two agree.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * Trainer state, devices and metric logging are dropped; beta, gamma and the
    token counts become parameters.
  * CPO's label smoothing and AlphaPO reward transform are dropped (both default
    to off); ``average_log_prob=True`` is represented by explicit token counts.
  * DPO's f-divergence variants and weighting are dropped; only the reverse-KL
    ``ipo`` branch remains.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["trl_ipo_loss", "trl_simpo_loss"]


def trl_simpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                   chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor,
                   beta: float, simpo_gamma: float) -> torch.Tensor:
    """Transcribed from CPOTrainer.cpo_loss with loss_type="simpo"; returns per-example losses."""
    policy_chosen_logps = policy_chosen_logps / chosen_tokens
    policy_rejected_logps = policy_rejected_logps / rejected_tokens
    logits = policy_chosen_logps - policy_rejected_logps
    gamma_logratios = simpo_gamma / beta
    logits = logits - gamma_logratios
    return -F.logsigmoid(beta * logits)


def trl_ipo_loss(chosen_logps: torch.Tensor, rejected_logps: torch.Tensor,
                 ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
                 chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor, beta: float) -> torch.Tensor:
    """Transcribed from the DPOTrainer ipo branch; returns per-sequence losses."""
    chosen_scores = chosen_logps - ref_chosen_logps
    rejected_scores = rejected_logps - ref_rejected_logps
    chosen_avg_score = chosen_scores / chosen_tokens.clamp(min=1.0)
    rejected_avg_score = rejected_scores / rejected_tokens.clamp(min=1.0)
    ipo_delta = chosen_avg_score - rejected_avg_score
    return (ipo_delta - 1 / (2 * beta)) ** 2
