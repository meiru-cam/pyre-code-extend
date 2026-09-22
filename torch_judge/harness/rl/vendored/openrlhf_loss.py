"""Vendored loss oracles transcribed from OpenRLHF.

Upstream:   https://github.com/OpenRLHF/OpenRLHF
File:       openrlhf/models/loss.py
Symbols:    PolicyLoss.forward, ValueLoss.forward, aggregate_loss, masked_mean
Retrieved:  2026-09-13
License:    Apache License 2.0
            Copyright the OpenRLHF authors. Licensed under the Apache License,
            Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
These functions are ground truth for the PPO-family exercises. The exercises are
graded against an oracle, and an oracle written by the same author as the
reference solution can be wrong in the same way twice. Transcribing the upstream
formulas gives an independent check.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Everything removed is infrastructure
that cannot run in the offline grading sandbox (torch and numpy only):

  * ``nn.Module`` wrappers became plain functions; the classes only stored
    constructor arguments that are now parameters.
  * The vLLM importance-sampling correction branch (``enable_vllm_is_correction``
    and its three ``vllm_is_correction_type`` modes) is dropped. It needs rollout
    log-probabilities from a vLLM engine.
  * ``dp_size``, ``batch_num_tokens`` and ``global_batch_size`` are dropped. They
    exist to rescale a loss across data-parallel ranks; single-process grading
    always takes the ``None`` path, which is plain masked-mean.
  * The auxiliary returns of ``PolicyLoss.forward`` (``clip_ratio``, ``ppo_kl``,
    ``vllm_kl``) are kept, since they are cheap and make a disagreement easier to
    localize.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch

__all__ = [
    "OPENRLHF_LOG_RATIO_CLAMP",
    "aggregate_loss",
    "masked_mean",
    "openrlhf_policy_loss",
    "openrlhf_value_loss",
    "openrlhf_gspo_ratio",
]

# PolicyLoss clamps the log-ratio to this band before exponentiating. It is a
# defensive bound against overflow, not part of the published PPO objective; the
# `ppo_clipped_policy_loss` exercise deliberately omits it. See that task's
# description for the reasoning. Oracle comparisons must stay inside this band.
OPENRLHF_LOG_RATIO_CLAMP = 20.0


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Transcribed from openrlhf/models/utils.py."""
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def aggregate_loss(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    token_level_loss: bool = True,
    batch_num_tokens: float | None = None,
) -> torch.Tensor:
    """Transcribed from openrlhf/models/loss.py::aggregate_loss.

    The ``dp_size`` and ``global_batch_size`` rescaling is dropped; both are 1 or
    ``None`` in a single-process run.
    """
    if token_level_loss:
        if batch_num_tokens is None:
            return masked_mean(loss, loss_mask, dim=None)
        return (loss * loss_mask).sum() / batch_num_tokens

    token_counts = loss_mask.sum(dim=-1)
    seq_loss = (loss * loss_mask).sum(dim=-1) / (token_counts + 1e-8)
    seq_mask = (token_counts > 0).float()
    return masked_mean(seq_loss, seq_mask, dim=None)


def openrlhf_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.2,
    dual_clip: float | None = None,
    token_level_loss: bool = True,
    apply_log_ratio_clamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transcribed from openrlhf/models/loss.py::PolicyLoss.forward, ppo branch.

    Returns ``(loss, clip_ratio, ppo_kl)``. ``apply_log_ratio_clamp`` is the one
    added knob: it exists so an exercise that deliberately omits the clamp can
    still be compared against this oracle outside the clamp band.
    """
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError(f"dual_clip must be > 1.0, got {dual_clip}")

    mask = action_mask.to(log_probs.dtype)
    raw_policy_log_ratio = log_probs - old_log_probs
    if apply_log_ratio_clamp:
        policy_log_ratio = raw_policy_log_ratio.clamp(
            min=-OPENRLHF_LOG_RATIO_CLAMP, max=OPENRLHF_LOG_RATIO_CLAMP
        )
    else:
        policy_log_ratio = raw_policy_log_ratio
    ratio = policy_log_ratio.exp()

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages

    if dual_clip is None:
        loss = -torch.min(surr1, surr2)
    else:
        clip1 = torch.min(surr1, surr2)
        clip2 = torch.max(clip1, dual_clip * advantages)
        loss = -torch.where(advantages < 0, clip2, clip1)

    loss = aggregate_loss(loss, mask, token_level_loss=token_level_loss)
    clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), mask, dim=None)
    ppo_kl = masked_mean(-raw_policy_log_ratio.detach(), mask, dim=None)
    return loss, clip_ratio, ppo_kl


def openrlhf_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float | None = 0.2,
    token_level_loss: bool = True,
) -> torch.Tensor:
    """Transcribed from openrlhf/models/loss.py::ValueLoss.forward."""
    mask = action_mask.to(values.dtype)
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2

    loss = aggregate_loss(loss, mask, token_level_loss=token_level_loss)
    return 0.5 * loss


def openrlhf_gspo_ratio(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Transcribed from openrlhf/models/loss.py::PolicyLoss.forward, gspo branch.

    Upstream immediately broadcasts the per-sequence ratio back over the mask
    (``ratio.exp().unsqueeze(-1) * action_mask``) because it feeds a per-token
    surrogate. The `gspo_sequence_ratio` exercise stops at the per-sequence
    value, so the broadcast is left to the caller.
    """
    mask = action_mask.to(log_probs.dtype)
    log_ratio = log_probs - old_log_probs
    ratio = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return ratio.exp()
