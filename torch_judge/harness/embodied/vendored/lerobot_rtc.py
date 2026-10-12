"""Vendored Real-Time Chunking (RTC) prefix weights and guided denoising step from LeRobot.

Upstream:   https://github.com/huggingface/lerobot
Commit:     75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee
Files:      src/lerobot/policies/rtc/modeling_rtc.py
Symbols:    RTCProcessor.get_prefix_weights (line 256), RTCProcessor._linweights,
            RTCProcessor._add_trailing_zeros, RTCProcessor._add_leading_ones,
            RTCProcessor.denoise_step (line 122)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2025 The HuggingFace Inc. team. All rights reserved. Licensed
            under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

The upstream module says it is based on Physical Intelligence's Kinetix code,
https://github.com/Physical-Intelligence/real-time-chunking-kinetix (MIT), whose
``src/model.py::get_prefix_weights`` and ``realtime_action`` were read at commit
9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b as a cross-check. Kinetix is not
transcribed here; its differences are pinned in
``tests/quality/test_rtc_prefix_guidance_oracle.py``.

WHY THIS FILE EXISTS
--------------------
Ground truth for the rtc_prefix_guidance exercise, so its reference solution is
not checked only against an oracle written by the same author.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``RTCProcessor`` and ``RTCConfig`` are dropped. The config fields the
    arithmetic reads (``prefix_attention_schedule``, ``max_guidance_weight``,
    ``execution_horizon``) become explicit arguments, so ``execution_horizon`` no
    longer falls back to a config default.
  * The schedule is a lowercase string (``"zeros"``, ``"ones"``, ``"linear"``,
    ``"exp"``) instead of the ``RTCAttentionSchedule`` enum (``"ZEROS"`` ...).
    An unknown schedule raises ``ValueError``; upstream falls through and raises
    ``UnboundLocalError`` on ``weights``.
  * The debug ``Tracker`` and its ``track`` call are dropped; they only record
    intermediate tensors.
  * ``.to(x_t.device)`` calls are kept; the oracle runs on CPU.

Upstream behavior deliberately kept, because it is what the oracle documents:

  * ``original_denoise_step_partial(x_t)`` runs *before* ``x_t.requires_grad_``,
    so the velocity carries no graph back to ``x_t`` and the vector-Jacobian
    product reduces to the weighted error itself.
  * The zero padding is a float32 tensor whatever the dtype of ``x_t``, and it
    pads a narrower action dimension as well as a shorter chunk.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["lerobot_denoise_step", "lerobot_get_prefix_weights"]


def _linweights(start, end, total):
    skip_steps_at_end = max(total - end, 0)

    linspace_steps = total - skip_steps_at_end - start

    if end <= start or linspace_steps <= 0:
        return torch.tensor([])

    return torch.linspace(1, 0, linspace_steps + 2)[1:-1]


def _add_trailing_zeros(weights, total, end):
    zeros_len = total - end

    if zeros_len <= 0:
        return weights

    zeros = torch.zeros(zeros_len)
    return torch.cat([weights, zeros])


def _add_leading_ones(weights, start, total):
    ones_len = min(start, total)

    if ones_len <= 0:
        return weights

    ones = torch.ones(ones_len)
    return torch.cat([ones, weights])


def lerobot_get_prefix_weights(start, end, total, schedule):
    """Transcribed from RTCProcessor.get_prefix_weights."""
    start = min(start, end)

    if schedule == "zeros":
        weights = torch.zeros(total)
        weights[:start] = 1.0
    elif schedule == "ones":
        weights = torch.ones(total)
        weights[end:] = 0.0
    elif schedule == "linear":
        lin_weights = _linweights(start, end, total)
        weights = _add_trailing_zeros(lin_weights, total, end)
        weights = _add_leading_ones(weights, start, total)
    elif schedule == "exp":
        lin_weights = _linweights(start, end, total)
        lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
        weights = _add_trailing_zeros(lin_weights, total, end)
        weights = _add_leading_ones(weights, start, total)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    return weights


def lerobot_denoise_step(
    x_t,
    prev_chunk_left_over,
    inference_delay,
    time,
    original_denoise_step_partial,
    execution_horizon,
    max_guidance_weight,
    schedule,
) -> Tensor:
    """Transcribed from RTCProcessor.denoise_step; returns the guided velocity."""
    # In the original implementation, the time goes from 0 to 1 and
    # In our implementation, the time goes from 1 to 0
    # So we need to invert the time
    tau = 1 - time

    if prev_chunk_left_over is None:
        # First step, no guidance - return v_t
        v_t = original_denoise_step_partial(x_t)
        return v_t

    x_t = x_t.clone().detach()

    squeezed = False
    if len(x_t.shape) < 3:
        # Add batch dimension
        x_t = x_t.unsqueeze(0)
        squeezed = True

    if len(prev_chunk_left_over.shape) < 3:
        # Add batch dimension
        prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

    # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
    # because there is nothing to merge
    if execution_horizon > prev_chunk_left_over.shape[1]:
        execution_horizon = prev_chunk_left_over.shape[1]

    batch_size = x_t.shape[0]
    action_chunk_size = x_t.shape[1]
    action_dim = x_t.shape[2]

    if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
        padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
        padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
        prev_chunk_left_over = padded

    assert prev_chunk_left_over.shape == x_t.shape, (
        "The padded previous chunk must be the same size as the input tensor"
    )

    weights = (
        lerobot_get_prefix_weights(inference_delay, execution_horizon, action_chunk_size, schedule)
        .to(x_t.device)
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    with torch.enable_grad():
        v_t = original_denoise_step_partial(x_t)
        x_t.requires_grad_(True)

        x1_t = x_t - time * v_t  # noqa: N806
        err = (prev_chunk_left_over - x1_t) * weights
        grad_outputs = err.clone().detach()
        correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

    max_guidance_weight = torch.as_tensor(max_guidance_weight)
    tau_tensor = torch.as_tensor(tau)
    squared_one_minus_tau = (1 - tau_tensor) ** 2
    inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
    c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
    guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

    result = v_t - guidance_weight * correction

    # Remove the batch dimension if it was added
    if squeezed:
        result = result.squeeze(0)

    return result
