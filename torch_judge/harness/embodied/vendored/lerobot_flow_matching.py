"""Vendored flow-matching training-input and Euler-sampling oracles from LeRobot.

Upstream:   https://github.com/huggingface/lerobot
Commit:     75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee
Files:      src/lerobot/policies/common/flow_matching.py
Symbols:    FlowConvention, make_flow_matching_inputs, euler_integrate
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2026 The HuggingFace Inc. team. All rights reserved.
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

Cross-check: https://github.com/Physical-Intelligence/openpi
Commit:     215abfb217dbac7d5f1273282331b9b1866c0479
Files:      src/openpi/models/pi0.py (JAX)
Symbols:    Pi0.compute_loss, Pi0.sample_actions
License:    Apache License 2.0 (repo LICENSE; nothing Gemma-derived is used here)

The openpi original agrees with LeRobot's NOISE_AT_ONE branch:
``time = beta(1.5, 1) * 0.999 + 0.001``, ``x_t = time * noise + (1 - time) * actions``,
``u_t = noise - actions`` in ``compute_loss``; ``dt = -1.0 / num_steps`` and a
``jax.lax.while_loop`` from ``(noise, 1.0)`` that steps ``x_t + dt * v_t, time + dt`` while
``time >= -dt / 2`` in ``sample_actions``. That loop is transcribed below as
``openpi_euler_times`` (JAX to Python floats) so the oracle test can pin that both
schedules take exactly ``num_steps`` steps at the same times.

WHY THIS FILE EXISTS
--------------------
Ground truth for pi0_flow_matching_chunk, so that exercise is not checked only
against an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``euler_integrate`` drops the real-time-chunking guidance hook (``rtc_processor``,
    ``rtc_enabled``, ``inference_delay``, ``prev_chunk_left_over``, ``execution_horizon``)
    and debug tracking, ``step_aware``, ``time_grid``, ``velocity_scale`` and
    ``precompute_times``. With those at their defaults the loop below is the upstream
    loop: uniform schedule, clamp before the velocity call, Euler update, clamp after.
  * ``sample_beta``, ``sample_noise`` and ``sample_time_beta`` are not transcribed: the
    exercise receives ``time`` and ``noise`` as inputs, so it has no RNG.
  * ``openpi_euler_times`` replaces the JAX ``while_loop`` with a Python loop over floats;
    it returns only the times at which the velocity is evaluated.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import enum
from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["FlowConvention", "lerobot_euler_integrate", "lerobot_make_flow_matching_inputs", "openpi_euler_times"]


class FlowConvention(str, enum.Enum):
    """Which end of the ``[0, 1]`` schedule holds the noise (flow_matching.py line 49)."""

    NOISE_AT_ONE = "noise_at_one"
    NOISE_AT_ZERO = "noise_at_zero"


def lerobot_make_flow_matching_inputs(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    *,
    convention: FlowConvention,
    prefix_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Transcribed from flow_matching.py::make_flow_matching_inputs (line 188)."""
    if not isinstance(convention, FlowConvention):
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
    if time.shape != actions.shape[:1]:
        raise ValueError(f"time must have shape (batch,), got {tuple(time.shape)}")
    if prefix_mask is not None and prefix_mask.shape != actions.shape[:2]:
        raise ValueError(f"prefix_mask must have shape (batch, horizon), got {tuple(prefix_mask.shape)}")
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    if prefix_mask is None:
        model_time = time
        expanded_time = time[:, None, None]
    else:
        # Clean tokens sit at the far end of the schedule from the noise.
        clean_time = 0.0 if noise_at_one else 1.0
        model_time = time[:, None].expand_as(prefix_mask)
        model_time = torch.where(prefix_mask, torch.full_like(model_time, clean_time), model_time)
        expanded_time = model_time.unsqueeze(-1)

    if noise_at_one:
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        velocity_target = noise - actions
    else:
        x_t = (1 - expanded_time) * noise + expanded_time * actions
        velocity_target = actions - noise

    return x_t, velocity_target, model_time


def lerobot_euler_integrate(
    denoise_fn: Callable[..., Tensor],
    noise: Tensor,
    num_steps: int,
    *,
    convention: FlowConvention = FlowConvention.NOISE_AT_ONE,
    hard_prefix: Tensor | None = None,
    hard_prefix_mask: Tensor | None = None,
) -> Tensor:
    """Transcribed from flow_matching.py::euler_integrate (line 256), uniform schedule, no RTC.

    ``hard_prefix_mask`` is shaped like ``noise`` (upstream indexes ``[..., 0]`` for the time).
    """
    bsize = noise.shape[0]
    device = noise.device
    if not isinstance(convention, FlowConvention):
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    # Clean tokens sit at the far end of the schedule from the noise.
    clean_time = 0.0 if noise_at_one else 1.0
    uniform_dt = (-1.0 if noise_at_one else 1.0) / num_steps

    def uniform_time(step: int) -> float:
        return (1.0 + step * uniform_dt) if noise_at_one else step * uniform_dt

    x_t = noise
    for step in range(num_steps):
        dt = uniform_dt
        time_tensor = torch.full((bsize,), uniform_time(step), dtype=torch.float32, device=device)

        if hard_prefix is not None:
            if hard_prefix_mask is None:
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            time_tensor[hard_prefix_mask[..., 0]] = clean_time

        v_t = denoise_fn(x_t, time_tensor)

        update = dt * v_t
        x_t = x_t + update

        if hard_prefix is not None:
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)

    return x_t


def openpi_euler_times(num_steps: int) -> list[float]:
    """The times at which openpi's pi0.py::Pi0.sample_actions evaluates the velocity."""
    dt = -1.0 / num_steps
    times = []
    time = 1.0
    while time >= -dt / 2:  # cond
        times.append(time)
        time = time + dt  # step returns (x_t + dt * v_t, time + dt)
    return times
