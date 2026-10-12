"""Vendored Diffusion Forcing noise-level bookkeeping, plus the open-oasis sampling levels.

Upstream:   https://github.com/buoyancy99/diffusion-forcing
Commit:     475e0bcab87545e48b24b39fb46a81fe59d80594
Files:      algorithms/diffusion_forcing/df_base.py
            algorithms/diffusion_forcing/models/diffusion.py
Symbols:    DiffusionForcingBase._generate_noise_levels (df_base.py line 172),
            DiffusionForcingBase._generate_scheduling_matrix (line 188),
            DiffusionForcingBase._generate_pyramid_scheduling_matrix (line 199),
            DiffusionForcingBase._generate_trapezoid_scheduling_matrix (line 208),
            DiffusionForcingBase.validation_step, noise-level lines (lines 128-151),
            Diffusion.sample_step, level conversion (diffusion.py lines 310-314),
            Diffusion.ddim_sample_step, stabilization and update mask
            (diffusion.py lines 391-396 and 445-450)
License:    MIT License, Copyright (c) 2024 Boyuan Chen

Upstream:   https://github.com/etched-ai/open-oasis
Commit:     f59deef2c019c212bd0c5a3a5b986a51f3701847
Files:      generate.py
Symbols:    main, noise-level lines of the sampling loop (lines 50-54 and 87-106)
License:    MIT License, Copyright (c) 2024 Etched & Decart

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for diffusion_forcing_schedule, so that exercise is not checked
only against an oracle written by the same author as its reference solution.
df_base.py is the contract. open-oasis is a second, independent use of the same
level conventions (an MIT world model that cites Diffusion Forcing) and is used
as a cross-check of the context stabilization level and the step mapping.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * The methods read ``self.cfg``, ``self.timesteps``, ``self.sampling_timesteps``
    and ``self.frame_stack``; they become plain function arguments.
  * ``_generate_noise_levels`` keeps the global-RNG ``torch.randint`` call. The
    ``device=xs.device`` argument is dropped (CPU only), and ``xs`` is replaced
    by its two leading sizes. The ``einops.rearrange(masks.bool(),
    "(t fs) b -> t b fs")`` call is rewritten as ``reshape`` + ``permute``
    because einops is not installed in the grading sandbox.
  * ``_generate_scheduling_matrix`` keeps its ``match`` statement and numpy
    int64 output; it returns ``None`` for an unknown kind, as upstream does.
  * ``df_step_levels`` lifts the noise-level lines out of the validation loop
    (no batch axis: upstream only repeats the same column over the batch).
  * ``df_model_levels`` lifts the level conversion from ``sample_step`` and the
    clean-to-stabilization substitution and update mask from
    ``ddim_sample_step``; the model call, q_sample and DDIM update are dropped.
  * ``oasis_step_levels`` lifts the per-step ``t`` / ``t_next`` construction
    out of the open-oasis loop. ``torch.full`` with a float fill from
    ``noise_range`` and ``dtype=torch.long`` is kept as upstream.

UPSTREAM QUIRK THE EXERCISE DOES NOT COPY
-----------------------------------------
``_generate_trapezoid_scheduling_matrix`` fills columns ``t`` and ``-t`` for
``t < (horizon + 1) // 2``. For an even horizon the column ``horizon // 2`` is
never written and stays at level 0 (clean) in every row. The exercise leaves
the trapezoid kind out; the oracle test pins the quirk.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

__all__ = [
    "df_generate_noise_levels",
    "df_generate_scheduling_matrix",
    "df_generate_pyramid_scheduling_matrix",
    "df_generate_trapezoid_scheduling_matrix",
    "df_step_levels",
    "df_model_levels",
    "oasis_step_levels",
]


def df_generate_noise_levels(
    num_frames: int,
    batch_size: int,
    timesteps: int,
    masks: Optional[torch.Tensor] = None,
    frame_stack: int = 1,
) -> torch.Tensor:
    """df_base.py _generate_noise_levels, noise_level == "random_all"."""
    noise_levels = torch.randint(0, timesteps, (num_frames, batch_size))

    if masks is not None:
        # for frames that are not available, treat as full noise
        m = masks.bool()
        m = m.reshape(num_frames, frame_stack, m.shape[-1]).permute(0, 2, 1)  # (t fs) b -> t b fs
        discard = torch.all(~m, -1)
        noise_levels = torch.where(discard, torch.full_like(noise_levels, timesteps - 1), noise_levels)

    return noise_levels


def df_generate_scheduling_matrix(kind: str, horizon: int, sampling_timesteps: int, uncertainty_scale: float):
    """df_base.py _generate_scheduling_matrix."""
    match kind:
        case "pyramid":
            return df_generate_pyramid_scheduling_matrix(horizon, uncertainty_scale, sampling_timesteps)
        case "full_sequence":
            return np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        case "autoregressive":
            return df_generate_pyramid_scheduling_matrix(horizon, sampling_timesteps, sampling_timesteps)
        case "trapezoid":
            return df_generate_trapezoid_scheduling_matrix(horizon, uncertainty_scale, sampling_timesteps)


def df_generate_pyramid_scheduling_matrix(horizon: int, uncertainty_scale: float, sampling_timesteps: int):
    """df_base.py _generate_pyramid_scheduling_matrix."""
    height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
        for t in range(horizon):
            scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

    return np.clip(scheduling_matrix, 0, sampling_timesteps)


def df_generate_trapezoid_scheduling_matrix(horizon: int, uncertainty_scale: float, sampling_timesteps: int):
    """df_base.py _generate_trapezoid_scheduling_matrix (kept to pin its quirk)."""
    height = sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
        for t in range((horizon + 1) // 2):
            scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m
            scheduling_matrix[m, -t] = sampling_timesteps + int(t * uncertainty_scale) - m

    return np.clip(scheduling_matrix, 0, sampling_timesteps)


def df_step_levels(curr_frame: int, scheduling_matrix: np.ndarray, m: int):
    """df_base.py validation_step lines 129-139, without the batch repeat."""
    from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))
    to_noise_levels = np.concatenate(
        (
            np.zeros((curr_frame,), dtype=np.int64),
            scheduling_matrix[m + 1],
        )
    )
    return torch.from_numpy(from_noise_levels), torch.from_numpy(to_noise_levels)


def df_model_levels(
    curr_noise_level: torch.Tensor,
    next_noise_level: torch.Tensor,
    timesteps: int,
    sampling_timesteps: int,
    stabilization_level: int,
):
    """diffusion.py sample_step conversion + ddim_sample_step stabilization and update mask.

    Returns (clipped_curr_noise_level, curr_noise_level, next_noise_level, update), all in
    real diffusion steps (-1 means clean); ``update`` is True where ddim_sample_step writes
    x_pred (upstream keeps orig_x where ``mask`` is True).
    """
    real_steps = torch.linspace(-1, timesteps - 1, steps=sampling_timesteps + 1).long()

    # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
    curr_noise_level = real_steps[curr_noise_level]
    next_noise_level = real_steps[next_noise_level]

    # convert noise level -1 to self.stabilization_level - 1
    clipped_curr_noise_level = torch.where(
        curr_noise_level < 0,
        torch.full_like(curr_noise_level, stabilization_level - 1, dtype=torch.long),
        curr_noise_level,
    )

    # only update frames where the noise level decreases
    mask = curr_noise_level == next_noise_level
    return clipped_curr_noise_level, curr_noise_level, next_noise_level, ~mask


def oasis_step_levels(
    i: int,
    noise_idx: int,
    ddim_noise_steps: int,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    B: int = 1,
):
    """open-oasis generate.py: the levels fed to the model for frame i at DDIM step noise_idx."""
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long)
    t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long)
    t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long)
    t_next = torch.where(t_next < 0, t, t_next)
    t = torch.cat([t_ctx, t], dim=1)
    t_next = torch.cat([t_ctx, t_next], dim=1)
    return t, t_next
