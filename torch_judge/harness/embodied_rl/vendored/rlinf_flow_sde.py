"""Vendored Flow-SDE step, Gaussian log-density and log-prob reductions from RLinf.

Upstream:   https://github.com/RLinf/RLinf
Commit:     0e9438dee0088be7673a1fb4789018aea168db1a
Files:      rlinf/models/embodiment/openpi/sampling/rl_sampler.py,
            rlinf/models/embodiment/openpi/tasks/rl.py,
            rlinf/algorithms/utils.py
Symbols:    get_timesteps, sample_mean_var, gaussian_logprob (rl_sampler.py);
            Pi0RL.default_forward, the log-prob recompute (rl.py);
            preprocess_loss_inputs, the logprob_type branches (utils.py)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2025-2026 The RLinf Authors. Licensed under the Apache
            License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for flow_sde_chain_logprob, so that exercise is not checked only
against an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``get_timesteps``, ``sample_mean_var`` and ``gaussian_logprob`` are copied
    unchanged apart from the ``rlinf_`` name prefix. All four noise methods are
    kept, though the exercise only uses ``flow_ode`` and ``flow_sde``.
  * ``rlinf_recompute_logprobs`` is the log-prob half of
    ``Pi0RL.default_forward``. The pi0 network (prefix KV cache, ``run_suffix``,
    ``velocity_from_suffix``) is replaced by a ``velocity_fn(x, t)`` callable;
    the noise head, value head and entropy are dropped. ``rl_cfg`` fields
    become arguments, and ``self.action_chunk`` / ``self.action_env_dim``
    become ``action_chunk`` / ``action_env_dim``. Upstream casts the chain to
    float32 before each step; this transcription keeps the chain's dtype so a
    float64 comparison is possible.
  * ``rlinf_reduce_logprobs`` keeps only the ``token_level``, ``action_level``
    and ``chunk_level`` reshapes of ``logprobs`` from ``preprocess_loss_inputs``.
    ``old_logprobs``, advantages, masks, values, versions and the
    ``sequence_token_level`` and OPD branches are dropped; ``single_action_dim``
    keeps its upstream name.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

__all__ = [
    "rlinf_gaussian_logprob",
    "rlinf_get_timesteps",
    "rlinf_recompute_logprobs",
    "rlinf_reduce_logprobs",
    "rlinf_sample_mean_var",
]


def rlinf_get_timesteps(num_steps: int, device: torch.device | str) -> torch.Tensor:
    """Transcribed from rl_sampler.py::get_timesteps: ``[1, (N-1)/N, ..., 1/N, 0]``."""
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device)
    return torch.cat([timesteps, torch.zeros(1, device=device)])


def rlinf_sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    idx: torch.Tensor,
    *,
    noise_method: str,
    noise_level: float,
    num_steps: int,
    noise_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribed from rl_sampler.py::sample_mean_var; returns ``(x_t_mean, x_t_std)``."""
    device = x_t.device
    timesteps = rlinf_get_timesteps(num_steps, device).to(x_t.dtype)
    t_input = timesteps[idx][:, None, None].expand_as(x_t)
    delta = (timesteps[idx] - timesteps[idx + 1])[:, None, None].expand_as(x_t)
    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1 - t_input)

    if noise_method == "flow_ode":
        x0_weight = 1 - (t_input - delta)
        x1_weight = t_input - delta
        x_t_std = torch.zeros_like(t_input)
    elif noise_method == "flow_sde":
        denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1 - denom_timesteps)
        sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
        sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
        x0_weight = torch.ones_like(t_input) - (t_input - delta)
        x1_weight = (t_input - delta) - sigma_i * sigma_i * delta / (2 * t_input)
        x_t_std = torch.sqrt(delta) * sigma_i
    elif noise_method == "flow_cps":
        pi = torch.pi
        level = torch.as_tensor(noise_level, device=device, dtype=x_t.dtype)
        cos_term = torch.cos(pi * level / 2)
        sin_term = torch.sin(pi * level / 2)
        x0_weight = torch.ones_like(t_input) - (t_input - delta)
        x1_weight = (t_input - delta) * cos_term
        x_t_std = (t_input - delta) * sin_term
    elif noise_method == "flow_noise":
        if noise_std is None:
            raise ValueError(
                "noise_method='flow_noise' requires noise_std from ExploreNoiseNet."
            )
        x0_weight = 1 - (t_input - delta)
        x1_weight = t_input - delta
        x_t_std = noise_std.to(dtype=x_t.dtype)
    else:
        raise NotImplementedError(
            f"noise_method={noise_method!r} is not implemented. "
            "Supported: 'flow_ode', 'flow_sde', 'flow_cps', 'flow_noise'."
        )

    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    return x_t_mean, x_t_std


def rlinf_gaussian_logprob(sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Transcribed from rl_sampler.py::gaussian_logprob; zero wherever ``sigma == 0``."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    log_two_pi = math.log(2 * math.pi)
    log_prob = (
        -torch.log(sigma_safe)
        - 0.5 * log_two_pi
        - 0.5 * ((sample - mu) / sigma_safe) ** 2
    )
    return torch.where(mask, torch.zeros_like(log_prob), log_prob)


def rlinf_recompute_logprobs(
    chains: torch.Tensor,
    denoise_inds: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_steps: int,
    noise_method: str,
    noise_level: float,
    joint_logprob: bool,
    action_chunk: int,
    action_env_dim: int,
) -> torch.Tensor:
    """Log-prob half of rl.py::Pi0RL.default_forward.

    ``chains`` is ``[B, num_steps + 1, H, D]``; ``denoise_inds`` is ``[B, num_steps]``
    (``arange(num_steps)`` per row in joint mode, the chosen SDE index repeated
    otherwise). Returns ``[B, action_chunk, action_env_dim]``.
    """
    B = chains.shape[0]
    device = chains.device
    dtype = chains.dtype

    timesteps = rlinf_get_timesteps(num_steps, device)
    if joint_logprob:
        num_recompute = num_steps
        initial = rlinf_gaussian_logprob(
            chains[:, 0],
            torch.zeros_like(chains[:, 0]),
            torch.ones_like(chains[:, 0]),
        )
        step_logprobs = [initial]
    else:
        num_recompute = 1
        step_logprobs = []

    arange_B = torch.arange(B, device=device)
    for idx in range(num_recompute):
        denoise_ind = denoise_inds[:, idx].to(torch.long)
        chains_pre = chains[arange_B, denoise_ind]
        chains_next = chains[arange_B, denoise_ind + 1]
        t_input = timesteps[denoise_ind].to(dtype)
        v_t = velocity_fn(chains_pre, t_input)
        x_t_mean, x_t_std = rlinf_sample_mean_var(
            chains_pre,
            v_t,
            denoise_ind,
            noise_method=noise_method,
            noise_level=noise_level,
            num_steps=num_steps,
        )
        step_logprobs.append(rlinf_gaussian_logprob(chains_next, x_t_mean, x_t_std))

    log_probs = torch.stack(step_logprobs, dim=1)
    log_probs = log_probs[:, :, :action_chunk, :action_env_dim]
    if joint_logprob:
        log_probs = log_probs.mean(dim=1)
    else:
        log_probs = log_probs[:, 0]
    return log_probs


def rlinf_reduce_logprobs(logprobs: torch.Tensor, logprob_type: str, single_action_dim: int) -> torch.Tensor:
    """The ``logprobs`` reshapes of utils.py::preprocess_loss_inputs."""
    bsz = logprobs.shape[0]
    if logprob_type == "token_level":
        # logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
    elif logprob_type == "action_level":
        # logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
    elif logprob_type == "chunk_level":
        # logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
    return logprobs
