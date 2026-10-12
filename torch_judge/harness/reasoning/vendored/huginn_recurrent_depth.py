"""Vendored recurrent-depth unroll and iteration sampler from Huginn (recurrent-pretraining).

Upstream:   https://github.com/seal-rg/recurrent-pretraining
Commit:     1ea7220ec7eb42d13e89db0663df254d0bcdc28e
Files:      recpre/raven_modeling_minimal.py
Symbols:    RavenForCausalLM.iterate_forward, RavenForCausalLM.core_block_forward,
            RavenForCausalLM.randomized_iteration_sampler, RavenForCausalLM.initialize_state
Retrieved:  2026-10-12
License:    Apache License 2.0
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for recurrent_depth_unroll, so its no-grad/grad unroll and its
iteration split are not checked only against code written by the same author as
the reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic and control flow are transcribed line for line. The methods become
free functions, and the model attributes they read become arguments:

  * ``self.transformer.adapter``, ``self.transformer.core_block`` (a list of
    blocks) and ``self.transformer.ln_f`` are passed in as callables.
  * ``self.config.mean_recurrence``, ``self.config.mean_backprop_depth``,
    ``self.config.init_values["std"]``, ``self.emb_scale`` and ``self.training``
    are passed in as values.
  * RoPE frequencies, attention masks, the KV cache and ``block_idx`` are dropped;
    they only feed the transformer blocks, which are opaque callables here.
  * ``_maybe_inject_noise`` is dropped: ``test_time_noise`` defaults to 0 in
    ``RavenConfig``, which makes it the identity.
  * ``_maybe_checkpoint_core_block`` calls ``core_block_forward`` directly.
    Activation checkpointing changes memory, not values or gradients.
  * The meta-tensor branch of ``randomized_iteration_sampler`` (FLOP counting) is
    dropped.
  * The two lines of ``randomized_iteration_sampler`` that split the sampled total
    ``p`` into ``(n, k)`` are factored into ``huginn_split_total`` so they can be
    checked for any ``p``; the sampler calls it unchanged.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

__all__ = [
    "huginn_core_block_forward",
    "huginn_initialize_state",
    "huginn_iterate_forward",
    "huginn_randomized_iteration_sampler",
    "huginn_split_total",
]


def huginn_core_block_forward(
    x: torch.Tensor,
    input_embeds: torch.Tensor,
    adapter: Callable[[torch.Tensor], torch.Tensor],
    core_block: Sequence[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Transcribed from core_block_forward (noise injection is the identity at the default config)."""
    x = adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
    for block in core_block:
        x = block(x)
    return x


def huginn_iterate_forward(
    input_embeds: torch.Tensor,
    input_states: torch.Tensor | None,
    adapter: Callable[[torch.Tensor], torch.Tensor],
    core_block: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ln_f: Callable[[torch.Tensor], torch.Tensor],
    num_steps=None,
    *,
    mean_recurrence: int = 32,
    mean_backprop_depth: int = 8,
    training: bool = True,
    init_std: float = 1.0,
    init_scale: float = 1.0,
    emb_scale: float = 1.0,
):
    """Transcribed from iterate_forward; returns (ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach())."""
    x = xk = (
        huginn_initialize_state(input_embeds, init_std, scale=init_scale, emb_scale=emb_scale)
        if input_states is None
        else input_states.clone()
    )
    if num_steps is None:
        num_steps_no_grad, num_steps_with_grad = huginn_randomized_iteration_sampler(
            mean_recurrence, mean_backprop_depth, training
        )
    elif hasattr(num_steps, "__len__") and len(num_steps) > 1:
        num_steps_no_grad, num_steps_with_grad = num_steps
    else:
        num_steps_no_grad, num_steps_with_grad = num_steps, torch.tensor(0)

    with torch.no_grad():
        for _no_grad_step in range(num_steps_no_grad):
            xk = x
            x = huginn_core_block_forward(xk, input_embeds, adapter, core_block)

    for _grad_step in range(num_steps_with_grad):
        xk = x
        x = huginn_core_block_forward(xk, input_embeds, adapter, core_block)
    return ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach()


def huginn_split_total(p: torch.Tensor, s) -> tuple[torch.Tensor, torch.Tensor]:
    """The two split lines of randomized_iteration_sampler, for a sampled total p."""
    n = torch.clamp(p - s, min=0)
    k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
    return n, k


def huginn_randomized_iteration_sampler(
    mean_recurrence: int, mean_backprop_depth: int, training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribed from randomized_iteration_sampler; draws from the global torch RNG."""
    t = max(mean_recurrence - mean_backprop_depth, 0)
    s = mean_backprop_depth
    if training:
        sigma = 0.5
        mu = math.log(t + s) - (sigma**2 / 2)
        rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma)
        p = torch.poisson(torch.tensor([rate], dtype=torch.float)) + 1
        n, k = huginn_split_total(p, s)
    else:
        n, k = torch.as_tensor(mean_recurrence), torch.as_tensor(0)

    return n.to(dtype=torch.long), k.to(dtype=torch.long)


def huginn_initialize_state(
    input_embeds: torch.Tensor, init_std: float, scale: float = 1.0, emb_scale: float = 1.0
) -> torch.Tensor:
    """Transcribed from initialize_state; init_std is config.init_values['std']."""
    x = torch.randn_like(input_embeds)
    std = init_std * scale
    if std > 0:
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if emb_scale != 1:
            x = x * emb_scale
    else:
        x.zero_()
    return x
