"""Vendored DreamerV3 RSSM KL oracles: the JAX original (in torch) and the PyTorch port.

Upstream:   https://github.com/danijar/dreamerv3
Commit:     e01491fad6434b2245a3b8ca201dd7faedcc458c
Files:      dreamerv3/rssm.py, embodied/jax/outs.py, dreamerv3/agent.py,
            dreamerv3/configs.yaml
Symbols:    RSSM.loss (lines 120-131), RSSM._dist (line 173),
            outs.Categorical (line 208), outs.OneHot, outs.Agg.kl,
            Agent.loss total (agent.py line 240), loss_scales and
            rssm.free_nats / rssm.unimix (configs.yaml lines 86 and 91)
License:    MIT License, Copyright (c) 2023 Danijar Hafner

Upstream:   https://github.com/NM512/dreamerv3-torch
Commit:     6ef8646d807cd10ce0c88e10a7e943211e7fc44c
Files:      networks.py, tools.py, configs.yaml
Symbols:    RSSM.kl_loss (networks.py line 272), RSSM.get_dist discrete
            branch (networks.py line 161), OneHotDist.__init__ (tools.py
            line 425), dyn_scale / rep_scale / kl_free / unimix_ratio
            (configs.yaml lines 57-61)
License:    MIT License, Copyright (c) 2023 NM512

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for rssm_kl_balance, so that exercise is not checked only against
an oracle written by the same author as its reference solution. The JAX
original is the contract. The PyTorch port is kept as a cross-check and to pin
where it differs.

WHAT WAS CHANGED, AND WHY
-------------------------
JAX original (``danijar_*`` functions):

  * ``jax.nn.softmax`` / ``log_softmax`` / ``jnp.log`` become their torch
    equivalents, ``jax.lax.stop_gradient`` becomes ``.detach()`` and
    ``jnp.maximum`` becomes ``torch.maximum`` (both split the gradient in half
    at an exact tie). The arithmetic order is kept.
  * ``Categorical.__init__`` casts the logits to float32 (``f32(logits)``).
    The oracle keeps the caller's dtype instead, so float64 comparisons are
    exact; pass float32 inputs to reproduce upstream precision.
  * The ``OneHot`` -> ``Categorical`` and ``Agg(..., 1, jnp.sum)`` wrapper
    classes are flattened into ``danijar_categorical_logits`` (the unimix
    step), ``danijar_categorical_kl`` (``Categorical.kl``) and a ``.sum(-1)``
    over the stoch groups (``Agg.kl``).
  * ``RSSM.loss`` receives the posterior and prior logits directly instead of
    running ``observe`` and ``_prior``; entropy metrics are dropped.
    ``self.free_nats`` and ``self.unimix`` become arguments.
  * The total in ``Agent.loss`` is ``sum(v.mean() * scales[k])`` over every
    loss; only the ``dyn`` and ``rep`` terms are kept, with their scales as
    arguments (upstream defaults dyn 1.0, rep 0.1).

PyTorch port (``nm512_kl_loss``):

  * ``OneHotDist.__init__`` and the discrete branch of ``get_dist`` are kept
    as upstream, built on ``torch.distributions``. Sampling and ``mode`` are
    dropped; the KL never calls them.
  * ``kl_loss`` takes logits instead of state dicts (the dicts only carry
    ``logit`` in the discrete case) and is otherwise unchanged. It returns the
    per-(B, T) loss, as upstream; the caller there takes the mean.

DIVERGENCES BETWEEN THE TWO UPSTREAMS
-------------------------------------
  * Default dyn scale: 1.0 in danijar ``configs.yaml``, 0.5 in NM512
    ``configs.yaml`` (the DreamerV3 paper changed it from 0.5 to 1 between
    arXiv versions). The exercise takes both betas as parameters.
  * Free bits: danijar uses ``jnp.maximum`` (half the gradient at an exact
    tie) and skips it when ``free_nats`` is 0; NM512 uses ``torch.clip``
    (the full gradient at a tie) and always applies it.
  * Unimix gate: danijar mixes when ``unimix`` is truthy, NM512 when
    ``unimix_ratio > 0``. Equivalent for the non-negative values used here.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
import torch.distributions as torchd
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "danijar_categorical_logits",
    "danijar_categorical_kl",
    "danijar_rssm_loss",
    "NM512OneHotDist",
    "nm512_kl_loss",
]


# ---------------------------------------------------------------- danijar/dreamerv3 (JAX)

def danijar_categorical_logits(logits: Tensor, unimix: float = 0.0) -> Tensor:
    """outs.Categorical.__init__."""
    if unimix:
        probs = torch.softmax(logits, -1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - unimix) * probs + unimix * uniform
        logits = torch.log(probs)
    return logits


def danijar_categorical_kl(self_logits: Tensor, other_logits: Tensor) -> Tensor:
    """outs.Categorical.kl."""
    logprob = torch.log_softmax(self_logits, -1)
    logother = torch.log_softmax(other_logits, -1)
    prob = torch.softmax(self_logits, -1)
    return (prob * (logprob - logother)).sum(-1)


def _dist(logits: Tensor, unimix: float) -> Tensor:
    """RSSM._dist: OneHot(logits, unimix) wrapped in Agg(out, 1, jnp.sum)."""
    return danijar_categorical_logits(logits, unimix)


def _agg_kl(self_logits: Tensor, other_logits: Tensor) -> Tensor:
    """Agg.kl: the per-group KL summed over the stoch axis."""
    return danijar_categorical_kl(self_logits, other_logits).sum(-1)


def danijar_rssm_loss(
    post: Tensor,
    prior: Tensor,
    free_nats: float = 1.0,
    unimix: float = 0.01,
    dyn_scale: float = 1.0,
    rep_scale: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """RSSM.loss plus the dyn/rep part of Agent.loss. Returns (total, dyn, rep)."""
    sg = lambda x: x.detach()
    dyn = _agg_kl(_dist(sg(post), unimix), _dist(prior, unimix))
    rep = _agg_kl(_dist(post, unimix), _dist(sg(prior), unimix))
    if free_nats:
        floor = torch.tensor(free_nats, dtype=dyn.dtype, device=dyn.device)
        dyn = torch.maximum(dyn, floor)
        rep = torch.maximum(rep, floor)
    losses = {"dyn": dyn, "rep": rep}
    scales = {"dyn": dyn_scale, "rep": rep_scale}
    loss = sum([v.mean() * scales[k] for k, v in losses.items()])
    return loss, dyn, rep


# ---------------------------------------------------------------- NM512/dreamerv3-torch

class NM512OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)


def _nm512_get_dist(logit: Tensor, unimix_ratio: float):
    return torchd.independent.Independent(NM512OneHotDist(logit, unimix_ratio=unimix_ratio), 1)


def nm512_kl_loss(
    post: Tensor,
    prior: Tensor,
    free: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    unimix_ratio: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """RSSM.kl_loss, discrete branch. Returns (loss, value, dyn_loss, rep_loss), all per (B, T)."""
    kld = torchd.kl.kl_divergence
    dist = lambda x: _nm512_get_dist(x, unimix_ratio)
    sg = lambda x: x.detach()

    rep_loss = value = kld(dist(post), dist(sg(prior)))
    dyn_loss = kld(dist(sg(post)), dist(prior))
    # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
    rep_loss = torch.clip(rep_loss, min=free)
    dyn_loss = torch.clip(dyn_loss, min=free)
    loss = dyn_scale * dyn_loss + rep_scale * rep_loss

    return loss, value, dyn_loss, rep_loss
