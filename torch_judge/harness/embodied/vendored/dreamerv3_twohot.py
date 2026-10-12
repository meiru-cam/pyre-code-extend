"""Vendored DreamerV3 symlog, symexp, two-hot bins, two-hot loss and two-hot mean.

Upstream:   https://github.com/danijar/dreamerv3
Commit:     e01491fad6434b2245a3b8ca201dd7faedcc458c
Files:      embodied/jax/nets.py, embodied/jax/outs.py, embodied/jax/heads.py (JAX)
Symbols:    nets.symlog, nets.symexp, outs.TwoHot (pred, loss), heads.Head.symexp_twohot (bins)
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2024 Danijar Hafner
            Permission is hereby granted, free of charge, to any person obtaining a copy of
            this software and associated documentation files, to deal in the Software
            without restriction, subject to the conditions of the MIT License.

Cross-check: https://github.com/NM512/dreamerv3-torch
Commit:     6ef8646d807cd10ce0c88e10a7e943211e7fc44c
Files:      tools.py (PyTorch)
Symbols:    symlog, symexp, DiscDist (mean, log_prob)
License:    MIT License, Copyright (c) 2023 NM512

WHY THIS FILE EXISTS
--------------------
Ground truth for symlog_twohot, so that exercise is not checked only against an oracle
written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The danijar arithmetic is transcribed line for line from JAX to torch:

  * ``jnp`` calls become their torch counterparts (``jnp.log1p`` -> ``torch.log1p``,
    ``jax.nn.one_hot`` -> ``F.one_hot``, ``jax.scipy.special.logsumexp`` ->
    ``torch.logsumexp``, ``[..., ::-1]`` -> ``.flip(-1)``, ``astype(i32)`` -> ``.to(torch.int32)``).
  * ``sg`` (``jax.lax.stop_gradient``) becomes ``.detach()``.
  * The clipped ``below`` and ``above`` indices are cast with ``.long()``, because
    ``F.one_hot`` needs int64 indices where ``jax.nn.one_hot`` accepts int32. The same cast is
    added in the NM512 transcription for the same reason.
  * ``TwoHot`` no longer forces float32: upstream casts ``logits`` with ``f32(...)`` and asserts
    float32 bins and targets. The transcription keeps the caller's dtype so the oracle test can
    compare in float64.
  * ``TwoHot.target`` is new: it is the first half of ``TwoHot.loss`` (from ``squash`` to the
    one-hot sum), split out verbatim so the target distribution can be compared directly.
    ``loss`` calls it, so the loss arithmetic is unchanged.
  * ``danijar_twohot_bins`` is the bin block of ``Head.symexp_twohot`` without the linear layer.
    Its ``transform`` argument defaults to ``symexp`` as upstream; passing the identity returns
    the symlog-space bin positions before ``symexp``. Because ``symexp`` is odd, mirroring before
    or after it gives the same values.

Upstream's default head uses ``TwoHot(logits, symexp_bins)`` with no squash, so it interpolates
between exponentially spaced bins in raw space. The exercise follows the configuration
``TwoHot(logits, symlog_space_bins, squash=symlog, unsquash=symexp)``, which the same class
supports and which the NM512 port uses; the oracle test pins where the two differ.

The NM512 functions are transcribed unchanged except that ``device="cuda"`` is dropped (the
buckets are built on the logits' device and in the logits' dtype instead of float32),
``mode``, ``log_prob_target`` and ``tensorboard`` are left out, and the ``.long()`` cast above. They
keep three divergences from danijar that the oracle test pins: ``symlog`` uses
``log(|x| + 1)`` rather than ``log1p``, ``mean`` is a plain left-to-right sum, and the buckets
are a single ``linspace(low, high, 255)`` instead of two mirrored halves.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "NM512DiscDist",
    "TwoHot",
    "danijar_twohot_bins",
    "nm512_symexp",
    "nm512_symlog",
    "symexp",
    "symlog",
]


# ---------------------------------------------------------------- danijar/dreamerv3

def symlog(x: Tensor) -> Tensor:
    """embodied/jax/nets.py lines 59-60."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """embodied/jax/nets.py lines 63-64."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def danijar_twohot_bins(bins: int, transform=symexp) -> Tensor:
    """embodied/jax/heads.py lines 135-142 (Head.symexp_twohot)."""
    f32 = torch.float32
    if bins % 2 == 1:
        half = torch.linspace(-20, 0, (bins - 1) // 2 + 1, dtype=f32)
        half = transform(half)
        out = torch.cat([half, -half[:-1].flip(0)], 0)
    else:
        half = torch.linspace(-20, 0, bins // 2, dtype=f32)
        half = transform(half)
        out = torch.cat([half, -half.flip(0)], 0)
    return out


class TwoHot:
    """embodied/jax/outs.py line 273."""

    def __init__(self, logits: Tensor, bins: Tensor, squash=None, unsquash=None):
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.bins = torch.as_tensor(bins, dtype=logits.dtype)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)

    def pred(self) -> Tensor:
        # Upstream comment: the naive left-to-right sum is non-zero for symmetric bins and
        # uniform probabilities, so a symmetric sum keeps predictions exactly zero at init.
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m: m + 1]
            p3 = self.probs[..., m + 1:]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m: m + 1]
            b3 = self.bins[..., m + 1:]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., :n // 2]
            p2 = self.probs[..., n // 2:]
            b1 = self.bins[..., :n // 2]
            b2 = self.bins[..., n // 2:]
            wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)
            return self.unsquash(wavg)

    def target(self, target: Tensor) -> Tensor:
        target = self.squash(target).detach()
        below = (self.bins <= target[..., None]).to(torch.int32).sum(-1) - 1
        above = len(self.bins) - (
            self.bins > target[..., None]).to(torch.int32).sum(-1)
        below = torch.clip(below, 0, len(self.bins) - 1).long()
        above = torch.clip(above, 0, len(self.bins) - 1).long()
        equal = (below == above)
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - target))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return (
            F.one_hot(below, len(self.bins)) * weight_below[..., None] +
            F.one_hot(above, len(self.bins)) * weight_above[..., None])

    def loss(self, target: Tensor) -> Tensor:
        target = self.target(target)
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return -(target * log_pred).sum(-1)


# ---------------------------------------------------------------- NM512/dreamerv3-torch

def nm512_symlog(x: Tensor) -> Tensor:
    """tools.py line 23."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def nm512_symexp(x: Tensor) -> Tensor:
    """tools.py line 27."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class NM512DiscDist:
    """tools.py line 452."""

    def __init__(self, logits: Tensor, low=-20.0, high=20.0, transfwd=nm512_symlog, transbwd=nm512_symexp):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=logits.device, dtype=logits.dtype)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self) -> Tensor:
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1).long()
        above = torch.clip(above, 0, len(self.buckets) - 1).long()
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)
