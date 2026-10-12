"""Vendored finite scalar quantizer (FSQ) oracle from NVIDIA Cosmos-Tokenizer.

Upstream:   https://github.com/NVIDIA/Cosmos-Tokenizer
Commit:     3584ae752ce8ebdbe06a420bf60d7513c0e878cc
Files:      cosmos_tokenizer/modules/quantizers.py, cosmos_tokenizer/modules/utils.py
Symbols:    FSQuantizer (__init__, bound, quantize, _scale_and_shift,
            _scale_and_shift_inverse, codes_to_indices, indices_to_codes, forward),
            round_ste
Retrieved:  2026-10-12
License:    Apache License 2.0
            SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0
            Upstream FSQuantizer is itself adapted from lucidrains/vector-quantize-pytorch
            at 9502a1f447876d53fd37685b226bf28f250dc4a3 (MIT, Copyright (c) 2020 Phil Wang),
            which follows the JAX listing in Mentzer et al. 2023, Appendix A.1.

WHY THIS FILE EXISTS
--------------------
Ground truth for the fsq_quantizer exercise, so it is not checked only against an
oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic of bound, quantize, the scale-and-shift pair, codes_to_indices and
indices_to_codes is transcribed line for line. Removed or changed infrastructure:

  * ``nn.Module`` and ``register_buffer`` are replaced by plain attributes; there are
    no parameters, so nothing about the arithmetic changes.
  * The optional ``project_in`` / ``project_out`` linear layers, ``num_codebooks``,
    ``keep_num_codebooks_dim`` and ``scale`` are dropped: the oracle covers the
    default single-codebook case with ``dim == len(levels)``, where both
    projections are ``nn.Identity``.
  * The einops packing of image/video tensors (``b d ... -> b ... d``) is dropped.
    The oracle takes channels-last input of any shape ``(..., d)`` directly.
  * ``forward`` returns ``(indices, codes)`` without the all-zero ``dummy_loss``.
  * Upstream casts outputs to ``self.dtype`` (bfloat16 by default). The oracle
    keeps a ``dtype`` argument but defaults it to ``None``, meaning "no cast", so
    comparisons are not limited by bfloat16 resolution.
  * ``round_ste`` is inlined from ``cosmos_tokenizer/modules/utils.py``.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch

__all__ = ["CosmosFSQuantizer", "cosmos_round_ste"]


def cosmos_round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class CosmosFSQuantizer:
    """Transcribed from quantizers.py::FSQuantizer (single codebook, no projections)."""

    def __init__(self, levels: list[int], dtype: torch.dtype | None = None):
        self.dtype = dtype
        self._levels = torch.tensor(levels, dtype=torch.int32)
        self._basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.codebook_dim = len(levels)
        self.codebook_size = self._levels.prod().item()
        self.implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size))

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = cosmos_round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes if self.dtype is None else codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert z.shape[-1] == self.codebook_dim, (
            f"expected dimension of {self.codebook_dim} but found dimension of {z.shape[-1]}"
        )
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        out = codes if self.dtype is None else codes.to(self.dtype)
        return indices, out

    __call__ = forward
