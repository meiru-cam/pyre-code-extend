"""Vendored open-genie space-time attention: an UNOFFICIAL reimplementation of Genie's ST block.

Upstream:   https://github.com/myscience/open-genie
Commit:     732b9f9b746f18fff1a0fb22f83638224f2f7cc6
Files:      genie/module/attention.py
Symbols:    Adapter (line 105), Attention (line 154), SpatialAttention (line 241),
            TemporalAttention (line 309), SpaceTimeAttention (line 373)
Retrieved:  2026-10-12
License:    MIT License, Copyright (c) 2020 Phil Wang (the LICENSE file shipped
            in myscience/open-genie at the commit above). Permission is granted
            free of charge to use, copy, modify, merge, publish and distribute
            the software, provided the copyright and permission notice are
            included in all copies or substantial portions of it.

WHY THIS FILE EXISTS
--------------------
Genie (Bruce et al. 2024, arXiv 2402.15391) has no official code release. The
st_transformer_attention exercise therefore takes its contract from the paper
(Section 2 "Methodology", ST-transformer paragraph and Figure 3) and uses this
unofficial MIT reimplementation as an independent cross-check, so the exercise
is not graded solely against an oracle written by the same author as its
reference solution. Treat it as a second opinion, not as Genie itself.

WHAT WAS CHANGED, AND WHY
-------------------------
The attention arithmetic is transcribed line for line: projections through
``Adapter``, ``scaled_dot_product_attention`` with ``is_causal`` and ``scale``,
the head merge and output projection, the spatial (per frame) and temporal (per
patch position) regrouping, and the residual sums in ``SpaceTimeAttention``.

  * einops is not a pyre-code dependency, so every ``rearrange`` / ``pack`` /
    ``unpack`` / ``Rearrange`` is rewritten as the equivalent torch
    ``reshape`` / ``permute`` / ``transpose``. ``Adapter``'s
    ``pack([q, k, v], '* n d')`` followed by ``rearrange(.., 'qkv n (h d) ->
    qkv h n d')`` and ``unpack`` is the same as splitting each of q, k, v into
    heads with ``view(b, n, h, d).transpose(1, 2)``. ``Attention.to_out`` keeps
    its two-entry ``nn.Sequential`` (head merge, then Linear or Identity) so
    ``to_out[1]`` still names the output projection.
  * ``RotaryEmbedding`` is not vendored. ``embed`` must be False; True raises
    NotImplementedError, and ``SpaceTimeAttention``'s default for ``embed``
    becomes False. The exercise has no positional embedding.
  * The ``cond`` (cross-attention key) path is not vendored: ``cond`` must be
    None. Upstream's spatial ``repeat(cond, .., t=t)`` passes a list for ``t``
    and is not exercised by self-attention anyway.
  * ``SpaceTimeAttention`` drops its feed-forward block (``ForwardBlock`` from
    ``genie/module/misc.py``, a grouped Conv3d) and ``ffn_skip``; ``forward``
    returns right after the temporal attention and its skip, and the
    FFN-only constructor arguments ``hid_dim`` and ``kernel_size`` are gone.
    The exercise covers the two attention layers only. ``space_skip`` is an
    Identity whenever ``d_inp`` is None or equals ``n_head * d_head``; its
    Conv3d branch for other widths is not vendored and raises.
  * The ``transpose`` option (channels-first video) is kept.

UPSTREAM QUIRKS THE EXERCISE DOES NOT COPY
------------------------------------------
  * The shape unpacking reads ``inp.shape`` where upstream reads ``video.shape``.
    Upstream's version is wrong only on the conditioning path, which is not
    vendored, so this changes nothing the oracle test exercises.
  * Default scale: ``default(scale, n_head * d_head ** -0.5)`` parses as
    ``n_head / sqrt(d_head)``, not ``1 / sqrt(d_head)``. Kept verbatim here; the
    oracle test passes ``scale`` explicitly and pins the difference.
  * ``Adapter`` builds the q/k/v projections, and ``Attention`` the output
    projection, only when the input width differs from ``n_head * d_head``;
    otherwise they are ``nn.Identity``. Kept verbatim; the oracle test installs
    Linear layers carrying the exercise's weights.
  * ``Attention`` applies a pre-LayerNorm to the query input. Kept verbatim; the
    exercise has no normalization, so the oracle test swaps it for Identity and
    pins that the default output differs.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

__all__ = ["Adapter", "Attention", "SpatialAttention", "TemporalAttention", "SpaceTimeAttention"]


def exists(var) -> bool:
    return var is not None


def default(var, val):
    return var if exists(var) else val


class Adapter(nn.Module):
    def __init__(
        self,
        qry_dim: int,
        n_head: int,
        d_head: int,
        key_dim: int | None = None,
        val_dim: int | None = None,
        block: type = nn.Linear,
        qry_kwargs: dict = {},
        key_kwargs: dict = {},
        val_kwargs: dict = {},
        bias: bool = False,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        if issubclass(block, nn.Module):
            block = (block, block, block)

        self.to_q = block[0](qry_dim, n_head * d_head, bias=bias, **qry_kwargs) if qry_dim != n_head * d_head else nn.Identity()
        self.to_k = block[1](key_dim, n_head * d_head, bias=bias, **key_kwargs) if key_dim != n_head * d_head else nn.Identity()
        self.to_v = block[2](val_dim, n_head * d_head, bias=bias, **val_kwargs) if val_dim != n_head * d_head else nn.Identity()

        self.n_head = n_head

    def forward(self, qry: Tensor, key: Tensor | None = None, val: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        key = default(key, qry)
        val = default(val, key)

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        # einops: pack([q, k, v], '* n d'); rearrange('qkv n (h d) -> qkv h n d'); unpack.
        def heads(t: Tensor) -> Tensor:
            b, n, hd = t.shape
            return t.view(b, n, self.n_head, hd // self.n_head).transpose(1, 2)

        return heads(q), heads(k), heads(v)


class _MergeHeads(nn.Module):
    """einops Rearrange('b h n d -> b n (h d)')."""

    def forward(self, x: Tensor) -> Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)


class Attention(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.d_inp = default(d_inp, n_head * d_head)
        self.d_out = default(d_out, self.d_inp)

        self.norm = nn.LayerNorm(n_head * d_head)
        self.embed = nn.Identity()

        self.to_qkv = Adapter(
            qry_dim=self.d_inp,
            n_head=n_head,
            d_head=d_head,
            bias=bias,
            **kwargs,
        )

        self.to_out = nn.Sequential(
            _MergeHeads(),
            nn.Linear(n_head * d_head, self.d_out, bias=bias) if self.d_out != n_head * d_head else nn.Identity(),
        )

        # Upstream quirk, kept verbatim: this is n_head / sqrt(d_head).
        self.scale = default(scale, n_head * d_head ** -0.5)
        self.causal = causal
        self.dropout = dropout

    def forward(self, qry: Tensor, key: Tensor | None = None, val: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        qry = self.embed(qry)
        qry = self.norm(qry)

        key = default(key, qry)
        val = default(val, key)

        q, k, v = self.to_qkv(qry, key, val)

        attn = scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=self.causal,
            dropout_p=self.dropout,
            scale=self.scale,
        )

        return self.to_out(attn)


class SpatialAttention(Attention):
    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        embed: bool = True,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(n_head, d_head, d_inp, d_out, bias, scale, causal, dropout, **kwargs)
        if embed:
            raise NotImplementedError("RotaryEmbedding is not vendored; pass embed=False")
        self.embed = nn.Identity()
        self.transpose = transpose

    def forward(self, video: Tensor, cond: Tensor | None = None, mask: Tensor | None = None, transpose: bool | None = None) -> Tensor:
        if exists(cond):
            raise NotImplementedError("the cond path is not vendored")
        transpose = default(transpose, self.transpose)

        # einops: rearrange(video, 'b c t h w -> b t h w c') when transposed.
        inp = video.permute(0, 2, 3, 4, 1) if transpose else video
        b, t, h, w, c = inp.shape

        # einops: pack([inp], '* h w c'); pack([inp], 'b * c')  ->  (b t, h w, c).
        inp = inp.reshape(b * t, h * w, c)

        out = super().forward(inp, key=None, mask=mask)

        # einops: unpack(.., 'b * c'); unpack(.., '* h w c').
        out = out.reshape(b, t, h, w, out.shape[-1])
        return out.permute(0, 4, 1, 2, 3) if transpose else out


class TemporalAttention(Attention):
    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        embed: bool = True,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(n_head, d_head, d_inp, d_out, bias, scale, causal, dropout, **kwargs)
        if embed:
            raise NotImplementedError("RotaryEmbedding is not vendored; pass embed=False")
        self.embed = nn.Identity()
        self.transpose = transpose

    def forward(self, video: Tensor, cond: Tensor | None = None, mask: Tensor | None = None, transpose: bool | None = None) -> Tensor:
        if exists(cond):
            raise NotImplementedError("the cond path is not vendored")
        transpose = default(transpose, self.transpose)

        # einops: rearrange(video, '{pattern} -> b h w t c').
        inp = video.permute(0, 3, 4, 2, 1) if transpose else video.permute(0, 2, 3, 1, 4)
        b, h, w, t, c = inp.shape
        # einops: pack([inp], '* t c').
        inp = inp.reshape(b * h * w, t, c)

        out = super().forward(inp, key=None, mask=mask)

        # einops: unpack(.., '* t c'); rearrange('b h w t c -> {pattern}').
        out = out.reshape(b, h, w, t, out.shape[-1])
        return out.permute(0, 4, 3, 1, 2) if transpose else out.permute(0, 3, 1, 2, 4)


class SpaceTimeAttention(nn.Module):
    def __init__(
        self,
        n_head: int | Tuple[int, int],
        d_head: int | Tuple[int, int],
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        embed: bool | Tuple[bool, bool] = False,
        scale: float | None = None,
        dropout: float = 0.0,
        transpose: bool = False,
        time_attn_kw: dict = {},
        space_attn_kw: dict = {},
    ) -> None:
        super().__init__()

        if isinstance(n_head, int):
            n_head = (n_head, n_head)
        if isinstance(d_head, int):
            d_head = (d_head, d_head)
        if isinstance(embed, bool):
            embed = (embed, embed)

        self.space_attn = SpatialAttention(
            n_head=n_head[0],
            d_head=d_head[0],
            d_inp=d_inp,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[0],
            causal=False,
            dropout=dropout,
            transpose=transpose,
            **space_attn_kw,
        )

        self.temp_attn = TemporalAttention(
            n_head=n_head[1],
            d_head=d_head[1],
            d_inp=None,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[1],
            # * Causal attention for temporal attention
            causal=True,
            dropout=dropout,
            transpose=transpose,
            **time_attn_kw,
        )

        space_hid = d_head[0] * n_head[0]
        self.time_skip = nn.Identity()  # Don't need this at the moment
        if exists(d_inp) and d_inp != space_hid:
            raise NotImplementedError("the Conv3d space_skip is not vendored; use d_inp == n_head * d_head")
        self.space_skip = nn.Identity()

    def forward(self, video: Tensor, cond=None, mask: Tensor | None = None) -> Tensor:
        if exists(cond):
            raise NotImplementedError("the cond path is not vendored")
        # We feed the video first through the spatial attention
        # and then through the temporal attention mechanism.
        video = self.space_attn(video, mask=mask) + self.space_skip(video)
        video = self.temp_attn(video, mask=mask) + self.time_skip(video)
        # Upstream continues with: video = self.ffn(video) + self.ffn_skip(video)  (not vendored).
        return video
