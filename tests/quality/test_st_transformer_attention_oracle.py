"""st_transformer_attention's reference, cross-checked against open-genie's SpaceTimeAttention.

Genie has no official code, so the contract comes from the paper (Section 2, the
ST-transformer paragraph and Figure 3) and this unofficial MIT reimplementation,
transcribed in `torch_judge/harness/embodied/vendored/open_genie_space_time_attention.py`,
is the independent cross-check. The vendored modules are built with the exercise's
weights installed as Linear layers, the pre-LayerNorm swapped for Identity and the
scale passed explicitly; the tests at the bottom pin each of those upstream quirks.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from torch_judge.harness.embodied.vendored.open_genie_space_time_attention import (
    Attention,
    SpaceTimeAttention,
)
from torch_judge.tasks import get_task

TASK_ID = "st_transformer_attention"
ATOL = 1e-10


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]], namespace["st_mask"]


def weights(D: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, D, D, generator=g, dtype=torch.float64) / math.sqrt(D) for _ in range(4)]


def linear(W: torch.Tensor) -> nn.Linear:
    """nn.Linear computing z @ W, as the exercise's projections do."""
    layer = nn.Linear(W.shape[0], W.shape[1], bias=False, dtype=W.dtype)
    with torch.no_grad():
        layer.weight.copy_(W.T)
    return layer


def install(attn: Attention, Wq, Wk, Wv, Wo, *, keep_norm: bool = False) -> None:
    attn.to_qkv.to_q, attn.to_qkv.to_k, attn.to_qkv.to_v = linear(Wq), linear(Wk), linear(Wv)
    attn.to_out[1] = linear(Wo)
    if not keep_norm:
        attn.norm = nn.Identity()


def oracle(W, n_heads: int, D: int, *, scale: float | str | None = "exact", keep_norm: bool = False) -> SpaceTimeAttention:
    d = D // n_heads
    module = SpaceTimeAttention(
        n_head=n_heads, d_head=d, d_inp=D, embed=False,
        scale=1.0 / math.sqrt(d) if scale == "exact" else scale,
    ).double()
    Wq, Wk, Wv, Wo = W
    install(module.space_attn, Wq[0], Wk[0], Wv[0], Wo[0], keep_norm=keep_norm)
    install(module.temp_attn, Wq[1], Wk[1], Wv[1], Wo[1], keep_norm=keep_norm)
    return module.eval()


def upstream_out(module: SpaceTimeAttention, x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """open-genie takes (B, T, H, W, C); the exercise's N patches are an H x W grid, row-major."""
    B, T, N, D = x.shape
    with torch.no_grad():
        return module(x.reshape(B, T, *grid, D)).reshape(B, T, N, D)


@pytest.mark.parametrize(
    "seed,B,T,grid,D,n_heads",
    [
        (0, 2, 4, (2, 3), 8, 2),
        (1, 1, 6, (3, 3), 12, 3),
        (2, 3, 3, (1, 5), 16, 4),
        (3, 2, 5, (2, 2), 8, 1),
        (4, 1, 8, (4, 2), 6, 6),
        (5, 2, 2, (3, 1), 10, 5),
        (6, 1, 7, (2, 4), 16, 8),
    ],
)
def test_matches_open_genie_space_time_attention(seed, B, T, grid, D, n_heads):
    st_attention, _ = reference()
    N = grid[0] * grid[1]
    g = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(B, T, N, D, generator=g, dtype=torch.float64)
    W = weights(D, seed)
    got = st_attention(x, *W, n_heads)
    want = upstream_out(oracle(W, n_heads, D), x, grid)
    assert torch.allclose(got, want, atol=ATOL, rtol=0), (got - want).abs().max()


@pytest.mark.parametrize("T,grid", [(1, (2, 3)), (5, (1, 1)), (1, (1, 1))])
def test_matches_open_genie_on_single_frame_or_patch(T, grid):
    st_attention, _ = reference()
    N, D, n_heads = grid[0] * grid[1], 8, 2
    x = torch.randn(2, T, N, D, generator=torch.Generator().manual_seed(T * 10 + N), dtype=torch.float64)
    W = weights(D, 20 + T + N)
    got = st_attention(x, *W, n_heads)
    want = upstream_out(oracle(W, n_heads, D), x, grid)
    assert torch.allclose(got, want, atol=ATOL, rtol=0), (got - want).abs().max()


def test_matches_open_genie_channels_first_layout():
    """open-genie's transpose=True takes (B, C, T, H, W); the arithmetic is unchanged."""
    st_attention, _ = reference()
    B, T, grid, D, n_heads = 2, 4, (2, 2), 8, 4
    x = torch.randn(B, T, 4, D, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    W = weights(D, 31)
    module = oracle(W, n_heads, D)
    module.space_attn.transpose = module.temp_attn.transpose = True
    with torch.no_grad():
        video = x.reshape(B, T, *grid, D).permute(0, 4, 1, 2, 3)
        want = module(video).permute(0, 2, 3, 4, 1).reshape(B, T, 4, D)
    assert torch.allclose(st_attention(x, *W, n_heads), want, atol=ATOL, rtol=0)


def test_open_genie_agrees_with_the_mask_form():
    """The paper's factorization, as open-genie runs it, equals masked full attention with st_mask."""
    _, st_mask = reference()
    B, T, grid, D, n_heads = 2, 4, (1, 3), 8, 2
    N = 3
    x = torch.randn(B, T, N, D, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    W = weights(D, 41)
    spatial, temporal = st_mask(T, N)
    flat = x.reshape(B, T * N, D)
    d = D // n_heads

    def masked(z, Wq, Wk, Wv, Wo, allowed):
        q, k, v = ((z @ M).view(B, T * N, n_heads, d).transpose(1, 2) for M in (Wq, Wk, Wv))
        s = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(~allowed, float("-inf"))
        return (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T * N, D) @ Wo

    Wq, Wk, Wv, Wo = W
    h = flat + masked(flat, Wq[0], Wk[0], Wv[0], Wo[0], spatial)
    y = (h + masked(h, Wq[1], Wk[1], Wv[1], Wo[1], temporal)).reshape(B, T, N, D)
    assert torch.allclose(upstream_out(oracle(W, n_heads, D), x, grid), y, atol=ATOL, rtol=0)


def test_pins_open_genie_default_scale_quirk():
    """Upstream's default scale n_head * d_head ** -0.5 is n_head / sqrt(d_head).

    It equals the exercise with every query projection multiplied by n_head, and differs
    from the exercise whenever n_head > 1.
    """
    st_attention, _ = reference()
    B, T, grid, D, n_heads = 2, 3, (2, 2), 8, 4
    x = torch.randn(B, T, 4, D, generator=torch.Generator().manual_seed(51), dtype=torch.float64)
    W = weights(D, 51)
    default_scale = upstream_out(oracle(W, n_heads, D, scale=None), x, grid)
    assert oracle(W, n_heads, D, scale=None).space_attn.scale == pytest.approx(n_heads / math.sqrt(D // n_heads))
    assert not torch.allclose(default_scale, st_attention(x, *W, n_heads), atol=1e-6)
    Wq_scaled = W[0] * n_heads
    assert torch.allclose(default_scale, st_attention(x, Wq_scaled, *W[1:], n_heads), atol=ATOL, rtol=0)


def test_pins_open_genie_identity_projections_and_prenorm():
    """Built with default projections, open-genie has no q/k/v/out weights at all when the
    width equals n_head * d_head, and it pre-normalizes each layer's input. The exercise
    has explicit weights and no normalization, so its output differs from that module."""
    st_attention, _ = reference()
    D, n_heads = 8, 2
    plain = SpaceTimeAttention(n_head=n_heads, d_head=D // n_heads, d_inp=D, embed=False)
    for attn in (plain.space_attn, plain.temp_attn):
        assert isinstance(attn.to_qkv.to_q, nn.Identity) and isinstance(attn.to_out[1], nn.Identity)
        assert isinstance(attn.norm, nn.LayerNorm)

    x = torch.randn(1, 3, 4, D, generator=torch.Generator().manual_seed(61), dtype=torch.float64)
    W = weights(D, 61)
    with_norm = upstream_out(oracle(W, n_heads, D, keep_norm=True), x, (2, 2))
    assert not torch.allclose(with_norm, st_attention(x, *W, n_heads), atol=1e-6)


def test_open_genie_temporal_layer_is_causal_and_spatial_is_not():
    """Same causality structure as the exercise: future frames never leak backward."""
    st_attention, _ = reference()
    B, T, grid, D, n_heads = 1, 5, (2, 2), 8, 2
    x = torch.randn(B, T, 4, D, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    W = weights(D, 71)
    module = oracle(W, n_heads, D)
    y = x.clone()
    y[:, 3:] += 5.0
    up_a, up_b = upstream_out(module, x, grid), upstream_out(module, y, grid)
    ours_a, ours_b = st_attention(x, *W, n_heads), st_attention(y, *W, n_heads)
    assert torch.allclose(up_a[:, :3], up_b[:, :3], atol=1e-12)
    assert torch.allclose(ours_a[:, :3], ours_b[:, :3], atol=1e-12)
    assert torch.allclose(up_b, ours_b, atol=ATOL, rtol=0)
