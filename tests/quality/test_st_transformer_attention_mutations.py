"""Mutation gate for the Genie spatiotemporal (ST-transformer) attention exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "st_transformer_attention"

SPATIAL_CALL = "_attend(frames, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, causal=False)"
TEMPORAL_CALL = "_attend(tracks, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, causal=True)"
SPLIT = ".view(S, L, n_heads, d).transpose(1, 2)"
MERGE = "out.transpose(1, 2).reshape(S, L, D)"


def _mutation(name: str, *edits: tuple[str, str], count: int = 1) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, count)
    return Mutation(name, solution)


# No factorization: both layers attend over the whole flattened clip with a frame-causal mask.
FLATTENED = '''import math

import torch


def st_mask(T, N):
    for value in (T, N):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("bad size")
    idx = torch.arange(T * N)
    frame, patch = idx // N, idx % N
    return frame[:, None] == frame[None, :], (patch[:, None] == patch[None, :]) & (frame[None, :] <= frame[:, None])


def _attend(z, Wq, Wk, Wv, Wo, n_heads, allowed):
    S, L, D = z.shape
    d = D // n_heads
    q = (z @ Wq).view(S, L, n_heads, d).transpose(1, 2)
    k = (z @ Wk).view(S, L, n_heads, d).transpose(1, 2)
    v = (z @ Wv).view(S, L, n_heads, d).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(~allowed, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(S, L, D) @ Wo


def st_attention(x, Wq, Wk, Wv, Wo, n_heads):
    if not torch.is_tensor(x) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError("bad x")
    B, T, N, D = x.shape
    for W in (Wq, Wk, Wv, Wo):
        if not torch.is_tensor(W) or W.shape != (2, D, D) or W.dtype != x.dtype:
            raise ValueError("bad weight")
    if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads < 1 or D % n_heads:
        raise ValueError("bad n_heads")
    frame = torch.arange(T * N) // N
    allowed = frame[None, :] <= frame[:, None]
    flat = x.reshape(B, T * N, D)
    h = flat + _attend(flat, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, allowed)
    return (h + _attend(h, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, allowed)).view(B, T, N, D)
'''


MUTATIONS = [
    _mutation("temporal attention not causal", (TEMPORAL_CALL, TEMPORAL_CALL.replace("causal=True", "causal=False"))),
    _mutation("spatial attention causal over patches", (SPATIAL_CALL, SPATIAL_CALL.replace("causal=False", "causal=True"))),
    Mutation("attention over the flattened T*N sequence, no factorization", FLATTENED),
    _mutation("temporal layer reuses the spatial weights", (TEMPORAL_CALL, TEMPORAL_CALL.replace("[1]", "[0]"))),
    _mutation("scores scaled by sqrt(D) instead of sqrt(d)", ("/ math.sqrt(d)", "/ math.sqrt(D)")),
    _mutation("open-genie default scale n_heads * d ** -0.5", ("/ math.sqrt(d)", "* (n_heads * d ** -0.5)")),
    _mutation(
        "heads take interleaved channels",
        (SPLIT, ".view(S, L, d, n_heads).permute(0, 3, 1, 2)"),
        (MERGE, "out.permute(0, 2, 3, 1).reshape(S, L, D)"),
        count=3,
    ),
    _mutation(
        "temporal layer runs before the spatial layer",
        ("    frames = x.reshape(B * T, N, D)\n"
         "    h = x + _attend(frames, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, causal=False).view(B, T, N, D)\n",
         "    tr = x.transpose(1, 2).reshape(B * N, T, D)\n"
         "    h = x + _attend(tr, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, causal=True).view(B, N, T, D).transpose(1, 2)\n"),
        ("    tracks = h.transpose(1, 2).reshape(B * N, T, D)\n"
         "    t_out = _attend(tracks, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, causal=True)\n"
         "    return h + t_out.view(B, N, T, D).transpose(1, 2)\n",
         "    frames = h.reshape(B * T, N, D)\n"
         "    return h + _attend(frames, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, causal=False).view(B, T, N, D)\n"),
    ),
    _mutation("causal mask also hides the diagonal", (".triu(1)", ".triu(0)")),
    _mutation(
        "temporal fold without the transpose",
        ("tracks = h.transpose(1, 2).reshape(B * N, T, D)", "tracks = h.reshape(B * N, T, D)"),
        ("return h + t_out.view(B, N, T, D).transpose(1, 2)", "return h + t_out.view(B, T, N, D)"),
    ),
    _mutation("no residual after the temporal layer", ("return h + t_out.view", "return t_out.view")),
    _mutation("no residual after the spatial layer", ("h = x + _attend(frames", "h = _attend(frames")),
    _mutation("pre-LayerNorm like open-genie", (
        "    S, L, D = z.shape\n",
        "    S, L, D = z.shape\n    z = torch.nn.functional.layer_norm(z, (D,))\n",
    )),
    _mutation("st_mask flattens patch-major", ("frame, patch = idx // N, idx % N", "patch, frame = idx // T, idx % T")),
    _mutation("st_mask temporal excludes the diagonal", ("(frame[None, :] <= frame[:, None])", "(frame[None, :] < frame[:, None])")),
    _mutation("st_mask temporal looks at the future", ("(frame[None, :] <= frame[:, None])", "(frame[None, :] >= frame[:, None])")),
    _mutation("computes in float32", (
        "    # Spatial: every frame",
        "    x = x.float()\n    Wq, Wk, Wv, Wo = (W.float() for W in (Wq, Wk, Wv, Wo))\n    # Spatial: every frame",
    )),
    _mutation("accepts an n_heads that does not divide D", (" or n_heads < 1 or D % n_heads:", " or n_heads < 1:")),
    _mutation("accepts weights of another dtype", (" or W.dtype != x.dtype:", ":")),
    _mutation("st_mask accepts sizes below 1", (" or value < 1:", ":")),
]


def test_st_transformer_attention_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])
    assert "paper plus an unofficial reimplementation" in task["sources"][0]["note"]


@pytest.mark.parametrize("_repeat", range(3))
def test_st_transformer_attention_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
