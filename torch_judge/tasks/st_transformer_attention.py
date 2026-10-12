"""Genie ST-transformer attention: spatial attention per frame, then causal temporal attention per patch."""

from ._interview import interview

# Masked full attention over the flattened T * N sequence, written independently of the
# reference solution. The factorized layer must match it exactly.
_FULL = r"""
import math, torch

def masked_mha(x, Wq, Wk, Wv, Wo, n_heads, allowed):
    B, S, D = x.shape
    d = D // n_heads
    q = (x @ Wq).view(B, S, n_heads, d).transpose(1, 2)
    k = (x @ Wk).view(B, S, n_heads, d).transpose(1, 2)
    v = (x @ Wv).view(B, S, n_heads, d).transpose(1, 2)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    s = s.masked_fill(~allowed, float("-inf"))
    out = torch.softmax(s, dim=-1) @ v
    return out.transpose(1, 2).reshape(B, S, D) @ Wo

def masked_st(x, Wq, Wk, Wv, Wo, n_heads, spatial, temporal):
    B, T, N, D = x.shape
    flat = x.reshape(B, T * N, D)
    h = flat + masked_mha(flat, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, spatial)
    y = h + masked_mha(h, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, temporal)
    return y.reshape(B, T, N, D)

def own_masks(T, N):
    idx = torch.arange(T * N)
    t, n = idx // N, idx % N
    return t[:, None] == t[None, :], (n[:, None] == n[None, :]) & (t[None, :] <= t[:, None])

def weights(D, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, D, D, generator=g, dtype=dtype) / math.sqrt(D) for _ in range(4)]

def video(B, T, N, D, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(B, T, N, D, generator=g, dtype=dtype)
"""

TASK = {
    "title": "Genie Spatiotemporal Attention",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "st_attention",
    "description_en": r"""A video world model sees `T` frames of `N` patch tokens each. Full self-attention over all `T * N` tokens costs `(T * N)^2`. Genie's ST-transformer factorizes it: a spatial layer attends within each frame, then a temporal layer attends across frames at each patch position, with a causal mask. Implement that attention pair and the boolean masks that describe it on the flattened sequence.

**Signatures:** define both functions at the top level of your code.
- `st_attention(x, Wq, Wk, Wv, Wo, n_heads) -> Tensor`
- `st_mask(T, N) -> (spatial, temporal)`

**Parameters:**
- `x` — floating tensor `(B, T, N, D)`: batch, frames, patches per frame, channels.
- `Wq`, `Wk`, `Wv`, `Wo` — tensors of shape `(2, D, D)` with the dtype of `x`. Index 0 holds the spatial layer's weights and index 1 the temporal layer's. The two layers never share weights.
- `n_heads` — positive integer that divides `D`. Each head has `d = D // n_heads` channels.

**One attention layer** with weights `(Wq, Wk, Wv, Wo)` on a group of `L` tokens `z` of shape `(L, D)`:

    q, k, v = z @ Wq, z @ Wk, z @ Wv          # split channels into heads: head i is channels [i*d, (i+1)*d)
    scores  = q_head @ k_head.T / sqrt(d)     # per head, (L, L)
    out     = concat_heads(softmax(scores) @ v_head) @ Wo

There are no biases, no normalization and no positional embedding.

**The ST pair:**
- Spatial: for every `(b, t)`, one layer with index-0 weights over the `N` tokens `x[b, t]`. No mask: every patch sees every patch of its own frame. `h = x + spatial(x)`.
- Temporal: for every `(b, n)`, one layer with index-1 weights over the `T` tokens `h[b, :, n]`, with a causal mask: the query at frame `t` sees frames `0..t` only. `y = h + temporal(h)`.
- Return `y`, shape `(B, T, N, D)`, dtype of `x`.

**Masks.** Flatten frame-major: token `t * N + n` is patch `n` of frame `t`, the order of `x.reshape(B, T * N, D)`. `st_mask` returns two bool tensors of shape `(T * N, T * N)`, `True` where query `i` may attend to key `j`:
- `spatial[i, j]` — `i` and `j` are in the same frame.
- `temporal[i, j]` — `i` and `j` are the same patch position and the frame of `j` is at most the frame of `i`.

Ordinary attention over the flattened sequence, with the scores at `False` set to `-inf`, using `spatial` for the first layer and `temporal` for the second (each with its residual), gives exactly the output of `st_attention`.

**Constraints:**
- No output at frame `t` depends on any input at a frame after `t`.
- `T = 1` and `N = 1` are valid. With `T = 1` the temporal layer returns `h @ Wv[1] @ Wo[1]`; with `N = 1` the spatial layer returns `x @ Wv[0] @ Wo[0]`.
- float64 inputs are checked to 1e-10 and float32 inputs to 1e-4.
- Do not modify any input in place.
- `st_attention` raises `ValueError` if `x` is not a 4-D floating tensor, if a weight is not `(2, D, D)` with the dtype of `x`, or if `n_heads` is not a positive integer dividing `D`.
- `st_mask` raises `ValueError` unless `T` and `N` are integers `>= 1`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why factorize.** For `T = 16` frames of `16 x 16` patches, full attention compares 4096 tokens with 4096 tokens. The spatial layer compares 256 with 256 per frame and the temporal layer 16 with 16 per patch, so the dominant cost grows linearly with the number of frames instead of quadratically. Two layers still let information travel from any patch of an earlier frame to any patch of a later frame: first within the earlier frame to the right patch position, then along that position across frames.

**Why the temporal mask is causal.** Genie's latent action model, video tokenizer and dynamics model all use the ST-transformer. The causal temporal mask lets one forward pass over a whole clip produce a prediction for every next frame, the same trick as a decoder-only language model, and lets a frame be generated from the past without seeing the future.

**What the paper leaves out.** The Genie paper describes the block (spatial layer, temporal layer, then one feed-forward layer) but does not release code. This exercise covers the two attention layers with their residuals; the feed-forward layer, normalization and positional embeddings are left out because they do not change which tokens can see which. The unofficial open-genie reimplementation adds rotary embeddings and a pre-LayerNorm to each layer.""",
    "advisory_prerequisites": ["mha", "causal_attention", "jepa_block_masks_and_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If you fold the batch and frame axes together, what shape does the spatial layer see, and which axis does it attend over? Which axes do you need to swap and fold so that the temporal layer attends over frames for one patch position? In the flattened order t * N + n, how do you recover t and n from a token index?"},
        {"level": 2, "kind": "analysis", "content": "Write one helper that runs multi-head attention over the middle axis of an (S, L, D) tensor with an optional causal mask, then call it twice: on x reshaped to (B*T, N, D) with the index-0 weights, and on the residual output with T and N swapped and folded to (B*N, T, D) with the index-1 weights. Undo the swap before adding the second residual. For the masks, compare token frame and patch indices with broadcasting. Pitfalls: a causal mask that also hides the diagonal, folding T and N in the wrong order, and scaling by sqrt(D) instead of sqrt(d)."},
    ],
    "model_connections": [
        "Genie (Bruce et al. 2024) builds its latent action model, video tokenizer (ST-ViViT) and MaskGIT dynamics model from ST-transformer blocks with causal temporal attention.",
        "The factorization follows Xu et al. 2020's spatial-temporal transformer and mirrors ViViT's factorized self-attention; TimeSformer calls the same pattern divided space-time attention.",
        "The unofficial open-genie SpaceTimeAttention runs spatial attention with causal=False, then temporal attention with causal=True, each with its own weights and a residual.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Cost per block is about T * N^2 + N * T^2 instead of (T * N)^2, so long clips fit in memory.",
            "The causal temporal mask gives next-frame predictions for every frame in one pass, like a language model.",
        ],
        "cons": [
            "A token in one layer only sees its own frame or its own patch position, so mixing across both needs a stack of blocks.",
            "Two attention layers per block double the projection parameters compared with one full-attention layer.",
        ],
    },
    "sources": [
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.15391",
            "section": "Bruce et al. 2024, Genie: Generative Interactive Environments, Section 2 Methodology, ST-transformer paragraph and Figure 3",
            "note": "The ORACLE for this exercise is the paper plus an unofficial reimplementation; Genie has no official code. "
                    "The paper specifies a spatial layer over the H x W tokens of one time step, then a temporal layer over the T time steps "
                    "of the same token with a causal mask, then one feed-forward layer.",
        },
        {
            "kind": "code",
            "url": "https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py",
            "commit": "732b9f9b746f18fff1a0fb22f83638224f2f7cc6",
            "path": "genie/module/attention.py",
            "symbol": "SpaceTimeAttention (with Attention, Adapter, SpatialAttention, TemporalAttention)",
            "license": "MIT",
            "adapted": "UNOFFICIAL reimplementation, used as a cross-check of the paper-derived contract: separate spatial (causal=False) and "
                       "temporal (causal=True) attention modules, each with its own projections, applied in that order with a residual after each.",
            "simplifications": "einops rewritten as torch reshapes. Left out: rotary embeddings, the pre-LayerNorm in each attention layer, the "
                               "Conv3d feed-forward block and the space_skip/ffn_skip Conv3d skips, cross-attention conditioning, dropout, and the in_channels/out_channels attributes; "
                               "the spatial module accepts only 5-D input. Projections are explicit (2, D, D) weights; "
                               "upstream silently uses nn.Identity for q/k/v and output when the width equals n_head * d_head. "
                               "Divergence: upstream's default scale is n_head * d_head ** -0.5, which is n_head / sqrt(d_head); "
                               "the exercise uses 1 / sqrt(d_head), as the oracle test pins.",
        },
    ],
    "tests": [
        {"name": "st_mask on two frames of two patches", "behavior": "attention.masking", "code": r"""
import torch
st_mask = {fn}.__globals__.get("st_mask")
assert callable(st_mask), "define st_mask next to st_attention"
spatial, temporal = st_mask(2, 2)
assert spatial.dtype == torch.bool and spatial.shape == (4, 4), (spatial.dtype, spatial.shape)
# token order: (frame 0, patch 0), (frame 0, patch 1), (frame 1, patch 0), (frame 1, patch 1)
T_, F_ = True, False
assert spatial.tolist() == [[T_, T_, F_, F_], [T_, T_, F_, F_], [F_, F_, T_, T_], [F_, F_, T_, T_]], spatial
assert temporal.tolist() == [[T_, F_, F_, F_], [F_, T_, F_, F_], [T_, F_, T_, F_], [F_, T_, F_, T_]], temporal
"""},
        {"name": "Factorized attention equals masked full attention", "behavior": "attention.masking", "code": _FULL + r"""
st_mask = {fn}.__globals__.get("st_mask")
assert callable(st_mask), "define st_mask next to st_attention"
x = video(2, 3, 4, 8, seed=0)
Wq, Wk, Wv, Wo = weights(8, seed=0)
out = {fn}(x, Wq, Wk, Wv, Wo, 2)
assert out.shape == (2, 3, 4, 8) and out.dtype == torch.float64, (out.shape, out.dtype)
spatial, temporal = st_mask(3, 4)
want = masked_st(x, Wq, Wk, Wv, Wo, 2, spatial, temporal)
assert torch.allclose(out, want, atol=1e-10, rtol=0), (out - want).abs().max()
"""},
        {"name": "Matches masked full attention across shapes and head counts", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Run spatial attention per frame with the index-0 weights, then causal temporal attention per patch with the index-1 weights, each plus its residual, scaling scores by 1/sqrt(D // n_heads).",
         "code": _FULL + r"""
for seed, (B, T, N, D, H) in enumerate([(1, 4, 3, 8, 1), (2, 5, 4, 8, 4), (3, 3, 6, 12, 3), (2, 6, 2, 16, 2), (1, 2, 9, 6, 6)]):
    x = video(B, T, N, D, seed)
    W = weights(D, seed)
    out = {fn}(x, *W, H)
    assert out.shape == (B, T, N, D), out.shape
    want = masked_st(x, *W, H, *own_masks(T, N))
    assert torch.allclose(out, want, atol=1e-10, rtol=0), (seed, (out - want).abs().max())
"""},
        {"name": "st_mask is exact for any T and N", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Flatten frame-major (token t * N + n); spatial allows the same frame, temporal allows the same patch at the same or an earlier frame, diagonal included.",
         "code": _FULL + r"""
st_mask = {fn}.__globals__["st_mask"]
for T, N in ((1, 1), (1, 5), (4, 1), (3, 5), (5, 2)):
    spatial, temporal = st_mask(T, N)
    want_s, want_t = own_masks(T, N)
    assert spatial.dtype == torch.bool and temporal.dtype == torch.bool, (spatial.dtype, temporal.dtype)
    assert spatial.shape == temporal.shape == (T * N, T * N), (spatial.shape, temporal.shape)
    assert torch.equal(spatial, want_s), (T, N, spatial)
    assert torch.equal(temporal, want_t), (T, N, temporal)
"""},
        {"name": "Earlier frames ignore future frames", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Mask the temporal scores so that the query at frame t only sees frames 0..t; never attend across frames in the spatial layer.",
         "code": _FULL + r"""
B, T, N, D, H = 2, 5, 3, 8, 2
x = video(B, T, N, D, seed=3)
W = weights(D, seed=3)
base = {fn}(x, *W, H)
for t0 in (1, 3, 4):
    y = x.clone()
    y[:, t0:] = y[:, t0:] * -3.0 + 2.0
    out = {fn}(y, *W, H)
    assert torch.allclose(out[:, :t0], base[:, :t0], atol=1e-12, rtol=0), (t0, (out[:, :t0] - base[:, :t0]).abs().max())
    assert (out[:, t0:] - base[:, t0:]).abs().amax(dim=-1).min() > 1e-6, "changed frames must change their outputs"
"""},
        {"name": "Spatial attention sees every patch of its frame", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "The spatial layer is bidirectional: every patch attends to every patch of the same frame, including later patch indices.",
         "code": _FULL + r"""
B, T, N, D, H = 2, 3, 5, 8, 2
x = video(B, T, N, D, seed=4)
W = weights(D, seed=4)
base = {fn}(x, *W, H)
y = x.clone()
y[0, 0, N - 1] += 4.0  # the last patch of the first frame
out = {fn}(y, *W, H)
assert (out[0, 0, 0] - base[0, 0, 0]).abs().max() > 1e-6, "patch 0 must see patch N - 1 of its frame"
assert torch.allclose(out[1], base[1], atol=1e-12, rtol=0), "batch items must not mix"
"""},
        {"name": "One frame: the temporal layer is its value path", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With T = 1 each patch attends only to itself in time, so the temporal layer returns h @ Wv[1] @ Wo[1]; keep both residuals.",
         "code": _FULL + r"""
B, N, D, H = 3, 4, 8, 4
x = video(B, 1, N, D, seed=5)
Wq, Wk, Wv, Wo = weights(D, seed=5)
flat = x.reshape(B, N, D)
h = flat + masked_mha(flat, Wq[0], Wk[0], Wv[0], Wo[0], H, torch.ones(N, N, dtype=torch.bool))
want = (h + h @ Wv[1] @ Wo[1]).reshape(B, 1, N, D)
out = {fn}(x, Wq, Wk, Wv, Wo, H)
assert torch.allclose(out, want, atol=1e-10, rtol=0), (out - want).abs().max()
"""},
        {"name": "One patch: the spatial layer is its value path", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With N = 1 each frame has one token, so the spatial layer returns x @ Wv[0] @ Wo[0]; the temporal layer stays causal.",
         "code": _FULL + r"""
B, T, D, H = 2, 6, 8, 2
x = video(B, T, 1, D, seed=6)
Wq, Wk, Wv, Wo = weights(D, seed=6)
flat = x.reshape(B, T, D)
h = flat + flat @ Wv[0] @ Wo[0]
causal = torch.ones(T, T, dtype=torch.bool).tril()
want = (h + masked_mha(h, Wq[1], Wk[1], Wv[1], Wo[1], H, causal)).reshape(B, T, 1, D)
out = {fn}(x, Wq, Wk, Wv, Wo, H)
assert torch.allclose(out, want, atol=1e-10, rtol=0), (out - want).abs().max()
"""},
        {"name": "Keeps dtype and does not mutate inputs", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Return the dtype of x (float32 stays float32, float64 stays float64) and never modify x or the weights in place.",
         "code": _FULL + r"""
x = video(2, 3, 4, 8, seed=7)
W = weights(8, seed=7)
copies = [x.clone()] + [w.clone() for w in W]
out64 = {fn}(x, *W, 2)
assert out64.dtype == torch.float64, out64.dtype
for a, b in zip([x] + W, copies):
    assert torch.equal(a, b), "st_attention modified an input"
out32 = {fn}(x.float(), *[w.float() for w in W], 2)
assert out32.dtype == torch.float32, out32.dtype
want = masked_st(x, *W, 2, *own_masks(3, 4))
assert torch.allclose(out32.double(), want, atol=1e-4, rtol=0), (out32.double() - want).abs().max()
"""},
        {"name": "Rejects bad shapes and arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a non 4-D or non-floating x, a weight that is not (2, D, D) in the dtype of x, an n_heads that does not divide D, or a T or N below 1.",
         "code": _FULL + r"""
st_mask = {fn}.__globals__["st_mask"]
x = video(1, 2, 3, 8, seed=8)
W = weights(8, seed=8)
def rejects(*args):
    try:
        {fn}(*args)
    except ValueError:
        return True
    return False
assert rejects(x[0], *W, 2), "accepted a 3-D x"
assert rejects(x.long(), *[w.long() for w in W], 2), "accepted an integer x"
assert rejects(x, W[0][0], *W[1:], 2), "accepted a (D, D) Wq"
assert rejects(x, W[0], W[1], W[2], torch.zeros(2, 8, 9, dtype=torch.float64), 2), "accepted a (2, D, D + 1) Wo"
assert rejects(x, W[0], W[1], W[2].float(), W[3], 2), "accepted a weight in another dtype"
for h in (3, 0, -2, 2.0):
    assert rejects(x, *W, h), f"accepted n_heads={h!r}"
for T, N in ((0, 2), (2, 0), (1.5, 2), (2, -1)):
    try:
        st_mask(T, N)
    except ValueError:
        pass
    else:
        raise AssertionError(f"st_mask accepted T={T!r}, N={N!r}")
"""},
    ],
    "solution": '''import math

import torch


def st_mask(T, N):
    for name, value in (("T", T), ("N", N)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    idx = torch.arange(T * N)
    frame, patch = idx // N, idx % N
    spatial = frame[:, None] == frame[None, :]
    temporal = (patch[:, None] == patch[None, :]) & (frame[None, :] <= frame[:, None])
    return spatial, temporal


def _attend(z, Wq, Wk, Wv, Wo, n_heads, causal):
    """Multi-head self-attention over axis 1 of z, shape (S, L, D)."""
    S, L, D = z.shape
    d = D // n_heads
    q = (z @ Wq).view(S, L, n_heads, d).transpose(1, 2)
    k = (z @ Wk).view(S, L, n_heads, d).transpose(1, 2)
    v = (z @ Wv).view(S, L, n_heads, d).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        future = torch.ones(L, L, dtype=torch.bool, device=z.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2).reshape(S, L, D) @ Wo


def st_attention(x, Wq, Wk, Wv, Wo, n_heads):
    if not torch.is_tensor(x) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor of shape (B, T, N, D)")
    B, T, N, D = x.shape
    for W in (Wq, Wk, Wv, Wo):
        if not torch.is_tensor(W) or W.shape != (2, D, D) or W.dtype != x.dtype:
            raise ValueError("each weight must be a (2, D, D) tensor with the dtype of x")
    if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads < 1 or D % n_heads:
        raise ValueError("n_heads must be a positive integer that divides D")

    # Spatial: every frame is one sequence of N tokens, bidirectional.
    frames = x.reshape(B * T, N, D)
    h = x + _attend(frames, Wq[0], Wk[0], Wv[0], Wo[0], n_heads, causal=False).view(B, T, N, D)

    # Temporal: every patch position is one sequence of T tokens, causal.
    tracks = h.transpose(1, 2).reshape(B * N, T, D)
    t_out = _attend(tracks, Wq[1], Wk[1], Wv[1], Wo[1], n_heads, causal=True)
    return h + t_out.view(B, N, T, D).transpose(1, 2)
''',
    "interview_questions": interview(
        concept=[
            "What does a spatial layer attend over in an ST-transformer, what does a temporal layer attend over, and why does the factorization save memory?",
            "Why is Genie's temporal attention causal while its spatial attention is bidirectional?",
        ],
        deep_dive=[
            "Write the boolean masks that make ordinary attention over the flattened T * N sequence equal to the spatial layer and to the temporal layer.",
            "How do you reshape a (B, T, N, D) tensor so that a standard attention routine runs the temporal layer, and what goes wrong if you fold the axes in the wrong order?",
            "After one spatial layer and one temporal layer, which input tokens can influence the output at frame t, patch n? What changes if the temporal layer runs first?",
        ],
        tradeoffs=[
            "Factorized space-time attention versus full attention over all T * N tokens: what is lost in expressiveness, and when does the cost saving matter?",
            "Separate weights for the spatial and temporal layers versus one shared set: what does each choice cost and buy?",
        ],
    ),
}
