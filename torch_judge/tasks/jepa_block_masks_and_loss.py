"""V-JEPA 2 training pieces: multi-block context/target masks, the latent regression loss, and the EMA target encoder."""

from ._interview import interview

# A brute-force model of the masks: Python sets over flat token indices.
_MASK_MODEL = r"""
import torch

def model_masks(grid, blocks, max_keep=None):
    T, H, W = grid
    targets, contexts = [], []
    for sample in blocks:
        covered = set()
        for (t0, t, top, h, left, w) in sample:
            for a in range(t0, t0 + t):
                for y in range(top, top + h):
                    for x in range(left, left + w):
                        covered.add((a * H + y) * W + x)
        targets.append(sorted(covered))
        contexts.append(sorted(set(range(T * H * W)) - covered))
    k_tgt = min(len(x) for x in targets)
    k_ctx = min(len(x) for x in contexts)
    if max_keep is not None:
        k_ctx = min(k_ctx, max_keep)
    return [c[:k_ctx] for c in contexts], [x[:k_tgt] for x in targets]

def draw(g, low, high):
    return int(torch.randint(low, high + 1, (1,), generator=g))

def random_blocks(g, grid, batch, max_blocks):
    T, H, W = grid
    while True:
        out = []
        for _ in range(batch):
            sample = []
            for _ in range(draw(g, 1, max_blocks)):
                t, h, w = draw(g, 1, T), draw(g, 1, H), draw(g, 1, W)
                sample.append((draw(g, 0, T - t), t, draw(g, 0, H - h), h, draw(g, 0, W - w), w))
            out.append(sample)
        if all(model_masks(grid, [s])[0][0] for s in out):
            return out

def check_masks(got, want, label):
    assert isinstance(got, tuple) and len(got) == 2, (label, "return (ctx_idx, tgt_idx)")
    ctx, tgt = got
    assert isinstance(ctx, torch.Tensor) and isinstance(tgt, torch.Tensor), label
    assert ctx.dtype == torch.int64 and tgt.dtype == torch.int64, (label, ctx.dtype, tgt.dtype)
    assert ctx.ndim == 2 and tgt.ndim == 2, (label, ctx.shape, tgt.shape)
    assert ctx.tolist() == want[0], (label, "context", ctx.tolist(), want[0])
    assert tgt.tolist() == want[1], (label, "target", tgt.tolist(), want[1])
"""

# An independent model of the loss: layer norm written out, gather by Python indexing.
_LOSS_MODEL = r"""
import torch

def normalized(feats):
    mu = feats.mean(-1, keepdim=True)
    var = feats.var(-1, unbiased=False, keepdim=True)
    return (feats - mu) / torch.sqrt(var + 1e-5)

def model_loss(pred, feats, idx, p):
    h = normalized(feats)
    picked = torch.stack([h[b, idx[b]] for b in range(h.shape[0])])
    return (pred - picked).abs().pow(p).mean() / p

def seeded(seed, B=3, N=10, K=4, D=6):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, N, D, generator=g, dtype=torch.float64) * 3 + 1
    idx = torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(B)])
    pred = torch.randn(B, K, D, generator=g, dtype=torch.float64)
    return pred, feats, idx
"""

_PARTS = [
    {
        "title": "Context and target masks",
        "description_en": r"""**Signature:** `make_masks(grid, blocks, max_keep=None) -> (ctx_idx, tgt_idx)`

- `grid = (T, H, W)` is the token grid: `T` time steps of `H x W` patches. Token `(t, y, x)` has flat index `t*H*W + y*W + x`.
- `blocks` has one entry per sample. Each entry is a non-empty list of blocks `(t0, t, top, h, left, w)`: the block covers times `t0 .. t0+t-1`, rows `top .. top+h-1` and columns `left .. left+w-1`.
- A sample's target is the union of its blocks: every token covered by at least one block. Its context is every other token. Both lists are in ascending flat-index order.
- Truncate every sample's target list to the shortest target list in the batch, and every context list to the shortest context list, keeping the first (lowest) indices. When `max_keep` is given, the context length is also capped at `max_keep`; the target length is not.
- Return two `torch.int64` tensors, `ctx_idx` of shape `(B, K_ctx)` and `tgt_idx` of shape `(B, K_tgt)`.
- Raise `ValueError` when any sample's blocks cover the whole grid (its context would be empty).
- Raise `ValueError` for a `grid` that is not three positive integers, an empty batch, a sample with no blocks, a block whose sizes are not at least 1 or that does not fit inside the grid, or a `max_keep` that is not a positive integer.

**Example:** `make_masks((1, 3, 3), [[(0, 1, 0, 2, 0, 2)], [(0, 1, 1, 1, 0, 3), (0, 1, 0, 2, 2, 1), (0, 1, 2, 1, 0, 1)]])`
- Sample 0 targets `[0, 1, 3, 4]` and keeps context `[2, 5, 6, 7, 8]`.
- Sample 1 targets `[2, 3, 4, 5, 6]` (two blocks overlap at 5) and keeps context `[0, 1, 7, 8]`.
- Both truncate to length 4: `ctx_idx = [[2, 5, 6, 7], [0, 1, 7, 8]]`, `tgt_idx = [[0, 1, 3, 4], [2, 3, 4, 5]]`.""",
    },
    {
        "title": "The latent prediction loss",
        "description_en": r"""Keep Part 1. Add `jepa_loss(pred, target_feats, tgt_idx, loss_exp=1.0) -> Tensor`.

- `target_feats` is the target encoder's output for every token, shape `(B, N, D)`. Layer-normalize it over the last dimension with `eps = 1e-5` and no learnable scale or shift, then gather the tokens named by `tgt_idx` (shape `(B, K)`, row `b` indexes sample `b`) to get `h` of shape `(B, K, D)`.
- The loss for one mask is `mean(|pred - h| ** loss_exp) / loss_exp`, a scalar, where `pred` has shape `(B, K, D)`.
- `pred` and `tgt_idx` may also be equal-length lists, one entry per mask, all indexing the same `target_feats`. The loss is then the plain average of the per-mask losses, not a mean over all their tokens pooled together.
- No gradient may reach `target_feats`. Gradients flow to `pred` as usual.
- Return a 0-d tensor with the dtype of `pred`. Tests use `float64` and compare with a tolerance of `1e-10`.
- Raise `ValueError` when `loss_exp <= 0`, when `target_feats` is not 3-D, when a `pred` does not have shape `(B, K, D)` for its `tgt_idx` of shape `(B, K)`, or when `pred` and `tgt_idx` are not both tensors or both lists of the same non-zero length.

**Example:** `target_feats = [[[1, 3], [0, 4], [2, 2]]]` normalizes to about `[[[-1, 1], [-1, 1], [0, 0]]]`. With `tgt_idx = [[2, 0]]` and `pred = [[[0.5, 0.5], [-1, 1]]]`, the loss is about `0.25` with `loss_exp = 1` and `0.0625` with `loss_exp = 2`.""",
    },
    {
        "title": "The EMA target encoder",
        "description_en": r"""Keep Parts 1–2. Add `ema_update(target_params, online_params, m) -> None` and `momentum(step, total_steps, start, end) -> float`.

- `ema_update` sets every target tensor to `m * target + (1 - m) * online`, in place: the same tensor objects hold the new values afterwards. The online tensors are not modified.
- Both arguments are iterables of tensors, such as `module.parameters()`; a generator may be iterated only once. Pair them in order.
- The update must work when target tensors have `requires_grad=True`, and must not record an autograd graph: run it without gradient tracking.
- Raise `ValueError`, before changing any tensor, when the two iterables have different lengths, when a pair's shapes differ, or when `m` is outside `[0, 1]`.
- `momentum` is the linear schedule `start + step * (end - start) / total_steps` for `step` in `0 .. total_steps`, so step 0 gives `start` and step `total_steps` gives `end`. Raise `ValueError` when `total_steps < 1` or `step` is outside that range. Tests compare with a tolerance of `1e-12`.

**Example:** with `target = [tensor([1., 2.]), tensor([[0.]])]` and `online = [tensor([3., 0.]), tensor([[10.]])]`, `ema_update(target, online, 0.75)` leaves `target` as `[tensor([1.5, 1.5]), tensor([[2.5]])]`. `momentum(5, 10, 0.99, 1.0)` is `0.995`.""",
    },
]

TESTS = [
    # Part 1: masks.
    {"name": "Part 1: union target, complement context, batch truncation", "part": 1, "behavior": "state.invariant",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
blocks = [[(0, 1, 0, 2, 0, 2)], [(0, 1, 1, 1, 0, 3), (0, 1, 0, 2, 2, 1), (0, 1, 2, 1, 0, 1)]]
check_masks(jepa.make_masks((1, 3, 3), blocks), ([[2, 5, 6, 7], [0, 1, 7, 8]], [[0, 1, 3, 4], [2, 3, 4, 5]]), "example")
# Flat index is t*H*W + y*W + x: time 1, row 0, columns 1-2 of a (2, 2, 3) grid.
ctx, tgt = jepa.make_masks((2, 2, 3), [[(1, 1, 0, 1, 1, 2)]])
assert tgt.tolist() == [[7, 8]], tgt
assert ctx.tolist() == [[0, 1, 2, 3, 4, 5, 6, 9, 10, 11]], ctx
"""},
    {"name": "Part 1: seeded batches match a brute-force mask", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "The target is every token covered by any block, the context is the rest, both ascending, each truncated to its batch minimum.",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
for seed, grid, batch, nblocks in ((0, (4, 5, 6), 3, 3), (1, (2, 3, 7), 4, 2), (2, (3, 4, 4), 2, 5), (3, (1, 6, 5), 5, 4), (4, (5, 2, 3), 3, 2)):
    g = torch.Generator().manual_seed(seed)
    blocks = random_blocks(g, grid, batch, nblocks)
    check_masks(jepa.make_masks(grid, blocks), model_masks(grid, blocks), seed)
"""},
    {"name": "Part 1: flat index order on a non-square grid", "part": 1, "visibility": "unshown", "behavior": "tensor.shape",
     "failure_message": "Flatten the (T, H, W) grid row-major: token (t, y, x) has index t*H*W + y*W + x, so x varies fastest.",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
for grid, block in (((3, 2, 5), (1, 2, 0, 1, 3, 2)), ((2, 4, 3), (0, 1, 1, 3, 2, 1)), ((2, 3, 2), (1, 1, 2, 1, 0, 2))):
    check_masks(jepa.make_masks(grid, [[block]]), model_masks(grid, [[block]]), grid)
"""},
    {"name": "Part 1: overlapping and repeated blocks count once", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Take the union of the blocks: a token covered by two blocks is one target token, and a token outside every block is context.",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
grid = (2, 4, 4)
one = (0, 2, 1, 2, 1, 2)
blocks = [[one, one, one], [(0, 1, 0, 3, 0, 3), (0, 2, 2, 2, 2, 2)], [one, (1, 1, 0, 4, 0, 1)]]
check_masks(jepa.make_masks(grid, blocks), model_masks(grid, blocks), "overlap")
ctx, tgt = jepa.make_masks(grid, [[one, one]])
assert tgt.shape == (1, 8) and ctx.shape == (1, 24), (tgt.shape, ctx.shape)
"""},
    {"name": "Part 1: truncation keeps the lowest indices", "part": 1, "visibility": "unshown", "behavior": "tensor.shape",
     "failure_message": "Cut each list to the batch minimum by keeping its first (lowest) indices, separately for context and target.",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
grid = (2, 3, 4)
blocks = [[(0, 2, 0, 3, 0, 1)], [(1, 1, 1, 2, 1, 3)], [(0, 1, 0, 1, 0, 4), (1, 1, 2, 1, 2, 2)]]
want = model_masks(grid, blocks)
ctx, tgt = jepa.make_masks(grid, blocks)
assert tgt.shape == (3, 6) and ctx.shape == (3, 18), (tgt.shape, ctx.shape)
check_masks((ctx, tgt), want, "lowest")
"""},
    {"name": "Part 1: max_keep caps the context only", "part": 1, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "max_keep caps the context length at min(batch minimum, max_keep) and never changes the target length.",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
g = torch.Generator().manual_seed(9)
grid = (3, 4, 5)
blocks = random_blocks(g, grid, 3, 3)
for max_keep in (1, 4, 7, 10 ** 6):
    check_masks(jepa.make_masks(grid, blocks, max_keep=max_keep), model_masks(grid, blocks, max_keep), max_keep)
ctx, tgt = jepa.make_masks(grid, blocks, max_keep=1)
assert ctx.shape[1] == 1 and tgt.shape[1] == min(len(x) for x in model_masks(grid, blocks)[1]), (ctx.shape, tgt.shape)
"""},
    {"name": "Part 1: an empty context raises, a one-token context does not", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Raise ValueError when a sample's blocks cover every token; a context of one token is valid and gives shape (B, 1).",
     "code": _MASK_MODEL + r"""
jepa = {fn}()
grid = (2, 2, 2)
for blocks in ([[(0, 2, 0, 2, 0, 2)]], [[(0, 1, 0, 1, 0, 1)], [(0, 2, 0, 2, 0, 1), (0, 2, 0, 2, 1, 1)]]):
    try:
        jepa.make_masks(grid, blocks)
    except ValueError:
        pass
    else:
        raise AssertionError(f"empty context accepted: {blocks}")
blocks = [[(0, 2, 0, 2, 0, 1), (0, 1, 0, 2, 1, 1), (1, 1, 1, 1, 1, 1)], [(0, 1, 0, 1, 0, 1)]]
ctx, tgt = jepa.make_masks(grid, blocks)
assert ctx.tolist() == [[5], [1]] and tgt.tolist() == [[0], [0]], (ctx.tolist(), tgt.tolist())
"""},
    {"name": "Part 1: rejects malformed grids and blocks", "part": 1, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "Raise ValueError for an empty batch, a sample with no blocks, a zero-size block, a block outside the grid, or max_keep < 1.",
     "code": r"""
jepa = {fn}()
grid = (2, 3, 3)
ok = (0, 1, 0, 1, 0, 1)
bad = [
    lambda: jepa.make_masks(grid, []),
    lambda: jepa.make_masks(grid, [[ok], []]),
    lambda: jepa.make_masks(grid, [[(0, 0, 0, 1, 0, 1)]]),
    lambda: jepa.make_masks(grid, [[(0, 1, 0, 1, 0, 0)]]),
    lambda: jepa.make_masks(grid, [[(1, 2, 0, 1, 0, 1)]]),
    lambda: jepa.make_masks(grid, [[(0, 1, 2, 2, 0, 1)]]),
    lambda: jepa.make_masks(grid, [[(0, 1, 0, 1, -1, 1)]]),
    lambda: jepa.make_masks(grid, [[(0, 1, 0, 1, 1, 3)]]),
    lambda: jepa.make_masks(grid, [[ok]], max_keep=0),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    # Part 2: loss.
    {"name": "Part 2: layer-normalized targets, L1 and L2", "part": 2, "behavior": "state.invariant",
     "code": r"""
import torch
jepa = {fn}()
feats = torch.tensor([[[1.0, 3.0], [0.0, 4.0], [2.0, 2.0]]], dtype=torch.float64)
idx = torch.tensor([[2, 0]])
pred = torch.tensor([[[0.5, 0.5], [-1.0, 1.0]]], dtype=torch.float64)
l1 = jepa.jepa_loss(pred, feats, idx)
l2 = jepa.jepa_loss(pred, feats, idx, loss_exp=2.0)
assert l1.shape == () and l2.shape == (), (l1.shape, l2.shape)
assert abs(l1.item() - 0.25) < 1e-5, l1
assert abs(l2.item() - 0.0625) < 1e-5, l2
"""},
    {"name": "Part 2: seeded losses match the formula", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Layer-normalize target_feats per token (eps 1e-5, no affine), gather row b with tgt_idx[b], then return mean(|pred - h| ** p) / p.",
     "code": _LOSS_MODEL + r"""
jepa = {fn}()
for seed in (0, 1, 2, 3, 4):
    pred, feats, idx = seeded(seed)
    for p in (1.0, 1.5, 2.0, 3.0):
        got = jepa.jepa_loss(pred, feats, idx, loss_exp=p)
        want = model_loss(pred, feats, idx, p)
        assert abs(got.item() - want.item()) < 1e-10, (seed, p, got.item(), want.item())
"""},
    {"name": "Part 2: predicting the normalized targets gives zero loss", "part": 2, "visibility": "unshown", "behavior": "numerics.stability",
     "failure_message": "Regress onto layer_norm(target_feats) over the feature dimension, not onto the raw features or a norm over tokens.",
     "code": _LOSS_MODEL + r"""
jepa = {fn}()
g = torch.Generator().manual_seed(5)
feats = torch.randn(2, 7, 5, generator=g, dtype=torch.float64) * 20 + 50
idx = torch.tensor([[6, 0, 3], [1, 1, 4]])
h = normalized(feats)
pred = torch.stack([h[b, idx[b]] for b in range(2)])
for p in (1.0, 2.0):
    loss = jepa.jepa_loss(pred, feats, idx, loss_exp=p)
    assert abs(loss.item()) < 1e-10, (p, loss.item())
"""},
    {"name": "Part 2: several masks are averaged per mask", "part": 2, "visibility": "unshown", "behavior": "metrics.averaging",
     "failure_message": "With lists, compute one loss per (pred, tgt_idx) pair and return their plain average, not a mean over all tokens pooled together.",
     "code": _LOSS_MODEL + r"""
jepa = {fn}()
g = torch.Generator().manual_seed(8)
feats = torch.randn(2, 12, 4, generator=g, dtype=torch.float64)
idx_a = torch.stack([torch.randperm(12, generator=g)[:2] for _ in range(2)])
idx_b = torch.stack([torch.randperm(12, generator=g)[:7] for _ in range(2)])
pred_a = torch.randn(2, 2, 4, generator=g, dtype=torch.float64) * 3
pred_b = torch.randn(2, 7, 4, generator=g, dtype=torch.float64)
for p in (1.0, 2.0):
    got = jepa.jepa_loss([pred_a, pred_b], feats, [idx_a, idx_b], loss_exp=p)
    want = (model_loss(pred_a, feats, idx_a, p) + model_loss(pred_b, feats, idx_b, p)) / 2
    assert abs(got.item() - want.item()) < 1e-10, (p, got.item(), want.item())
    single = jepa.jepa_loss([pred_b], feats, [idx_b], loss_exp=p)
    assert abs(single.item() - model_loss(pred_b, feats, idx_b, p).item()) < 1e-10, p
"""},
    {"name": "Part 2: gradients reach pred and never the targets", "part": 2, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "Compute the normalized targets without gradient (torch.no_grad or detach) so target_feats gets no gradient; pred still must.",
     "code": _LOSS_MODEL + r"""
jepa = {fn}()
pred, feats, idx = seeded(11)
pred = pred.clone().requires_grad_(True)
feats = feats.clone().requires_grad_(True)
loss = jepa.jepa_loss(pred, feats, idx, loss_exp=2.0)
loss.backward()
assert feats.grad is None, "a gradient reached target_feats"
h = normalized(feats.detach())
picked = torch.stack([h[b, idx[b]] for b in range(3)])
want = (pred.detach() - picked) / pred.numel()
assert pred.grad is not None and torch.allclose(pred.grad, want, atol=1e-12), "wrong gradient for pred"
"""},
    {"name": "Part 2: returns a scalar in pred's dtype", "part": 2, "visibility": "unshown", "behavior": "tensor.dtype_device",
     "failure_message": "Return a 0-d tensor in pred's dtype; do not call .item() or cast to float32.",
     "code": _LOSS_MODEL + r"""
jepa = {fn}()
pred, feats, idx = seeded(2)
loss = jepa.jepa_loss(pred, feats, idx)
assert isinstance(loss, torch.Tensor) and loss.shape == () and loss.dtype == torch.float64, (type(loss), getattr(loss, "shape", None))
loss32 = jepa.jepa_loss(pred.float(), feats.float(), idx)
assert loss32.dtype == torch.float32 and abs(loss32.item() - loss.item()) < 1e-5, loss32
"""},
    {"name": "Part 2: rejects bad exponents and shapes", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "Raise ValueError for loss_exp <= 0, a pred that is not (B, K, D), non-3-D target_feats, or mismatched lists.",
     "code": r"""
import torch
jepa = {fn}()
feats = torch.randn(2, 6, 3)
idx = torch.tensor([[0, 1], [2, 3]])
pred = torch.randn(2, 2, 3)
bad = [
    lambda: jepa.jepa_loss(pred, feats, idx, loss_exp=0.0),
    lambda: jepa.jepa_loss(pred, feats, idx, loss_exp=-1.0),
    lambda: jepa.jepa_loss(torch.randn(2, 3, 3), feats, idx),
    lambda: jepa.jepa_loss(torch.randn(1, 2, 3), feats, idx),
    lambda: jepa.jepa_loss(torch.randn(2, 2, 4), feats, idx),
    lambda: jepa.jepa_loss(pred, feats[0], idx),
    lambda: jepa.jepa_loss([pred, pred], feats, [idx]),
    lambda: jepa.jepa_loss([], feats, []),
    lambda: jepa.jepa_loss(pred, feats, [idx]),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    # Part 3: EMA target.
    {"name": "Part 3: one EMA step and the momentum schedule", "part": 3, "behavior": "optim.state",
     "code": r"""
import torch
jepa = {fn}()
target = [torch.tensor([1.0, 2.0]), torch.tensor([[0.0]])]
online = [torch.tensor([3.0, 0.0]), torch.tensor([[10.0]])]
first = target[0]
jepa.ema_update(target, online, 0.75)
assert target[0] is first, "update the target tensors in place"
assert torch.allclose(target[0], torch.tensor([1.5, 1.5])) and torch.allclose(target[1], torch.tensor([[2.5]])), target
assert abs(jepa.momentum(0, 10, 0.99, 1.0) - 0.99) < 1e-12
assert abs(jepa.momentum(5, 10, 0.99, 1.0) - 0.995) < 1e-12
assert abs(jepa.momentum(10, 10, 0.99, 1.0) - 1.0) < 1e-12
"""},
    {"name": "Part 3: seeded EMA is in place and leaves online untouched", "part": 3, "visibility": "unshown", "behavior": "optim.state",
     "failure_message": "Write m * target + (1 - m) * online into each target tensor in place; never write into the online tensors.",
     "code": r"""
import torch
jepa = {fn}()
for seed in (0, 1, 2):
    g = torch.Generator().manual_seed(seed)
    shapes = [(4, 3), (3,), (2, 2, 2)]
    target = [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]
    online = [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]
    t0, o0 = [t.clone() for t in target], [o.clone() for o in online]
    ptrs = [t.data_ptr() for t in target]
    objs = list(target)
    for m in (0.9, 0.99925, 0.3):
        want = [m * t + (1 - m) * o for t, o in zip(target, online)]
        jepa.ema_update(target, online, m)
        for t, w in zip(target, want):
            assert torch.allclose(t, w, atol=1e-12), (seed, m)
    assert all(a is b for a, b in zip(target, objs)) and [t.data_ptr() for t in target] == ptrs, "not in place"
    assert all(torch.equal(o, c) for o, c in zip(online, o0)), "online tensors were modified"
"""},
    {"name": "Part 3: momentum 1 keeps the target, 0 copies online", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "m weights the old target: m = 1 leaves it unchanged and m = 0 copies the online weights.",
     "code": r"""
import torch
jepa = {fn}()
g = torch.Generator().manual_seed(4)
target = [torch.randn(5, generator=g, dtype=torch.float64)]
online = [torch.randn(5, generator=g, dtype=torch.float64)]
before = target[0].clone()
jepa.ema_update(target, online, 1.0)
assert torch.equal(target[0], before), "m = 1 changed the target"
jepa.ema_update(target, online, 0.0)
assert torch.allclose(target[0], online[0], atol=1e-15), "m = 0 did not copy online"
"""},
    {"name": "Part 3: works on module parameters without building a graph", "part": 3, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "Run the update under torch.no_grad() so in-place writes to parameters that require grad are allowed and record nothing.",
     "code": r"""
import torch
import torch.nn as nn
jepa = {fn}()
torch.manual_seed(0)
online = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).double()
target = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).double()
want = [0.6 * t.detach() + 0.4 * o.detach() for t, o in zip(target.parameters(), online.parameters())]
jepa.ema_update(target.parameters(), online.parameters(), 0.6)
for t, w in zip(target.parameters(), want):
    assert t.requires_grad and t.grad_fn is None, "the update recorded a graph"
    assert torch.allclose(t.detach(), w, atol=1e-12), "wrong update through parameters() generators"
"""},
    {"name": "Part 3: mismatched inputs raise before anything changes", "part": 3, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "Check lengths, shapes and 0 <= m <= 1 first and raise ValueError without modifying any target tensor.",
     "code": r"""
import torch
jepa = {fn}()
a, b = torch.ones(3), torch.ones(2, 2)
cases = [
    ([a.clone(), b.clone()], [torch.zeros(3)], 0.5),
    ([a.clone()], [torch.zeros(3), torch.zeros(2, 2)], 0.5),
    ([a.clone(), b.clone()], [torch.zeros(3), torch.zeros(4)], 0.5),
    ([a.clone()], [torch.zeros(3)], 1.5),
    ([a.clone()], [torch.zeros(3)], -0.1),
]
for i, (target, online, m) in enumerate(cases):
    try:
        jepa.ema_update(target, online, m)
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
    assert all(torch.all(t == 1) for t in target), f"case {i} modified a target before raising"
"""},
    {"name": "Part 3: the momentum schedule is linear with both ends included", "part": 3, "visibility": "unshown", "behavior": "optim.state",
     "failure_message": "momentum(step, total, start, end) = start + step * (end - start) / total for 0 <= step <= total; raise ValueError outside.",
     "code": r"""
jepa = {fn}()
for start, end, total in ((0.996, 1.0, 1000), (0.99925, 0.99925, 3750), (0.5, 0.9, 7)):
    for step in sorted({0, 1, total // 3, total - 1, total}):
        want = start + step * (end - start) / total
        got = jepa.momentum(step, total, start, end)
        assert isinstance(got, float), type(got)
        assert abs(got - want) < 1e-12, (start, end, total, step, got, want)
for args in ((-1, 10, 0.9, 1.0), (11, 10, 0.9, 1.0), (0, 0, 0.9, 1.0)):
    try:
        jepa.momentum(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{args} was accepted")
"""},
]

TASK = {
    "title": "JEPA Block Masks, Latent Loss and EMA Target",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "JEPA",
    "description_en": r"""Write `JEPA`, the deterministic pieces of a V-JEPA 2 training step: which video tokens the context encoder sees and which ones the predictor must predict, the loss on the predicted features, and the slowly moving target encoder that produces those features.

The requirement arrives in parts. Each part keeps every earlier method, so one `JEPA` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `JEPA()` takes no arguments and keeps no state between calls.
- A video is a grid of `T x H x W` tokens; token `(t, y, x)` has flat index `t*H*W + y*W + x`.
- Block placements are given, so nothing is random. Tests are seeded and run on CPU.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The objective.** A JEPA (joint-embedding predictive architecture) hides part of the input and predicts the hidden part's representation, not its pixels. The context encoder sees only the context tokens; a predictor takes its output plus position tokens for the target and outputs one feature per target token. A target encoder sees the whole clip, and its layer-normalized features at the target positions are the regression targets. V-JEPA 2 uses an L1 loss (`loss_exp = 1`).

**Multi-block masks.** V-JEPA 2 draws several rectangular blocks per clip, each spanning all time steps, and predicts their union. Whole tubes through time stop the model from copying a patch from the neighbouring frame. The released ViT-L recipe mixes 8 small blocks (15% of the frame each) with 2 large ones (70% each), as two separate masks averaged in the loss. Upstream draws one block size per batch and a position per sample, so samples end up with different numbers of tokens; truncating to the batch minimum lets them stack into one tensor. The truncation keeps the lowest flat indices, which slightly favours early frames and top rows.

**Why the target is safe to chase.** The target branch gets no gradient and is updated only as an exponential moving average of the context encoder, `target = m * target + (1 - m) * online`. With the stop-gradient and the layer norm, the trivial solution of predicting a constant is not a fixed point the encoders can reach together. V-JEPA 2's configs keep `m = 0.99925` constant; the code builds a linear schedule from `ema[0]` to `ema[1]` over every step, so a ramp is one config change away.""",
    "parts": _PARTS,
    "advisory_prerequisites": ["vit_patch", "layernorm", "contrastive_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If you mark covered tokens in a (T, H, W) boolean grid and flatten it, what order do the indices come out in, and what is the context? Which tensor in the loss must be cut off from autograd, and why does averaging per mask differ from pooling tokens when masks have different sizes? How do you change a parameter's values without replacing the tensor object or recording a graph?"},
        {"level": 2, "kind": "analysis", "content": "Validate inputs first. For each sample, fill a boolean grid with every block's slice, flatten it, and take the indices of true entries as the target and of false entries as the context; raise if the context is empty. Find the two batch minimums (cap the context one with max_keep), slice each list and stack. In the loss, normalize the targets with layer_norm under no_grad, gather with the indices expanded over the feature dimension, and average the per-mask losses. For EMA, materialize both iterables into lists, check them, then multiply and add in place under no_grad."},
    ],
    "model_connections": [
        "V-JEPA 2's MaskCollator builds exactly these context and target index lists, and apply_masks gathers them from the encoder output.",
        "The same EMA target encoder appears in BYOL, DINO and I-JEPA; MoCo uses it for the key encoder.",
        "V-JEPA 2-AC, the action-conditioned world model, plans by comparing predicted latent states, which only works because the encoder was trained to be predictable in latent space.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Predicting features instead of pixels spends capacity on what is predictable and ignores unpredictable texture detail.",
            "Index-list masks let the context encoder skip masked tokens entirely, which cuts compute in proportion to the mask ratio.",
            "An EMA target needs no negatives and no second optimizer.",
        ],
        "cons": [
            "Truncating to the batch minimum throws away some context and target tokens, always the highest-indexed ones.",
            "Collapse is avoided by architecture and schedule choices, not by the loss itself, so bad hyperparameters can still collapse.",
            "The target encoder doubles parameter memory, and its lag makes early training slower to bootstrap.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/masks/multiseq_multiblock3d.py",
            "commit": "204698b45b3712590f06245fbfba32d3be539812",
            "path": "src/masks/multiseq_multiblock3d.py",
            "symbol": "_MaskGenerator._sample_block_mask, _MaskGenerator.__call__",
            "license": "MIT",
            "adapted": "Union of blocks as the prediction mask, its complement as the context, ascending flat indices, truncation of both lists to the batch minimum, and the max_keep cap on the context.",
            "simplifications": "Block sizes and positions are passed in instead of drawn, and each block may have its own size (upstream: one size per batch); "
                               "an empty context raises ValueError instead of redrawing the blocks; "
                               "upstream's .squeeze() turns a one-token mask into a 0-d tensor that len() rejects, while the exercise always returns (B, K); "
                               "max_context_frames_ratio, full_complement, pred_full_complement and inv_block are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/masks/utils.py",
            "commit": "204698b45b3712590f06245fbfba32d3be539812",
            "path": "src/masks/utils.py",
            "symbol": "apply_masks",
            "license": "MIT",
            "adapted": "Gathering target tokens by per-sample index rows.",
            "simplifications": "Used with concat=False only, inside the loss.",
        },
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa/train.py",
            "commit": "204698b45b3712590f06245fbfba32d3be539812",
            "path": "app/vjepa/train.py",
            "symbol": "forward_target, loss_fn, momentum_scheduler, momentum update of target_encoder",
            "license": "MIT",
            "adapted": "Layer norm on the target features under no_grad, mean(|z - h| ** loss_exp) / loss_exp averaged over masks, the in-place EMA update and the linear momentum schedule.",
            "simplifications": "One clip group instead of a list per frames-per-clip; the momentum schedule is a function of the step instead of a generator; "
                               "mismatched parameter lists raise instead of being cut short by zip; mixed precision and DDP are dropped.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2506.09985",
            "section": "Assran et al. 2025 (V-JEPA 2), Section 2.1 (Methodology) and Section 2.4 (Pretraining Recipe)",
            "note": "Mask-denoising in representation space with an L1 loss, stop-gradient and an EMA target encoder.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2404.08471",
            "section": "Bardes et al. 2024 (V-JEPA), Section 3.1 (Training Objective) and Section 3.2 (Prediction Task: multi-block masking)",
            "note": "Why the target encoder uses stop-gradient and EMA, and the short-range plus long-range block masks.",
        },
    ],
    "tests": TESTS,
    "solution": r'''import torch
import torch.nn.functional as F


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class JEPA:
    def make_masks(self, grid, blocks, max_keep=None):
        if len(grid) != 3 or not all(_positive_int(n) for n in grid):
            raise ValueError("grid must be three positive integers (T, H, W)")
        if not blocks:
            raise ValueError("blocks needs one entry per sample")
        if max_keep is not None and not _positive_int(max_keep):
            raise ValueError("max_keep must be a positive integer")
        T, H, W = grid
        contexts, targets = [], []
        for sample in blocks:
            if not sample:
                raise ValueError("every sample needs at least one block")
            covered = torch.zeros((T, H, W), dtype=torch.bool)
            for t0, t, top, h, left, w in sample:
                if min(t, h, w) < 1 or min(t0, top, left) < 0 or t0 + t > T or top + h > H or left + w > W:
                    raise ValueError(f"block {(t0, t, top, h, left, w)} does not fit in grid {grid}")
                covered[t0:t0 + t, top:top + h, left:left + w] = True
            covered = covered.flatten()  # row-major: index = t*H*W + y*W + x
            context = torch.nonzero(~covered).flatten()
            if context.numel() == 0:
                raise ValueError("the blocks cover the whole grid, so the context is empty")
            contexts.append(context)
            targets.append(torch.nonzero(covered).flatten())
        k_ctx = min(len(c) for c in contexts)
        if max_keep is not None:
            k_ctx = min(k_ctx, max_keep)
        k_tgt = min(len(x) for x in targets)
        ctx_idx = torch.stack([c[:k_ctx] for c in contexts])
        tgt_idx = torch.stack([x[:k_tgt] for x in targets])
        return ctx_idx, tgt_idx

    def jepa_loss(self, pred, target_feats, tgt_idx, loss_exp=1.0):
        if not loss_exp > 0:
            raise ValueError("loss_exp must be positive")
        if isinstance(pred, torch.Tensor) and isinstance(tgt_idx, torch.Tensor):
            preds, indices = [pred], [tgt_idx]
        elif isinstance(pred, (list, tuple)) and isinstance(tgt_idx, (list, tuple)):
            preds, indices = list(pred), list(tgt_idx)
        else:
            raise ValueError("pred and tgt_idx must both be tensors or both be lists")
        if not preds or len(preds) != len(indices):
            raise ValueError("pred and tgt_idx must be lists of the same non-zero length")
        if target_feats.ndim != 3:
            raise ValueError("target_feats must have shape (B, N, D)")
        B, _, D = target_feats.shape
        for z, idx in zip(preds, indices):
            if idx.ndim != 2 or idx.shape[0] != B or z.shape != (B, idx.shape[1], D):
                raise ValueError("each pred must be (B, K, D) for a tgt_idx of shape (B, K)")
        with torch.no_grad():
            h = F.layer_norm(target_feats, (D,))  # eps 1e-5, no affine
        loss = 0
        for z, idx in zip(preds, indices):
            h_masked = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, D))
            loss = loss + torch.mean(torch.abs(z - h_masked) ** loss_exp) / loss_exp
        return loss / len(preds)  # each mask counts once, whatever its size

    def ema_update(self, target_params, online_params, m):
        target_params, online_params = list(target_params), list(online_params)
        if len(target_params) != len(online_params):
            raise ValueError("target and online parameter lists differ in length")
        if any(t.shape != o.shape for t, o in zip(target_params, online_params)):
            raise ValueError("target and online parameter shapes differ")
        if not 0.0 <= m <= 1.0:
            raise ValueError("m must be in [0, 1]")
        with torch.no_grad():
            for target, online in zip(target_params, online_params):
                target.mul_(m).add_(online, alpha=1 - m)

    def momentum(self, step, total_steps, start, end):
        if total_steps < 1 or not 0 <= step <= total_steps:
            raise ValueError("need total_steps >= 1 and 0 <= step <= total_steps")
        return float(start + step * (end - start) / total_steps)
''',
    "interview_questions": interview(
        concept=[
            "Why does a JEPA predict the target encoder's features of the hidden region instead of reconstructing its pixels, as a masked autoencoder does?",
            "Both the prediction and its target come from learned encoders. What stops a JEPA from collapsing to a constant representation?",
        ],
        deep_dive=[
            "V-JEPA 2 masks the union of several blocks that each span every frame. Why mask whole tubes through time rather than independent random patches?",
            "Each sample's context and target index lists are truncated to the batch minimum. Why is that needed, and what bias does keeping the lowest indices introduce?",
            "Why are the target features layer-normalized and computed without gradient, and what goes wrong if gradients flow into the target branch?",
        ],
        tradeoffs=[
            "L1 versus L2 regression for the latent loss: how does loss_exp change the gradient and the sensitivity to outlier tokens?",
            "A constant EMA momentum such as 0.99925 versus a schedule that ramps toward 1: what do you trade between target stability and how quickly the target tracks the online encoder?",
        ],
    ),
}
