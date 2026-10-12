"""DreamerV3 reward and value head: symlog, symexp, two-hot targets in symlog space and a symmetric mean."""

from ._interview import interview

# Bins whose values are exactly symlog of known raw values, and are not evenly spaced.
_UNEVEN_BINS = r"""
import torch

def uneven_bins(seed, n_pos, dtype=torch.float64, middle=True):
    g = torch.Generator().manual_seed(seed)
    raw = torch.sort(torch.rand(n_pos, generator=g, dtype=dtype) * 50 + 0.01).values
    raw = torch.unique(raw)
    pos = torch.sign(raw) * torch.log1p(torch.abs(raw))
    zero = torch.zeros(1, dtype=dtype) if middle else torch.zeros(0, dtype=dtype)
    bins = torch.cat([-pos.flip(0), zero, pos])
    raws = torch.cat([-raw.flip(0), zero, raw])
    return bins, raws
"""

TASK = {
    "title": "DreamerV3 Symlog Two-Hot Head",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "twohot_loss",
    "description_en": r"""Build the pieces of the DreamerV3 reward and value head: the symlog squash, a two-hot target over symmetric bins in symlog space, its cross-entropy loss, and a prediction that is exactly zero when the head is uniform.

**Signatures:** define all six functions at the top level of your code.
- `symlog(x) -> Tensor` and `symexp(x) -> Tensor`
- `symmetric_bins(K, limit=20.0) -> Tensor`
- `twohot_target(y, bins) -> Tensor`
- `twohot_loss(logits, y, bins) -> Tensor`
- `twohot_mean(logits, bins) -> Tensor`

**symlog and symexp:**
- `symlog(x) = sign(x) * log1p(|x|)` and `symexp(x) = sign(x) * expm1(|x|)`, elementwise. They are inverses.
- Use `log1p` and `expm1`, not `log(1 + |x|)` and `exp(|x|) - 1`: in float64, `symlog(1e-20)` must be `1e-20` to a relative error of `1e-12`.

**symmetric_bins:**
- Returns `K` float32 bin positions in symlog space, equal to `torch.linspace(-limit, limit, K)` within `1e-5`.
- Exactly antisymmetric: `bins[K - 1 - i] == -bins[i]` bit for bit. For odd `K` the middle bin is exactly 0; for even `K` no bin is 0.
- Raise `ValueError` unless `K` is an integer of at least 2 and `limit > 0`.

**Bins in general.** `bins` is a 1-D tensor of `K >= 2` strictly increasing values in symlog space. It need not be evenly spaced, symmetric, or come from `symmetric_bins`. Cast it to the dtype of the other input before use.

**twohot_target:** `y` is a float tensor of any shape `(...)` in raw space. Return a `(..., K)` tensor in the dtype of `y`:
- Let `x = symlog(y)`. `below` is the index of the last bin `<= x`, and `above` the index of the first bin `> x`, both clipped to `0 .. K-1`.
- When `below != above`, put `|bins[above] - x| / (bins[above] - bins[below])` on `below` and the rest on `above`, so that `sum(target * bins) == x`.
- A target exactly on a bin puts all its mass on that bin. A target below the first bin or above the last puts all its mass on that edge bin.
- Every row sums to 1 and has at most two adjacent non-zero entries.

**twohot_loss:** `logits` has shape `(..., K)` and `y` shape `(...)`. Return the per-element cross-entropy `-(twohot_target(y, bins) * log_softmax(logits)).sum(-1)`, shape `(...)`, in the dtype of `logits`. Do not reduce over the batch. The target is a constant: no gradient flows into `y`.

**twohot_mean:** return `symexp` of the expected bin position under `softmax(logits)`, shape `(...)`, in the dtype of `logits`. Compute the expectation as a symmetric sum:
- Multiply each probability by its bin. Add the product at bin `i` to the product at bin `K - 1 - i` for every `i < K // 2`, then sum those pair sums. For odd `K`, add the middle bin's product.
- So when the bins are antisymmetric and `probs[i] == probs[K - 1 - i]` for all `i` (uniform logits, for example), the result is exactly `0.0`, not a rounding error like `-7e-8`. This must hold for even and odd `K`, in float32 and float64.

**Errors.** `twohot_target`, `twohot_loss` and `twohot_mean` raise `ValueError` when `bins` is not 1-D with at least 2 values, when `logits.shape[-1] != K`, or when `logits.shape[:-1] != y.shape`.

**Example:**

    bins = symmetric_bins(5, 2.0)                 # [-2., -1., 0., 1., 2.]
    twohot_target(symexp(torch.tensor(0.25)), bins)
    # [0., 0., 0.75, 0.25, 0.]   symlog(y) = 0.25, a quarter of the way from the bin at 0 to the bin at 1
    twohot_mean(torch.zeros(5), bins)             # exactly 0.0
    twohot_loss(torch.zeros(5), torch.tensor(0.0), bins)   # log(5)

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why symlog.** Rewards and returns range from fractions to thousands across tasks, and a squared-error head on raw values lets the large ones dominate the gradient. symlog compresses large magnitudes like a logarithm, stays close to the identity near zero, and is defined for negative values. DreamerV3 trains every reward and value head in this space, so one set of hyperparameters works across domains.

**Why two-hot instead of regression.** Predicting a categorical distribution over fixed bins turns regression into classification. The gradient of a cross-entropy has the same scale whatever the target's magnitude, and a distribution can represent several plausible returns. Two-hot encoding spreads a continuous target over its two neighbouring bins, with weights chosen so that the expected bin position is exactly the target. DreamerV3 uses 255 bins between -20 and 20 in symlog space, which covers raw values up to about 5e8.

**Why a symmetric sum.** The output layer of the reward and value heads is initialized to zero, so the first predictions are a uniform softmax. With a plain `(probs * bins).sum()` in float32, uniform probabilities over 255 symmetric bins give a value of order 1e-8 to 1e-7 instead of 0, and that small bias feeds back into bootstrapped value targets. Pairing each bin with its mirror makes every pair cancel exactly.

**Two places to interpolate.** This exercise interpolates linearly between bins in symlog space, as the first DreamerV3 paper and the PyTorch port do. The current DreamerV3 code instead places the bins at `symexp` of those positions and interpolates in raw space. Both decode a single two-hot target back to exactly `y`. They differ when returns are noisy: a symlog-space head learns the average of `symlog(y)`, and `symexp` of that average is smaller in magnitude than the average of `y`. A raw-space head learns the average of `y` itself.""",
    "advisory_prerequisites": ["cross_entropy", "softmax"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where must y be squashed before it is compared with the bins, and how do you find the two bins around it with comparisons rather than division by a bin width? Which neighbour should get the larger weight, the near one or the far one? For the mean, which pairs of terms cancel when the probabilities are symmetric?"},
        {"level": 2, "kind": "analysis", "content": "Count how many bins are <= symlog(y) and how many are > it to get the two indices, clip both to the valid range, and handle the case where they coincide so you never divide by zero. Each weight is the distance to the other bin over the gap. For the loss, detach y before building the target. For the mean, flip the lower half of the products, add it to the upper half, sum, and add the middle term only when K is odd."},
    ],
    "model_connections": [
        "DreamerV3 predicts rewards, critic values and its slow target critic with this head, and normalizes returns separately for the actor.",
        "The categorical value head in C51 and MuZero's value and reward heads also project a scalar onto a fixed support, and MuZero squashes it with an invertible transform first.",
        "Classification losses for value functions (Farebrother et al. 2024, 'Stop Regressing') scale better than mean-squared error in large networks; two-hot is their simplest variant.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Gradient scale does not depend on the reward scale, so one learning rate works across very different tasks.",
            "The symmetric mean is exactly zero at initialization, so a fresh critic does not inject bias into bootstrapped targets.",
        ],
        "cons": [
            "The prediction can never leave the bin range; targets beyond it are clipped to the edge bin.",
            "With noisy returns, symexp of the symlog-space average underestimates the average return in magnitude, a bias that raw-space interpolation avoids.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "embodied/jax/nets.py",
            "symbol": "symlog, symexp",
            "license": "MIT",
            "adapted": "symlog and symexp with log1p and expm1, transcribed from JAX to torch.",
            "simplifications": "None beyond the JAX to torch translation.",
        },
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "embodied/jax/outs.py",
            "symbol": "TwoHot.pred, TwoHot.loss",
            "license": "MIT",
            "adapted": "Two-hot target with clipped below/above indices, the cross-entropy loss and the symmetric weighted sum, transcribed from JAX to torch with squash=symlog and unsquash=symexp.",
            "simplifications": "Upstream forces float32; the exercise keeps the input dtype. "
                               "The target is returned by its own function. "
                               "The exercise always squashes with symlog, whereas the current default head passes no squash and symexp-spaced bins, which interpolates in raw space.",
        },
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "embodied/jax/heads.py",
            "symbol": "Head.symexp_twohot",
            "license": "MIT",
            "adapted": "Symmetric bins built so that the mirror of every bin is its exact negative.",
            "simplifications": "The exercise returns symlog-space positions (no symexp). "
                               "For even K upstream builds two halves of K/2 points that both end at 0, so it has two zero bins and a wider spacing; "
                               "the exercise uses linspace(-limit, limit, K) spacing with no zero bin. Odd K agrees with upstream.",
        },
        {
            "kind": "code",
            "url": "https://github.com/NM512/dreamerv3-torch",
            "commit": "6ef8646d807cd10ce0c88e10a7e943211e7fc44c",
            "path": "tools.py",
            "symbol": "symlog, symexp, DiscDist.mean, DiscDist.log_prob",
            "license": "MIT",
            "adapted": "Cross-check for the symlog-space two-hot target and loss in PyTorch.",
            "simplifications": "The port uses log(|x| + 1) and exp(|x|) - 1, which lose precision near zero; "
                               "its mean is a plain left-to-right sum that is not exactly zero for uniform logits; "
                               "its buckets are a fixed linspace(-20, 20, 255); log_prob returns the negative of the loss and expects a trailing singleton dimension.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2301.04104",
            "section": "Hafner et al. 2023, Mastering Diverse Domains through World Models, arXiv v1: 'Symlog Predictions' (symlog, symexp, two-hot targets over 255 bins) and 'Critic Learning' (zero-initialized output weights)",
            "note": "Defines symlog/symexp, the two-hot encoding of a continuous target, the 255-bin support and zero initialization of the reward and critic output layers.",
        },
    ],
    "tests": [
        {"name": "symlog, symexp and symmetric bins", "behavior": "state.invariant", "code": r"""
import torch
symlog = {fn}.__globals__["symlog"]
symexp = {fn}.__globals__["symexp"]
symmetric_bins = {fn}.__globals__["symmetric_bins"]
x = torch.tensor([-3.0, 0.0, 0.5, 100.0])
want = torch.tensor([-1.3862944, 0.0, 0.4054651, 4.6151205])
assert torch.allclose(symlog(x), want, atol=1e-6), symlog(x)
assert torch.allclose(symexp(symlog(x)), x, rtol=1e-6), symexp(symlog(x))
bins = symmetric_bins(5, 2.0)
assert bins.dtype == torch.float32 and torch.equal(bins, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])), bins
"""},
        {"name": "Two-hot target between bins, on a bin and outside the range", "behavior": "state.invariant", "code": r"""
import torch
symexp = {fn}.__globals__["symexp"]
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
bins = symmetric_bins(5, 2.0)
t = twohot_target(symexp(torch.tensor(0.25)), bins)
assert t.shape == (5,) and torch.allclose(t, torch.tensor([0.0, 0.0, 0.75, 0.25, 0.0]), atol=1e-6), t
t = twohot_target(torch.tensor([0.0, 1e6, -1e6]), bins)
want = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
assert t.shape == (3, 5) and torch.allclose(t, want), t
"""},
        {"name": "Uniform logits: loss is log K and the mean is exactly zero", "behavior": "state.invariant", "code": r"""
import math, torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_mean = {fn}.__globals__["twohot_mean"]
bins = symmetric_bins(5, 2.0)
loss = {fn}(torch.zeros(5), torch.tensor(0.0), bins)
assert loss.shape == () and abs(loss.item() - math.log(5)) < 1e-6, loss
bins = symmetric_bins(255)
m = twohot_mean(torch.zeros(3, 255), bins)
assert m.shape == (3,) and torch.equal(m, torch.zeros(3)), m
"""},
        {"name": "symlog and symexp keep precision near zero and invert each other", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Use sign(x) * log1p(|x|) and sign(x) * expm1(|x|); log(1 + |x|) and exp(|x|) - 1 lose tiny values.",
         "code": r"""
import torch
symlog = {fn}.__globals__["symlog"]
symexp = {fn}.__globals__["symexp"]
tiny = torch.tensor([1e-20, -3e-17, 1e-14], dtype=torch.float64)
assert torch.allclose(symlog(tiny), tiny, rtol=1e-12, atol=0), symlog(tiny)
assert torch.allclose(symexp(tiny), tiny, rtol=1e-12, atol=0), symexp(tiny)
g = torch.Generator().manual_seed(0)
x = torch.randn(200, generator=g, dtype=torch.float64) * torch.logspace(-6, 6, 200, dtype=torch.float64)
want = torch.sign(x) * torch.log1p(torch.abs(x))
assert torch.allclose(symlog(x), want, rtol=1e-12, atol=0), "symlog"
assert torch.allclose(symexp(want), x, rtol=1e-10, atol=0), "symexp"
assert torch.allclose(symexp(torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)), torch.tensor([-6.38905609893065, 0.0, 19.085536923187668], dtype=torch.float64)), "symexp values"
assert symlog(torch.tensor(0.0)).item() == 0.0 and symexp(torch.tensor(0.0)).item() == 0.0
assert symlog(x.float()).dtype == torch.float32 and symexp(x.float()).dtype == torch.float32
"""},
        {"name": "symmetric_bins for even and odd K", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Bins are float32, match linspace(-limit, limit, K), mirror exactly (bins[K-1-i] == -bins[i]), and include 0 only for odd K.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
for K in (2, 3, 4, 5, 6, 41, 254, 255, 256):
    for limit in (20.0, 3.5):
        b = symmetric_bins(K, limit)
        assert b.shape == (K,) and b.dtype == torch.float32, (K, b.shape, b.dtype)
        assert torch.allclose(b, torch.linspace(-limit, limit, K), atol=1e-5, rtol=0), (K, limit)
        assert torch.equal(b, -b.flip(0)), (K, limit, "not exactly antisymmetric")
        assert torch.all(b[1:] > b[:-1]), (K, "not strictly increasing")
        zeros = int((b == 0).sum())
        assert zeros == (K % 2), (K, "zero bins", zeros)
assert torch.allclose(symmetric_bins(255), torch.linspace(-20, 20, 255), atol=1e-5), "default limit is 20"
"""},
        {"name": "Targets exactly on a bin, including uneven bins", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "below is the last bin <= symlog(y) and above the first bin > symlog(y); a target exactly on bin i is one-hot at i, not split across its neighbours.",
         "code": _UNEVEN_BINS + r"""
twohot_target = {fn}.__globals__["twohot_target"]
for seed in (0, 1, 2):
    bins, raws = uneven_bins(seed, 7)
    t = twohot_target(raws, bins)
    assert t.shape == (len(bins), len(bins)), t.shape
    assert torch.allclose(t, torch.eye(len(bins), dtype=torch.float64), atol=1e-12), (seed, t)
bins, raws = uneven_bins(3, 6, dtype=torch.float32, middle=False)
t = twohot_target(raws, bins)
assert torch.allclose(t, torch.eye(len(bins)), atol=1e-6), t
"""},
        {"name": "Seeded targets interpolate in symlog space with the right weights", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Compare symlog(y), not y, with the bins, and give the nearer bin the larger weight so that sum(target * bins) == symlog(y).",
         "code": _UNEVEN_BINS + r"""
twohot_target = {fn}.__globals__["twohot_target"]
symlog = lambda v: torch.sign(v) * torch.log1p(torch.abs(v))
for seed in (4, 5, 6, 7, 8):
    bins, raws = uneven_bins(seed, 9)
    g = torch.Generator().manual_seed(100 + seed)
    y = (torch.rand(6, 5, generator=g, dtype=torch.float64) * 2 - 1) * raws[-1] * 0.999
    t = twohot_target(y, bins)
    assert t.shape == (6, 5, len(bins)) and t.dtype == torch.float64, (t.shape, t.dtype)
    assert torch.allclose(t.sum(-1), torch.ones(6, 5, dtype=torch.float64), atol=1e-12), seed
    assert torch.all(t >= 0), seed
    assert torch.allclose((t * bins).sum(-1), symlog(y), atol=1e-10), (seed, "expected bin != symlog(y)")
    nz = (t > 0).sum(-1)
    assert torch.all(nz <= 2), seed
    idx = torch.arange(len(bins), dtype=torch.float64)
    first = torch.where(t > 0, idx, torch.full_like(idx, 1e9)).min(-1).values
    last = torch.where(t > 0, idx, torch.full_like(idx, -1.0)).max(-1).values
    assert torch.all(last - first <= 1), (seed, "non-adjacent bins")
"""},
        {"name": "Zero target for even and odd K", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "y = 0 is one-hot on the middle bin for odd K and splits 0.5 / 0.5 between bins K/2 - 1 and K/2 for even K.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
for K in (3, 5, 255):
    t = twohot_target(torch.zeros(2), symmetric_bins(K))
    want = torch.zeros(2, K); want[:, K // 2] = 1.0
    assert torch.equal(t, want), (K, t.nonzero())
for K in (2, 4, 256):
    t = twohot_target(torch.zeros(2), symmetric_bins(K))
    want = torch.zeros(2, K); want[:, K // 2 - 1] = 0.5; want[:, K // 2] = 0.5
    assert torch.allclose(t, want, atol=1e-6), (K, t[0, K // 2 - 2:K // 2 + 2])
"""},
        {"name": "Targets outside the bin range go to the edge bin", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Clip below and above to 0..K-1; when they coincide put weight 1 on that bin instead of dividing by a zero gap.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
for K in (4, 5, 255):
    bins = symmetric_bins(K, 2.0)
    y = torch.tensor([7.0, -7.0, 1e30, -1e30, 6.389056, -6.389056])
    t = twohot_target(y, bins)
    assert torch.isfinite(t).all(), (K, t)
    want = torch.zeros(6, K)
    want[0::2, -1] = 1.0
    want[1::2, 0] = 1.0
    assert torch.allclose(t, want, atol=1e-5), (K, t)
"""},
        {"name": "Seeded per-element loss matches the cross-entropy", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Return -(target * log_softmax(logits, -1)).sum(-1) for every element, shape (...), without averaging over the batch.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
for seed, shape, K in ((0, (4,), 255), (1, (2, 3), 6), (2, (5,), 7), (3, (), 9)):
    g = torch.Generator().manual_seed(seed)
    bins = symmetric_bins(K, 5.0).double()
    logits = torch.randn(*shape, K, generator=g, dtype=torch.float64) * 2
    y = torch.randn(shape, generator=g, dtype=torch.float64) * 30
    loss = {fn}(logits, y, bins)
    assert loss.shape == shape and loss.dtype == torch.float64, (loss.shape, loss.dtype)
    want = -(twohot_target(y, bins) * torch.log_softmax(logits, -1)).sum(-1)
    assert torch.allclose(loss, want, atol=1e-12), (seed, loss, want)
    assert torch.all(loss > 0), seed
"""},
        {"name": "Loss gradient reaches the logits but not y", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Detach y before building the target: the gradient with respect to the logits is softmax(logits) - target, and y gets none.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
g = torch.Generator().manual_seed(9)
bins = symmetric_bins(11, 4.0).double()
logits = torch.randn(3, 4, 11, generator=g, dtype=torch.float64).requires_grad_()
y = (torch.randn(3, 4, generator=g, dtype=torch.float64) * 5).requires_grad_()
loss = {fn}(logits, y, bins)
loss.sum().backward()
assert y.grad is None or torch.all(y.grad == 0), y.grad
want = torch.softmax(logits.detach(), -1) - twohot_target(y.detach(), bins)
assert torch.allclose(logits.grad, want, atol=1e-12), (logits.grad - want).abs().max()
"""},
        {"name": "Symmetric probabilities give exactly zero for even and odd K", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Add the product at bin i to the product at bin K-1-i before summing (plus the middle bin for odd K); a plain left-to-right sum is not exactly 0.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_mean = {fn}.__globals__["twohot_mean"]
g = torch.Generator().manual_seed(5)
for K in (4, 5, 41, 254, 255, 256):
    for dtype in (torch.float32, torch.float64):
        bins = symmetric_bins(K).to(dtype)
        for logits in (torch.zeros(2, 3, K, dtype=dtype), torch.full((4, K), 3.7, dtype=dtype)):
            m = twohot_mean(logits, bins)
            assert m.dtype == dtype and m.shape == logits.shape[:-1], (m.shape, m.dtype)
            assert torch.equal(m, torch.zeros_like(m)), (K, dtype, m.abs().max().item())
        half = torch.randn(6, K // 2, generator=g, dtype=dtype)
        mid = torch.randn(6, K % 2, generator=g, dtype=dtype)
        logits = torch.cat([half, mid, half.flip(-1)], -1)
        m = twohot_mean(logits, bins)
        assert torch.equal(m, torch.zeros_like(m)), (K, dtype, "symmetric logits", m.abs().max().item())
"""},
        {"name": "Seeded mean is symexp of the expected bin", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "twohot_mean returns symexp(sum(softmax(logits) * bins)) in raw space, not the symlog-space value.",
         "code": _UNEVEN_BINS + r"""
twohot_mean = {fn}.__globals__["twohot_mean"]
twohot_target = {fn}.__globals__["twohot_target"]
symexp = lambda v: torch.sign(v) * torch.expm1(torch.abs(v))
for seed in (0, 1, 2, 3, 4):
    bins, raws = uneven_bins(seed, 5 + seed, middle=seed % 2 == 0)
    g = torch.Generator().manual_seed(50 + seed)
    logits = torch.randn(3, 4, len(bins), generator=g, dtype=torch.float64) * 3
    m = twohot_mean(logits, bins)
    want = symexp((torch.softmax(logits, -1) * bins).sum(-1))
    assert m.shape == (3, 4) and torch.allclose(m, want, rtol=1e-10, atol=1e-12), (seed, m, want)
    y = (torch.rand(7, generator=g, dtype=torch.float64) * 2 - 1) * raws[-1] * 0.9
    back = twohot_mean(twohot_target(y, bins).log(), bins)
    assert torch.allclose(back, y, rtol=1e-9, atol=1e-9), (seed, back, y)
for K in (5, 6, 9):
    g = torch.Generator().manual_seed(K)
    bins = torch.cumsum(torch.rand(K, generator=g, dtype=torch.float64) + 0.1, 0) - 1.0
    logits = torch.randn(4, K, generator=g, dtype=torch.float64)
    want = symexp((torch.softmax(logits, -1) * bins).sum(-1))
    assert torch.allclose(twohot_mean(logits, bins), want, rtol=1e-10, atol=1e-12), (K, "asymmetric bins")
"""},
        {"name": "Output dtypes follow the inputs", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Cast bins to the input dtype: twohot_target follows y, twohot_loss and twohot_mean follow logits.",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
twohot_mean = {fn}.__globals__["twohot_mean"]
bins = symmetric_bins(7, 3.0)
y64 = torch.tensor([0.3, -2.0, 15.0], dtype=torch.float64)
t = twohot_target(y64, bins)
assert t.dtype == torch.float64, t.dtype
assert twohot_target(y64.float(), bins.double()).dtype == torch.float32
l64 = torch.zeros(3, 7, dtype=torch.float64)
assert {fn}(l64, y64, bins).dtype == torch.float64
assert {fn}(l64.float(), y64.float(), bins).dtype == torch.float32
assert twohot_mean(l64, bins).dtype == torch.float64
assert twohot_mean(l64.float(), bins.double()).dtype == torch.float32
"""},
        {"name": "Rejects bad bins and mismatched shapes", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for K < 2 or a non-positive limit in symmetric_bins, bins that are not 1-D with 2+ values, logits whose last dim is not K, or y not shaped like logits[..., 0].",
         "code": r"""
import torch
symmetric_bins = {fn}.__globals__["symmetric_bins"]
twohot_target = {fn}.__globals__["twohot_target"]
twohot_mean = {fn}.__globals__["twohot_mean"]
bins = torch.linspace(-2.0, 2.0, 5)
bad = [
    lambda: symmetric_bins(1),
    lambda: symmetric_bins(0),
    lambda: symmetric_bins(5.0),
    lambda: symmetric_bins(5, 0.0),
    lambda: symmetric_bins(5, -1.0),
    lambda: twohot_target(torch.zeros(3), bins[None]),
    lambda: twohot_target(torch.zeros(3), bins[:1]),
    lambda: {fn}(torch.zeros(3, 4), torch.zeros(3), bins),
    lambda: {fn}(torch.zeros(3, 5), torch.zeros(4), bins),
    lambda: {fn}(torch.zeros(3, 5), torch.zeros(3, 1), bins),
    lambda: twohot_mean(torch.zeros(3, 6), bins),
    lambda: twohot_mean(torch.zeros(3, 5), bins.reshape(5, 1)),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def symmetric_bins(K, limit=20.0):
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer >= 2")
    if not limit > 0:
        raise ValueError("limit must be positive")
    # Build the lower half and mirror it, so bins[K - 1 - i] == -bins[i] exactly.
    lower = torch.linspace(-limit, limit, K, dtype=torch.float32)[: K // 2]
    middle = torch.zeros(K % 2, dtype=torch.float32)
    return torch.cat([lower, middle, -lower.flip(0)])


def _check_bins(bins):
    if bins.ndim != 1 or bins.shape[0] < 2:
        raise ValueError("bins must be a 1-D tensor with at least 2 values")


def _check_logits(logits, bins):
    _check_bins(bins)
    if logits.shape[-1] != bins.shape[0]:
        raise ValueError("logits must have K = len(bins) entries in the last dimension")


def twohot_target(y, bins):
    _check_bins(bins)
    bins = bins.to(y.dtype)
    K = bins.shape[0]
    x = symlog(y)
    below = (bins <= x[..., None]).sum(-1) - 1
    above = K - (bins > x[..., None]).sum(-1)
    below = below.clamp(0, K - 1)
    above = above.clamp(0, K - 1)
    equal = below == above  # outside the range: both indices sit on the edge bin
    one = torch.ones_like(x)
    dist_to_below = torch.where(equal, one, torch.abs(bins[below] - x))
    dist_to_above = torch.where(equal, one, torch.abs(bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    return (F.one_hot(below, K).to(y.dtype) * weight_below[..., None]
            + F.one_hot(above, K).to(y.dtype) * weight_above[..., None])


def twohot_loss(logits, y, bins):
    _check_logits(logits, bins)
    if logits.shape[:-1] != y.shape:
        raise ValueError("y must have shape logits.shape[:-1]")
    target = twohot_target(y.detach().to(logits.dtype), bins)
    return -(target * torch.log_softmax(logits, -1)).sum(-1)


def twohot_mean(logits, bins):
    _check_logits(logits, bins)
    bins = bins.to(logits.dtype)
    K = bins.shape[0]
    half = K // 2
    weighted = torch.softmax(logits, -1) * bins
    # Pair bin i with bin K - 1 - i so symmetric probabilities cancel exactly.
    pairs = weighted[..., :half].flip(-1) + weighted[..., K - half:]
    total = pairs.sum(-1)
    if K % 2 == 1:
        total = weighted[..., half] + total
    return symexp(total)
''',
    "interview_questions": interview(
        concept=[
            "What problem does symlog solve for DreamerV3's reward and value heads, and why not simply normalize rewards instead?",
            "How does a two-hot encoding turn a continuous target into a categorical one, and what property do its two weights guarantee?",
        ],
        deep_dive=[
            "Walk through building the two-hot target for a value that lies between two bins, exactly on a bin, and outside the bin range. Where can an implementation divide by zero?",
            "Why does DreamerV3 compute the expected bin with a symmetric sum, and what goes wrong with a plain sum when the output layer is zero-initialized?",
            "What is the gradient of the two-hot cross-entropy with respect to the logits, and why is the target treated as a constant?",
        ],
        tradeoffs=[
            "Interpolating between bins in symlog space versus in raw space between symexp-spaced bins: what does each make exact, and where does each introduce bias?",
            "A two-hot categorical value head versus mean-squared error on symlog targets: compare gradient scale, multimodality, range limits and compute.",
        ],
    ),
}
