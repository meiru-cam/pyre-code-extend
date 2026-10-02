Worth confirming with the interviewer: whether accumulation must stay in `float64` regardless of the input's own dtype (assumed below — `float32` changes the rescale precision, see the last follow-up), and whether `block_size` is fixed for a whole call or may change from chunk to chunk (Part 4 below handles both).

### Part 1

```python
import numpy as np


def softmax_entropy(logits):
    x = np.asarray(logits, dtype=np.float64)
    exp_x = np.exp(x)                       # NOTE: np.exp(1000.0) is inf, with a RuntimeWarning
                                             #       "overflow encountered in exp"
    p = exp_x / np.sum(exp_x)               # NOTE: once exp_x holds an inf, p is inf / inf = nan
                                             #       ("invalid value encountered in divide")
    return float(-np.sum(p * np.log(p)))    # NOTE: even without overflow, a p_i that underflows to
                                             #       exactly 0.0 makes log(p_i) = -inf ("divide by zero
                                             #       encountered in log"), and 0.0 * -inf is nan too
                                             #       ("invalid value encountered in multiply")
```

### Part 2

Write $Z = \sum_k e^{x_k}$, so $p_i = e^{x_i}/Z$ and $\log p_i = x_i - \log Z$. Factor out the largest exponent $m = \max_k x_k$: since $e^{x_i} = e^m e^{x_i - m}$,

$$Z = e^m s, \qquad \log Z = m + \log s, \qquad s = \sum_k e^{x_k - m}.$$

Every exponent $x_k - m$ is $\le 0$, and the one at the maximizing $k$ is exactly $0$, so every term of $s$ lies in $(0, 1]$ and $s \ge 1$: the sum can never overflow and never vanish, no matter how large or spread out $x$ is. That is the reason for subtracting $m$ before exponentiating.

Writing $\log p_i$ as $(x_i - m) - \log s$ never calls `log` on a probability — it only takes the log of $s$, which is always finite and $\ge 1$. Computing $p_i$ first and calling `np.log` on it is different: $p_i = e^{x_i - m}/s$ can underflow to exactly `0.0` when $x_i - m$ is very negative, and `np.log(0.0)` is `-inf`, the same failure as in Part 1. The centered form sidesteps this because $p_i$ itself is only formed afterwards, from `np.exp(log_p_i)`; even if that underflows to `0.0`, the product with `log_p_i` is `0.0` times a finite number, not `0.0` times `-inf`.

```python
def log_softmax_stable(logits):
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    shifted = x - m                          # NOTE: shifted <= 0 everywhere, so np.exp below never overflows
    total = np.exp(shifted).sum()
    return shifted - np.log(total)           # NOTE: never calls log on a value that could be 0


def softmax_entropy_stable(logits):
    log_p = log_softmax_stable(logits)
    p = np.exp(log_p)                        # NOTE: p can still underflow to exactly 0.0 here, but log_p
                                              #       stays finite, so p * log_p below is 0.0 * finite, not nan
    return float(-np.sum(p * log_p))
```

### Part 3

The block-wise parts reuse one identity. Substituting $\log p_i = x_i - \log Z$ into the definition of $H$ and using $\sum_i p_i = 1$,

$$H(p) = -\sum_i p_i(x_i - \log Z) = \log Z - \sum_i p_i x_i.$$

So $H$ only needs $\log Z$ and the probability-weighted sum of the logits, neither of which requires materializing $p$. With $m$ and $s$ as in Part 2, $\log Z = m + \log s$. For the second term substitute $x_i = m + (x_i - m)$ and divide by $Z = e^m s$:

$$\sum_i p_i x_i = \frac{1}{Z}\sum_i e^{x_i} x_i = \frac{1}{s}\sum_i e^{x_i - m}\bigl(m + (x_i - m)\bigr)
= m + \frac{u}{s}, \qquad u = \sum_i e^{x_i - m}(x_i - m).$$

Substituting both into $H$ cancels $m$:

$$H(p) = \log s - \frac{u}{s}.$$

$s$ and $u$ are both sums over $i$ and can be accumulated block by block once $m$ is known. A first pass over the blocks gets $m$; a second accumulates $s$ and $u$ together — no third pass is needed, since $u/s$ already gives the probability-weighted average of the logits without ever forming an individual $p_i$.

```python
def _chunks(x, block_size):
    for start in range(0, x.shape[0], block_size):
        yield x[start:start + block_size]                      # NOTE: at most block_size elements live at
                                                                 #       once; no array of size N is ever built


def softmax_entropy_blockwise(logits, block_size=2):
    x = np.asarray(logits, dtype=np.float64)

    m = -np.inf
    for chunk in _chunks(x, block_size):          # pass 1: the global maximum
        m = max(m, float(np.max(chunk)))

    s = u = 0.0
    for chunk in _chunks(x, block_size):           # pass 2: s and u together, given m
        shifted = chunk - m
        w = np.exp(shifted)
        s += float(np.sum(w))
        u += float(np.sum(w * shifted))

    return float(np.log(s) - u / s)
```

### Part 4

Each chunk can supply its own local $(m, s, u)$, computed exactly as in Part 3 but relative to its own maximum. Merging an already-processed triple $(m_0, s_0, u_0)$ with a new chunk's triple $(m_1, s_1, u_1)$ into one triple relative to $m = \max(m_0, m_1)$ means re-centering every sum on $m$. For $s$: $\sum e^{x_k - m} = \sum e^{x_k - m_0} e^{m_0 - m} = s_0 e^{m_0 - m}$ for the terms already centered on $m_0$, and likewise for the $m_1$-centered terms, so $s = s_0 e^{m_0 - m} + s_1 e^{m_1 - m}$.

For $u$, write $\delta = m_0 - m \le 0$ and substitute $x_k - m = (x_k - m_0) + \delta$:

$$\sum_k e^{x_k - m}(x_k - m) = e^{\delta}\sum_k e^{x_k - m_0}\bigl((x_k - m_0) + \delta\bigr)
= e^{\delta}\bigl(u_0 + \delta s_0\bigr),$$

and symmetrically for the $m_1$-centered terms. Folding a new chunk into the running triple is exactly this merge, and the final triple gives $H$ by the Part 3 formula.

```python
def _centered_stats(x):
    m = float(np.max(x))
    shifted = x - m
    w = np.exp(shifted)
    return m, float(np.sum(w)), float(np.sum(w * shifted))


def _fold(acc, chunk):
    m0, s0, u0 = acc
    m1, s1, u1 = chunk
    if s0 == 0.0:                 # NOTE: acc is still the empty accumulator (m0 = -inf). Without this guard,
        return chunk               #       exp(m0 - m) is 0.0 and s0 * (m0 - m) is 0.0 * -inf = nan below
    m = max(m0, m1)
    a, b = np.exp(m0 - m), np.exp(m1 - m)
    s = s0 * a + s1 * b
    u = (u0 + s0 * (m0 - m)) * a + (u1 + s1 * (m1 - m)) * b
    return m, s, u


def softmax_entropy_online(blocks):
    acc = (-np.inf, 0.0, 0.0)
    for block in blocks:
        acc = _fold(acc, _centered_stats(np.asarray(block, dtype=np.float64)))
    m, s, u = acc
    return float(np.log(s) - u / s)
```

### Follow-ups

- Two passes over an in-memory array (Part 3) is simpler than one online pass (Part 4) and uses the same O(1) space; reach for Part 4's rescale only when the data genuinely cannot be revisited.
- The rescale inside `_fold` is the same recurrence behind online softmax / FlashAttention: there it folds in an output-vector chunk with a running max, instead of the two scalars `s` and `u` here.
- For a batch of shape `(B, N)`, replace the scalars `m, s, u` with arrays of shape `(B,)`, add `axis=-1, keepdims=True` to every reduction, and let each chunk have shape `(B, block_size)`.
- In `float32`, `s` and `u` lose precision once many chunks of very different scale have been folded in; keep the accumulators in `float64` even when `logits` itself is `float32`.

```python
from scipy.special import softmax as reference_softmax
from scipy.stats import entropy as reference_entropy


def iter_blocks(x, block_size):
    for start in range(0, len(x), block_size):
        yield x[start:start + block_size]


assert round(softmax_entropy(np.array([0.5, 2.5, -1.0])), 4) == 0.4761
assert round(softmax_entropy_stable(np.array([1000.0, 1002.0, 998.0])), 4) == 0.4411
example = np.array([1.5, -2.0, 4.0, -0.5, 3.0])
assert round(softmax_entropy_blockwise(example, block_size=2), 4) == 0.8168
assert round(softmax_entropy_online(iter_blocks(example, block_size=2)), 4) == 0.8168

rng = np.random.default_rng(0)
cases = [
    np.array([0.5, 2.5, -1.0]),
    np.array([1000.0, 1002.0, 998.0]),
    np.array([0.0, -800.0, -900.0]),
    np.array([3.0]),
    example,
    np.array([1e6, 1e6, 1e6]),
    rng.normal(size=7),
    rng.normal(scale=500.0, size=9),
    rng.normal(scale=1e4, size=11),
]
for x in cases:
    reference = reference_entropy(reference_softmax(x))
    assert np.allclose(softmax_entropy_stable(x), reference, atol=1e-9)
    for block_size in (1, 2, 3, 4, len(x) + 5):               # block_size need not divide len(x)
        assert np.allclose(softmax_entropy_blockwise(x, block_size), reference, atol=1e-9)
        assert np.allclose(softmax_entropy_online(iter_blocks(x, block_size)), reference, atol=1e-9)
    assert np.allclose(softmax_entropy_online([x]), softmax_entropy_stable(x), atol=1e-12)  # one big chunk
    mid = len(x) // 2                                          # chunks of different lengths in one stream
    chunks = [c for c in (x[:mid], x[mid:mid + 1], x[mid + 1:]) if len(c)]
    assert np.allclose(softmax_entropy_online(chunks), reference, atol=1e-9)

mild = np.array([0.5, 2.5, -1.0, 1.5])                          # Part 1 only agrees away from over/underflow
assert np.allclose(softmax_entropy(mild), softmax_entropy_stable(mild))
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    assert np.isnan(softmax_entropy(np.array([1000.0, 1002.0, 998.0])))    # overflow -> inf / inf
    assert np.isnan(softmax_entropy(np.array([0.0, -800.0, -900.0])))      # underflow -> 0.0 * log(0.0)
```
