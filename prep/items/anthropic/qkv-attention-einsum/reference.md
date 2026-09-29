Worth confirming with the interviewer: the convention for `mask` (`True` = may attend is assumed here, matching `torch.nn.functional.scaled_dot_product_attention`'s `attn_mask`) and what a fully masked row should return (the zero vector, assumed here, rather than raising); that `d_model` is always divisible by `n_heads`; and, in Part 3, that only the four stated bugs need fixing and every other line is correct as given.

### Part 1

The two contractions are each a batched matrix product. `np.einsum('bqd,bkd->bqk', Q, K)` sums over the shared feature axis `d`, leaving one score per `(batch, query, key)` triple; `np.einsum('bqk,bkd->bqd', weights, V)` sums over the key axis, leaving one output vector per `(batch, query)`. Subtracting the row maximum before `np.exp` leaves the softmax unchanged in exact arithmetic: for any constant $m$ that does not depend on the summed index $j$,

$$\frac{e^{s_j - m}}{\sum_k e^{s_k - m}} = \frac{e^{-m}e^{s_j}}{e^{-m}\sum_k e^{s_k}} = \frac{e^{s_j}}{\sum_k e^{s_k}},$$

while keeping every exponent $\le 0$, so `np.exp` never overflows. A fully masked row has every score $-\infty$, so its maximum is $-\infty$ too, and $-\infty - (-\infty)$ is `nan`; the implementation replaces that row's $m$ with $0$ before subtracting, which still leaves every entry of the row at $\exp(-\infty) = 0$ (the row's score is $-\infty$ regardless of what finite value is subtracted from it), and separately guards the softmax denominator — also exactly $0$ on that row — against a $0/0$.

```python
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d)            # (B, T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    m = np.max(scores, axis=-1, keepdims=True)                       # (B, T_q, 1)
    # NOTE: a fully masked row has every score -inf, so m is -inf too, and -inf - (-inf) is nan.
    #       Replace m with 0 there; exp(scores - 0) is still exp(-inf) = 0 on that row.
    m = np.where(np.isneginf(m), 0.0, m)
    exp_scores = np.exp(scores - m)                                  # (B, T_q, T_k)
    denom = np.sum(exp_scores, axis=-1, keepdims=True)                # (B, T_q, 1)
    # NOTE: denom is exactly 0 on a fully masked row; dividing 0 / 0 is nan, so guard the denominator
    #       (the numerator is 0 there regardless, so the guard value only avoids the nan).
    weights = exp_scores / np.where(denom == 0.0, 1.0, denom)
    return np.einsum('bqk,bkd->bqd', weights, V)
```

Complexity: $O(BT_qT_kd)$ time for the score contraction, $O(BT_qT_kd_v)$ for the weighted-sum contraction, and $O(BT_qT_k)$ memory for `scores`/`weights`, on top of the inputs and the output.

### Part 2

$Q$, $K$, $V$ are each one contraction with the corresponding weight matrix, `np.einsum('btd,de->bte', X, W_q)` and likewise for `W_k`, `W_v` — shape `(B, T, d_model)`. Splitting the last axis into $(H, d_{\text{head}})$ is a `reshape`, not an `einsum` (`einsum` permutes and sums axes that already exist; it cannot split one axis into two), giving shape `(B, T, H, d_head)`; moving the head axis before the time axis is a pure permutation, `np.einsum('bthe->bhte', ...)`, giving `(B, H, T, d_head)`. Per-head scores are `np.einsum('bhqe,bhke->bhqk', Qh, Kh)`, shape `(B, H, T, T)` — the same contraction as Part 1's, with an extra batch-like axis `h`; the weighted sum of $V$ is `np.einsum('bhqk,bhke->bhqe', weights, Vh)`, shape `(B, H, T, d_head)`. Merging heads reverses the split: `np.einsum('bhte->bthe', ...)` moves the head axis back after the time axis, `(B, T, H, d_head)`, and `reshape` collapses the last two axes into `(B, T, d_model)`; the final `np.einsum('btd,de->bte', merged, W_o)` gives `(B, T, d_model)`. Because $i \le i$ always, a query always sees at least itself, so `scores.max(axis=-1)` is never $-\infty$ here and the fully masked case of Part 1 cannot arise in a causal block.

**The $1/\sqrt{d_{\text{head}}}$ scale.** Let $q, k \in \mathbb{R}^{d_{\text{head}}}$ have independent components, each with mean $0$ and variance $1$, and let every $q_i$ be independent of every $k_j$. Then $q \cdot k = \sum_{i=0}^{d_{\text{head}}-1} q_ik_i$, and

$$\mathrm{E}[q \cdot k] = \sum_i \mathrm{E}[q_i]\,\mathrm{E}[k_i] = 0, \qquad
\mathrm{Var}(q\cdot k) = \mathrm{E}\Bigl[\bigl(\textstyle\sum_i q_ik_i\bigr)^2\Bigr] = \sum_i\sum_j \mathrm{E}[q_ik_iq_jk_j].$$

For $i \ne j$, $q_i, k_i, q_j, k_j$ are four mutually independent, zero-mean variables, so $\mathrm{E}[q_ik_iq_jk_j] = \mathrm{E}[q_i]\mathrm{E}[k_i]\mathrm{E}[q_j]\mathrm{E}[k_j] = 0$; for $i = j$, $\mathrm{E}[q_i^2k_i^2] = \mathrm{E}[q_i^2]\,\mathrm{E}[k_i^2] = 1 \cdot 1 = 1$, using the independence of $q_i$ and $k_i$ and $\mathrm{E}[q_i^2] = \mathrm{Var}(q_i) = 1$. So $\mathrm{Var}(q\cdot k) = \sum_{i=0}^{d_{\text{head}}-1} 1 = d_{\text{head}}$: on this model the dot product's standard deviation, and so its typical magnitude, grows as $\sqrt{d_{\text{head}}}$. As $d_{\text{head}}$ grows, unscaled scores spread further apart, softmax saturates toward a one-hot vector, and its Jacobian $\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$ shrinks toward $0$ at every entry — the gradient with respect to the scores vanishes. Dividing $q \cdot k$ by $\sqrt{d_{\text{head}}}$ rescales its variance back to exactly $1$ for every $d_{\text{head}}$, independent of the head size, which is what keeps softmax in a well-conditioned regime regardless of how many features each head has.

```python
def multi_head_self_attention(X, W_q, W_k, W_v, W_o, n_heads):
    B, T, d_model = X.shape
    d_head = d_model // n_heads

    Q = np.einsum('btd,de->bte', X, W_q)                              # (B, T, d_model)
    K = np.einsum('btd,de->bte', X, W_k)                              # (B, T, d_model)
    V = np.einsum('btd,de->bte', X, W_v)                              # (B, T, d_model)

    Qh = np.einsum('bthe->bhte', Q.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)
    Kh = np.einsum('bthe->bhte', K.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)
    Vh = np.einsum('bthe->bhte', V.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)

    scores = np.einsum('bhqe,bhke->bhqk', Qh, Kh) / np.sqrt(d_head)   # (B, H, T, T)
    visible = np.tril(np.ones((T, T), dtype=bool))                    # (T, T); visible[i, j] = query i sees key j
    scores = np.where(visible, scores, -np.inf)

    m = np.max(scores, axis=-1, keepdims=True)                        # (B, H, T, 1); never -inf: i sees itself
    exp_scores = np.exp(scores - m)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, H, T, T)

    head_out = np.einsum('bhqk,bhke->bhqe', weights, Vh)               # (B, H, T, d_head)
    merged = np.einsum('bhte->bthe', head_out).reshape(B, T, d_model)  # (B, T, d_model)
    return np.einsum('btd,de->bte', merged, W_o)                       # (B, T, d_model)
```

Complexity: the four `(d_model, d_model)` projections cost $O(BTd_{\text{model}}^2)$ together; the per-head contractions cost $O(BHT^2d_{\text{head}}) = O(BT^2d_{\text{model}})$, since $H \cdot d_{\text{head}} = d_{\text{model}}$; `scores`/`weights` use $O(BHT^2)$ memory. For long sequences ($T \gg d_{\text{model}}$) the quadratic-in-$T$ term dominates; for short ones the quadratic-in-$d_{\text{model}}$ projections do.

### Part 3

Fixing the bugs one at a time, in the order below, surfaces one symptom after another on the random input used in the checks ($B=3$, $T=5$, $d=4$):

- Bugs fixed so far: none · Result: `ValueError: operands could not be broadcast together with shapes (3,5,4) (3,5)` · Points to: `layer_norm`'s `var`, missing `keepdims`
- Bugs fixed so far: `var`'s `keepdims` · Result: runs, but differs from the formula by as much as $1.32$ · Points to: `layer_norm`'s `mean`, wrong axis
- Bugs fixed so far: + `mean`'s axis · Result: still differs, by as much as $0.56$ · Points to: the softmax `max`, wrong axis
- Bugs fixed so far: + the softmax `max` · Result: still differs, by as much as $0.17$ · Points to: the score scale, `//`
- Bugs fixed so far: all four · Result: matches to floating-point precision

**`layer_norm`'s `var`.** `x.mean(axis=-1, keepdims=True)` already has shape `(B, T, 1)`, but `((x - mean) ** 2).mean(axis=-1)` drops the reduced axis instead of keeping it as size $1$, shape `(B, T)`. Dividing the `(B, T, d)` array `x - mean` by it right-aligns the trailing axes, `d` against `T`; whenever $d \ne T$ NumPy has no way to broadcast them and raises.

**`layer_norm`'s `mean`.** `axis=0` reduces over the batch axis instead of the feature axis, so every row is centered on the mean of the *same feature across every sequence in the batch* rather than the mean of its *own $d$ features* — with $\gamma=1,\beta=0$ and a batch of size $1$ this is especially visible, since the mean of one element is that element, so `x - mean` is identically $0$ and `layer_norm` returns the constant $\beta$ regardless of `x`.

**The softmax `max`.** `axis=-2` maximizes over the query axis instead of the key axis. The identity used in Part 1 to justify subtracting a constant before `np.exp` requires that constant not to depend on the index being summed over (the key axis, `axis=-1`); a maximum taken over `axis=-2` varies with the key index, so subtracting it changes the ratio between a row's entries and produces a different softmax, not merely a differently-computed one.

**The score scale.** `//` floors every entry of $QK^\top$ to the next lower integer before dividing — and, since NumPy's `//` on floats is true floor division, before the softmax ever sees a fractional score. This is silently wrong: no exception, no warning, just numbers rounded down by up to $1$ before they are ever exponentiated.

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)                             # NOTE: was axis=0
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)               # NOTE: was missing keepdims=True
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def causal_self_attention(x, W_q, W_k, W_v):
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)                 # NOTE: was // np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)                             # NOTE: was axis=-2
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


def transformer_block_forward(x, gamma, beta, W_q, W_k, W_v):
    return x + causal_self_attention(layer_norm(x, gamma, beta), W_q, W_k, W_v)
```

### Follow-ups

- **Complexity.** The four `(d_model, d_model)` projections of Part 2 cost $O(BTd_{\text{model}}^2)$; the per-head score and weighted-value contractions cost $O(BT^2d_{\text{model}})$ together; `scores`/`weights` use $O(BHT^2)$ memory. Long sequences make the $T^2$ term dominate; short ones make the $d_{\text{model}}^2$ projections dominate.
- **KV caching.** Under a causal mask, position $j$'s key and value never depend on any position after $j$. Decoding one new token at a time can therefore cache every earlier position's $K$ and $V$: a new step needs $Q$ for the new position only, appends one row to the cached $K$ and $V$, and costs $O(T)$ instead of recomputing all $T$ positions and an $O(T^2)$ attention from scratch.
- **`-inf` versus a large negative number.** `-inf` makes a masked entry's weight exactly $0$, which is what makes a fully masked row's `denom` exactly $0$ and therefore detectable. A finite stand-in like `-1e9` avoids ever producing `nan` from an all-masked row (its softmax becomes a silent uniform distribution instead), but is itself representable only in a wide enough type: cast to `float16` (whose largest finite magnitude is about $65504$), `-1e9` overflows to `-inf` anyway.
- **Multi-query and grouped-query attention.** Give $K$ and $V$ fewer heads than $Q$ — one shared head (multi-query) or a handful of groups, each shared by several query heads (grouped-query) — by projecting to `g < d_model` features, `np.einsum('btd,dg->btg', X, W_k)`, reshaped to `(B, T, H_kv, d_head)`. Every query head in a group contracts against the same `Kh`/`Vh` slice; the payoff is a KV cache that stores $H_{kv}$ heads' worth of keys and values instead of $H$.
- **`einsum` versus `@`.** `np.einsum(subscripts, *arrays)` without `optimize=True` evaluates the contraction left to right without dispatching to BLAS, and can run several times slower than `@` on the same arrays; passing `optimize=True` (or writing the contraction as `@`/`np.matmul`, which always uses BLAS) closes nearly all of the gap. `einsum`'s explicit subscripts are worth it for clarity on multi-axis contractions like the ones above — just benchmark before leaving the default `optimize` inside a hot loop.

```python
import torch
import torch.nn.functional as F


def loop_attention(Q, K, V, mask=None):                           # Part 1, straight from the formula
    B, Tq, d = Q.shape
    Tk, dv = K.shape[1], V.shape[2]
    out = np.zeros((B, Tq, dv))
    for b in range(B):
        for i in range(Tq):
            visible = [j for j in range(Tk) if mask is None or mask[b, i, j]]
            if not visible:
                continue                                          # out[b, i] stays the zero vector
            scores = [np.dot(Q[b, i], K[b, j]) / np.sqrt(d) for j in range(Tk)]
            m = max(scores[j] for j in visible)
            exps = {j: np.exp(scores[j] - m) for j in visible}
            s = sum(exps.values())
            for j in visible:
                out[b, i] += (exps[j] / s) * V[b, j]
    return out


# Part 1: the worked example of the statement
Q = np.array([[[1., 0.], [0., 1.]]])
K = np.array([[[1., 0.], [0., 1.], [1., 1.]]])
V = np.array([[[10.], [20.], [30.]]])
mask = np.array([[[False, False, False], [True, False, True]]])
out = scaled_dot_product_attention(Q, K, V, mask)
assert np.array_equal(out[0, 0], [0.0])                            # fully masked row
assert round(float(out[0, 1, 0]), 4) == 23.3952
assert np.allclose(out, loop_attention(Q, K, V, mask))

# Part 1: random trials against the loop version, exercising the fully masked case
rng = np.random.default_rng(0)
fully_masked = 0
for _ in range(200):
    B, Tq, Tk, d, dv = (int(rng.integers(1, 4)) for _ in range(5))
    Qr, Kr, Vr = rng.normal(size=(B, Tq, d)), rng.normal(size=(B, Tk, d)), rng.normal(size=(B, Tk, dv))
    m = rng.random((B, Tq, Tk)) < 0.5 if rng.random() < 0.7 else None
    if m is not None:
        fully_masked += int(np.sum(~m.any(axis=-1)))
    assert np.allclose(scaled_dot_product_attention(Qr, Kr, Vr, m), loop_attention(Qr, Kr, Vr, m), atol=1e-8)
assert fully_masked > 20                                           # the fully masked case was really exercised

# Part 1: random trials against torch (no mask, so torch's own fully-masked-row nan never comes up)
for _ in range(20):
    B, Tq, Tk, d, dv = (int(rng.integers(1, 4)) for _ in range(5))
    Qr, Kr, Vr = rng.normal(size=(B, Tq, d)), rng.normal(size=(B, Tk, d)), rng.normal(size=(B, Tk, dv))
    theirs = F.scaled_dot_product_attention(torch.tensor(Qr), torch.tensor(Kr), torch.tensor(Vr)).numpy()
    assert np.allclose(scaled_dot_product_attention(Qr, Kr, Vr), theirs, atol=1e-8)


def loop_multi_head(X, W_q, W_k, W_v, W_o, n_heads):                # Part 2, straight from the formula
    B, T, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    out = np.zeros((B, T, d_model))
    for b in range(B):
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            for i in range(T):
                scores = [np.dot(Q[b, i, sl], K[b, j, sl]) / np.sqrt(d_head) for j in range(i + 1)]
                m = max(scores)
                exps = [np.exp(s - m) for s in scores]
                s = sum(exps)
                for j, e in enumerate(exps):
                    out[b, i, sl] += (e / s) * V[b, j, sl]
    return out @ W_o


def split_heads_torch(t, n_heads):
    b, t_, dm = t.shape
    return t.reshape(b, t_, n_heads, dm // n_heads).transpose(1, 2)


# Part 2: the worked example of the statement
X = np.array([[[1., 0., 2., 0.], [0., 1., 0., 2.]]])
I4 = np.eye(4)
out = multi_head_self_attention(X, I4, I4, I4, I4, n_heads=2)
assert np.array_equal(out[0, 0], [1., 0., 2., 0.])                 # a token that can only see itself reproduces itself
assert [round(float(v), 4) for v in out[0, 1]] == [0.3302, 0.6698, 0.1116, 1.8884]

# Part 2: random trials against the loop version and against torch
for _ in range(20):
    B, T, H = int(rng.integers(1, 4)), int(rng.integers(1, 6)), int(rng.choice([1, 2, 4]))
    d_model = H * int(rng.integers(1, 4))
    X = rng.normal(size=(B, T, d_model)) * 0.5
    W_q, W_k, W_v, W_o = (rng.normal(size=(d_model, d_model)) * 0.5 for _ in range(4))
    ours = multi_head_self_attention(X, W_q, W_k, W_v, W_o, n_heads=H)
    assert np.allclose(ours, loop_multi_head(X, W_q, W_k, W_v, W_o, H), atol=1e-8)

    Xt, Wqt, Wkt, Wvt, Wot = (torch.tensor(a) for a in (X, W_q, W_k, W_v, W_o))
    heads = F.scaled_dot_product_attention(split_heads_torch(Xt @ Wqt, H), split_heads_torch(Xt @ Wkt, H),
                                            split_heads_torch(Xt @ Wvt, H), is_causal=True)
    theirs = (heads.transpose(1, 2).reshape(B, T, d_model) @ Wot).numpy()
    assert np.allclose(ours, theirs, atol=1e-6)

# Part 2: the causal-mask property -- changing a later token never changes an earlier output
X = rng.normal(size=(2, 5, 6))
weights = [rng.normal(size=(6, 6)) * 0.5 for _ in range(4)]
before = multi_head_self_attention(X, *weights, n_heads=3)
X_changed = X.copy()
X_changed[:, -1, :] += rng.normal(size=6)
after = multi_head_self_attention(X_changed, *weights, n_heads=3)
assert np.allclose(before[:, :-1], after[:, :-1], atol=1e-8)
assert not np.allclose(before[:, -1], after[:, -1], atol=1e-4)     # the changed token's own output does move

# Part 2: the 1/sqrt(d_head) variance derivation, checked by simulation
for d_head in (8, 16, 64):
    q = rng.normal(size=(200_000, d_head))
    k = rng.normal(size=(200_000, d_head))
    dot = np.sum(q * k, axis=1)
    assert abs(dot.var() - d_head) < 0.05 * d_head                 # Var(q . k) ~= d_head, unscaled
    assert abs((dot / np.sqrt(d_head)).var() - 1.0) < 0.05          # Var(scaled) ~= 1, for every d_head


def loop_block(x, gamma, beta, W_q, W_k, W_v, eps=1e-5):            # Part 3, straight from the formula, no broadcasting
    B, T, d = x.shape
    normed = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            row = x[b, t]
            mean = sum(row) / d
            var = sum((v - mean) ** 2 for v in row) / d
            for k_ in range(d):
                normed[b, t, k_] = (row[k_] - mean) / np.sqrt(var + eps) * gamma[k_] + beta[k_]
    Q, K, V = normed @ W_q, normed @ W_k, normed @ W_v
    out = np.zeros_like(x)
    for b in range(B):
        for i in range(T):
            scores = [np.dot(Q[b, i], K[b, j]) / np.sqrt(d) for j in range(i + 1)]
            m = max(scores)
            exps = [np.exp(s - m) for s in scores]
            s = sum(exps)
            for j, e in enumerate(exps):
                out[b, i] += (e / s) * V[b, j]
            out[b, i] += x[b, i]
    return out


# Part 3: the worked example of the statement
x = np.array([[[1., 0.], [0., 1.]]])
gamma, beta = np.array([1., 1.]), np.array([0., 0.])
I2 = np.eye(2)
out = transformer_block_forward(x, gamma, beta, I2, I2, I2)
assert [round(float(v), 4) for v in out[0, 0]] == [2.0, -1.0]
assert [round(float(v), 4) for v in out[0, 1]] == [-0.8884, 1.8884]

# Part 3: random trials against the loop version, and the causal property
B, T, d = 3, 5, 4
x = rng.normal(size=(B, T, d))
gamma, beta = rng.normal(size=d), rng.normal(size=d)
W_q, W_k, W_v = rng.normal(size=(d, d)), rng.normal(size=(d, d)), rng.normal(size=(d, d))
fixed = transformer_block_forward(x, gamma, beta, W_q, W_k, W_v)
reference = loop_block(x, gamma, beta, W_q, W_k, W_v)
assert np.allclose(fixed, reference, atol=1e-8)
x_changed = x.copy()
x_changed[:, -1, :] += 3.0
assert np.allclose(fixed[:, :-1], transformer_block_forward(x_changed, gamma, beta, W_q, W_k, W_v)[:, :-1], atol=1e-8)


# Part 3: reintroducing any one of the four bugs breaks the output, on the same input
def layer_norm_bug_axis(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)                            # bug: was axis=-1
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def layer_norm_bug_keepdims(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1)                            # bug: missing keepdims=True
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def attention_bug_max_axis(x, W_q, W_k, W_v):
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-2, keepdims=True)                          # bug: was axis=-1
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


def attention_bug_floor_div(x, W_q, W_k, W_v):
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) // np.sqrt(d)              # bug: was / np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


out_bug_axis = x + causal_self_attention(layer_norm_bug_axis(x, gamma, beta), W_q, W_k, W_v)
assert not np.allclose(out_bug_axis, reference, atol=1e-8)

try:
    layer_norm_bug_keepdims(x, gamma, beta)                          # d != T, so this cannot silently broadcast
    assert False, "expected a broadcasting error"
except ValueError:
    pass

out_bug_max = x + attention_bug_max_axis(layer_norm(x, gamma, beta), W_q, W_k, W_v)
assert not np.allclose(out_bug_max, reference, atol=1e-8)

out_bug_div = x + attention_bug_floor_div(layer_norm(x, gamma, beta), W_q, W_k, W_v)
assert not np.allclose(out_bug_div, reference, atol=1e-8)


def attention_both_bugs(x, W_q, W_k, W_v):                          # softmax-max axis and floor-div together
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) // np.sqrt(d)              # bug: was / np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-2, keepdims=True)                          # bug: was axis=-1
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


# Part 3: the cumulative fixing order of the solution's table, on this same input
out_var_only = x + attention_both_bugs(layer_norm_bug_axis(x, gamma, beta), W_q, W_k, W_v)
assert round(float(np.max(np.abs(out_var_only - reference))), 2) == 1.32

out_var_and_mean = x + attention_both_bugs(layer_norm(x, gamma, beta), W_q, W_k, W_v)
assert round(float(np.max(np.abs(out_var_and_mean - reference))), 2) == 0.56

assert round(float(np.max(np.abs(out_bug_div - reference))), 2) == 0.17

print("all checks passed")
```
