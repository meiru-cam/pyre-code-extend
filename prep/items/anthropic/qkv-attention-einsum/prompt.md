*Scaled dot-product attention* maps a set of *query* vectors to a weighted average of a set of *value* vectors, one weight per value, computed from a compatibility score between the query and that value's associated *key* vector. Implement the following three parts in NumPy.

### Part 1 — Scaled dot-product attention

`Q` is a float array of shape `(B, T_q, d)`: `B` independent batch elements, each with `T_q` query vectors in $\mathbb{R}^d$. `K` has shape `(B, T_k, d)`: `T_k` key vectors in the same $\mathbb{R}^d$ as the queries. `V` has shape `(B, T_k, d_v)`: the `T_k` value vectors, one per key, whose dimension $d_v$ need not equal $d$. For batch element $b$, query $i$ and key $j$,

$$\mathrm{score}_{b,i,j} = \frac{Q_{b,i} \cdot K_{b,j}}{\sqrt{d}}, \qquad
\mathrm{weights}_{b,i,:} = \mathrm{softmax}(\mathrm{score}_{b,i,:}), \qquad \mathrm{output}_{b,i} = \sum_{j=0}^{T_k-1} \mathrm{weights}_{b,i,j}\, V_{b,j},$$

the softmax taken over $j$, the last axis of `score`. An optional boolean `mask` of shape `(B, T_q, T_k)` restricts which keys a query may use: `mask[b, i, j]` is `True` exactly when query $i$ of batch $b$ may attend to key $j$, and `False` when it may not; `mask=None` means every query may attend to every key. A masked key is excluded from the softmax by setting its score to $-\infty$ before exponentiating, never by zeroing its weight after normalizing. If every key is masked out for some query ($\mathrm{mask}_{b,i,:}$ all `False`), that query's weights are all $0$ and its output is the zero vector of shape $(d_v,)$, rather than $0/0$.

Compute the two contractions above — $QK^\top$ and $\mathrm{weights} \cdot V$ — with `np.einsum`; softmax is a reduction over one axis, not a contraction, and stays ordinary NumPy. Make the softmax numerically stable by subtracting the row maximum before calling `np.exp`.

```py
def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: np.ndarray | None = None) -> np.ndarray:
    """Q: (B, T_q, d), K: (B, T_k, d), V: (B, T_k, d_v). mask: (B, T_q, T_k) bool or None,
    True where the query may attend to the key. Returns (B, T_q, d_v)."""
```

Example with $B=1$, $T_q=2$, $T_k=3$, $d=2$, $d_v=1$:

```text
Q[0] = [[1, 0],      K[0] = [[1, 0],      V[0] = [[10],
        [0, 1]]              [0, 1],             [20],
                              [1, 1]]             [30]]

mask[0] = [[False, False, False],   # query 0: every key masked out
           [True,  False, True ]]   # query 1: may attend to keys 0 and 2, not key 1

query 0 -> weights[0, 0] = [0, 0, 0], output[0, 0] = [0]                       # fully masked row

query 1 -> raw scores Q[0,1].K[0,j] for j = 0, 1, 2 are 0, 1, 1; scaled by 1/sqrt(2): 0, 0.7071, 0.7071
           key 1 is masked, so only keys 0 and 2 enter the softmax
           weights[0, 1] ~= [0.3302, 0, 0.6698]
           output[0, 1] = 0.3302 * 10 + 0.6698 * 30 ~= [23.3952]
```

### Part 2 — Multi-head self-attention

*Self-attention* is scaled dot-product attention in which the queries, keys and values all come from the same sequence. `X` is a float array of shape `(B, T, d_model)`. Weight matrices `W_q`, `W_k`, `W_v`, each of shape `(d_model, d_model)`, produce $Q = XW_q$, $K = XW_k$, $V = XW_v$, every one of shape `(B, T, d_model)`. `d_model` must be divisible by the number of *heads*, $H$, each of size $d_{\text{head}} = d_{\text{model}} / H$. Feature $e$ of head $h$ ($0 \le h < H$, $0 \le e < d_{\text{head}}$) of $Q$ is column $h \cdot d_{\text{head}} + e$ of $Q$, and likewise for $K$ and $V$. Head $h$ runs scaled dot-product attention (Part 1's formula, with $d_{\text{head}}$ in place of $d$) among the $T$ positions, using only its own slice of $Q$, $K$, $V$, under a *causal mask*: position $i$ may attend to position $j$ exactly when $j \le i$ (a query may not attend to a later position). Concatenating the $H$ heads' outputs along the feature axis and multiplying by an output projection $W_o$ of shape `(d_model, d_model)` gives the final output, shape `(B, T, d_model)`.

Implement the head split and the head merge with `np.reshape` (to split or combine the head and feature axes) and `np.einsum` (to move the head axis before the time axis, and back); implement every contraction — the three input projections, the per-head scores, the per-head weighted sum of $V$, and the output projection — with `np.einsum`. State the shape of every intermediate array. Before scaling the scores, derive why they are divided by $\sqrt{d_{\text{head}}}$: if every one of the $d_{\text{head}}$ components of a query vector $q$ and of a key vector $k$ is an independent random variable with mean $0$ and variance $1$, find $\mathrm{Var}(q \cdot k)$, and explain what dividing by $\sqrt{d_{\text{head}}}$ does to it.

```py
def multi_head_self_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                               W_o: np.ndarray, n_heads: int) -> np.ndarray:
    """X: (B, T, d_model). W_q, W_k, W_v, W_o: (d_model, d_model). d_model must be divisible by
    n_heads; d_head = d_model // n_heads. Applies a causal mask. Returns (B, T, d_model)."""
```

Example with $B=1$, $T=2$, $d_{\text{model}}=4$, $H=2$ (so $d_{\text{head}}=2$), $W_q=W_k=W_v=W_o=I_4$:

```text
X[0] = [[1, 0, 2, 0],       # head 0 = features [0, 1], head 1 = features [2, 3]
        [0, 1, 0, 2]]       # Q = K = V = X here, since the weight matrices are the identity

query 0 may attend to key 0 only (causal) -> output[0, 0] = V[0, 0] = X[0, 0] = [1, 0, 2, 0]

query 1 may attend to keys 0 and 1:
  head 0: raw scores 0, 1 scaled by 1/sqrt(2) -> 0, 0.7071; weights ~= [0.3302, 0.6698]; head output ~= [0.3302, 0.6698]
  head 1: raw scores 0, 4 scaled by 1/sqrt(2) -> 0, 2.8284; weights ~= [0.0558, 0.9442]; head output ~= [0.1116, 1.8884]
  concatenate the heads and apply W_o = I -> output[0, 1] ~= [0.3302, 0.6698, 0.1116, 1.8884]
```

### Part 3 — Debugging: layer normalization and causal self-attention

*Layer normalization* rescales a $d$-dimensional row vector $v$ to zero mean and unit variance, then applies a learned elementwise affine map with scale $\gamma \in \mathbb{R}^d$ and shift $\beta \in \mathbb{R}^d$:

$$\mathrm{LN}(v)_k = \frac{v_k - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_k + \beta_k, \qquad
\mu = \frac{1}{d}\sum_{k=0}^{d-1} v_k, \qquad \sigma^2 = \frac{1}{d}\sum_{k=0}^{d-1} (v_k - \mu)^2,$$

with a small constant $\epsilon > 0$ added for numerical safety. Applied to an array `x` of shape `(B, T, d)`, it normalizes every one of the $B \times T$ rows independently, each over its own $d$ features. The function below is meant to normalize its input `x` this way, run single-head causal self-attention (Part 2's formula with $H=1$) on the normalized result, and add that back to `x` as a residual connection — but it has four bugs: one in the axis `layer_norm` reduces over for the mean, one in the shape of the variance it computes, one in the axis `causal_self_attention` maximizes over before exponentiating, and one in how it scales the raw scores. Find and fix all four; nothing else in the code is wrong. A fixed implementation matches, to `atol=1e-8`, a direct implementation of the same formula written with explicit loops over batch, time and feature indices (no NumPy broadcasting), for every input, and it has the causal property of Part 2: changing token $T-1$ of `x` leaves the output at every position before it unchanged.

```py
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """x: (B, T, d). gamma, beta: (d,). Normalizes each (batch, time) row over its d features."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def causal_self_attention(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    """x: (B, T, d). W_q, W_k, W_v: (d, d). Single-head causal self-attention. Returns (B, T, d)."""
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) // np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-2, keepdims=True)
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


def transformer_block_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                               W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    """x: (B, T, d). Pre-norm residual block: x + causal_self_attention(layer_norm(x))."""
    return x + causal_self_attention(layer_norm(x, gamma, beta), W_q, W_k, W_v)
```

Example with $B=1$, $T=2$, $d=2$, $\gamma=(1,1)$, $\beta=(0,0)$, $W_q=W_k=W_v=I_2$:

```text
x[0] = [[1, 0],
        [0, 1]]

query 0 may attend to key 0 only (causal) -> transformer_block_forward(x, ...)[0, 0] ~= [2.0000, -1.0000]
query 1 may attend to both keys           -> transformer_block_forward(x, ...)[0, 1] ~= [-0.8884, 1.8884]
```
