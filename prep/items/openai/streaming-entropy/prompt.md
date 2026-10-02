`logits` is a one-dimensional float array of shape `(N,)`, and every entry is finite; $N \ge 1$. Let $x$ denote `logits` and $p = \mathrm{softmax}(x)$,

$$p_i = \frac{e^{x_i}}{\sum_{k=0}^{N-1} e^{x_k}}, \qquad i = 0, \dots, N-1,$$

and let $H(p) = -\sum_i p_i \log p_i$ be the *Shannon entropy* of $p$, with $\log$ the natural logarithm throughout. Every $p_i \in (0, 1]$, so $H(p) \ge 0$, and every part below returns this non-negative value (not the raw sum $\sum_i p_i \log p_i$, which is $\le 0$). Implement the following four parts in NumPy.

### Part 1 — Direct entropy

Compute $H(p)$ straight from the definition: exponentiate, normalize, sum.

```py
def softmax_entropy(logits: np.ndarray) -> float:
    """logits: shape (N,) float array, N >= 1, every entry finite. Returns H(softmax(logits))."""
```

For `logits = [0.5, 2.5, -1.0]` the function returns approximately `0.4761`.

### Part 2 — Numerically stable entropy

On some inputs Part 1 is unusable:

```py
softmax_entropy(np.array([1000.0, 1002.0, 998.0]))
# nan
```

Implement `softmax_entropy_stable(logits)`, which must return the same value as Part 1 up to floating-point rounding whenever Part 1 does not fail, and a finite, correct value on every finite input, including the one above (approximately `0.4411`). The implementation must (a) subtract the maximum logit before calling `np.exp`, and (b) compute $\log p_i$ in the centered form $(x_i - m) - \log \sum_k e^{x_k - m}$, where $m = \max_k x_k$, rather than as $\log\bigl(e^{x_i} / \sum_k e^{x_k}\bigr)$.

```py
def softmax_entropy_stable(logits: np.ndarray) -> float:
    """Same contract as softmax_entropy, correct and finite for every finite input."""
```

### Part 3 — Entropy in fixed-size blocks, O(1) extra space

*O(1) extra space* here means the code must not allocate any array whose size depends on $N$ — in particular, no length-$N$ vector of probabilities. Split `logits` into consecutive slices of a fixed `block_size` (e.g. `2`, which need not divide $N$, so the last slice may be shorter), and keep only a fixed number of running scalars between slices. The whole array is available and may be read more than once.

```py
def softmax_entropy_blockwise(logits: np.ndarray, block_size: int = 2) -> float:
    """Same contract as softmax_entropy_stable, using O(1) extra space for a fixed block_size."""
```

For `logits = [1.5, -2.0, 4.0, -0.5, 3.0]` and `block_size = 2`, the three slices are `[1.5, -2.0]`, `[4.0, -0.5]`, `[3.0]`, and the function returns approximately `0.8168`.

### Part 4 — Single-pass online entropy

Now `logits` is not one array but a sequence of chunks that arrive one at a time, for example from a generator, and each chunk can be read once only: an earlier chunk cannot be revisited once the next one has arrived. Implement `softmax_entropy_online(blocks)`, again with O(1) extra space, that consumes `blocks` once and returns the entropy of the concatenation of all chunks.

```py
def softmax_entropy_online(blocks) -> float:
    """blocks: an iterable of 1-D float arrays, the logits split into chunks of any length, consumed
    once, in order. Returns the same value as softmax_entropy_stable on their concatenation."""
```

With the same three slices as the Part 3 example, arriving one at a time, the function again returns approximately `0.8168`.
