`logits` is a float array of shape `(B, T, V)`: the unnormalized scores a language model assigns to `V` vocabulary classes, for each of `T` positions in each of `B` sequences. `targets` is an integer array of shape `(B, T)`; `targets[b, t]` is the id of the correct class at that position, an integer in `[0, V)`. Write $z = \text{logits}[b, t] \in \mathbb{R}^V$ and $\operatorname{softmax}(z)_i = e^{z_i} / \sum_k e^{z_k}$. The *cross-entropy loss* at one position is the negative log-probability the model assigns to the correct class:

$$\ell(b, t) = -\log \operatorname{softmax}(z)_{y}, \qquad y = \text{targets}[b, t].$$

Complete the following four parts in NumPy.

### Part 1 — Numerically stable cross-entropy

Implement `cross_entropy(logits, targets)`, returning the mean of $\ell(b, t)$ over all $B \times T$ positions. The function must stay finite and correct for logits as large as `1000.0`. Start from this skeleton:

```python
import numpy as np


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """logits: shape (B, T, V) float array of per-position class scores. targets: shape (B, T) int
    array, each entry a class id in [0, V). Returns the mean of -log softmax(logits)[b, t, targets[b, t]]
    over all B * T positions. Must stay finite and correct for logits as large as 1000."""
    raise NotImplementedError
```

Example, with $B = 1$, $T = 2$, $V = 3$:

```text
logits  = [[[2.0, 1.0, 0.1],
            [0.0, 0.0, 5.0]]]
targets = [[0, 2]]
```

`cross_entropy(logits, targets)` returns approximately `0.2152`.

### Part 2 — Excluding positions with a mask

`mask` is an array of shape `(B, T)`, holding either `0`/`1` integers or booleans; `mask[b, t] == 0` marks a position — for example padding at the end of a shorter sequence in the batch — that must not contribute to the loss. At such a position `targets[b, t]` may hold any integer, including a value outside `[0, V)`, and it must not be read. Implement `cross_entropy_masked(logits, targets, mask)`: the mean of $\ell(b, t)$ now divides only by the number of positions with `mask == 1`, not by $B \times T$. If every position is masked, return `0.0`.

```py
def cross_entropy_masked(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Same contract as cross_entropy, except positions with mask == 0 (shape (B, T), 0/1 or bool) are
    excluded from both the sum and the count in the mean. targets at a masked position may be any
    integer, including a value outside [0, V). Returns 0.0 when every position is masked."""
```

Example, extending Part 1's example with a third, masked position whose target is deliberately out of range:

```text
logits  = [[[2.0, 1.0, 0.1],
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 1.0]]]
targets = [[0, 2, 7]]
mask    = [[1, 1, 0]]
```

`cross_entropy_masked(logits, targets, mask)` returns the same `0.2152` as Part 1, since the third position does not count.

### Part 3 — Label smoothing

With *label smoothing*, the target distribution used at an unmasked position is no longer the one-hot distribution on `targets[b, t]` (the distribution with all its mass on class $y$), but

$$p = (1 - \varepsilon) \cdot \text{one-hot}(y) + \frac{\varepsilon}{V} \mathbf{1}, \qquad y = \text{targets}[b, t],$$

where $\varepsilon \in [0, 1)$ is `label_smoothing` and $\mathbf{1} \in \mathbb{R}^V$ is the all-ones vector. The loss at that position becomes $-\sum_v p_v \log \operatorname{softmax}(z)_v$ instead of $-\log \operatorname{softmax}(z)_y$. Implement `cross_entropy_smoothed(logits, targets, mask, label_smoothing)` with the same masking contract as Part 2; `label_smoothing = 0.0` must reproduce `cross_entropy_masked` exactly.

```py
def cross_entropy_smoothed(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray,
                            label_smoothing: float) -> float:
    """Same contract as cross_entropy_masked, plus label smoothing with parameter label_smoothing = eps:
    the target distribution at an unmasked position is (1 - eps) * one_hot(targets[b, t]) + eps / V."""
```

On the example of Part 2 with `label_smoothing = 0.2`, `cross_entropy_smoothed` returns approximately `0.6452`.

### Part 4 — Cross-entropy, KL divergence, and the gradient

Fix a position $(b, t)$ and write $p$ for the target distribution used there (one-hot in Parts 1–2, the smoothed distribution of Part 3) and $q = \operatorname{softmax}(z)$ for the model's predicted distribution, so that $\ell(b, t) = H(p, q) = -\sum_v p_v \log q_v$.

**(a)** Derive $H(p, q) = H(p) + \mathrm{KL}(p \Vert q)$, where $H(p) = -\sum_v p_v \log p_v$ is the entropy of $p$ and $\mathrm{KL}(p \Vert q) = \sum_v p_v \log(p_v / q_v)$ is the KL divergence. Use the identity to explain why, with one-hot targets, minimizing the loss over `logits` is exactly minimizing $\mathrm{KL}(p \Vert q)$.

**(b)** With label smoothing ($\varepsilon > 0$), give a closed-form expression for $H(p)$ and state the value the loss of Part 3 can no longer go below at that position, no matter how `logits` is chosen. Check the bound numerically.

**(c)** Derive $\partial \ell(b, t) / \partial z$ for the loss of Part 3 (which reduces to Part 1's when `label_smoothing = 0` and every position is unmasked), and implement `cross_entropy_grad`, returning the gradient of the mean loss with respect to every entry of `logits`, including the effect of masking and of dividing by the count of unmasked positions.

```py
def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray,
                        label_smoothing: float) -> np.ndarray:
    """Gradient of cross_entropy_smoothed(logits, targets, mask, label_smoothing) with respect to
    logits, shape (B, T, V). The gradient row at a masked position is all zero."""
```

Verify the result against central finite differences and against automatic differentiation.
