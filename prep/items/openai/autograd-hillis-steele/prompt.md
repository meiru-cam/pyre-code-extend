`W` is a floating-point tensor of shape `(N, D, D)` with $N \ge 1$. It holds $N$ square matrices $W_0, W_1, \dots, W_{N-1}$ of size $D \times D$, where `W[i]` is $W_i$. The $i$-th *prefix product* is the product of the first $i + 1$ matrices, taken from left to right:

$$P_i = W_0 W_1 \cdots W_i, \qquad i = 0, 1, \dots, N-1,$$

so $P_0 = W_0$ and $P_i = P_{i-1} W_i$ for $i \ge 1$. The output is a tensor `P` of shape `(N, D, D)` with `P[i]` equal to $P_i$. All $N$ prefix products are returned. For example, with $N = 3$ and $D = 2$:

```math
W_0 = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix},\qquad
W_1 = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix},\qquad
W_2 = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}
```

```math
P_0 = W_0 = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix},\qquad
P_1 = P_0 W_1 = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix},\qquad
P_2 = P_1 W_2 = \begin{pmatrix} 5 & 1 \\ 3 & 1 \end{pmatrix}
```

Implement the following four parts in PyTorch.

### Part 1 — Forward pass with indexed assignment

**(a)** Implement `prefix_products_inplace(W)`. Allocate the output with `P = torch.empty_like(W)` and fill it in a `for` loop with `P[i] = P[i - 1] @ W[i]`. Indexed assignment overwrites the memory of the existing tensor `P`. Operations of this kind are called *in-place* operations.

```py
def prefix_products_inplace(W: torch.Tensor) -> torch.Tensor:
    """W: float tensor of shape (N, D, D). Returns P of shape (N, D, D) with P[i] = W[0] @ ... @ W[i]."""
```

**(b)** PyTorch's automatic differentiation engine, *autograd*, records the operations applied to a tensor created with `requires_grad=True`. When `loss.backward()` is called on a scalar `loss`, it applies the chain rule backwards through the recorded operations and stores the gradient of `loss` with respect to `W` in `W.grad`. Under autograd, the function from (a) fails:

```py
W = torch.randn(4, 3, 3, requires_grad=True)
P = prefix_products_inplace(W)      # the values in P are correct
loss = P.sum()
loss.backward()
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

Explain why PyTorch raises this error.

### Part 2 — Forward pass that autograd can differentiate

Rewrite the function as `prefix_products(W)` without any in-place operation, so that the code of Part 1(b) runs and `W.grad` holds the correct gradient.

### Part 3 — Backward pass by hand

Let $L$ be a scalar loss that depends on all the $P_i$. The *upstream gradient* $G_i = \partial L / \partial P_i$ is given. It has the same shape as $P_i$, and its entry $(a, b)$ is $\partial L / \partial (P_i)_{ab}$. Without using autograd, derive a formula for $\partial L / \partial W_i$ and implement it:

```py
def prefix_products_backward(W: torch.Tensor, P: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """W, P, G: shape (N, D, D). P is the forward output and G[i] = dL/dP[i].
    Returns dW of shape (N, D, D) with dW[i] = dL/dW[i]."""
```

Verify the result against the `W.grad` that autograd computes for the function of Part 2, or against central finite differences $\bigl(L(W + \varepsilon E) - L(W - \varepsilon E)\bigr) / 2\varepsilon$, where $E$ is 1 in a single entry and 0 elsewhere.

### Part 4 — Hillis–Steele parallel scan

The loop of Parts 1–3 performs $N - 1$ products one after another, because step $i$ needs the result of step $i - 1$. Matrix multiplication is associative, so the same prefix products can be computed in $\lceil \log_2 N \rceil$ rounds with the Hillis–Steele scan. Start from $x_i = W_i$. Round $k = 0, 1, 2, \dots$ uses the stride $s = 2^k$ and is executed while $s < N$. In one round, simultaneously for every $i \ge s$,

$$x_i \leftarrow x_{i-s}\, x_i ,$$

where the right-hand side uses the values from before the round, and the entries with $i < s$ stay unchanged. After the last round $x_i = P_i$. The products inside one round do not depend on each other and can run in parallel. For $N = 4$:

- : start · $x_0$: $W_0$ · $x_1$: $W_1$ · $x_2$: $W_2$ · $x_3$: $W_3$
- : after $s = 1$ · $x_0$: $W_0$ · $x_1$: $W_0 W_1$ · $x_2$: $W_1 W_2$ · $x_3$: $W_2 W_3$
- : after $s = 2$ · $x_0$: $W_0$ · $x_1$: $W_0 W_1$ · $x_2$: $W_0 W_1 W_2$ · $x_3$: $W_0 W_1 W_2 W_3$

**(a)** Implement `scan_forward(W)`, which returns the same `P` as Part 2. Each round must be a single batched matrix multiplication, with no Python loop over $i$.

**(b)** Derive and implement the backward pass of this scan: given `G`, return $\partial L / \partial W$ in $\lceil \log_2 N \rceil$ rounds.
