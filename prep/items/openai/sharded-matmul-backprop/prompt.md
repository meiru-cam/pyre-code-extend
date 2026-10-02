Let $X$ have shape `(B, d_in)`, $W$ have shape `(d_in, d_out)`, and $Y = XW$ have shape `(B, d_out)`. A *device* is one entry of a Python list holding a NumPy array; the whole computation still runs on a single machine, and every exchange of data between devices goes through one of two explicit functions:

- `all_gather(shards, axis)`: `shards[k]` is device $k$'s piece of a logically single array, split along `axis`. Returns a list of length $n$ in which every entry is the same array, the concatenation of all the shards along `axis`.
- `all_reduce(shards)`: `shards[k]` is device $k$'s own array, all of the same shape, each holding a partial contribution to a sum. Returns a list of length $n$ in which every entry is the same array, the elementwise sum of all the shards.

A *shard* of a matrix is a contiguous slice along one of its axes, assigned to one device. In *column-parallel* sharding, $W$ is split along its output axis (`axis=1`, size `d_out`) into shards $W_0, \dots, W_{n-1}$, and every device holds a full copy of $X$. In *row-parallel* sharding, $W$ is split along its input axis (`axis=0`, size `d_in`) into shards $W_0, \dots, W_{n-1}$, and $X$ is split along its own feature axis (also size `d_in`) to match, so device $k$ holds only $X_k$, the columns of $X$ that line up with the rows of $W_k$. When the length of the axis being split is not divisible by $n$, the remainder is spread one entry at a time over the first shards: with `d_out = 7` and $n = 3$ devices the column widths are $3, 2, 2$, so device 0's shard of $W$ has shape `(d_in, 3)` and devices 1 and 2 have shape `(d_in, 2)`. The sizes here are small — `B`, `d_in` and `d_out` at most 50, entries within $\pm 100$ — and $n$ never exceeds the length of the axis being split, so no device ends up with an empty shard.

### Part 1 — Forward and backward by hand

For **column-parallel** sharding, state what device $k$ computes in the forward pass and how the full $Y$ is assembled. Then, given the upstream gradient $\bar Y = \partial L / \partial Y$ (the full array, identical on every device), derive $\bar W_k = \partial L / \partial W_k$ and $\bar X = \partial L / \partial X$ for each device, and say at which step, if any, communication is needed and of which kind. Do the same for **row-parallel** sharding.

### Part 2 — Implement the sharded layer

Implement `all_gather`, `all_reduce`, and the forward and backward pass of both sharding schemes:

```py
def all_gather(shards: list, axis: int) -> list: ...
def all_reduce(shards: list) -> list: ...

def column_parallel_forward(X, W_shards: list) -> list:
    """X: (B, d_in), a full copy on every device. W_shards[k]: (d_in, d_out_k).
    Returns Y_shards[k]: (B, d_out_k), device k's own shard of Y (not gathered)."""

def column_parallel_backward(X, W_shards: list, dY_shards: list) -> tuple:
    """dY_shards[k]: (B, d_out_k), dL/dY restricted to device k's columns.
    Returns (dW_shards, dX_shards)."""

def row_parallel_forward(X_shards: list, W_shards: list) -> list:
    """X_shards[k]: (B, d_in_k). W_shards[k]: (d_in_k, d_out).
    Returns Y_shards[k]: (B, d_out), the full Y, identical on every device."""

def row_parallel_backward(X_shards: list, W_shards: list, dY) -> tuple:
    """dY: (B, d_out), identical on every device. Returns (dW_shards, dX_shards)."""
```

Shard $W$, and for row-parallel also $X$, with `np.array_split` so that `d_out` or `d_in` need not be divisible by $n$. Verify both schemes, forward and backward, against an unsharded NumPy reference and against PyTorch autograd, in `float64`, with `np.allclose`; include at least one case where the split is uneven.

### Part 3 — Debug a sharded MLP

The file below implements a two-layer MLP split across $n$ devices: layer 1 is a column-parallel Linear from `d_in` to `d_hidden` followed by `tanh`, layer 2 is a row-parallel Linear from `d_hidden` to `d_out`, and the loss is the mean squared error against a target. `check_gradients` compares the sharded forward and backward pass against an unsharded NumPy reference built from the same weights, and returns the maximum absolute error of $Y$ and of each of the three gradients.

The file contains 5 bugs, each independent of the others: fixing one does not require touching the others, and each has its own effect on `check_gradients`. Find and fix all 5; for each, say where it is and why it produces the effect you observe. Once all 5 are fixed, `check_gradients()` returns errors below `1e-8` for every one of $Y$, `dW1`, `dW2`, `dX`, for any random seed.

```python
import numpy as np

B, D_IN, D_HIDDEN, D_OUT, N = 5, 4, 7, 3, 3   # NOTE: D_HIDDEN is not divisible by N


def all_gather(shards, axis):
    full = np.concatenate(shards, axis=axis)
    return [full for _ in shards]


def all_reduce(shards):
    total = sum(shards)
    return [total for _ in shards]


def shard_columns(A, n):
    return np.array_split(A, n, axis=1)


def shard_rows(A, n):
    chunk = A.shape[0] // n
    return [A[i * chunk:(i + 1) * chunk] for i in range(n)]


def layer1_forward(X, W1_shards):
    Z1_shards = [X @ W1k for W1k in W1_shards]
    H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]
    return Z1_shards, H_shards


def layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards):
    dZ1_shards = [dHk * (1 - Z1k ** 2) for dHk, Z1k in zip(dH_shards, Z1_shards)]
    dW1_shards = [dZ1k.T @ X for dZ1k in dZ1_shards]
    dX_shards = [dZ1k @ W1k.T for dZ1k, W1k in zip(dZ1_shards, W1_shards)]
    return dW1_shards, dX_shards


def layer2_forward(H_shards, W2_shards):
    Y_partial = [Hk @ W2k for Hk, W2k in zip(H_shards, W2_shards)]
    return Y_partial


def layer2_backward(H_shards, W2_shards, dY):
    dW2_shards = [Hk.T @ dY for Hk in H_shards]
    dH_shards = [dY @ W2k.T for W2k in W2_shards]
    return dW2_shards, dH_shards


def mse_loss_and_grad(Y, target):
    diff = Y - target
    loss = np.mean(diff ** 2)
    dY = 2 * diff / diff.size
    return loss, dY


def check_gradients(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, D_IN))
    W1 = rng.standard_normal((D_IN, D_HIDDEN)) * 0.5
    W2 = rng.standard_normal((D_HIDDEN, D_OUT)) * 0.5
    target = rng.standard_normal((B, D_OUT))
    W1_shards = shard_columns(W1, N)
    W2_shards = shard_rows(W2, N)

    Z1_shards, H_shards = layer1_forward(X, W1_shards)
    Y_shards = layer2_forward(H_shards, W2_shards)
    Y = Y_shards[0]
    loss, dY = mse_loss_and_grad(Y, target)
    dW2_shards, dH_shards = layer2_backward(H_shards, W2_shards, dY)
    dW1_shards, dX_shards = layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards)

    W1_full = all_gather(W1_shards, axis=1)[0]
    W2_full = all_gather(W2_shards, axis=0)[0]
    Z1 = X @ W1_full
    H = np.tanh(Z1)
    Y_ref = H @ W2_full
    loss_ref, dY_ref = mse_loss_and_grad(Y_ref, target)
    dW2_ref = H.T @ dY_ref
    dH_ref = dY_ref @ W2_full.T
    dZ1_ref = dH_ref * (1 - H ** 2)
    dW1_ref = X.T @ dZ1_ref
    dX_ref = dZ1_ref @ W1_full.T

    return {
        "Y": np.abs(Y - Y_ref).max(),
        "dW1": np.abs(np.concatenate(dW1_shards, axis=1) - dW1_ref).max(),
        "dW2": np.abs(np.concatenate(dW2_shards, axis=0) - dW2_ref).max(),
        "dX": np.abs(dX_shards[0] - dX_ref).max(),
    }


if __name__ == "__main__":
    print(check_gradients())
```
