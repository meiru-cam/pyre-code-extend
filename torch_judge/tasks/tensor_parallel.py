"""Tensor parallelism by hand: shards and collectives, then column-parallel and row-parallel linear layers, then a sharded two-layer MLP."""

from ._interview import interview

# A spy subclass that counts collective calls, and helpers for random shapes and shardings.
_HELP = r"""
import numpy as np

def spy(cls):
    class Spy(cls):
        def __init__(self):
            self.calls = []
        def all_gather(self, shards, axis):
            self.calls.append("all_gather")
            return super().all_gather(shards, axis)
        def all_reduce(self, shards):
            self.calls.append("all_reduce")
            return super().all_reduce(shards)
    return Spy()

def close(a, b):
    return np.asarray(a).shape == np.asarray(b).shape and np.allclose(a, b, rtol=1e-10, atol=1e-10)

def pieces(A, n, axis):
    size = A.shape[axis]
    out, start = [], 0
    for k in range(n):
        width = size // n + (1 if k < size % n else 0)
        out.append(np.take(A, range(start, start + width), axis=axis))
        start += width
    return out
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "tensor.shape", "code": r"""
import numpy as np
tp = {fn}()
A = np.arange(14.0).reshape(2, 7)
parts = tp.split(A, 3, 1)
assert [p.shape for p in parts] == [(2, 3), (2, 2), (2, 2)]
assert np.array_equal(parts[1], [[3.0, 4.0], [10.0, 11.0]])
copies = tp.all_gather(parts, 1)
assert len(copies) == 3 and all(np.array_equal(c, A) for c in copies)
sums = tp.all_reduce([np.ones(2), 2 * np.ones(2), 4 * np.ones(2)])
assert len(sums) == 3 and all(np.array_equal(s, [7.0, 7.0]) for s in sums)
sums[0][0] = -1.0
assert sums[1][0] == 7.0, "each device gets its own copy"
"""},
    {"name": "Part 1: shapes, axes and copies", "part": 1, "visibility": "unshown", "behavior": "tensor.shape",
     "failure_message": "split must give n contiguous shards with the remainder spread one entry at a time over the first shards, along either axis; all_gather must concatenate along the given axis; all_reduce must sum elementwise; every device must get its own array, separate from the inputs and from the other devices.",
     "code": _HELP + r"""
rng = np.random.default_rng(1)
tp = {fn}()
for trial in range(100):
    A = rng.standard_normal((rng.integers(1, 9), rng.integers(1, 9)))
    axis = int(rng.integers(0, 2))
    n = int(rng.integers(1, A.shape[axis] + 1))
    parts = tp.split(A, n, axis)
    assert len(parts) == n and all(close(p, q) for p, q in zip(parts, pieces(A, n, axis))), (trial, A.shape, n, axis)
    gathered = tp.all_gather(parts, axis)
    assert len(gathered) == n and all(close(g, A) for g in gathered)
    gathered[0] += 1
    assert close(parts[0], pieces(A, n, axis)[0]) and (n == 1 or close(gathered[-1], A))
    terms = [rng.standard_normal(A.shape) for _ in range(n)]
    summed = tp.all_reduce(terms)
    assert len(summed) == n and all(close(s, sum(terms)) for s in summed)
    summed[0] += 1
    assert n == 1 or close(summed[-1], sum(terms))
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "gradient.flow", "code": r"""
import numpy as np
tp = {fn}()
X = np.array([[1.0, 2.0], [0.0, -1.0]])
W = np.array([[1.0, 0.0, 2.0], [3.0, -1.0, 1.0]])
W_shards = [W[:, :2], W[:, 2:]]
Y_shards = tp.column_forward(X, W_shards)
assert np.allclose(Y_shards[0], [[7.0, -2.0], [-3.0, 1.0]]) and np.allclose(Y_shards[1], [[4.0], [-1.0]])
dY_shards = [np.ones((2, 2)), np.ones((2, 1))]
dW_shards, dX_shards = tp.column_backward(X, W_shards, dY_shards)
assert np.allclose(dW_shards[0], [[1.0, 1.0], [1.0, 1.0]]) and np.allclose(dW_shards[1], [[1.0], [1.0]])
assert len(dX_shards) == 2 and all(np.allclose(d, [[3.0, 3.0], [3.0, 3.0]]) for d in dX_shards)
"""},
    {"name": "Part 2: random column shards against autograd", "part": 2, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "Column-parallel results differed from torch autograd on the unsharded layer, or the communication was wrong: column_forward must not communicate, and column_backward must combine the devices' dX contributions with exactly one call to self.all_reduce.",
     "code": _HELP + r"""
rng = np.random.default_rng(2)
for trial in range(60):
    B, d_in, d_out = (int(v) for v in rng.integers(1, 9, size=3))
    n = int(rng.integers(1, d_out + 1))
    X, W, dY = rng.standard_normal((B, d_in)), rng.standard_normal((d_in, d_out)), rng.standard_normal((B, d_out))
    tp = spy({fn})
    W_shards, dY_shards = pieces(W, n, 1), pieces(dY, n, 1)
    Y_shards = tp.column_forward(X, W_shards)
    assert tp.calls == [], ("column_forward communicated", tp.calls)
    Xt, Wt = torch.tensor(X, requires_grad=True), torch.tensor(W, requires_grad=True)
    Yt = Xt @ Wt
    Yt.backward(torch.tensor(dY))
    assert close(np.concatenate(Y_shards, axis=1), Yt.detach().numpy()), trial
    dW_shards, dX_shards = tp.column_backward(X, W_shards, dY_shards)
    assert tp.calls == ["all_reduce"], ("column_backward collectives", tp.calls)
    assert all(close(a, b) for a, b in zip(dW_shards, pieces(Wt.grad.numpy(), n, 1))), trial
    assert len(dX_shards) == n and all(close(d, Xt.grad.numpy()) for d in dX_shards), trial
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "gradient.flow", "code": r"""
import numpy as np
tp = {fn}()
X = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]])
W = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]])
X_shards, W_shards = [X[:, :2], X[:, 2:]], [W[:2], W[2:]]
Y_shards = tp.row_forward(X_shards, W_shards)
assert len(Y_shards) == 2 and all(np.allclose(y, [[-2.0, 5.0], [3.0, 0.0]]) for y in Y_shards)
dY = np.array([[1.0, 0.0], [0.0, 2.0]])
dW_shards, dX_shards = tp.row_backward(X_shards, W_shards, dY)
assert np.allclose(dW_shards[0], [[1.0, 0.0], [2.0, 2.0]]) and np.allclose(dW_shards[1], [[-1.0, 2.0]])
assert np.allclose(dX_shards[0], [[1.0, 0.0], [4.0, 2.0]]) and np.allclose(dX_shards[1], [[3.0], [-2.0]])
"""},
    {"name": "Part 3: random row shards against autograd", "part": 3, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "Row-parallel results differed from torch autograd on the unsharded layer, or the communication was wrong: row_forward must sum the devices' partial outputs with exactly one call to self.all_reduce, and row_backward must not communicate.",
     "code": _HELP + r"""
rng = np.random.default_rng(3)
for trial in range(60):
    B, d_in, d_out = (int(v) for v in rng.integers(1, 9, size=3))
    n = int(rng.integers(1, d_in + 1))
    X, W, dY = rng.standard_normal((B, d_in)), rng.standard_normal((d_in, d_out)), rng.standard_normal((B, d_out))
    tp = spy({fn})
    X_shards, W_shards = pieces(X, n, 1), pieces(W, n, 0)
    Xt, Wt = torch.tensor(X, requires_grad=True), torch.tensor(W, requires_grad=True)
    Yt = Xt @ Wt
    Yt.backward(torch.tensor(dY))
    Y_shards = tp.row_forward(X_shards, W_shards)
    assert tp.calls == ["all_reduce"], ("row_forward collectives", tp.calls)
    assert len(Y_shards) == n and all(close(y, Yt.detach().numpy()) for y in Y_shards), trial
    dW_shards, dX_shards = tp.row_backward(X_shards, W_shards, dY)
    assert tp.calls == ["all_reduce"], ("row_backward communicated", tp.calls)
    assert all(close(a, b) for a, b in zip(dW_shards, pieces(Wt.grad.numpy(), n, 0))), trial
    assert all(close(a, b) for a, b in zip(dX_shards, pieces(Xt.grad.numpy(), n, 1))), trial
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "gradient.flow", "code": r"""
import numpy as np
tp = {fn}()
X = np.array([[0.5, -1.0]])
W1 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
W2 = np.array([[1.0], [2.0], [-1.0]])
loss, dW1_shards, dW2_shards, dX = tp.mlp_step(X, [W1[:, :2], W1[:, 2:]], [W2[:2], W2[2:]], np.array([[0.0]]))
H = np.tanh(X @ W1)
Y = H @ W2
assert np.isclose(loss, float(Y[0, 0] ** 2))
dZ = (2 * Y @ W2.T) * (1 - H ** 2)
assert np.allclose(np.concatenate(dW1_shards, axis=1), X.T @ dZ)
assert np.allclose(np.concatenate(dW2_shards, axis=0), H.T @ (2 * Y))
assert np.allclose(dX, dZ @ W1.T)
"""},
    {"name": "Part 4: random MLPs against autograd", "part": 4, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "The sharded MLP's loss or gradients differed from torch autograd on the unsharded model (tanh after layer 1, mean squared error over every entry), with uneven splits, or it did not use exactly two all_reduce calls: one in the forward pass, one for dX.",
     "code": _HELP + r"""
rng = np.random.default_rng(4)
for trial in range(60):
    B, d_in, d_hidden, d_out = (int(v) for v in rng.integers(1, 9, size=4))
    n = int(rng.integers(1, d_hidden + 1))
    X, target = rng.standard_normal((B, d_in)), rng.standard_normal((B, d_out))
    W1, W2 = rng.standard_normal((d_in, d_hidden)) * 0.7, rng.standard_normal((d_hidden, d_out)) * 0.7
    tp = spy({fn})
    loss, dW1_shards, dW2_shards, dX = tp.mlp_step(X, pieces(W1, n, 1), pieces(W2, n, 0), target)
    assert tp.calls == ["all_reduce", "all_reduce"], ("mlp_step collectives", tp.calls)
    Xt, W1t, W2t = (torch.tensor(a, requires_grad=True) for a in (X, W1, W2))
    lt = ((torch.tanh(Xt @ W1t) @ W2t - torch.tensor(target)) ** 2).mean()
    lt.backward()
    assert np.isclose(loss, lt.item(), rtol=1e-10, atol=1e-12), trial
    assert all(close(a, b) for a, b in zip(dW1_shards, pieces(W1t.grad.numpy(), n, 1))), trial
    assert all(close(a, b) for a, b in zip(dW2_shards, pieces(W2t.grad.numpy(), n, 0))), trial
    assert close(dX, Xt.grad.numpy()), trial
"""},
]

TASK = {
    "title": "Tensor-Parallel Linear Layers",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "TensorParallel",
    "description_en": r"""Build `TensorParallel`, which simulates devices as list entries: first shards and collectives, then the forward and backward pass of a column-parallel and a row-parallel linear layer, then a sharded two-layer MLP.

The requirement arrives in parts. Each part adds methods to the same `TensorParallel` class and keeps the earlier ones working. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A device is one entry of a Python list; entry `k` holds device `k`'s NumPy array. All arrays are `float64`.
- A layer computes `Y = X @ W` with `X` of shape `(B, d_in)` and `W` of shape `(d_in, d_out)`.
- A shard is a contiguous slice along one axis. With `n` devices and an axis of length `L`, the first `L % n` shards hold `L // n + 1` entries and the rest `L // n`, so `np.array_split` order. `n` never exceeds `L`.
- Every exchange of data between devices goes through `self.all_gather` or `self.all_reduce`. The tests count these calls, so call them on `self`, and only where the data must move.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it checks that you can derive matmul gradients by hand and say where communication is unavoidable. Each later part adds one requirement: the column split, the row split, and the two composed into the MLP block every tensor-parallel transformer uses.

**Where it is used:** Megatron-LM's tensor parallelism splits attention and MLP weights this way, with one all-reduce in the forward pass and one in the backward pass of each block; DeepSpeed and PyTorch's DTensor offer the same layouts.

Adapted from the sharded matmul question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as methods of one class with shorter names. The source's derivation part becomes interview questions, its bug hunt becomes Part 4, writing the MLP step correctly, and the collectives return separate copies and are counted.""",
    "parts": [
        {
            "title": "Shards and collectives",
            "description_en": r"""**Signatures:** `split(A, n, axis) -> list`, `all_gather(shards, axis) -> list`, `all_reduce(shards) -> list`

- `split` cuts `A` into `n` shards along `axis` by the shard rule.
- `all_gather` concatenates the shards along `axis` and gives every device the full array.
- `all_reduce` sums same-shaped arrays elementwise and gives every device the sum.
- Each returned entry is its own array: changing one device's copy changes no other device and no input.

**Example:** `A = np.arange(14.0).reshape(2, 7)` and `n = 3` along axis `1`:
- the shards have shapes `(2, 3)`, `(2, 2)` and `(2, 2)`; the middle one is `[[3, 4], [10, 11]]`
- `all_gather` of them along axis `1` gives `A` three times
- `all_reduce` of `[1, 1]`, `[2, 2]` and `[4, 4]` gives `[7, 7]` three times""",
        },
        {
            "title": "Column parallel",
            "description_en": r"""Keep Part 1. `W` is split along `d_out`, and every device holds all of `X`.

- `column_forward(X, W_shards) -> list` returns each device's own columns of `Y`, without gathering them.
- `column_backward(X, W_shards, dY_shards) -> (dW_shards, dX_shards)`: `dY_shards[k]` is `dL/dY` for device `k`'s columns. `dW_shards[k]` matches `W_shards[k]`, and every entry of `dX_shards` is the full `dL/dX`.

**Example:** `X = [[1, 2], [0, -1]]` and `W = [[1, 0, 2], [3, -1, 1]]`, split into two columns and one:
- the output shards are `[[7, -2], [-3, 1]]` and `[[4], [-1]]`
- with every entry of `dY` equal to `1`, the `dW` shards are all ones, and both devices get `dX = [[3, 3], [3, 3]]`, the sum of each device's share""",
        },
        {
            "title": "Row parallel",
            "description_en": r"""Keep Parts 1–2. `W` is split along `d_in`, and `X` is split along its columns to match.

- `row_forward(X_shards, W_shards) -> list` gives every device the full `Y`.
- `row_backward(X_shards, W_shards, dY) -> (dW_shards, dX_shards)`: `dY` is the full `dL/dY`, already on every device. `dW_shards[k]` matches `W_shards[k]`, and `dX_shards[k]` matches `X_shards[k]`.

**Example:** `X = [[1, 2, -1], [0, 1, 1]]` and `W = [[1, 2], [0, 1], [3, -1]]`, split after the second entry of `d_in`:
- both devices get `Y = [[-2, 5], [3, 0]]`
- with `dY = [[1, 0], [0, 2]]`, the `dW` shards are `[[1, 0], [2, 2]]` and `[[-1, 2]]`
- the `dX` shards are `[[1, 0], [4, 2]]` and `[[3], [-2]]`""",
        },
        {
            "title": "A sharded MLP",
            "description_en": r"""Keep Parts 1–3. Add `mlp_step(X, W1_shards, W2_shards, target) -> (loss, dW1_shards, dW2_shards, dX)`.

- Layer 1 is column parallel from `d_in` to `d_hidden`, followed by `tanh`. Layer 2 is row parallel from `d_hidden` to `d_out`, taking each device's `tanh` output as its `X` shard.
- `loss` is the mean of `(Y - target) ** 2` over every entry, as a float. Return the gradients of `loss` for each shard of `W1` and `W2`, and the full `dX`.
- Use exactly two `all_reduce` calls and no `all_gather`.

**Example:** `X = [[0.5, -1]]`, `W1 = [[1, 0, 2], [0, 1, 1]]` split into two columns and one, `W2 = [[1], [2], [-1]]` split to match, `target = [[0]]`:
- `H = tanh(X @ W1)` and `Y = H @ W2`, so `loss = Y ** 2`
- `dW2` is `H.T @ (2 * Y)`, and the gradient reaching `X @ W1` is `(2 * Y @ W2.T) * (1 - H ** 2)`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Seven columns over three devices: which device gets the extra one, and where does each shard start? If every device received the same array object, what would happen when one device changed its copy?"},
        {"level": 2, "kind": "analysis", "content": "np.array_split already puts the extra entries in the first shards, along any axis. all_gather is np.concatenate along the axis, all_reduce is an elementwise sum of the list; return a fresh copy per device, for example [total.copy() for _ in shards], so no two devices share an array."},
    ],
    "model_connections": [
        "Megatron-style transformer blocks put a column-parallel layer before the nonlinearity and a row-parallel layer after it, so each MLP needs one all-reduce forward and one backward.",
        "Attention heads are split the same way: the QKV projection is column parallel and the output projection row parallel.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Column then row parallel lets the elementwise tanh run on local shards with no communication between the layers.",
            "Each device stores only its slice of the weights, so models too large for one device fit.",
            "The communication pattern is fixed and known in advance, which makes it easy to overlap with compute.",
        ],
        "cons": [
            "Every block still needs an all-reduce in each direction, which dominates when devices are linked by a slow network.",
            "Uneven splits leave some devices with more work than others.",
            "Elementwise operations that mix columns, such as layer norm, need the full activations and break the pattern.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import numpy as np


class TensorParallel:
    # -- part 1: shards and collectives
    def split(self, A, n, axis):
        """n contiguous shards along axis; the first len % n shards get one extra entry."""
        return [np.array(piece) for piece in np.array_split(A, n, axis=axis)]

    def all_gather(self, shards, axis):
        full = np.concatenate(shards, axis=axis)
        return [full.copy() for _ in shards]  # every device gets its own copy

    def all_reduce(self, shards):
        total = np.sum(shards, axis=0)
        return [total.copy() for _ in shards]

    # -- part 2: column parallel, W split along d_out
    def column_forward(self, X, W_shards):
        return [X @ Wk for Wk in W_shards]  # no communication: each device owns its output columns

    def column_backward(self, X, W_shards, dY_shards):
        dW_shards = [X.T @ dYk for dYk in dY_shards]
        partial = [dYk @ Wk.T for dYk, Wk in zip(dY_shards, W_shards)]  # each device's share of dX
        return dW_shards, self.all_reduce(partial)

    # -- part 3: row parallel, W split along d_in and X split to match
    def row_forward(self, X_shards, W_shards):
        return self.all_reduce([Xk @ Wk for Xk, Wk in zip(X_shards, W_shards)])

    def row_backward(self, X_shards, W_shards, dY):
        dW_shards = [Xk.T @ dY for Xk in X_shards]
        dX_shards = [dY @ Wk.T for Wk in W_shards]  # no communication: dY is already on every device
        return dW_shards, dX_shards

    # -- part 4: a two-layer MLP, column parallel then row parallel
    def mlp_step(self, X, W1_shards, W2_shards, target):
        H_shards = [np.tanh(Zk) for Zk in self.column_forward(X, W1_shards)]
        Y = self.row_forward(H_shards, W2_shards)[0]
        diff = Y - target
        loss = float(np.mean(diff ** 2))
        dY = 2 * diff / diff.size
        dW2_shards, dH_shards = self.row_backward(H_shards, W2_shards, dY)
        dZ_shards = [dHk * (1 - Hk ** 2) for dHk, Hk in zip(dH_shards, H_shards)]  # tanh' = 1 - tanh^2
        dW1_shards, dX_shards = self.column_backward(X, W1_shards, dZ_shards)
        return loss, dW1_shards, dW2_shards, dX_shards[0]
''',
    "interview_questions": interview(
        concept=[
            "With seven columns and three devices, how wide is each shard, and why give the extra columns to the first devices?",
            "What is the difference between all_gather and all_reduce, and what does each device hold afterwards?",
        ],
        deep_dive=[
            "Why should a collective return a separate array for every device, and what bug can a shared array cause?",
        ],
        tradeoffs=[
            "In column-parallel backward, why does dX need an all-reduce while dW needs no communication?",
            "In row-parallel forward, why must partial outputs be summed, and why does the backward pass need no communication?",
            "Why put the column-parallel layer before tanh and the row-parallel layer after it, and what would the opposite order cost?",
            "How would you check sharded gradients against an unsharded reference, and which tolerances would you use?",
        ],
    ),
}
