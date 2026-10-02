"""Prefix products of matrices: autograd-safe forward, hand-written backward, then a parallel scan."""

from ._interview import interview

# Helpers: count matrix products and forbid autograd inside the learner's backward passes.
_HELPERS = r"""
import contextlib, math

class Counter:
    n = 0

@contextlib.contextmanager
def counting():
    names = [(torch, "matmul"), (torch, "bmm"), (torch, "mm"), (torch, "einsum"),
             (torch.Tensor, "__matmul__"), (torch.Tensor, "__rmatmul__"), (torch.Tensor, "matmul"),
             (torch.Tensor, "bmm"), (torch.Tensor, "mm")]
    saved = [(obj, name, getattr(obj, name)) for obj, name in names]
    Counter.n = 0
    def wrap(f):
        def g(*a, **k):
            Counter.n += 1
            return f(*a, **k)
        return g
    for obj, name, f in saved:
        setattr(obj, name, wrap(f))
    try:
        yield Counter
    finally:
        for obj, name, f in saved:
            setattr(obj, name, f)

@contextlib.contextmanager
def no_autograd():
    def boom(*a, **k):
        raise AssertionError("compute this gradient by hand, without autograd")
    saved = (torch.autograd.grad, torch.Tensor.backward, torch.autograd.backward)
    torch.autograd.grad, torch.Tensor.backward, torch.autograd.backward = boom, boom, boom
    try:
        yield
    finally:
        torch.autograd.grad, torch.Tensor.backward, torch.autograd.backward = saved

def loop_products(W):
    out = [W[0]]
    for i in range(1, W.shape[0]):
        out.append(out[-1] @ W[i])
    return torch.stack(out)

def true_grad(W, G):
    W = W.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad((loop_products(W) * G).sum(), W)
    return g

EXAMPLE = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[1.0, 3.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
"""

TASK = {
    "title": "Prefix Products and Parallel Scan",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "PrefixScan",
    "description_en": r"""Build `PrefixScan`, which computes all prefix products of a stack of matrices in PyTorch, then their gradient by hand, then both with a parallel scan.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `PrefixScan` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `W` is a float tensor of shape `(N, D, D)` with `N >= 1`, holding the matrices `W[0]` to `W[N - 1]`. `torch` is available.
- The prefix products are `P[i] = W[0] @ W[1] @ ... @ W[i]`, multiplied left to right, so `P[0] = W[0]` and `P[i] = P[i - 1] @ W[i]`. `P` has the shape of `W`.
- `G` has the shape of `W`: `G[i]` is the gradient of a scalar loss `L` with respect to `P[i]`.
- Never change a tensor you are given.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the forward loop is three lines; the work is knowing what autograd saves, deriving a backward pass by hand, and trading sequential steps for parallel work, and each later part adds one requirement.

**Where it is used:** linear recurrences such as state-space models and linear attention are trained with parallel scans over an associative operator, which is this scan with matrices.

Adapted from the autograd and Hillis-Steele scan question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The source's in-place first attempt is folded into Part 1's requirement that autograd works.""",
    "parts": [
        {
            "title": "A forward pass autograd can follow",
            "description_en": r"""**Signature:** `PrefixScan()`, `prefix_products(W) -> Tensor`

- Return `P` as defined above.
- When `W.requires_grad` is `True`, `prefix_products(W)` must work with autograd: `(prefix_products(W) * G).sum().backward()` runs and fills `W.grad` with the correct gradient.
- Writing results into a preallocated tensor with `P[i] = ...` breaks this, because autograd saved `P[i - 1]` for the backward pass and the write changes it. Build new tensors instead.

**Example**, `N = 3`, `D = 2`:
- `W[0] = [[2, 0], [1, 1]]`, `W[1] = [[1, 3], [0, 1]]`, `W[2] = [[0, 1], [1, 0]]`
- `P[1] = [[2, 6], [1, 4]]` and `P[2] = [[6, 2], [4, 1]]`""",
        },
        {
            "title": "The backward pass by hand",
            "description_en": r"""Keep Part 1 and compute the gradient yourself.

**Signature:** `backward(W, P, G) -> Tensor`

- `P` is the output of `prefix_products(W)`. Return `dW` with `dW[i]` the gradient of `L` with respect to `W[i]`.
- Do not use autograd: no `backward()` and no `torch.autograd.grad`.
- Use `O(N)` matrix products.

**Example**, the matrices of Part 1 with every `G[i]` the all-ones matrix: `dW[0] = [[9, 3], [9, 3]]`, the same as `W.grad` after `P.sum().backward()`.""",
        },
        {
            "title": "Hillis-Steele forward",
            "description_en": r"""Keep Parts 1–2 and compute `P` in a logarithmic number of rounds.

**Signature:** `scan_forward(W) -> Tensor`

- Return the same `P` as Part 1.
- Start with `x = W`. Run rounds with stride `s = 1, 2, 4, …` while `s < N`. In a round, every `x[i]` with `i >= s` becomes `x[i - s] @ x[i]`, using the values from before the round; the others stay.
- Each round is one batched matrix product. The whole call makes at most `ceil(log2(N))` calls to `@`, `torch.matmul`, `torch.bmm`, `torch.mm` or `torch.einsum`.

**Example**, `N = 4`: after stride `1`, `x` is `W0`, `W0 W1`, `W1 W2`, `W2 W3`; after stride `2`, it is `P[0]` to `P[3]`, in two rounds.""",
        },
        {
            "title": "Hillis-Steele backward",
            "description_en": r"""Keep Parts 1–3 and compute the gradient through the scan.

**Signature:** `scan_backward(W, G) -> Tensor`

- Return the same `dW` as Part 2, without autograd.
- Undo the rounds from last to first. The whole call makes at most `3 * ceil(log2(N))` calls to the matrix product functions listed in Part 3.

**Example:** for `N = 33` that is at most `18` products, where Part 2's loop needs about `64`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "For Y = A @ B, which inputs does autograd keep to compute the gradients later? If you then write into the tensor that held A, what does the backward pass find? How can you build P without writing into any existing tensor?"},
        {"level": 2, "kind": "analysis", "content": "Keep a Python list: out = [W[0]]; for i in range(1, N): out.append(out[-1] @ W[i]). Each product makes a new tensor, so nothing autograd saved is overwritten. Return torch.stack(out), which is also differentiable."},
    ],
    "model_connections": [
        "State-space models such as S4 and Mamba, and linear attention, train recurrences in parallel with associative scans.",
        "Writing custom backward passes, as in torch.autograd.Function or fused kernels, needs exactly this kind of hand-derived gradient.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Building new tensors instead of writing in place keeps every value autograd saved intact.",
            "The hand-written backward pass uses one carry and O(N) products, the same cost as the forward pass.",
            "The scan needs only ceil(log2 N) dependent rounds, so a parallel device finishes long sequences faster.",
        ],
        "cons": [
            "The scan does O(N log N) products instead of O(N), so on one core it is slower.",
            "Keeping each round's input for the backward pass costs log N copies of W.",
            "Long products of matrices can overflow or vanish; real models keep them stable with structure or normalisation.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "gradient.flow", "code": _HELPERS + r"""
s = {fn}()
P = s.prefix_products(EXAMPLE)
assert torch.equal(P[1], torch.tensor([[2.0, 6.0], [1.0, 4.0]])) and torch.equal(P[2], torch.tensor([[6.0, 2.0], [4.0, 1.0]]))
W = EXAMPLE.clone().requires_grad_(True)
s.prefix_products(W).sum().backward()
assert torch.allclose(W.grad, true_grad(EXAMPLE, torch.ones(3, 2, 2)))
"""},
        {"name": "Part 1: random stacks under autograd", "part": 1, "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "For a random stack, the prefix products were wrong, the input changed, or backward() failed or gave the wrong gradient; do not write into a tensor autograd saved.",
         "code": _HELPERS + r"""
s = {fn}()
torch.manual_seed(0)
for N in (1, 2, 3, 6, 9):
    for D in (1, 3):
        W0 = torch.randn(N, D, D, dtype=torch.float64)
        G = torch.randn(N, D, D, dtype=torch.float64)
        W = W0.clone().requires_grad_(True)
        P = s.prefix_products(W)
        assert P.shape == W.shape and torch.allclose(P.detach(), loop_products(W0)), (N, D)
        (P * G).sum().backward()
        assert torch.allclose(W.grad, true_grad(W0, G)), (N, D)
        assert torch.equal(W.detach(), W0), "W was changed"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "gradient.flow", "code": _HELPERS + r"""
s = {fn}()
G = torch.ones(3, 2, 2)
with no_autograd():
    dW = s.backward(EXAMPLE, loop_products(EXAMPLE), G)
assert torch.allclose(dW[0], torch.tensor([[9.0, 3.0], [9.0, 3.0]])), dW[0]
assert torch.allclose(dW, true_grad(EXAMPLE, G))
"""},
        {"name": "Part 2: random gradients by hand", "part": 2, "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "The hand-written gradient differed from autograd's, used autograd, changed an input, or used far more than O(N) matrix products.",
         "code": _HELPERS + r"""
s = {fn}()
torch.manual_seed(1)
for N in (1, 2, 4, 7, 12):
    for D in (1, 2, 4):
        W = torch.randn(N, D, D, dtype=torch.float64)
        G = torch.randn(N, D, D, dtype=torch.float64)
        P = loop_products(W)
        copies = (W.clone(), P.clone(), G.clone())
        with no_autograd():
            dW = s.backward(W, P, G)
        assert dW.shape == W.shape and torch.allclose(dW, true_grad(W, G)), (N, D)
        assert all(torch.equal(a, b) for a, b in zip((W, P, G), copies)), "an input was changed"
W = torch.randn(40, 3, 3, dtype=torch.float64) * 0.5
with no_autograd(), counting() as c:
    s.backward(W, loop_products(W), torch.randn(40, 3, 3, dtype=torch.float64))
assert c.n <= 4 * 40, f"{c.n} matrix products for N = 40"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "performance.complexity", "code": _HELPERS + r"""
s = {fn}()
W = torch.cat([EXAMPLE, EXAMPLE[:1]])
with counting() as c:
    P = s.scan_forward(W)
assert torch.allclose(P, loop_products(W)) and c.n <= 2, c.n
"""},
        {"name": "Part 3: logarithmic rounds", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "scan_forward gave different products from the loop (keep lower indices on the left), changed W, or made more than ceil(log2 N) matrix product calls.",
         "code": _HELPERS + r"""
s = {fn}()
torch.manual_seed(2)
for N in (1, 2, 3, 5, 8, 9, 16, 33):
    W = torch.randn(N, 3, 3, dtype=torch.float64) * 0.7
    copy = W.clone()
    with counting() as c:
        P = s.scan_forward(W)
    assert torch.allclose(P, loop_products(W)), N
    assert c.n <= math.ceil(math.log2(N)), f"{c.n} product calls for N = {N}"
    assert torch.equal(W, copy), "W was changed"
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "gradient.flow", "code": _HELPERS + r"""
s = {fn}()
G = torch.ones(3, 2, 2)
with no_autograd():
    dW = s.scan_backward(EXAMPLE, G)
assert torch.allclose(dW, true_grad(EXAMPLE, G))
"""},
        {"name": "Part 4: gradients through the scan", "part": 4, "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "scan_backward differed from autograd's gradient, used autograd, changed an input, or made more than 3 * ceil(log2 N) matrix product calls.",
         "code": _HELPERS + r"""
s = {fn}()
torch.manual_seed(3)
for N in (1, 2, 3, 5, 8, 11, 16, 33):
    for D in (1, 3):
        W = torch.randn(N, D, D, dtype=torch.float64) * 0.7
        G = torch.randn(N, D, D, dtype=torch.float64)
        copies = (W.clone(), G.clone())
        with no_autograd(), counting() as c:
            dW = s.scan_backward(W, G)
        assert torch.allclose(dW, true_grad(W, G)), (N, D)
        assert c.n <= 3 * math.ceil(math.log2(N)), f"{c.n} product calls for N = {N}"
        assert torch.equal(W, copies[0]) and torch.equal(G, copies[1]), "an input was changed"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import torch


class PrefixScan:
    def prefix_products(self, W):
        out = [W[0]]
        for i in range(1, W.shape[0]):
            out.append(out[-1] @ W[i])  # new tensors only, so autograd keeps every input it saved
        return torch.stack(out)

    def backward(self, W, P, G):
        dW = []
        carry = torch.zeros_like(W[0])  # dL/dP[i]: from the loss and from P[i + 1] = P[i] @ W[i + 1]
        for i in reversed(range(W.shape[0])):
            carry = carry + G[i]
            dW.append(carry if i == 0 else P[i - 1].transpose(-1, -2) @ carry)
            carry = carry @ W[i].transpose(-1, -2)
        return torch.stack(dW[::-1])

    def _rounds(self, W):
        """Runs the scan; returns P and the input of every round, for the backward pass."""
        X, inputs, stride = W, [], 1
        while stride < W.shape[0]:
            inputs.append((stride, X))
            # one batched product per round; lower indices stay on the left
            X = torch.cat([X[:stride], X[:-stride] @ X[stride:]])
            stride *= 2
        return X, inputs

    def scan_forward(self, W):
        return self._rounds(W)[0]

    def scan_backward(self, W, G):
        H = G
        for stride, X in reversed(self._rounds(W)[1]):
            # Y[j] = X[j] for j < stride, else X[j - stride] @ X[j]; X[j] may be used in both roles
            right = X[:-stride].transpose(-1, -2) @ H[stride:]  # X[j] as the right factor of Y[j]
            left = H[stride:] @ X[stride:].transpose(-1, -2)    # X[j] as the left factor of Y[j + stride]
            dX = torch.cat([H[:stride], right]) + torch.cat([left, torch.zeros_like(H[:stride])])
            H = dX
        return H
''',
    "interview_questions": interview(
        concept=[
            "Why does writing P[i] = P[i - 1] @ W[i] into a preallocated tensor break backward()?",
            "Which tensors does autograd save for a matrix product, and why those?",
        ],
        deep_dive=[
            "What does the forward loop cost in products and in sequential steps for N matrices of size D?",
        ],
        tradeoffs=[
            "Why is the gradient flowing into P[i] a sum of two terms, and what are they?",
            "Why does the Hillis-Steele scan need associativity but not commutativity, and where does order matter?",
            "When is the scan faster than the loop, given that it does more total work?",
            "How would you cut the scan's memory, for example by recomputing rounds or using a Blelloch scan?",
        ],
    ),
}
