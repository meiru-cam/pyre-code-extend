"""Muon optimizer update: momentum followed by Newton-Schulz orthogonalization."""

from ._interview import interview

TASK = {
    "title": "Muon Newton-Schulz Update",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "muon_update",
    "description_en": r"""Compute one Muon update direction for a 2-D weight matrix.

**Signature:** `muon_update(grad, momentum, beta=0.95, nesterov=True, ns_steps=5, eps=1e-7) -> Tensor`

**Parameters:**
- `grad` — float tensor `(R, K)`, the current gradient.
- `momentum` — float tensor `(R, K)`, the momentum buffer. Update it in place.
- `beta` — momentum coefficient in `[0, 1)`.
- `nesterov` — whether to use the Nesterov look-ahead.
- `ns_steps` — number of Newton-Schulz iterations.

**Steps:**
- Momentum: `momentum = beta * momentum + (1 - beta) * grad`, in place.
- Direction: `G = (1 - beta) * grad + beta * momentum` if `nesterov`, otherwise `G = momentum`.
- Newton-Schulz: let `X = G / (frobenius_norm(G) + eps)`. If `R > K`, work on `X.T` and transpose back at the end. Repeat `ns_steps` times with `a, b, c = 3.4445, -4.7750, 2.0315`:

    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X

- Scale: return `X * sqrt(max(1, R / K))`.

**Returns:** tensor with the same shape and dtype as `grad`.

**Constraints:**
- Do not modify `grad`.
- Compute in the input dtype; do not use an SVD or matrix inverse.
- Raise `ValueError` when `grad` is not 2-D, `momentum` has a different shape, `beta` is outside `[0, 1)`, or `ns_steps < 1`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**What Muon does.** For a gradient with SVD `U S V.T`, the ideal Muon step is the orthogonal factor `U V.T`: every singular direction moves by the same amount, so rare but important directions are not drowned out by a few dominant ones. Dividing by the Frobenius norm puts all singular values in `[0, 1]`, and the quintic iteration pushes them toward 1 without an SVD.

**Why these coefficients.** They make the iteration grow small singular values as fast as possible. The price is that the result is only approximately orthogonal: singular values land roughly between 0.7 and 1.2 instead of exactly 1, which works just as well in practice and runs stably in bfloat16.

**Where it is used.** Muon updates hidden 2-D weight matrices. Embeddings, the LM head, biases and norm gains still use AdamW. Kimi's Moonlight and K2 scaled it to large language models.""",
    "advisory_prerequisites": ["sgd_momentum", "adam"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Why normalize by the Frobenius norm before iterating? Why transpose when R > K? What happens to each singular value under the polynomial a*s + b*s^3 + c*s^5?"},
        {"level": 2, "kind": "analysis", "content": "Validate, then momentum.mul_(beta).add_(grad, alpha=1 - beta). Form G without touching grad. Normalize, transpose if tall, loop ns_steps times with A = X @ X.T, B = b*A + c*A@A, X = a*X + B@X, transpose back, and scale by sqrt(max(1, R / K))."},
    ],
    "model_connections": [
        "Keller Jordan's Muon set NanoGPT speedrun records; Moonshot's Moonlight and Kimi K2 trained with Muon on hidden matrices and AdamW elsewhere.",
    ],
    "pro_con_analysis": {
        "pros": ["Faster convergence than AdamW per step at similar cost, one momentum buffer instead of two, and a well-conditioned update direction."],
        "cons": ["Only for 2-D hidden weights, extra matmuls per step, and the orthogonalization needs the full matrix, which complicates sharding under FSDP or tensor parallelism."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/KellerJordan/Muon',
      'commit': 'f98f1cacc0263b04290753e32be8d498c1efc806',
      'path': 'muon.py',
      'symbol': 'zeropower_via_newtonschulz5 and muon_update',
      'license': 'MIT',
      'adapted': 'Momentum lerp, Nesterov look-ahead, Frobenius normalization, transposing tall matrices, the quintic '
                 'coefficients and the sqrt(max(1, R / K)) scale.',
      'simplifications': 'Computes in the input dtype instead of bfloat16, does not overwrite grad in place, and supports '
                         '2-D matrices only.'}],
    "tests": [
        {"name": "Update of a diagonal gradient", "behavior": "optim.state", "code": r"""
import torch
grad = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
momentum = torch.zeros(2, 2, dtype=torch.float64)
out = {fn}(grad, momentum, beta=0.5, nesterov=False, ns_steps=5)
assert torch.allclose(momentum, 0.5 * grad), momentum
x = torch.tensor([2.0, 1.0], dtype=torch.float64) * 0.5
x = x / (x.norm() + 1e-7)
for _ in range(5):
    x = 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5
assert out.shape == (2, 2) and torch.allclose(out, torch.diag(x), atol=1e-12), (out, x)
assert out[0, 1] == 0 and out[1, 0] == 0
"""},
        {"name": "Matches a seeded reference iteration", "visibility": "unshown", "behavior": "optim.state", "failure_message": "Follow the stated momentum, Nesterov, normalization, transpose, iteration and scaling steps exactly.", "code": r"""
import math, torch
def oracle(grad, buf, beta, nesterov, steps):
    buf = beta * buf + (1 - beta) * grad
    G = (1 - beta) * grad + beta * buf if nesterov else buf
    R, K = G.shape
    X = G / (torch.sqrt((G * G).sum()) + 1e-7)
    tall = R > K
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    if tall:
        X = X.T
    return X * math.sqrt(max(1.0, R / K)), buf
for seed in (4, 36, 90):
    g = torch.Generator().manual_seed(seed)
    R, K = (int(v) for v in torch.randint(2, 9, (2,), generator=g))
    buf = torch.randn(R, K, generator=g, dtype=torch.float64)
    for nesterov, beta, steps in ((True, 0.95, 5), (False, 0.8, 3)):
        grad = torch.randn(R, K, generator=g, dtype=torch.float64)
        grad_copy = grad.clone(); state = buf.clone()
        want, want_buf = oracle(grad, buf.clone(), beta, nesterov, steps)
        out = {fn}(grad, state, beta=beta, nesterov=nesterov, ns_steps=steps)
        assert torch.equal(grad, grad_copy), "grad was modified"
        assert torch.allclose(state, want_buf, atol=1e-12), (seed, "momentum")
        assert out.shape == (R, K) and out.dtype == torch.float64, out.shape
        assert torch.allclose(out, want, atol=1e-9), (seed, nesterov, (out - want).abs().max())
"""},
        {"name": "Approximately orthogonalizes a well-conditioned matrix", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "The result should share the gradient's singular vectors with singular values pushed toward 1.", "code": r"""
import torch
g = torch.Generator().manual_seed(11)
U, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
V, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64))
S = torch.tensor([3.0, 2.0, 1.5, 1.0], dtype=torch.float64)
grad = U[:, :4] @ torch.diag(S) @ V.T
out = {fn}(grad, torch.zeros_like(grad), beta=0.0, nesterov=False, ns_steps=5) / (6 / 4) ** 0.5
core = U[:, :4].T @ out @ V
off = core - torch.diag(torch.diagonal(core))
assert off.abs().max() < 1e-9, off
sv = torch.diagonal(core)
assert torch.all(sv > 0.6) and torch.all(sv < 1.25), sv
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Non-2-D gradients, mismatched buffers, beta outside [0, 1) and ns_steps < 1 must raise ValueError.", "code": r"""
import torch
cases = [
    (torch.zeros(3), torch.zeros(3), {}),
    (torch.zeros(2, 3), torch.zeros(3, 2), {}),
    (torch.zeros(2, 3), torch.zeros(2, 3), {"beta": 1.0}),
    (torch.zeros(2, 3), torch.zeros(2, 3), {"ns_steps": 0}),
]
for grad, buf, kwargs in cases:
    try:
        {fn}(grad, buf, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {tuple(grad.shape)} {tuple(buf.shape)} {kwargs}")
"""},
    ],
    "solution": '''import math
import torch

def muon_update(grad, momentum, beta=0.95, nesterov=True, ns_steps=5, eps=1e-7):
    if grad.ndim != 2 or momentum.shape != grad.shape:
        raise ValueError("grad and momentum must share one 2-D shape")
    if not 0 <= beta < 1 or ns_steps < 1:
        raise ValueError("need 0 <= beta < 1 and ns_steps >= 1")
    momentum.mul_(beta).add_(grad, alpha=1 - beta)
    G = (1 - beta) * grad + beta * momentum if nesterov else momentum.clone()

    a, b, c = 3.4445, -4.7750, 2.0315
    rows, cols = G.shape
    X = G / (G.norm() + eps)
    tall = rows > cols
    if tall:
        X = X.T
    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if tall:
        X = X.T
    return X * math.sqrt(max(1.0, rows / cols))
''',
    "interview_questions": interview(
        concept=[
            "What does Muon do differently from Adam, and what is the ideal update it approximates?",
            "Why would equalizing the singular values of an update help training?",
        ],
        deep_dive=[
            "Walk through the Newton-Schulz iteration. Why normalize by the Frobenius norm first, and why transpose tall matrices?",
            "What happens to one singular value under the quintic polynomial, and why do the tuned coefficients not converge exactly to 1?",
            "Why is Muon applied only to hidden 2-D weights, and what optimizes embeddings, the LM head and norm gains?",
        ],
        tradeoffs=[
            "Muon versus AdamW: memory, compute per step and behaviour under distributed sharding?",
            "Newton-Schulz versus an exact SVD or polar decomposition: speed, numerical precision and bfloat16 stability?",
        ],
    ),
}
