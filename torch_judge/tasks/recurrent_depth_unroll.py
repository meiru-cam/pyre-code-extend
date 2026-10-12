"""Recurrent-depth unroll: iterate a shared core block with truncated backpropagation."""

from ._interview import interview

TASK = {
    "title": "Recurrent-Depth Unroll with Truncated Backprop",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "recurrent_forward",
    "description_en": r"""A recurrent-depth model (Huginn) reasons in latent space by applying one shared core block many times. The prelude embeds the input once as `e`; each iteration concatenates the current state with `e`, maps it back to width `h` with an adapter, and runs the core. Training samples the number of iterations and backpropagates through only the last few. Implement the unroll and the split of a sampled iteration count.

**Signature:** `recurrent_forward(e, s0, adapter, core, n_no_grad, k_grad) -> Tensor`

**Parameters:**
- `e` — floating tensor of shape `(..., h)`, the prelude output. It may require grad.
- `s0` — floating tensor with the same shape as `e`, the initial latent state.
- `adapter` — callable mapping a `(..., 2h)` tensor to `(..., h)`.
- `core` — callable mapping `(..., h)` to `(..., h)`.
- `n_no_grad`, `k_grad` — integers `>= 0`.

**Returns:** the state after `n_no_grad + k_grad` iterations of

    s = core(adapter(torch.cat([s, e], dim=-1)))

starting from `s = s0`. The state comes first in the concatenation, and `e` is injected in every iteration.

**Second function:** `split_iterations(p, mean_backprop_depth) -> (n_no_grad, k_grad)` takes a sampled total iteration count `p` and the backprop depth `s` and returns the plain Python ints

    n_no_grad = max(p - s, 0)
    k_grad    = min(s, p)

so `n_no_grad + k_grad == p` and at most `s` iterations carry gradient.

**Constraints:**
- Run the first `n_no_grad` iterations inside `torch.no_grad()` and the last `k_grad` iterations with autograd enabled. Call `adapter` and `core` exactly once per iteration.
- The core and adapter parameters receive exactly the gradient of the last `k_grad` iterations. With `k_grad = 0` the output carries no graph and they receive no gradient.
- `e` keeps its graph: every gradient-carrying iteration sends gradient back through `e` to the prelude. Do not detach `e`.
- Do not detach `s0` or the state between the gradient-carrying iterations. With `n_no_grad = 0` the gradient reaches `s0`; with `n_no_grad > 0` it does not.
- Do not apply a final norm; with `n_no_grad = k_grad = 0` return a tensor equal to `s0`.
- Results keep the dtype of the inputs; float64 values and gradients are checked to 1e-10.
- `recurrent_forward` raises `ValueError` when `n_no_grad` or `k_grad` is not an integer `>= 0` or when `s0.shape != e.shape`. `split_iterations` raises `ValueError` when `p` or `mean_backprop_depth` is not an integer `>= 0`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The architecture.** Geiping et al. split the model into a prelude `P`, a recurrent core `R` and a coda `C`: `e = P(x)`, `s_0` is random noise, `s_i = R(e, s_(i-1))` for `i = 1..r`, and the coda reads `s_r`. Injecting `e` every step keeps the recurrence anchored to the input, so the state can settle to a fixed point that does not depend on `s_0`.

**Sampling the depth.** During training `r` is drawn from a log-normal Poisson distribution with a heavy tail, so the model learns to run for more iterations at test time. Upstream draws `p` and then splits it exactly as `split_iterations` does, with `mean_backprop_depth = 8` and `mean_recurrence = 32` by default. At evaluation it runs `mean_recurrence` iterations with none of them carrying gradient.

**Truncated backpropagation.** Only the last `k` iterations keep activations, so memory and backward cost do not grow with the sampled depth. The earlier iterations still move the state; they just do not teach the core. The prelude is still trained through every gradient-carrying step, because `e` enters each one.""",
    "advisory_prerequisites": ["mamba_ssm", "gradient_accumulation"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which context manager stops autograd from recording an operation, and what does a tensor produced inside it look like to a later backward pass? If the no-grad iterations come first, do you need to detach anything by hand at the boundary? How many times does e enter the graph when k_grad is 3?"},
        {"level": 2, "kind": "analysis", "content": "Validate the counts and shapes, then keep one running state starting at s0. Run the first loop inside a no-grad block and the second loop outside it, using the same update both times. Watch the concatenation order, injecting e in every iteration, and not adding detaches that cut the gradient-carrying chain. The split is two lines of integer arithmetic; check the case where p is smaller than the backprop depth."},
    ],
    "model_connections": [
        "Huginn-0125 (Geiping et al. 2025) runs a 4-layer core block a sampled number of times and backpropagates through the last 8 iterations.",
        "Deep-thinking networks (Bansal et al. 2022) introduced recalling the input at every iteration so long unrolls stay stable.",
        "Truncated backpropagation through time for RNNs makes the same memory trade along the time axis instead of depth.",
    ],
    "pro_con_analysis": {
        "pros": ["Activation memory and backward cost are fixed by k, so training can sample very deep unrolls, and test-time compute scales by just running more iterations."],
        "cons": ["The core never gets gradient from the early iterations, so it relies on reaching a stable fixed point; the gradient is biased whenever early steps matter."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/seal-rg/recurrent-pretraining",
            "commit": "1ea7220ec7eb42d13e89db0663df254d0bcdc28e",
            "path": "recpre/raven_modeling_minimal.py",
            "symbol": "RavenForCausalLM.iterate_forward, core_block_forward, randomized_iteration_sampler, initialize_state",
            "license": "Apache-2.0",
            "adapted": "The no-grad then grad unroll of iterate_forward and the update of core_block_forward, with the adapter and core passed in as callables; the n/k split of randomized_iteration_sampler with the sampled total p passed in.",
            "simplifications": "No final ln_f on the returned state (upstream normalizes it inside iterate_forward); no test-time noise, RoPE, masks, KV cache or activation checkpointing; "
                               "the log-normal Poisson draw happens outside, so split_iterations takes p and also accepts p = 0 (upstream always samples p >= 1); "
                               "counts are Python ints instead of long tensors; the evaluation branch (n = mean_recurrence, k = 0) is a call with k_grad = 0 instead of part of the split.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2502.05171",
            "section": "Geiping et al. 2025, Section 3.1 Macroscopic Design and Section 3.3 Training Objective (Truncated Backpropagation, equations 1 and 2)",
            "note": "The prelude/core/coda recurrence, the log-normal Poisson depth sampler and backpropagating through only the last k = 8 iterations.",
        },
    ],
    "tests": [
        {"name": "No-grad iterations first, then gradient-carrying ones", "behavior": "gradient.flow", "code": r"""
import torch
modes = []
def adapter(x):
    return x[..., :2] + 0.5 * x[..., 2:]
def core(x):
    modes.append(torch.is_grad_enabled())
    return 0.5 * x + 1.0
e = torch.tensor([[1.0, -2.0]])
s0 = torch.zeros(1, 2)
out = {fn}(e, s0, adapter, core, 2, 1)
want = s0
for _ in range(3):
    want = 0.5 * (want + 0.5 * e) + 1.0
assert torch.allclose(out, want), (out, want)
assert modes == [False, False, True], modes
"""},
        {"name": "Splitting a sampled iteration count", "behavior": "edge.empty_or_boundary", "code": r"""
split_iterations = {fn}.__globals__["split_iterations"]
assert split_iterations(40, 8) == (32, 8)
assert split_iterations(8, 8) == (0, 8)
assert split_iterations(5, 8) == (0, 5)
"""},
        {"name": "Matches a seeded unroll", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Each iteration must compute core(adapter(cat([s, e], dim=-1))) with the state first and e injected every time, for exactly n_no_grad + k_grad iterations.", "code": r"""
import torch
for seed, n, k in ((0, 3, 2), (1, 0, 4), (2, 5, 1), (3, 1, 0)):
    torch.manual_seed(seed)
    h = 3
    A = torch.nn.Linear(2 * h, h).double()
    W = torch.nn.Linear(h, h).double()
    core = lambda x: torch.tanh(W(x))
    e = torch.randn(2, 4, h, dtype=torch.float64)
    s0 = torch.randn(2, 4, h, dtype=torch.float64)
    out = {fn}(e, s0, A, core, n, k)
    want = s0
    with torch.no_grad():
        for _ in range(n + k):
            want = torch.tanh(W(A(torch.cat([want, e], dim=-1))))
    assert out.shape == e.shape and out.dtype == torch.float64, (out.shape, out.dtype)
    assert torch.allclose(out.detach(), want, atol=1e-10, rtol=0), (seed, n, k)
"""},
        {"name": "Grad mode schedule per call", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "The first n_no_grad iterations run under torch.no_grad() and the last k_grad with autograd, calling adapter and core once each per iteration.", "code": r"""
import torch
for n, k in ((0, 3), (4, 0), (3, 2), (1, 5)):
    a_modes, c_modes = [], []
    def adapter(x):
        a_modes.append(torch.is_grad_enabled())
        return x[..., :3] - 0.25 * x[..., 3:]
    def core(x):
        c_modes.append(torch.is_grad_enabled())
        return torch.sin(x)
    e = torch.randn(2, 3, requires_grad=True)
    {fn}(e, torch.zeros(2, 3), adapter, core, n, k)
    want = [False] * n + [True] * k
    assert a_modes == want and c_modes == want, (n, k, a_modes, c_modes)
"""},
        {"name": "Core gradient comes from the last k iterations only", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Core and adapter parameters must get exactly the gradient of the last k_grad iterations: no detach between them and no gradient from the no-grad ones.", "code": r"""
import torch
for seed, n, k in ((10, 3, 3), (11, 2, 1), (12, 0, 4)):
    torch.manual_seed(seed)
    h = 3
    A = torch.nn.Linear(2 * h, h).double()
    W = torch.nn.Linear(h, h).double()
    params = list(A.parameters()) + list(W.parameters())
    core = lambda x: torch.tanh(W(x))
    e = torch.randn(4, h, dtype=torch.float64)
    s0 = torch.randn(4, h, dtype=torch.float64)
    g = torch.randn(4, h, dtype=torch.float64)
    got = torch.autograd.grad(({fn}(e, s0, A, core, n, k) * g).sum(), params)
    def unroll(steps_no_grad, steps_grad):
        s = s0
        with torch.no_grad():
            for _ in range(steps_no_grad):
                s = torch.tanh(W(A(torch.cat([s, e], dim=-1))))
        for _ in range(steps_grad):
            s = torch.tanh(W(A(torch.cat([s, e], dim=-1))))
        return torch.autograd.grad((s * g).sum(), params)
    want = unroll(n, k)
    full = unroll(0, n + k)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-10, rtol=0), (seed, n, k)
    if n > 0:
        assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(want, full))
"""},
        {"name": "Prelude gets gradient through e on every gradient step", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Keep e attached and inject it in every gradient-carrying iteration, so the prelude receives the sum of gradients from all k_grad steps.", "code": r"""
import torch
for seed, n, k in ((20, 2, 3), (21, 4, 1), (22, 0, 2)):
    torch.manual_seed(seed)
    h = 3
    P = torch.nn.Linear(5, h).double()
    A = torch.nn.Linear(2 * h, h).double()
    W = torch.nn.Linear(h, h).double()
    core = lambda x: torch.tanh(W(x))
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    s0 = torch.randn(2, 3, h, dtype=torch.float64)
    g = torch.randn(2, 3, h, dtype=torch.float64)
    got = torch.autograd.grad(({fn}(P(x), s0, A, core, n, k) * g).sum(), list(P.parameters()))
    e = P(x)
    s = s0
    with torch.no_grad():
        for _ in range(n):
            s = torch.tanh(W(A(torch.cat([s, e], dim=-1))))
    for _ in range(k):
        s = torch.tanh(W(A(torch.cat([s, e], dim=-1))))
    want = torch.autograd.grad((s * g).sum(), list(P.parameters()))
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-10, rtol=0), (seed, n, k)
"""},
        {"name": "Gradient reaches s0 only without no-grad iterations", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Do not detach s0: with n_no_grad = 0 its gradient flows through every iteration, and with n_no_grad > 0 the no-grad phase cuts it.", "code": r"""
import torch
torch.manual_seed(30)
h = 2
A = torch.nn.Linear(2 * h, h).double()
W = torch.nn.Linear(h, h).double()
core = lambda x: torch.tanh(W(x))
e = torch.randn(3, h, dtype=torch.float64)
s0 = torch.randn(3, h, dtype=torch.float64, requires_grad=True)
(got,) = torch.autograd.grad({fn}(e, s0, A, core, 0, 3).sum(), [s0])
s = s0
for _ in range(3):
    s = torch.tanh(W(A(torch.cat([s, e], dim=-1))))
(want,) = torch.autograd.grad(s.sum(), [s0])
assert torch.allclose(got, want, atol=1e-10, rtol=0), (got, want)
out = {fn}(e, s0, A, core, 2, 3)
(gs0, gw) = torch.autograd.grad(out.sum(), [s0, W.weight], allow_unused=True)
assert gs0 is None or torch.count_nonzero(gs0) == 0, gs0
assert gw is not None and torch.count_nonzero(gw) > 0
"""},
        {"name": "k_grad = 0 carries no graph", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "With k_grad = 0 every iteration runs under no_grad, so the output has no graph; with no iterations at all, return s0's values.", "code": r"""
import torch
torch.manual_seed(40)
h = 3
A = torch.nn.Linear(2 * h, h).double()
W = torch.nn.Linear(h, h).double()
core = lambda x: torch.tanh(W(x))
e = torch.randn(2, h, dtype=torch.float64, requires_grad=True)
s0 = torch.randn(2, h, dtype=torch.float64)
out = {fn}(e, s0, A, core, 5, 0)
assert not out.requires_grad and out.grad_fn is None, out
want = s0
with torch.no_grad():
    for _ in range(5):
        want = torch.tanh(W(A(torch.cat([want, e], dim=-1))))
assert torch.allclose(out, want, atol=1e-10, rtol=0)
same = {fn}(e, s0, A, core, 0, 0)
assert torch.equal(same, s0), same
"""},
        {"name": "Split covers p below, at and above the backprop depth", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Return (max(p - s, 0), min(s, p)) as Python ints, so the counts sum to p and at most s iterations carry gradient.", "code": r"""
split_iterations = {fn}.__globals__["split_iterations"]
for s in range(0, 11):
    for p in range(0, 41):
        n, k = split_iterations(p, s)
        assert type(n) is int and type(k) is int, (p, s, type(n), type(k))
        assert (n, k) == (max(p - s, 0), min(s, p)), (p, s, n, k)
"""},
        {"name": "Rejects invalid counts and shapes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError for negative or non-integer counts and for s0 whose shape differs from e.", "code": r"""
import torch
split_iterations = {fn}.__globals__["split_iterations"]
ident = lambda x: x[..., :2]
e = torch.zeros(3, 2)
cases = [
    lambda: {fn}(e, torch.zeros(3, 2), ident, ident, -1, 2),
    lambda: {fn}(e, torch.zeros(3, 2), ident, ident, 2, -1),
    lambda: {fn}(e, torch.zeros(3, 2), ident, ident, 1.5, 2),
    lambda: {fn}(e, torch.zeros(3, 3), ident, ident, 1, 1),
    lambda: split_iterations(-1, 8),
    lambda: split_iterations(5, -2),
    lambda: split_iterations(5.0, 8),
]
for i, case in enumerate(cases):
    try:
        case()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch


def _check_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")


def recurrent_forward(e, s0, adapter, core, n_no_grad, k_grad):
    _check_count("n_no_grad", n_no_grad)
    _check_count("k_grad", k_grad)
    if s0.shape != e.shape:
        raise ValueError("s0 must have the same shape as e")
    s = s0
    with torch.no_grad():
        for _ in range(n_no_grad):
            s = core(adapter(torch.cat([s, e], dim=-1)))
    for _ in range(k_grad):
        s = core(adapter(torch.cat([s, e], dim=-1)))
    return s


def split_iterations(p, mean_backprop_depth):
    _check_count("p", p)
    _check_count("mean_backprop_depth", mean_backprop_depth)
    n_no_grad = max(p - mean_backprop_depth, 0)
    k_grad = min(mean_backprop_depth, p)
    return n_no_grad, k_grad
''',
    "interview_questions": interview(
        concept=[
            "What are the prelude, the recurrent core and the coda in a recurrent-depth model, and what does each iteration compute?",
            "Why does the core receive the prelude output e in every iteration instead of only as the initial state?",
        ],
        deep_dive=[
            "Why does running the first iterations under no_grad give exactly the gradient of the last k iterations, without an explicit detach?",
            "How does the prelude get trained when only the last k iterations carry gradient, and how many gradient contributions does it receive?",
            "How do you split a sampled depth p into no-grad and gradient-carrying iterations, and what happens when p is smaller than the backprop depth?",
        ],
        tradeoffs=[
            "Truncated backpropagation through the last k iterations versus full backpropagation or activation checkpointing: memory, compute and gradient bias?",
            "Concatenating the state with e through an adapter versus adding e to the state: what does each cost, and when does the choice matter?",
        ],
    ),
}
