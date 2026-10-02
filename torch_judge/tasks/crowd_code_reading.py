"""Read a crowd-label PyTorch training script: count its costs, estimate annotator reliability from learned confusion matrices, and vectorize its double-loop loss."""

from ._interview import interview

# The loop loss as written in the script, and random inputs of matching shapes.
_MODEL = r"""
import math, random, time
import torch

def nll_loop(p, M, Y):
    total = torch.zeros((), dtype=p.dtype)
    for i in range(Y.shape[0]):
        for a in range(Y.shape[1]):
            lab = Y[i, a].item()
            if lab < 0:
                continue
            q = p[i] @ M[a]
            total = total + -torch.log(q[lab] + 1e-8)
    return total

def rand_inputs(g, N, A, C, missing=0.4):
    p = torch.softmax(torch.randn(N, C, generator=g, dtype=torch.float64), -1).requires_grad_()
    M = torch.softmax(torch.randn(A, C, C, generator=g, dtype=torch.float64) * 2, -1).requires_grad_()
    Y = torch.randint(0, C, (N, A), generator=g)
    Y[torch.rand(N, A, generator=g) < missing] = -1
    return p, M, Y

def learner(fn, name):
    f = fn.__globals__.get(name)
    assert callable(f), f"define {name} at module level"
    return f
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "performance.complexity", "code": _MODEL + r"""
got = {fn}(600, 12, 3, 7, 32, 1500)
want = {"net_macs": 600 * (12 * 32 + 32 * 3), "net_outputs": 600 * 35, "nll_macs": 1500 * 9, "loop_iterations": 4200, "dense_elements": 12600}
assert got == want, got
"""},
    {"name": "Part 1: random sizes", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "op_counts disagreed with the counts read off the script for random N, D, C, A, H and P: forward multiply-adds of the two linear layers, their output elements, multiply-adds of the loss, inner-loop iterations, or elements of the all-pairs product.",
     "code": _MODEL + r"""
for seed in range(100):
    rng = random.Random(seed)
    N, D, C, A, H = (rng.randint(1, 5000) for _ in range(5))
    P = rng.randint(0, N * A)
    got = {fn}(N, D, C, A, H, P)
    want = {"net_macs": N * (D * H + H * C), "net_outputs": N * (H + C), "nll_macs": P * C * C, "loop_iterations": N * A, "dense_elements": N * A * C}
    assert got == want, (seed, got, want)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "metrics.averaging", "code": _MODEL + r"""
reliability, least_reliable = learner({fn}, "reliability"), learner({fn}, "least_reliable")
M = torch.tensor([[[0.9, 0.1], [0.3, 0.7]], [[0.5, 0.5], [0.4, 0.6]], [[0.6, 0.4], [0.0, 1.0]]])
r = reliability(M)
assert r.shape == (3,) and torch.allclose(r, torch.tensor([0.8, 0.55, 0.8])), r
assert least_reliable(r) == 1
"""},
    {"name": "Part 2: random confusion matrices", "part": 2, "visibility": "unshown", "behavior": "metrics.averaging",
     "failure_message": "reliability did not return the mean of each annotator's diagonal as an (A,) tensor for random A and C, or least_reliable did not return the first index of the minimum as a Python int.",
     "code": _MODEL + r"""
reliability, least_reliable = learner({fn}, "reliability"), learner({fn}, "least_reliable")
g = torch.Generator().manual_seed(0)
for trial in range(50):
    A, C = int(torch.randint(1, 9, (1,), generator=g)), int(torch.randint(1, 7, (1,), generator=g))
    M = torch.softmax(torch.randn(A, C, C, generator=g, dtype=torch.float64), -1)
    want = torch.tensor([sum(M[a, c, c].item() for c in range(C)) / C for a in range(A)], dtype=torch.float64)
    r = reliability(M)
    assert r.shape == (A,) and torch.allclose(r.double(), want), (trial, r, want)
    worst = min(range(A), key=lambda a: (want[a].item(), a))
    got = least_reliable(want)
    assert type(got) is int and got == worst, (trial, got, worst)
assert least_reliable(torch.tensor([0.5, 0.2, 0.2])) == 1
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "gradient.flow", "code": _MODEL + r"""
nll_vectorized = learner({fn}, "nll_vectorized")
p = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
M = torch.tensor([[[0.9, 0.1], [0.2, 0.8]], [[0.6, 0.4], [0.5, 0.5]]], dtype=torch.float64)
Y = torch.tensor([[0, -1], [1, 1]])
got = nll_vectorized(p, M, Y)
want = -(math.log(0.69 + 1e-8) + math.log(0.66 + 1e-8) + math.log(0.48 + 1e-8))
assert got.dim() == 0 and abs(got.item() - want) < 1e-9, (got, want)
"""},
    {"name": "Part 3: values, gradients and speed", "part": 3, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "nll_vectorized differed from the double loop in value or in its gradients with respect to p and M, was not a 0-dim tensor (0 when nothing is observed), or took more than 2 seconds on 200000 samples and 10 annotators, which a Python loop over pairs cannot meet.",
     "code": _MODEL + r"""
nll_vectorized = learner({fn}, "nll_vectorized")
g = torch.Generator().manual_seed(1)
for trial in range(20):
    N, A, C = int(torch.randint(1, 30, (1,), generator=g)), int(torch.randint(1, 6, (1,), generator=g)), int(torch.randint(2, 6, (1,), generator=g))
    p, M, Y = rand_inputs(g, N, A, C)
    want = nll_loop(p, M, Y)
    got = nll_vectorized(p, M, Y)
    assert got.dim() == 0 and torch.allclose(got.double(), want), (trial, got, want)
    if Y.ge(0).any():
        gw = torch.autograd.grad(want, (p, M))
        gg = torch.autograd.grad(got, (p, M))
        assert all(torch.allclose(x, y) for x, y in zip(gg, gw)), (trial, "gradients differ")
p, M, Y = rand_inputs(g, 3, 2, 3)
Y[:] = -1
assert nll_vectorized(p, M, Y).item() == 0.0
p, M, Y = rand_inputs(g, 200000, 10, 4)
start = time.perf_counter()
nll_vectorized(p.detach(), M.detach(), Y)
took = time.perf_counter() - start
assert took < 2.0, f"took {took:.2f}s"
"""},
]

TASK = {
    "title": "Crowd Labels Code Reading",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "op_counts",
    "description_en": r"""A training script learns a classifier from labels given by several annotators, with no true labels. Read what it does, then extend and speed it up. Each part asks for module-level functions.

The requirement arrives in parts. Each part keeps every earlier function, so one file passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- There are `N` samples with `D` features, `C` classes and `A` annotators. `Y` is an `(N, A)` integer tensor: `Y[i, a]` is the class annotator `a` gave sample `i`, or `-1` if `a` did not label `i`. `P` is the number of entries of `Y` that are not `-1`.
- The classifier `net` is `Linear(D, H)`, ReLU, then `Linear(H, C)`. A softmax of its output gives `p`, an `(N, C)` tensor of class probabilities.
- `M` is an `(A, C, C)` tensor of confusion matrices: `M[a, c, o]` is the chance that annotator `a` reports `o` when the true class is `c`, so every row sums to 1.
- The script's loss loops over every `i`, then every `a`. It skips `Y[i, a] < 0`, and otherwise adds `-log((p[i] @ M[a])[Y[i, a]] + 1e-8)`. That loop is called the double loop below.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** code-reading rounds hand you an unfamiliar training script and ask where the time and memory go before you change anything. Each later part adds one requirement: a small extension on the learned parameters, then a refactor that must not change the numbers.

**Where it is used:** labels from crowds or from several model graders disagree, and confusion-matrix layers learn how far to trust each source. Removing Python loops from a loss like this one is everyday work in training code.

Adapted from the PyTorch code-reading question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded. The written answers about shapes and complexity become `op_counts`, and `reliability` takes the confusion matrices instead of being attached to the crowd layer. The script itself, its data generator and the bonus operator questions are left out; the vectorized loss is graded on random inputs and on speed.""",
    "parts": [
        {
            "title": "Reading the costs",
            "description_en": r"""**Signature:** `op_counts(N, D, C, A, H, P) -> dict`

Return exact counts for one training step, ignoring biases and the softmax:
- `net_macs`: multiply-adds in the forward pass of `net` on all `N` samples.
- `net_outputs`: elements in the outputs of the two linear layers, which backward keeps.
- `nll_macs`: multiply-adds the double loop spends on `p[i] @ M[a]`, over the `P` observed pairs.
- `loop_iterations`: times the inner loop body starts, missing labels included.
- `dense_elements`: elements of `p[i] @ M[a]` computed for every `(i, a)` pair at once, before masking.

**Example:** `op_counts(600, 12, 3, 7, 32, 1500)` returns:
- `net_macs` `249600` and `net_outputs` `21000`
- `nll_macs` `13500`, `loop_iterations` `4200` and `dense_elements` `12600`""",
        },
        {
            "title": "Annotator reliability",
            "description_en": r"""Keep Part 1. Add `reliability(M)` and `least_reliable(r)`.

- `reliability(M)` returns an `(A,)` tensor: annotator `a`'s reliability is the mean over classes `c` of `M[a, c, c]`, the chance it reports the true class.
- `least_reliable(r)` returns the index of the smallest entry as a Python `int`, the first such index on a tie.

**Example:** with confusion matrices whose diagonals are `(0.9, 0.7)`, `(0.5, 0.6)` and `(0.6, 1.0)`:
- `reliability` returns `[0.8, 0.55, 0.8]`
- `least_reliable` returns `1`""",
        },
        {
            "title": "Vectorize the loss",
            "description_en": r"""Keep Parts 1–2. Add `nll_vectorized(p, M, Y)`.

- Return the double loop's value as a 0-dim tensor, with the same gradients with respect to `p` and `M`, up to floating-point error. With no observed label it returns `0`.
- Use no Python loop over samples or over observed pairs: on `200000` samples and `10` annotators it must finish within `2` seconds.

**Example:** `p = [[0.7, 0.3], [0.2, 0.8]]`, `M[0] = [[0.9, 0.1], [0.2, 0.8]]`, `M[1] = [[0.6, 0.4], [0.5, 0.5]]` and `Y = [[0, -1], [1, 1]]`:
- the observed pairs give probabilities `0.69`, `0.66` and `0.48`
- the result is `-(log(0.69 + 1e-8) + log(0.66 + 1e-8) + log(0.48 + 1e-8))`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "A Linear(D, H) applied to N rows: how many multiply-adds and how many output elements? For one observed pair, what are the shapes of p[i] and M[a], and how many multiply-adds does their product take? Does the inner loop body start for a missing label?"},
        {"level": 2, "kind": "analysis", "content": "Each linear layer costs rows times inputs times outputs multiply-adds and leaves rows times outputs elements: N*D*H plus N*H*C, and N*H plus N*C. A (C,) by (C, C) product costs C*C, paid once per observed pair. The loops start N*A times, and the all-pairs product holds N*A*C numbers."},
    ],
    "model_connections": [
        "Crowd-label and multi-grader setups learn a confusion matrix per source to weigh noisy labels, as in reward models trained on disagreeing raters.",
        "Replacing per-element Python loops with gathers and einsum is the usual first speed-up when a loss is slow to train.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Counting multiply-adds and Python iterations separately explains why a tiny loss can dominate step time.",
            "Gathering only the observed pairs keeps memory at P*C instead of N*A*C.",
            "Reliability read off the diagonal needs no true labels.",
        ],
        "cons": [
            "Multiply-add counts ignore Python overhead and kernel launches, which dominate small loops.",
            "Mean diagonal reliability hides annotators who are good on some classes and bad on others.",
            "Vectorized code is harder to read than the loop it replaces, so a test against the loop must stay.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import torch


def op_counts(N, D, C, A, H, P):
    """Costs of one training step of the crowd-label classifier, read off the code."""
    return {
        "net_macs": N * (D * H + H * C),  # two linear layers on N rows
        "net_outputs": N * (H + C),  # forward outputs kept for backward
        "nll_macs": P * C * C,  # one (C,) @ (C, C) product per observed pair
        "loop_iterations": N * A,  # the double loop visits missing pairs too
        "dense_elements": N * A * C,  # p[i] @ M[a] for every pair at once
    }


def reliability(M):
    """(A, C, C) row-stochastic confusion matrices -> (A,) mean of each diagonal."""
    return M.diagonal(dim1=-2, dim2=-1).mean(dim=-1)


def least_reliable(r):
    return int(torch.argmin(r))  # the first index on ties


def nll_vectorized(p, M, Y):
    """Same scalar as the double loop: -sum over observed (i, a) of log((p[i] @ M[a])[Y[i, a]] + 1e-8)."""
    i, a = (Y >= 0).nonzero(as_tuple=True)  # observed pairs only: O(P * C) memory, not O(N * A * C)
    lab = Y[i, a]
    q = torch.einsum("pc,pc->p", p[i], M[a, :, lab])
    return -torch.log(q + 1e-8).sum()
''',
    "interview_questions": interview(
        concept=[
            "Trace the data from X to the optimizer step: where do Y and the true labels enter, if at all?",
            "What are the shapes of net(X), p and M, and of the two operands of p[i] @ M[a]?",
        ],
        deep_dive=[
            "The forward pass of net does far more multiply-adds than the loss loop, yet the loop takes longer. Why?",
        ],
        tradeoffs=[
            "Why is mean diagonal reliability a reasonable estimate without true labels, and when does it mislead?",
            "Computing p @ M for every pair before masking needs N*A*C memory. How do you avoid it?",
            "How do you prove the vectorized loss matches the loop, gradients included?",
            "What would you check before trusting a learned confusion matrix, given that p and M can trade off against each other?",
        ],
    ),
}
