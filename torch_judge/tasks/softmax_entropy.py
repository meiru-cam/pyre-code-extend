"""Softmax entropy: direct, numerically stable, in blocks with constant memory, and online over chunks."""

from ._interview import interview

# A high-precision model and memory probes.
_HELPERS = r"""
import random, tracemalloc

def model_entropy(x):
    x = [float(v) for v in x]
    m = max(x)
    w = [math.exp(v - m) for v in x]
    s = math.fsum(w)
    return max(0.0, math.log(s) - math.fsum(wi * (v - m) for wi, v in zip(w, x)) / s)

def close(a, b):
    return abs(float(a) - b) <= 1e-9 * max(1.0, abs(b))

def peak_bytes(f):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = f()
        return out, tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()

def random_logits(rng, n, spread):
    centre = rng.uniform(-spread, spread)
    return np.array([centre + rng.gauss(0, rng.choice([0.1, 1.0, 5.0, 40.0])) for _ in range(n)])
"""

TASK = {
    "title": "Streaming Softmax Entropy",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "EntropyMeter",
    "description_en": r"""Build `EntropyMeter`, which computes the entropy of the softmax of a vector of logits, first directly and then stably, in blocks and in one pass over chunks.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `EntropyMeter` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `logits` is a 1-D float array with at least one entry. Every entry is finite and at most `1e300` in size. NumPy is available as `np`.
- With `p = softmax(logits)`, return the entropy `H = -sum(p_i * log(p_i))` with the natural log, as a `float`. It is never negative.
- Results must match the exact value to about `1e-9` relative error.
- Never change `logits`.
- The examples write arrays as Python lists for brevity; the tests pass NumPy float arrays.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the formula is one line; the work is keeping it finite for large logits and computing it without materialising the distribution, and each later part adds one requirement.

**Where it is used:** the entropy of next-token distributions drives sampling heuristics, uncertainty estimates and entropy bonuses in RL fine-tuning, over vocabularies too large to hold many copies of.

Adapted from the streaming softmax entropy question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Direct entropy",
            "description_en": r"""**Signature:** `EntropyMeter()`, `entropy(logits) -> float`

- In this part every entry of `logits` is between `-30` and `30`, so the definition works as written: exponentiate, normalise, sum.

**Example:**
- `entropy([2.0, -1.0, 0.5, 3.5])` is about `0.6659`
- `entropy([4.0])` is `0.0`, and three equal logits give `log(3)`""",
        },
        {
            "title": "Stable entropy",
            "description_en": r"""Keep Part 1. `entropy` must now work for every input allowed by the rules.

- With `m` the largest logit, compute `exp(x_i - m)` instead of `exp(x_i)`, and `log(p_i)` as `(x_i - m) - log(sum_k exp(x_k - m))` rather than as the log of a quotient.
- The result is finite and correct even where `exp(x_i)` overflows or every `exp(x_i)` underflows to `0`.

**Example:**
- `entropy([710.0, 713.0, 711.5])` is about `0.6216`; the direct formula returns `nan` here
- `entropy([-800.0, -801.0, -805.0, -799.5])` is about `0.9665`""",
        },
        {
            "title": "Blocks in constant memory",
            "description_en": r"""Keep Parts 1–2 and work through the array in fixed-size blocks.

**Signature:** `entropy_blockwise(logits, block_size=2) -> float`

- Same result as `entropy`.
- Read `logits` in consecutive slices of `block_size` entries; the last slice may be shorter. You may read the array more than once.
- Keep only a fixed number of scalars between slices. Never build an array whose size grows with `len(logits)`, such as all the probabilities. A full copy counts too: `logits.astype(...)` copies everything, so convert each slice instead.

**Example:** `entropy_blockwise([0.0, 1.0, -3.0, 2.5, 1.5, -0.5, 4.0], 3)` reads `[0.0, 1.0, -3.0]`, `[2.5, 1.5, -0.5]` and `[4.0]`, and returns about `0.9171`.""",
        },
        {
            "title": "One pass over chunks",
            "description_en": r"""Keep Parts 1–3. Now the logits arrive as chunks you can read only once.

**Signature:** `entropy_online(chunks) -> float`

- `chunks` is an iterable of 1-D float arrays, any of which may be empty, consumed once in order. Together they hold at least one logit.
- Return the entropy of their concatenation, with the same accuracy as `entropy`.
- Keep only a fixed number of scalars between chunks; do not store the chunks. When a chunk raises the running maximum, rescale what you kept.

**Example:** the three slices of the Part 3 example, given one at a time, again give about `0.9171`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Entropy is -sum(p * log p) with p = exp(x) / sum(exp(x)). Which arrays do you need, in which order? What should one logit give, and what should N equal logits give?"},
        {"level": 2, "kind": "analysis", "content": "x = np.asarray(logits, dtype=float); e = np.exp(x); p = e / e.sum(); return float(-(p * np.log(p)).sum()). Check it on one logit (0.0) and on equal logits (log N)."},
    ],
    "model_connections": [
        "Sampling strategies such as entropy-based temperature or min-p, and uncertainty estimates, use the entropy of the next-token distribution.",
        "Policy-gradient fine-tuning adds an entropy bonus, computed per position over vocabularies of 100k+ tokens without materialising extra copies.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Subtracting the maximum keeps every exponent at or below zero, so nothing overflows and the largest term is exactly 1.",
            "Writing log p in centred form avoids log(0) when tiny probabilities underflow.",
            "Three running scalars, rescaled when the maximum grows, give the exact entropy in one pass with constant memory.",
        ],
        "cons": [
            "A Python loop over small blocks is far slower than one vectorised pass; blocks trade speed for memory.",
            "Rescaling the running sums at every new maximum adds rounding error on adversarial orderings.",
            "Online processing can only report the final value; per-position probabilities still need a second pass.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "numerics.stability", "code": _HELPERS + r"""
e = {fn}()
assert abs(e.entropy(np.array([2.0, -1.0, 0.5, 3.5])) - 0.6659165347987293) < 1e-9
assert e.entropy(np.array([4.0])) == 0.0
assert abs(e.entropy(np.array([1.0, 1.0, 1.0])) - math.log(3)) < 1e-12
"""},
        {"name": "Part 1: random moderate logits", "part": 1, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "For logits between -30 and 30, the entropy differed from -sum(p log p) of the softmax, came out negative, or changed the input.",
         "code": _HELPERS + r"""
e = {fn}()
rng = random.Random(1)
for _ in range(300):
    x = np.clip(random_logits(rng, rng.randint(1, 60), 5.0), -30, 30)
    copy = x.copy()
    got = e.entropy(x)
    assert isinstance(got, float) and got >= 0.0 and close(got, model_entropy(x)), (list(x), got)
    assert (x == copy).all(), "logits were changed"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "numerics.stability", "code": _HELPERS + r"""
e = {fn}()
assert abs(e.entropy(np.array([710.0, 713.0, 711.5])) - model_entropy([710.0, 713.0, 711.5])) < 1e-9
assert abs(e.entropy(np.array([-800.0, -801.0, -805.0, -799.5])) - model_entropy([-800.0, -801.0, -805.0, -799.5])) < 1e-9
"""},
        {"name": "Part 2: extreme logits", "part": 2, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "For very large or very negative logits, or a huge spread between them, the entropy was nan, inf, negative or wrong; subtract the maximum and use the centred log form.",
         "code": _HELPERS + r"""
e = {fn}()
rng = random.Random(2)
for _ in range(300):
    x = random_logits(rng, rng.randint(1, 60), rng.choice([1e3, 1e5, 1e9]))
    got = e.entropy(x)
    assert math.isfinite(got) and got >= 0.0 and close(got, model_entropy(x)), (list(x)[:5], got)
for x in ([1e300, -1e300, 3.0], [-1e300, -1e300], [5.0, 5.0 - 800.0, 5.0 - 1600.0], [0.0] * 1000):
    got = e.entropy(np.array(x))
    assert math.isfinite(got) and close(got, model_entropy(x)), (x[:3], got)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "numerics.stability", "code": _HELPERS + r"""
x = np.array([0.0, 1.0, -3.0, 2.5, 1.5, -0.5, 4.0])
assert abs({fn}().entropy_blockwise(x, 3) - model_entropy(x)) < 1e-9
"""},
        {"name": "Part 3: blocks of every size", "part": 3, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "entropy_blockwise differed from the entropy for some block size (including blocks that do not divide the length and blocks longer than the array), or for extreme logits.",
         "code": _HELPERS + r"""
e = {fn}()
rng = random.Random(3)
for _ in range(200):
    x = random_logits(rng, rng.randint(1, 40), rng.choice([10.0, 1e4, 1e9]))
    block = rng.choice([1, 2, 3, 7, 64])
    got = e.entropy_blockwise(x, block)
    assert math.isfinite(got) and close(got, model_entropy(x)), (block, list(x)[:5], got)
assert close(e.entropy_blockwise(np.array([1e300, -1e300, 2.0]), 1), model_entropy([1e300, -1e300, 2.0]))
assert close(e.entropy_blockwise(np.array([3.0, 1.0, 2.0])), model_entropy([3.0, 1.0, 2.0])), "block_size defaults to 2"
"""},
        {"name": "Part 3: memory does not grow with the length", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "entropy_blockwise allocated memory proportional to the number of logits; keep a few running scalars and touch one block at a time.",
         "code": _HELPERS + r"""
x = np.random.default_rng(4).normal(0, 3, 200_000)
e = {fn}()
got, extra = peak_bytes(lambda: e.entropy_blockwise(x, 1000))
assert close(got, model_entropy(x))
assert extra < 400_000, f"used {extra / 1e6:.1f} MB extra; the logits alone are 1.6 MB"
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "numerics.stability", "code": _HELPERS + r"""
chunks = iter([np.array([0.0, 1.0, -3.0]), np.array([2.5, 1.5, -0.5]), np.array([4.0])])
assert abs({fn}().entropy_online(chunks) - model_entropy([0.0, 1.0, -3.0, 2.5, 1.5, -0.5, 4.0])) < 1e-9
"""},
        {"name": "Part 4: one pass, constant memory", "part": 4, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "entropy_online was wrong for some chunking (empty chunks, a later chunk with a much larger maximum), read a chunk twice, or kept memory proportional to the number of logits.",
         "code": _HELPERS + r"""
e = {fn}()
rng = random.Random(5)
for _ in range(200):
    x = random_logits(rng, rng.randint(1, 50), rng.choice([10.0, 1e4, 1e9]))
    cuts = sorted(rng.randint(0, len(x)) for _ in range(rng.randint(0, 6)))
    pieces = [x[a:b] for a, b in zip([0] + cuts, cuts + [len(x)])]
    got = e.entropy_online(p for p in pieces)
    assert math.isfinite(got) and close(got, model_entropy(x)), (len(pieces), got)
rising = [np.array([float(v)]) for v in range(0, 3000, 3)]
assert close(e.entropy_online(iter(rising)), model_entropy([float(v) for v in range(0, 3000, 3)]))
def chunks():
    g = np.random.default_rng(6)
    for _ in range(200):
        yield g.normal(0, 3, 1000)
want = model_entropy(np.concatenate(list(chunks())))
got, extra = peak_bytes(lambda: e.entropy_online(chunks()))
assert close(got, want)
assert extra < 400_000, f"used {extra / 1e6:.1f} MB extra; storing every chunk takes 1.6 MB"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import math

import numpy as np


class EntropyMeter:
    def entropy(self, logits):
        x = np.asarray(logits, dtype=float)
        shifted = x - x.max()  # the largest term becomes exp(0) = 1, so nothing overflows
        log_z = math.log(np.exp(shifted).sum())
        log_p = shifted - log_z  # log p_i without ever dividing tiny numbers
        return float(max(0.0, -(np.exp(log_p) * log_p).sum()))

    def _update(self, state, block):
        """Folds one block into (m, s, v): s = sum exp(x - m), v = sum exp(x - m) * (x - m)."""
        block = np.asarray(block, dtype=float)
        if block.size == 0:
            return state
        m, s, v = state
        new_m = max(m, float(block.max()))
        if s:  # rescale the old sums to the new maximum
            scale, delta = math.exp(m - new_m), m - new_m
            s, v = s * scale, (v + delta * s) * scale
        shifted = block - new_m
        e = np.exp(shifted)
        return new_m, s + float(e.sum()), v + float((e * shifted).sum())

    @staticmethod
    def _finish(state):
        _, s, v = state
        return max(0.0, math.log(s) - v / s)  # H = log Z - E[x - m]

    def entropy_blockwise(self, logits, block_size=2):
        x = np.asarray(logits, dtype=float)
        state = (-math.inf, 0.0, 0.0)
        for start in range(0, len(x), block_size):
            state = self._update(state, x[start:start + block_size])  # a view, not a copy
        return self._finish(state)

    def entropy_online(self, blocks):
        state = (-math.inf, 0.0, 0.0)
        for block in blocks:
            state = self._update(state, block)
        return self._finish(state)
''',
    "interview_questions": interview(
        concept=[
            "Why is the entropy of a softmax never negative, and what values does it take for one logit and for N equal logits?",
            "Which intermediate arrays does the direct formula create, and in what order?",
        ],
        deep_dive=[
            "What does the direct computation cost in time and memory for N logits?",
        ],
        tradeoffs=[
            "Why does subtracting the maximum logit leave the softmax unchanged, and what does it fix?",
            "Why compute log p in the centred form instead of taking the log of the probability?",
            "Which running scalars give the entropy in one pass, and how do you update them when the maximum grows?",
            "How would you compute per-row entropies for a batch of sequences over a 100k-token vocabulary on a GPU?",
        ],
    ),
}
