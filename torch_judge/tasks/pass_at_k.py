"""Unbiased pass@k estimator from n samples per problem."""

from ._interview import interview

TASK = {
    "title": "Unbiased pass@k",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "pass_at_k",
    "description_en": r"""Estimate pass@k from `n` sampled solutions per problem, of which `c` are correct.

**Signature:** `pass_at_k(num_samples, num_correct, k) -> float`

**Parameters:**
- `num_samples` — list of integers, `n_i` samples drawn for problem `i`.
- `num_correct` — list of integers of the same length, `c_i` correct samples for problem `i`.
- `k` — positive integer.

**Returns:** Python float, the mean over problems of the unbiased estimator

    pass@k_i = 1 - C(n_i - c_i, k) / C(n_i, k)

where `C(a, k)` is the binomial coefficient and `C(a, k) = 0` when `a < k`.

**Constraints:**
- Stay accurate for `n` in the thousands. Do not evaluate large binomial coefficients as floats; use the product form `C(n - c, k) / C(n, k) = prod over i from n - c + 1 to n of (1 - k / i)`.
- Raise `ValueError` for empty or mismatched inputs, `k < 1`, any `n_i < k`, or any `c_i` outside `[0, n_i]`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why not just sample k times.** Drawing exactly k samples and checking for any success has high variance. Drawing n larger than k and counting successes gives a lower-variance estimate of the same quantity.

**Why not 1 - (1 - c / n) ** k.** Plugging the empirical success rate into the formula for independent draws is biased, because it samples with replacement from the n generations. The combinatorial form counts subsets of size k drawn without replacement and is unbiased. This is the estimator from the Codex paper.""",
    "advisory_prerequisites": ["rl_eval_loop"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the probability that a random size-k subset of the n samples contains no correct sample? What happens when fewer than k samples are wrong?"},
        {"level": 2, "kind": "analysis", "content": "Validate first. For each problem, if n - c < k the estimate is 1.0; otherwise compute 1 - prod(1 - k / i for i in range(n - c + 1, n + 1)). Return the mean over problems."},
    ],
    "model_connections": [
        "HumanEval, MBPP and RLVR evaluation harnesses report pass@1 and pass@k with this estimator from the Codex paper.",
    ],
    "pro_con_analysis": {
        "pros": ["Unbiased and lower variance than drawing exactly k samples, and one set of n samples gives every k up to n."],
        "cons": ["Measures only whether any sample is correct; it hides how often the model is right, which pass@1 and majority voting capture."],
    },
    "tests": [
        {"name": "Small cases by hand", "behavior": "metrics.averaging", "code": r"""
out = {fn}([5, 4], [2, 0], 2)
want = ((1 - 3 / 10) + 0.0) / 2
assert isinstance(out, float) and abs(out - want) < 1e-12, (out, want)
assert abs({fn}([10], [3], 1) - 0.3) < 1e-12
"""},
        {"name": "Matches exact rational arithmetic on seeded inputs", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "Use the combinatorial estimator without replacement, averaged over problems.", "code": r"""
import random
from fractions import Fraction
from math import comb
for seed in (7, 23, 91):
    rng = random.Random(seed)
    k = rng.randint(1, 8)
    ns = [rng.randint(k, 40) for _ in range(rng.randint(3, 8))]
    cs = [rng.randint(0, n) for n in ns]
    want = sum(1 - Fraction(comb(n - c, k), comb(n, k)) for n, c in zip(ns, cs)) / len(ns)
    out = {fn}(ns, cs, k)
    assert abs(out - float(want)) < 1e-12, (seed, out, float(want))
"""},
        {"name": "Accurate for thousands of samples", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Large binomial coefficients overflow floats; use the product form.", "code": r"""
from fractions import Fraction
from math import comb
for n, c, k in ((2000, 3, 100), (5000, 10, 500), (400, 395, 20), (3000, 0, 50)):
    want = 1 - Fraction(comb(n - c, k), comb(n, k))
    out = {fn}([n], [c], k)
    assert abs(out - float(want)) < 1e-10, (n, c, k, out, float(want))
"""},
        {"name": "Rejects invalid counts", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Empty or mismatched inputs, k < 1, n < k and c outside [0, n] must raise ValueError.", "code": r"""
cases = [([], [], 1), ([5], [1, 2], 1), ([5], [1], 0), ([3], [1], 4), ([5], [6], 1), ([5], [-1], 1)]
for args in cases:
    try:
        {fn}(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {args}")
"""},
    ],
    "solution": '''def pass_at_k(num_samples, num_correct, k):
    if not num_samples or len(num_samples) != len(num_correct):
        raise ValueError("need one correct count per problem")
    if k < 1:
        raise ValueError("k must be at least 1")
    total = 0.0
    for n, c in zip(num_samples, num_correct):
        if n < k or c < 0 or c > n:
            raise ValueError("need k <= n and 0 <= c <= n")
        if n - c < k:
            total += 1.0
            continue
        miss = 1.0
        for i in range(n - c + 1, n + 1):
            miss *= 1.0 - k / i
        total += 1.0 - miss
    return total / len(num_samples)
''',
    "interview_questions": interview(
        concept=[
            "What does pass@k measure, and why is it the usual metric for code and math generation?",
            "Why is 1 - (1 - c / n) ** k a biased estimate of pass@k?",
        ],
        deep_dive=[
            "Derive the unbiased estimator by counting size-k subsets of the n samples.",
            "Why does evaluating C(n, k) directly fail for large n, and how does the product form avoid it?",
            "What should the estimate be when fewer than k samples are wrong?",
        ],
        tradeoffs=[
            "RL often raises pass@1 while pass@k at large k stays flat or drops. What does that say about what RL is learning?",
            "pass@k versus majority voting versus a verifier-reranked best-of-n: what does each reward, and which matches deployment?",
        ],
    ),
}
