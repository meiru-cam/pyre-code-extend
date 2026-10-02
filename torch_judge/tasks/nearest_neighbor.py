"""Vectorised 1-nearest-neighbour in NumPy, then the same prediction as an affine layer and a softmax."""

from ._interview import interview

# Exact distances from a slow model, used only where the nearest point is clear.
_HELPERS = r"""
import time, tracemalloc

def model_predict(Xt, y, Q):
    d = ((Q[:, None, :] - Xt[None, :, :]) ** 2).sum(axis=2)
    return y[np.argmin(d, axis=1)], d

def clear_rows(d, gap=1e-6):
    s = np.sort(d, axis=1)
    return np.ones(len(d), dtype=bool) if d.shape[1] == 1 else (s[:, 1] - s[:, 0] > gap * (1 + np.abs(s[:, 0])))

X_EX = np.array([[1.0, 1.0], [3.0, 1.0], [1.0, 4.0], [4.0, 4.0], [0.0, 2.0]])
Y_EX = np.array([1, 2, 0, 2, 1])
Q_EX = np.array([[2.0, 1.0], [2.0, 3.0], [4.0, 2.0], [0.0, 0.0]])
"""

TASK = {
    "title": "Nearest Neighbour as a Layer",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "NearestNeighbor",
    "description_en": r"""Build `NearestNeighbor`, which classifies points by their closest training point with whole-array NumPy code, then expresses the same rule as one affine layer and a softmax.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `NearestNeighbor` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `X_train` is a float array of shape `(n, d)` and `y_train` an int array of shape `(n,)`. `X_query` has shape `(m, d)`. `1 <= n, m <= 2000` and `1 <= d <= 50`. NumPy is available as `np`; no SciPy or scikit-learn.
- Distance is squared Euclidean. A query's nearest point is the training point with the smallest distance; on a tie, the smallest index wins. Its predicted label is that point's `y_train` entry.
- Never change the arrays you are given.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the loop version is obvious; the work is an all-pairs distance without an `(m, n, d)` array, and each later part adds one requirement.

**Where it is used:** retrieval over embeddings, nearest-neighbour baselines and k-NN language models all compute these distances in bulk, and the identity behind Part 2 is why dot-product search can stand in for distance search.

Adapted from the NumPy 1-NN and affine layer question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The ban on Python loops is checked by speed and memory on large inputs.""",
    "parts": [
        {
            "title": "Whole-array 1-NN",
            "description_en": r"""**Signature:** `NearestNeighbor()`, `predict(X_train, y_train, X_query) -> np.ndarray`

- Return the `m` predicted labels.
- Use whole-array operations, no Python loop over points, queries or coordinates. With `n = m = 2000` and `d = 50` it must finish in about a second.
- Never build an array of shape `(m, n, d)`: memory beyond the inputs stays `O((n + m) * d + n * m)`. Expand the squared distance into norms and a dot product instead.

**Example**, `n = 5`, `d = 2`, `m = 4`:
- `X_train = [[1, 1], [3, 1], [1, 4], [4, 4], [0, 2]]`, `y_train = [1, 2, 0, 2, 1]`
- `X_query = [[2, 1], [2, 3], [4, 2], [0, 0]]`
- the result is `[1, 0, 2, 1]`: query `[2, 1]` is at distance `1` from both point `0` and point `1`, and point `0` wins the tie""",
        },
        {
            "title": "The same rule as a layer",
            "description_en": r"""Keep Part 1. Build an affine layer whose softmax picks the nearest point.

**Signature:** `affine_layer(X_train) -> tuple[np.ndarray, np.ndarray]`, `predict_affine(X_train, y_train, X_query) -> tuple[np.ndarray, np.ndarray]`

- `affine_layer` returns `W` of shape `(d, n)` and `b` of shape `(n,)`, built from `X_train` alone. For every query `q`, the largest entry of `q @ W + b` must be at the nearest point's index, with ties at the same indices as the distances: the smallest index wins under `np.argmax`.
- `predict_affine` computes `logits = X_query @ W + b`, then the softmax of each row, and returns `(probs, labels)`: `probs` of shape `(m, n)` with rows summing to `1`, and `labels[j] = y_train[argmax(probs[j])]`.
- The softmax must stay finite for large logits: coordinates may reach `1000` in size.
- `labels` must equal `predict`'s output.

**Example**, same data: `probs` has shape `(4, 5)`, and `labels` is `[1, 0, 2, 1]` again.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Expand |q - x|^2 into three terms. Which term depends only on q, which only on x, and which on both? Which single matrix product gives the cross term for every pair at once, and what shapes do the other two terms need to broadcast?"},
        {"level": 2, "kind": "analysis", "content": "q2 = (Q ** 2).sum(1)[:, None] is (m, 1); x2 = (X ** 2).sum(1)[None, :] is (1, n); D = q2 - 2 * Q @ X.T + x2 is (m, n). np.argmin(D, axis=1) returns the first minimum, so ties go to the smaller index; return y_train[that]. Convert inputs to float first."},
    ],
    "model_connections": [
        "Vector databases and retrieval-augmented generation rank stored embeddings by distance to a query embedding in exactly this way.",
        "k-NN language models and nearest-neighbour evaluation of representations compute all-pairs distances between batches of embeddings.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Expanding the squared distance turns the work into one matrix product, which NumPy runs in optimised BLAS.",
            "Memory stays at the (m, n) distance matrix instead of an (m, n, d) difference tensor.",
            "Dropping the query's own norm turns distance ranking into a dot product plus a bias, which is a linear layer.",
        ],
        "cons": [
            "The expanded form subtracts large numbers, so near-ties can be decided by rounding instead of exact distance.",
            "An (m, n) matrix still grows with both sizes; very large indexes need batching or approximate search.",
            "Softmax over raw negative distances is very peaked, so probabilities are close to 0 or 1 and say little about confidence.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": _HELPERS + r"""
got = np.asarray({fn}().predict(X_EX, Y_EX, Q_EX))
assert list(got) == [1, 0, 2, 1], got
"""},
        {"name": "Part 1: random sets and ties", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A predicted label differed from the nearest training point by squared distance, with ties going to the smallest index, or an input array was changed.",
         "code": _HELPERS + r"""
c = {fn}()
rng = np.random.default_rng(0)
for trial in range(60):
    n, m, d = int(rng.integers(1, 40)), int(rng.integers(1, 30)), int(rng.integers(1, 6))
    if trial % 2:
        Xt, Q = rng.integers(-3, 4, (n, d)).astype(float), rng.integers(-3, 4, (m, d)).astype(float)  # many exact ties
    else:
        Xt, Q = rng.normal(0, 3, (n, d)), rng.normal(0, 3, (m, d))
    y = rng.integers(0, 5, n)
    copies = (Xt.copy(), y.copy(), Q.copy())
    got = np.asarray(c.predict(Xt, y, Q))
    want, dist = model_predict(Xt, y, Q)
    ok = clear_rows(dist) if trial % 2 == 0 else np.ones(m, dtype=bool)
    assert got.shape == (m,) and (got[ok] == want[ok]).all(), trial
    assert all((a == b).all() for a, b in zip((Xt, y, Q), copies)), "an input array was changed"
same = np.array([[2.0, 2.0], [2.0, 2.0], [5.0, 5.0]])
assert list(np.asarray(c.predict(same, np.array([7, 8, 9]), np.array([[2.0, 2.0]])))) == [7], "duplicates: the smallest index wins"
"""},
        {"name": "Part 1: large inputs, no (m, n, d) array", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "With n = m = 2000 and d = 50, prediction was too slow or allocated far more than an (m, n) matrix; expand |q - x|^2 = |q|^2 - 2 q.x + |x|^2 and use one matrix product.",
         "code": _HELPERS + r"""
rng = np.random.default_rng(1)
Xt, Q = rng.normal(size=(2000, 50)), rng.normal(size=(2000, 50))
y = rng.integers(0, 10, 2000)
c = {fn}()
tracemalloc.start()
t0 = time.perf_counter()
got = np.asarray(c.predict(Xt, y, Q))
elapsed = time.perf_counter() - t0
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
assert peak < 200_000_000, f"peak memory {peak / 1e6:.0f} MB; an (m, n) float matrix is 32 MB"
assert elapsed < 4.0, f"took {elapsed:.1f}s"
dist = (Q ** 2).sum(1)[:, None] - 2 * Q @ Xt.T + (Xt ** 2).sum(1)[None, :]
ok = clear_rows(dist)
assert (got[ok] == y[np.argmin(dist, axis=1)][ok]).all()
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "numerics.stability", "code": _HELPERS + r"""
c = {fn}()
W, b = c.affine_layer(X_EX)
assert np.asarray(W).shape == (2, 5) and np.asarray(b).shape == (5,)
probs, labels = c.predict_affine(X_EX, Y_EX, Q_EX)
probs = np.asarray(probs)
assert probs.shape == (4, 5) and np.allclose(probs.sum(axis=1), 1.0)
assert list(np.asarray(labels)) == [1, 0, 2, 1]
"""},
        {"name": "Part 2: the layer matches 1-NN", "part": 2, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "The affine layer's argmax was not the nearest point (ties to the smallest index), the softmax rows did not sum to 1 or overflowed for large coordinates, or labels differed from predict.",
         "code": _HELPERS + r"""
c = {fn}()
rng = np.random.default_rng(2)
for trial in range(60):
    n, m, d = int(rng.integers(1, 30)), int(rng.integers(1, 20)), int(rng.integers(1, 5))
    scale = [1, 1, 50, 1000][trial % 4]
    if trial % 3 == 0:
        Xt, Q = rng.integers(-3, 4, (n, d)).astype(float), rng.integers(-3, 4, (m, d)).astype(float)
    else:
        Xt, Q = rng.normal(0, 1, (n, d)) * scale, rng.normal(0, 1, (m, d)) * scale
    y = rng.integers(0, 4, n)
    W, b = (np.asarray(a, dtype=float) for a in c.affine_layer(Xt))
    assert W.shape == (d, n) and b.shape == (n,)
    want, dist = model_predict(Xt, y, Q)
    ok = clear_rows(dist) if trial % 3 else np.ones(m, dtype=bool)
    logits = Q @ W + b
    assert (np.argmax(logits, axis=1)[ok] == np.argmin(dist, axis=1)[ok]).all(), trial
    probs, labels = c.predict_affine(Xt, y, Q)
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    assert probs.shape == (m, n) and np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0), trial
    assert (labels == np.asarray(c.predict(Xt, y, Q))).all() and (labels[ok] == want[ok]).all(), trial
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import numpy as np


class NearestNeighbor:
    def _sq_dists(self, X_train, X_query):
        # |q - x|^2 = |q|^2 - 2 q.x + |x|^2: (m, 1) + (m, n) + (1, n), never an (m, n, d) array
        q2 = (X_query ** 2).sum(axis=1)[:, None]
        x2 = (X_train ** 2).sum(axis=1)[None, :]
        return q2 - 2.0 * X_query @ X_train.T + x2

    def predict(self, X_train, y_train, X_query):
        X_train, X_query = np.asarray(X_train, dtype=float), np.asarray(X_query, dtype=float)
        nearest = np.argmin(self._sq_dists(X_train, X_query), axis=1)  # first minimum: smallest index wins
        return np.asarray(y_train)[nearest]

    def affine_layer(self, X_train):
        # -|q - x_i|^2 = 2 x_i.q - |x_i|^2 - |q|^2, and -|q|^2 is the same for every i
        X_train = np.asarray(X_train, dtype=float)
        return 2.0 * X_train.T, -(X_train ** 2).sum(axis=1)

    def predict_affine(self, X_train, y_train, X_query):
        W, b = self.affine_layer(X_train)
        logits = np.asarray(X_query, dtype=float) @ W + b
        z = np.exp(logits - logits.max(axis=1, keepdims=True))  # shifting by the row max avoids overflow
        probs = z / z.sum(axis=1, keepdims=True)
        return probs, np.asarray(y_train)[np.argmax(probs, axis=1)]
''',
    "interview_questions": interview(
        concept=[
            "Why does |q - x|^2 = |q|^2 - 2 q.x + |x|^2 let you avoid an (m, n, d) array?",
            "What are the shapes of each intermediate array in the vectorised version?",
        ],
        deep_dive=[
            "Why does np.argmin give the right tie-break, and when could the expanded form break a tie the exact distance would not?",
        ],
        tradeoffs=[
            "Why can the |q|^2 term be dropped when building the layer, and what W and b remain?",
            "Why must the softmax subtract each row's maximum before exponentiating?",
            "How would you serve nearest-neighbour search over millions of points?",
            "What does softmax over negative distances say about confidence, and how would a temperature change it?",
        ],
    ),
}
