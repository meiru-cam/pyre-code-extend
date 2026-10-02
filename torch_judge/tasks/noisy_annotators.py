"""Majority votes, annotator reliability, filtering and weighted votes for noisy labels, in NumPy."""

from ._interview import interview

# A data generator with planted bad annotators, and a slow model written straight from the statement.
_HELPERS = r"""
import math
import numpy as np

def make_data(seed, n=400, d=8, k=3):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 2.0, (k, d))
    y = rng.integers(0, k, n + 300)
    X = means[y] + rng.normal(0, 1.6, (n + 300, d))
    ann = np.full((n, 5), -1)
    for i in range(n):
        for j in rng.choice(5, size=int(rng.integers(2, 5)), replace=False):
            if j == 3:
                ann[i, j] = rng.integers(0, k)          # guesses
            elif j == 4:
                ann[i, j] = 0                           # always says class 0
            else:
                ann[i, j] = y[i] if rng.random() < 0.9 else rng.choice([c for c in range(k) if c != y[i]])
    return X[:n], X[n:], y[n:], ann, k

def model_scores(ann, k, w=None):
    n, a = ann.shape
    w = [1.0] * a if w is None else list(w)
    scores = [[0.0] * k for _ in range(n)]
    for i in range(n):
        for j in range(a):
            if ann[i, j] != -1:
                scores[i][ann[i, j]] += w[j]
    return scores

def model_aggregate(ann, k, w=None):
    scores = model_scores(ann, k, w)
    return [max(range(k), key=lambda c: (s[c], -c)) for s in scores], [sum(s) > 0 for s in scores]

def model_reliability(ann, k):
    a = ann.shape[1]
    out = []
    for j in range(a):
        labels, has = model_aggregate(np.delete(ann, j, axis=1), k)
        hits = [ann[i, j] == labels[i] for i in range(len(ann)) if ann[i, j] != -1 and has[i]]
        out.append(sum(hits) / len(hits) if hits else 0.0)
    return out

def model_flag(r):
    mean = sum(r) / len(r)
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
    return [x < mean - std for x in r]

def model_weights(r, k):
    out = []
    for x in r:
        x = min(max(x, 0.01), 0.99)
        out.append(max(0.0, math.log(x * (k - 1) / (1 - x))))
    return out

def same_labels(got, want, scores=None):
    labels, has = np.asarray(got[0]), np.asarray(got[1], dtype=bool)
    assert has.shape == (len(want[0]),) and list(has) == list(want[1]), "has_label differs"
    for i, (g, w) in enumerate(zip(labels, want[0])):
        if not want[1][i]:
            continue
        if scores is not None:
            top = sorted(scores[i])
            if len(top) > 1 and top[-1] - top[-2] < 1e-9:
                continue  # a float tie: either order of summation may win
        assert g == w, f"sample {i}: expected class {w}, got {g}"

EXAMPLE = np.array([[0, 0, 0, 1], [1, 1, -1, 0], [0, -1, 0, 1], [1, 1, 1, 0], [-1, -1, 1, 0], [0, 0, -1, 1], [1, 0, 1, 1]])
"""

TASK = {
    "title": "Noisy Annotator Cleanup",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "AnnotationCleaner",
    "description_en": r"""Build `AnnotationCleaner`, which turns votes from several annotators of uneven quality into training labels.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `AnnotationCleaner` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `annotations` is an integer array of shape `(n, A)`: `annotations[i, j]` is the class annotator `j` gave sample `i`, from `0` to `n_classes - 1`, or `-1` if `j` did not label `i`. NumPy is available as `np`.
- Methods may return NumPy arrays or lists of the stated length.
- `aggregate(annotations, n_classes, weights=None)` returns `(labels, has_label)`. For each sample, add each voter's weight to the class it chose; `weights=None` means weight `1.0` for everyone. `labels[i]` is the class with the largest total, ties to the smaller class. `has_label[i]` is `True` exactly when that sample's total weight is above `0`, and `labels[i]` may be anything when it is `False`.
- Never change the arrays you are given.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** a majority vote is one line; the work is measuring each annotator without letting them agree with themselves, and each later part adds one requirement.

**Where it is used:** crowdsourced labels for classifiers and human preference data for language models both mix careful annotators with careless or biased ones.

Adapted from the noisy annotators question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class. scikit-learn is not available here, so Part 2 trains a nearest-centroid classifier instead of logistic regression, and Part 3's weight formula is stated rather than derived.""",
    "parts": [
        {
            "title": "Votes and reliability",
            "description_en": r"""**Signature:** `AnnotationCleaner()`, `aggregate(annotations, n_classes, weights=None)`, `reliability(annotations, n_classes)`, `flag(r)`

- `aggregate` follows the rules above.
- `reliability` returns one score per annotator. For annotator `j`, take the samples `j` labeled where the other annotators, with `j`'s column removed and equal weights, give a label. The score is the fraction of those samples where `j` agrees with that label, or `0.0` if there are none.
- `flag(r)` returns one `bool` per annotator: `True` when its score is below the mean of all scores minus their standard deviation. Use the population standard deviation, as `np.std` does by default.

**Example**, `n_classes = 2`, four annotators, seven samples (rows), `-1` meaning not labeled:
- `[[0, 0, 0, 1], [1, 1, -1, 0], [0, -1, 0, 1], [1, 1, 1, 0], [-1, -1, 1, 0], [0, 0, -1, 1], [1, 0, 1, 1]]`
- `aggregate` gives labels `[0, 1, 0, 1, 0, 0, 1]`, every sample labeled; sample `4` is a 1-to-1 tie, so class `0` wins
- `reliability` is about `[0.83, 0.6, 0.8, 0.14]`, and `flag` marks only annotator `3`""",
        },
        {
            "title": "Filter and train",
            "description_en": r"""Keep Part 1. Drop the flagged annotators and train on what is left.

**Signature:** `filtered_labels(annotations, n_classes)`, `centroid_accuracy(X_train, labels, has_label, X_test, y_test, n_classes) -> float`

- `filtered_labels` scores and flags the annotators of the full matrix with Part 1, removes the flagged columns, and returns `aggregate` of the rest with equal weights.
- `centroid_accuracy` trains on the rows of `X_train` whose `has_label` is `True`. Each class's centroid is the mean of its training rows. A test row is predicted as the class with the nearest centroid by Euclidean distance, ties to the smaller class; a class with no training rows is never predicted.
- It returns the fraction of test rows predicted correctly.

**Example**, same matrix:
- `filtered_labels` drops annotator `3`, and sample `4` changes from `0` to `1`
- with `X_train = [[0], [1], [4], [5]]`, labels `[0, 0, 1, 1]`, all labeled, `X_test = [[2], [3], [10]]` and `y_test = [0, 1, 0]`, the centroids are `0.5` and `4.5`, and the accuracy is `2 / 3`""",
        },
        {
            "title": "Weighted votes",
            "description_en": r"""Keep Parts 1–2. Down-weight annotators instead of dropping them.

**Signature:** `reliability_weights(r, n_classes)`, `weighted_labels(annotations, n_classes)`

- `reliability_weights` clips each score to `[0.01, 0.99]`, then sets `w = log(r * (n_classes - 1) / (1 - r))` with the natural log, then raises any negative `w` to `0`.
- `weighted_labels` computes `reliability` of the full matrix, turns it into weights, and returns `aggregate` with those weights.

**Example**, same matrix:
- the weights are about `[1.61, 0.41, 1.39, 0.0]`
- `weighted_labels` gives `[0, 1, 0, 1, 1, 0, 1]`, every sample labeled: sample `4` is now `1` without removing annotator `3`'s column""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If annotator j's own vote is part of the majority you compare j with, how does that inflate j's score on a sample with only two voters? Which column do you remove before computing the reference labels, and which samples then do not count?"},
        {"level": 2, "kind": "analysis", "content": "aggregate: scores = zeros((n, n_classes)); for each column j, add weights[j] at the rows it labeled and the class it chose; return argmax(scores, axis=1), which keeps the first maximum, and scores.sum(axis=1) > 0. reliability: for each j, aggregate np.delete(annotations, j, axis=1), then average agreement over rows j labeled that have a label."},
    ],
    "model_connections": [
        "Human preference datasets for RLHF are cleaned by checking each rater against the others and down-weighting raters who disagree with the consensus.",
        "Dawid-Skene and its successors estimate each annotator's reliability to build better training labels from crowdsourced votes.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Leaving an annotator out of their own reference majority keeps their score honest even with two voters per sample.",
            "A threshold based on the pool's mean and spread needs no retuning for a different number of classes.",
            "Log-odds weights keep every sample labeled and let a good annotator outvote two weak ones.",
        ],
        "cons": [
            "One score per annotator ignores that some annotators are good on one class and bad on another.",
            "If most annotators share a bias, the consensus is biased and the honest minority gets flagged.",
            "One pass of weighting is the first step of Dawid-Skene; iterating to convergence usually does better.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": _HELPERS + r"""
c = {fn}()
labels, has = c.aggregate(EXAMPLE, 2)
assert list(np.asarray(labels)) == [0, 1, 0, 1, 0, 0, 1] and all(np.asarray(has))
r = np.asarray(c.reliability(EXAMPLE, 2), dtype=float)
assert np.allclose(r, [5 / 6, 0.6, 0.8, 1 / 7]), r
assert list(np.asarray(c.flag(r))) == [False, False, False, True]
"""},
        {"name": "Part 1: random votes", "part": 1, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "On random vote matrices, a label, has_label, reliability score or flag differed from the rules; ties go to the smaller class, and an annotator is compared with the others only.",
         "code": _HELPERS + r"""
c = {fn}()
rng = np.random.default_rng(4)
for trial in range(120):
    n, a, k = int(rng.integers(1, 30)), int(rng.integers(1, 6)), int(rng.integers(2, 5))
    ann = rng.integers(-1, k, (n, a))
    copy = ann.copy()
    w = None if trial % 2 else rng.choice([0.0, 0.5, 1.0, 2.0], a)
    same_labels(c.aggregate(ann, k, w), model_aggregate(ann, k, w), model_scores(ann, k, w))
    r = np.asarray(c.reliability(ann, k), dtype=float)
    assert r.shape == (a,) and np.allclose(r, model_reliability(ann, k)), (r, model_reliability(ann, k))
    assert list(np.asarray(c.flag(r), dtype=bool)) == model_flag(list(r))
    assert (ann == copy).all(), "the annotations array was changed"
assert list(np.asarray(c.aggregate(np.array([[-1, -1]]), 3)[1])) == [False]
assert np.allclose(c.reliability(np.array([[0, -1], [-1, 1]]), 2), [0.0, 0.0]), "no counted sample gives 0.0"
assert list(np.asarray(c.flag(np.array([0.5, 0.5, 0.5])), dtype=bool)) == [False, False, False]
"""},
        {"name": "Part 1: planted bad annotators", "part": 1, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "On data with one guessing and one always-class-0 annotator, the reliability scores or flags differed from the rules.",
         "code": _HELPERS + r"""
c = {fn}()
for seed in range(3):
    _, _, _, ann, k = make_data(seed)
    r = np.asarray(c.reliability(ann, k), dtype=float)
    assert np.allclose(r, model_reliability(ann, k)), seed
    assert list(np.asarray(c.flag(r), dtype=bool)) == model_flag(list(r)), seed
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
c = {fn}()
labels, has = c.filtered_labels(EXAMPLE, 2)
assert list(np.asarray(labels)) == [0, 1, 0, 1, 1, 0, 1] and all(np.asarray(has))
acc = c.centroid_accuracy(np.array([[0.0], [1.0], [4.0], [5.0]]), np.array([0, 0, 1, 1]), np.array([True] * 4),
                          np.array([[2.0], [3.0], [10.0]]), np.array([0, 1, 0]), 2)
assert abs(acc - 2 / 3) < 1e-12, acc
"""},
        {"name": "Part 2: filtering and the classifier", "part": 2, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "filtered_labels did not drop exactly the flagged columns before voting, or centroid_accuracy differed from nearest-centroid on the labeled rows only.",
         "code": _HELPERS + r"""
c = {fn}()
for seed in range(4):
    X, X_test, y_test, ann, k = make_data(seed)
    keep = [not f for f in model_flag(model_reliability(ann, k))]
    want = model_aggregate(ann[:, keep], k)
    got = c.filtered_labels(ann, k)
    same_labels(got, want)
    labels, has = np.asarray(want[0]), np.asarray(want[1])
    cents = {cl: X[has & (labels == cl)].mean(axis=0) for cl in range(k) if (has & (labels == cl)).any()}
    pred = [min(cents, key=lambda cl: (float(((x - cents[cl]) ** 2).sum()), cl)) for x in X_test]
    expected = float(np.mean(np.array(pred) == y_test))
    assert abs(c.centroid_accuracy(X, labels, has, X_test, y_test, k) - expected) < 1e-12, seed
X = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 1.0]])
acc = c.centroid_accuracy(X, np.array([0, 2, 1]), np.array([True, False, True]), np.array([[8.0, 8.0], [0.2, 0.0]]), np.array([1, 0]), 3)
assert acc == 1.0, "an unlabeled row must not create a centroid, so class 2 is never predicted"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "numerics.stability", "code": _HELPERS + r"""
c = {fn}()
w = np.asarray(c.reliability_weights(np.asarray(c.reliability(EXAMPLE, 2)), 2), dtype=float)
assert np.allclose(w, [math.log(5), math.log(1.5), math.log(4), 0.0]), w
labels, has = c.weighted_labels(EXAMPLE, 2)
assert list(np.asarray(labels)) == [0, 1, 0, 1, 1, 0, 1] and all(np.asarray(has))
"""},
        {"name": "Part 3: weights and weighted votes", "part": 3, "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "reliability_weights did not clip to [0.01, 0.99], apply log(r * (K - 1) / (1 - r)) and floor at 0, or weighted_labels did not vote with those weights.",
         "code": _HELPERS + r"""
c = {fn}()
r = np.array([0.0, 0.005, 0.2, 0.5, 0.75, 0.995, 1.0])
for k in (2, 3, 5):
    assert np.allclose(c.reliability_weights(r, k), model_weights(list(r), k)), k
for seed in range(4):
    _, _, _, ann, k = make_data(seed)
    w = model_weights(model_reliability(ann, k), k)
    same_labels(c.weighted_labels(ann, k), model_aggregate(ann, k, w), model_scores(ann, k, w))
rng = np.random.default_rng(9)
for _ in range(60):
    n, a, k = int(rng.integers(1, 25)), int(rng.integers(2, 6)), int(rng.integers(2, 4))
    ann = rng.integers(-1, k, (n, a))
    w = model_weights(model_reliability(ann, k), k)
    same_labels(c.weighted_labels(ann, k), model_aggregate(ann, k, w), model_scores(ann, k, w))
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import numpy as np


class AnnotationCleaner:
    def aggregate(self, annotations, n_classes, weights=None):
        annotations = np.asarray(annotations)
        n, a = annotations.shape
        weights = np.ones(a) if weights is None else np.asarray(weights, dtype=float)
        scores = np.zeros((n, n_classes))
        for j in range(a):
            voted = annotations[:, j] != -1
            scores[voted, annotations[voted, j]] += weights[j]
        return np.argmax(scores, axis=1), scores.sum(axis=1) > 0  # argmax keeps the first, smallest class

    def reliability(self, annotations, n_classes):
        annotations = np.asarray(annotations)
        r = np.zeros(annotations.shape[1])
        for j in range(annotations.shape[1]):
            # compare with the other annotators only, so an annotator cannot agree with itself
            others, has_others = self.aggregate(np.delete(annotations, j, axis=1), n_classes)
            counted = (annotations[:, j] != -1) & has_others
            if counted.any():
                r[j] = np.mean(annotations[counted, j] == others[counted])
        return r

    def flag(self, r):
        r = np.asarray(r, dtype=float)
        return r < r.mean() - r.std()

    def filtered_labels(self, annotations, n_classes):
        annotations = np.asarray(annotations)
        keep = ~self.flag(self.reliability(annotations, n_classes))
        return self.aggregate(annotations[:, keep], n_classes)

    def centroid_accuracy(self, X_train, labels, has_label, X_test, y_test, n_classes):
        X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
        labels, has_label = np.asarray(labels), np.asarray(has_label, dtype=bool)
        dist = np.full((len(X_test), n_classes), np.inf)  # a class with no training point is never predicted
        for c in range(n_classes):
            members = X_train[has_label & (labels == c)]
            if len(members):
                dist[:, c] = ((X_test - members.mean(axis=0)) ** 2).sum(axis=1)
        return float(np.mean(np.argmin(dist, axis=1) == np.asarray(y_test)))

    def reliability_weights(self, r, n_classes):
        r = np.clip(np.asarray(r, dtype=float), 0.01, 0.99)  # avoids log(0) at r == 0 or r == 1
        return np.clip(np.log(r * (n_classes - 1) / (1 - r)), 0.0, None)  # worse than chance gets weight 0

    def weighted_labels(self, annotations, n_classes):
        w = self.reliability_weights(self.reliability(annotations, n_classes), n_classes)
        return self.aggregate(annotations, n_classes, weights=w)
''',
    "interview_questions": interview(
        concept=[
            "Why must an annotator be compared with the other annotators' majority rather than one that includes their own vote?",
            "Why use the pool's mean and standard deviation for flagging instead of a fixed threshold?",
        ],
        deep_dive=[
            "What does each method cost for n samples and A annotators, and where does the leave-one-out step add work?",
        ],
        tradeoffs=[
            "What do you lose when you drop a flagged annotator's whole column?",
            "Where does the weight log(r (K - 1) / (1 - r)) come from, and why floor it at zero?",
            "What goes wrong if most annotators share the same bias?",
            "How would you iterate this into Dawid-Skene, and what would each step estimate?",
        ],
    ),
}
