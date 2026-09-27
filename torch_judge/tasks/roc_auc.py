"""Binary ROC AUC as a rank statistic with correct tie handling."""

from ._interview import interview

TASK = {
    "title": "ROC AUC",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "roc_auc",
    "description_en": r"""Compute the area under the ROC curve for binary scores.

**Signature:** `roc_auc(scores, labels) -> float`

**Parameters:**
- `scores` — 1-D float tensor. Higher means more likely positive.
- `labels` — 1-D tensor of the same length with values 0 or 1. Boolean tensors are allowed.

**Returns:** a Python float. It equals the probability that a randomly chosen positive scores higher than a randomly chosen negative, with ties counted as one half:

    auc = ( #(s_pos > s_neg) + 0.5 * #(s_pos == s_neg) ) / (n_pos * n_neg)

where the counts run over every pair of one positive and one negative.

**Constraints:**
- Run in O(n log n) time. Do not build the n_pos by n_neg comparison matrix.
- Raise `ValueError` when the inputs are not 1-D, have different lengths, contain a label other than 0 or 1, contain a non-finite score, or lack either class.
- Do not use scikit-learn.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Two views of one number.** Sweeping a threshold from high to low traces the ROC curve of true-positive rate against false-positive rate; the trapezoidal area under it equals the pairwise probability above. The trapezoids are what give tied scores their half credit.

**The rank formula.** Sort all scores, give tied scores the average of their ranks (1-based), and sum the ranks of the positives. Then `auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)`. This is the Mann-Whitney U statistic divided by the number of pairs.""",
    "advisory_prerequisites": ["precision_recall_f1"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does AUC measure as a probability over pairs? How should two equal scores, one positive and one negative, count? How can sorting replace the pairwise comparison?"},
        {"level": 2, "kind": "analysis", "content": "Validate first. Sort scores, then walk runs of equal scores and assign each run the mean of its 1-based positions. Sum the ranks where the label is 1 and apply (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg). Accumulate in float64."},
    ],
    "model_connections": [
        "Reward-model and classifier evaluations report ROC AUC because it is threshold-free and insensitive to class balance.",
    ],
    "pro_con_analysis": {
        "pros": ["Threshold-free, invariant to any monotonic rescaling of scores, and unaffected by the positive rate."],
        "cons": ["Can look high on heavily imbalanced data where precision is poor; precision-recall AUC is more informative there."],
    },
    "tests": [
        {"name": "Perfect, reversed and partial rankings", "behavior": "metrics.ties", "code": r"""
import torch
labels = torch.tensor([0, 0, 1, 1])
assert abs({fn}(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) - 1.0) < 1e-12
assert abs({fn}(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) - 0.0) < 1e-12
out = {fn}(torch.tensor([0.1, 0.4, 0.35, 0.8]), labels)
assert isinstance(out, float) and abs(out - 0.75) < 1e-12, out
"""},
        {"name": "Ties between classes count as one half", "behavior": "metrics.ties", "code": r"""
import torch
out = {fn}(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0, 1, 0, 1]))
assert abs(out - 0.5) < 1e-12, out
out = {fn}(torch.tensor([0.2, 0.5, 0.5, 0.9]), torch.tensor([0, 0, 1, 1]))
assert abs(out - (1 + 0.5 + 1 + 1) / 4) < 1e-12, out
"""},
        {"name": "Matches a trapezoidal ROC oracle on seeded data with ties", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "The result must equal the pairwise probability with half credit for tied positive-negative pairs.", "code": r"""
import random, torch
def trapezoid(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    tp = fp = 0; prev_tpr = prev_fpr = 0.0; area = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]; fp += 1 - pairs[j][1]; j += 1
        tpr, fpr = tp / n_pos, fp / n_neg
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr; i = j
    return area
for seed in (2, 13, 71):
    rng = random.Random(seed)
    n = rng.randint(30, 80)
    labels = [rng.random() < 0.3 for _ in range(n)]
    labels[0], labels[1] = True, False
    labels = [int(x) for x in labels]
    scores = [round(rng.gauss(0.8 * y, 1.0), 1) for y in labels]
    out = {fn}(torch.tensor(scores, dtype=torch.float32), torch.tensor(labels))
    want = trapezoid(scores, labels)
    assert abs(out - want) < 1e-9, (seed, out, want)
"""},
        {"name": "Handles a large input efficiently", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "Use a sort-based O(n log n) computation; a pairwise comparison matrix does not fit at this size.", "code": r"""
import torch
n = 100_000
g = torch.Generator().manual_seed(0)
labels = (torch.rand(n, generator=g) < 0.5).long()
scores = torch.floor((labels.double() + torch.randn(n, generator=g, dtype=torch.float64)) * 4) / 4
order = torch.argsort(scores, descending=True, stable=True)
s, y = scores[order], labels[order].double()
uniq, counts = torch.unique_consecutive(s, return_counts=True)
ends = torch.cumsum(counts, 0) - 1
tp = torch.cumsum(y, 0)[ends]; fp = torch.cumsum(1 - y, 0)[ends]
tpr = torch.cat([torch.zeros(1, dtype=torch.float64), tp / tp[-1]])
fpr = torch.cat([torch.zeros(1, dtype=torch.float64), fp / fp[-1]])
want = float(((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2).sum())
out = {fn}(scores, labels)
assert abs(out - want) < 1e-9, (out, want)
"""},
        {"name": "Accepts boolean labels and is invariant to monotonic rescaling", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "AUC depends only on the ordering of scores; boolean labels are valid input.", "code": r"""
import torch
g = torch.Generator().manual_seed(4)
scores = torch.randn(50, generator=g)
labels = torch.rand(50, generator=g) < 0.4
labels[0], labels[1] = True, False
a = {fn}(scores, labels)
b = {fn}(torch.exp(3 * scores) + 7, labels.long())
assert abs(a - b) < 1e-12, (a, b)
"""},
        {"name": "Rejects inputs with one class or invalid values", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "AUC is undefined without both classes; invalid labels, shapes or scores must raise ValueError.", "code": r"""
import torch
bad = [
    (torch.tensor([0.1, 0.2]), torch.tensor([1, 1])),
    (torch.tensor([0.1, 0.2]), torch.tensor([0, 0])),
    (torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0, 1, 2, 0])),
    (torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0, 1])),
    (torch.tensor([[0.1, 0.2]]), torch.tensor([[0, 1]])),
    (torch.tensor([0.1, float("nan")]), torch.tensor([0, 1])),
]
for scores, labels in bad:
    try:
        {fn}(scores, labels)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {scores} {labels}")
"""},
    ],
    "solution": '''import torch

def roc_auc(scores, labels):
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be 1-D tensors of equal length")
    labels = labels.long()
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must be 0 or 1")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative")

    sorted_scores, order = torch.sort(scores.double())
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = torch.cumsum(counts, 0)
    average_rank = (ends - counts + 1 + ends).double() / 2
    ranks = torch.repeat_interleave(average_rank, counts)
    rank_sum = float(ranks[labels[order] == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
''',
    "interview_questions": interview(
        concept=[
            "What does the ROC curve plot, and what does the area under it mean as a probability?",
            "Why is ROC AUC called threshold-free, and why does it not change when you rescale scores monotonically?",
        ],
        deep_dive=[
            "How do you compute AUC in O(n log n) instead of comparing every positive with every negative?",
            "How should tied scores be handled, and why does the trapezoidal rule give them half credit?",
            "What should the function do when only one class is present, and why?",
        ],
        tradeoffs=[
            "Why can ROC AUC look good on heavily imbalanced data while precision is poor? When would you use PR AUC instead?",
            "AUC versus log loss versus accuracy for model selection: what does each reward?",
        ],
    ),
}
