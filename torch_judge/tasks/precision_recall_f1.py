"""Multiclass precision, recall and F1 with micro, macro and weighted averaging."""

from ._interview import interview

TASK = {
    "title": "Precision, Recall and F1",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "precision_recall_f1",
    "description_en": r"""Compute precision, recall and F1 for single-label multiclass predictions.

**Signature:** `precision_recall_f1(preds, targets, average="macro") -> dict`

**Parameters:**
- `preds` — 1-D integer tensor of predicted class ids.
- `targets` — 1-D integer tensor of true class ids, same length.
- `average` — one of `"micro"`, `"macro"`, `"weighted"`.

**Returns:** a dictionary with exactly the keys `precision`, `recall` and `f1`, each a Python float.

**Per class c**, counting over all samples:

    tp = #(pred == c and target == c)
    fp = #(pred == c and target != c)
    fn = #(pred != c and target == c)
    precision_c = tp / (tp + fp)      recall_c = tp / (tp + fn)
    f1_c = 2 * tp / (2 * tp + fp + fn)

Any ratio whose denominator is 0 is defined as 0.0.

**Averaging.** The classes are those that appear in `preds` or in `targets`.
- `micro` — sum `tp`, `fp` and `fn` over classes first, then apply the three formulas once.
- `macro` — unweighted mean of the per-class values. Macro F1 is the mean of `f1_c`, not the F1 of macro precision and macro recall.
- `weighted` — mean of the per-class values weighted by each class's support, the number of targets equal to `c`.

**Constraints:**
- Raise `ValueError` for empty inputs, mismatched lengths, tensors that are not 1-D, or an unknown `average`.
- Do not use scikit-learn.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Which average to report.** Micro averaging counts every sample equally, so frequent classes dominate; for single-label multiclass data micro precision, recall and F1 all equal accuracy. Macro averaging counts every class equally, so it exposes poor performance on rare classes. Weighted averaging sits in between and can hide a failing rare class.

**Why F1 is not the F1 of the averages.** Harmonic means do not commute with arithmetic means. Scikit-learn reports the mean of per-class F1, and this exercise follows it.""",
    "advisory_prerequisites": [],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which classes take part in the average? What happens to a class that is predicted but never appears in the targets? Is macro F1 computed from macro precision and recall?"},
        {"level": 2, "kind": "analysis", "content": "Take classes = unique values of torch.cat([preds, targets]). For each class count tp, fp, fn and support with boolean masks. Write a safe divide that returns 0.0 on a zero denominator. Micro sums the counts; macro and weighted average the per-class ratios."},
    ],
    "model_connections": [
        "Classification benchmarks and reward-model accuracy reports use exactly these averages; scikit-learn's precision_recall_fscore_support is the usual reference.",
    ],
    "pro_con_analysis": {
        "pros": ["Precision and recall separate false positives from false negatives, which accuracy hides on imbalanced data."],
        "cons": ["F1 ignores true negatives and depends on the decision threshold, so it can rank models differently from threshold-free metrics such as ROC AUC."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/scikit-learn/scikit-learn',
      'commit': '857849927da6e988d7d026b17aef214d43e5f26e',
      'path': 'sklearn/metrics/_classification.py',
      'symbol': 'precision_recall_fscore_support',
      'license': 'BSD-3-Clause',
      'adapted': 'Per-class counts, micro, macro and weighted averaging, macro F1 as the mean of per-class F1, and '
                 'zero_division=0.',
      'simplifications': 'Single-label multiclass only; no labels, pos_label, binary average or samples average; the class '
                         'set is the union of predictions and targets.'}],
    "tests": [
        {"name": "Macro average on a small example", "behavior": "metrics.averaging", "code": r"""
import torch
preds = torch.tensor([0, 1, 1, 2, 2, 2])
targets = torch.tensor([0, 1, 2, 2, 2, 0])
out = {fn}(preds, targets, "macro")
assert set(out) == {"precision", "recall", "f1"}
want = {"precision": (1 + 0.5 + 2/3) / 3, "recall": (0.5 + 1 + 2/3) / 3, "f1": (2/3 + 2/3 + 2/3) / 3}
for key in want:
    assert isinstance(out[key], float) and abs(out[key] - want[key]) < 1e-9, (key, out, want)
"""},
        {"name": "Micro average equals accuracy", "behavior": "metrics.averaging", "code": r"""
import torch
preds = torch.tensor([0, 1, 1, 2, 2, 2])
targets = torch.tensor([0, 1, 2, 2, 2, 0])
out = {fn}(preds, targets, "micro")
for key in ("precision", "recall", "f1"):
    assert abs(out[key] - 4 / 6) < 1e-9, (key, out)
"""},
        {"name": "Matches a seeded oracle for every average", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "Per-class counts, the class set and each averaging rule must follow the stated definitions.", "code": r"""
import random, torch
def oracle(preds, targets, average):
    classes = sorted(set(preds) | set(targets))
    rows = []
    for c in classes:
        tp = sum(p == c and t == c for p, t in zip(preds, targets))
        fp = sum(p == c and t != c for p, t in zip(preds, targets))
        fn = sum(p != c and t == c for p, t in zip(preds, targets))
        rows.append((tp, fp, fn, sum(t == c for t in targets)))
    div = lambda a, b: a / b if b else 0.0
    if average == "micro":
        tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
        return div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    per = [(div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)) for tp, fp, fn, _ in rows]
    weights = [1.0] * len(rows) if average == "macro" else [r[3] for r in rows]
    total = sum(weights)
    return tuple(div(sum(w * v[i] for w, v in zip(weights, per)), total) for i in range(3))
for seed in (1, 8, 42):
    rng = random.Random(seed)
    n = rng.randint(20, 60)
    targets = [rng.choice([0, 1, 1, 2, 5]) for _ in range(n)]
    preds = [t if rng.random() < 0.6 else rng.choice([0, 1, 2, 3, 5]) for t in targets]
    for average in ("micro", "macro", "weighted"):
        out = {fn}(torch.tensor(preds), torch.tensor(targets), average)
        want = oracle(preds, targets, average)
        got = (out["precision"], out["recall"], out["f1"])
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), (seed, average, got, want)
"""},
        {"name": "Macro F1 is the mean of per-class F1", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "Average the per-class F1 values; do not combine macro precision and macro recall into one F1.", "code": r"""
import torch
preds = torch.tensor([0, 0, 0, 0, 1])
targets = torch.tensor([0, 1, 1, 1, 1])
out = {fn}(preds, targets, "macro")
f1_0 = 2 * 1 / (2 * 1 + 3 + 0)
f1_1 = 2 * 1 / (2 * 1 + 0 + 3)
assert abs(out["f1"] - (f1_0 + f1_1) / 2) < 1e-9, out
"""},
        {"name": "Zero denominators give zero and predicted-only classes count", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "A class that is predicted but absent from targets still takes part in the average, with zero recall defined as 0.0.", "code": r"""
import torch
preds = torch.tensor([0, 2, 2])
targets = torch.tensor([0, 1, 1])
out = {fn}(preds, targets, "macro")
assert abs(out["precision"] - 1 / 3) < 1e-9, out
assert abs(out["recall"] - 1 / 3) < 1e-9, out
assert abs(out["f1"] - 1 / 3) < 1e-9, out
weighted = {fn}(preds, targets, "weighted")
assert abs(weighted["recall"] - 1 / 3) < 1e-9, weighted
"""},
        {"name": "Rejects invalid inputs", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Empty, misaligned or non-1-D inputs and unknown averages must raise ValueError.", "code": r"""
import torch
bad = [
    (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), "macro"),
    (torch.tensor([0, 1]), torch.tensor([0]), "macro"),
    (torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), "macro"),
    (torch.tensor([0, 1]), torch.tensor([0, 1]), "samples"),
]
for args in bad:
    try:
        {fn}(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {args}")
"""},
    ],
    "solution": '''import torch

def precision_recall_f1(preds, targets, average="macro"):
    if average not in ("micro", "macro", "weighted"):
        raise ValueError(f"unknown average {average!r}")
    if preds.ndim != 1 or targets.ndim != 1 or preds.shape != targets.shape or preds.numel() == 0:
        raise ValueError("preds and targets must be non-empty 1-D tensors of equal length")

    def divide(a, b):
        return a / b if b else 0.0

    classes = torch.unique(torch.cat([preds, targets])).tolist()
    counts = []
    for c in classes:
        pred_c, true_c = preds == c, targets == c
        tp = int((pred_c & true_c).sum())
        fp = int((pred_c & ~true_c).sum())
        fn = int((~pred_c & true_c).sum())
        counts.append((tp, fp, fn, int(true_c.sum())))

    if average == "micro":
        tp, fp, fn = (sum(row[i] for row in counts) for i in range(3))
        precision, recall, f1 = divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn)
    else:
        per_class = [
            (divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn))
            for tp, fp, fn, _ in counts
        ]
        weights = [1.0] * len(counts) if average == "macro" else [row[3] for row in counts]
        total = sum(weights)
        precision, recall, f1 = (
            divide(sum(w * values[i] for w, values in zip(weights, per_class)), total)
            for i in range(3)
        )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
''',
    "interview_questions": interview(
        concept=[
            "Define precision and recall in words. Give an application where each one matters more.",
            "Why is F1 a harmonic mean rather than an arithmetic mean of precision and recall?",
        ],
        deep_dive=[
            "Explain micro, macro and weighted averaging. Why do micro precision, recall and F1 all equal accuracy for single-label multiclass data?",
            "Is macro F1 the F1 of macro precision and macro recall? Construct a case where the two differ.",
            "What should happen when a class is never predicted or never appears in the targets?",
        ],
        tradeoffs=[
            "On a heavily imbalanced dataset, which average would you report and why?",
            "F1 depends on the decision threshold. How would you choose the threshold, and when would you prefer a threshold-free metric?",
        ],
    ),
}
