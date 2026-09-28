"""Numerically stable binary cross-entropy from logits, with an optional positive-class weight."""

from ._interview import interview

TASK = {
    "title": "Binary Cross-Entropy with Logits",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "binary_cross_entropy",
    "description_en": r"""Compute the mean binary cross-entropy loss directly from logits.

**Signature:** `binary_cross_entropy(logits, targets, pos_weight=None) -> Tensor`

**Parameters:**
- `logits` — float tensor of any shape. Raw scores before the sigmoid.
- `targets` — float tensor of the same shape. Values in `[0, 1]`; soft labels are allowed.
- `pos_weight` — `None`, a Python float, or a float tensor that broadcasts against `logits`. It multiplies the positive-class term only.

**Returns:** scalar tensor. With `p = sigmoid(logits)` and `w = pos_weight` (1 when `None`):

    loss = mean( -(w * targets * log(p) + (1 - targets) * log(1 - p)) )

**Constraints:**
- Work from logits. The result must stay finite for logits as large as 100 in magnitude.
- Do not call `F.binary_cross_entropy_with_logits`, `F.binary_cross_entropy` or `nn.BCEWithLogitsLoss`.
- Keep the result differentiable with respect to `logits`.
- Raise `ValueError` when `logits` and `targets` have different shapes.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why logits.** `sigmoid(100)` rounds to exactly 1.0 in float32, so `log(1 - p)` becomes `log(0)`. Rewriting the loss with `softplus(x) = log(1 + exp(x))` avoids ever forming `p`: `-log(p) = softplus(-x)` and `-log(1 - p) = softplus(x)`.

**Why pos_weight.** With rare positives, weighting the positive term trades precision for recall. A value near `num_negatives / num_positives` roughly balances the two classes' contribution to the gradient.""",
    "advisory_prerequisites": ["cross_entropy"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is -log(sigmoid(x)) written without a sigmoid? Which of the two log terms does pos_weight scale?"},
        {"level": 2, "kind": "analysis", "content": "Use -log(p) = softplus(-x) and -log(1 - p) = softplus(x). Per element: w * y * softplus(-x) + (1 - y) * softplus(x). F.softplus is stable for large inputs. Validate shapes first, then take the mean."},
    ],
    "model_connections": [
        "Reward models, classifier heads and multi-label taggers all train with this loss; `pos_weight` is PyTorch's own knob for class imbalance.",
    ],
    "pro_con_analysis": {
        "pros": ["Stable at any logit scale and supports soft labels and per-class positive weights."],
        "cons": ["Treats labels independently, so it cannot express that classes are mutually exclusive; use softmax cross-entropy for that."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/pytorch/pytorch',
      'commit': '2166a71ef9393436f33138e6d0d50a3fde30b699',
      'path': 'aten/src/ATen/native/Loss.cpp',
      'symbol': 'binary_cross_entropy_with_logits',
      'license': 'BSD-3-Clause',
      'adapted': 'Softplus-stable form with pos_weight scaling only the positive term, mean reduction.',
      'simplifications': 'No per-element weight or reduction argument; written in Python instead of ATen C++.'}],
    "tests": [
        {"name": "Matches the definition on small logits", "behavior": "numerics.stability", "code": r"""
import torch, math
x = torch.tensor([0.0, 2.0, -1.0])
y = torch.tensor([1.0, 0.0, 1.0])
p = [1 / (1 + math.exp(-v)) for v in x.tolist()]
want = -sum(t * math.log(q) + (1 - t) * math.log(1 - q) for q, t in zip(p, y.tolist())) / 3
out = {fn}(x, y)
assert out.ndim == 0 and abs(out.item() - want) < 1e-6, (out, want)
"""},
        {"name": "Applies pos_weight to the positive term", "behavior": "numerics.stability", "code": r"""
import torch, math
x = torch.tensor([0.5, -0.5])
y = torch.tensor([1.0, 0.0])
p = [1 / (1 + math.exp(-v)) for v in x.tolist()]
want = (-3.0 * math.log(p[0]) - math.log(1 - p[1])) / 2
out = {fn}(x, y, pos_weight=3.0)
assert abs(out.item() - want) < 1e-6, (out, want)
"""},
        {"name": "Stays finite and exact for extreme logits", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Large-magnitude logits must not overflow; avoid forming sigmoid(x) and then taking its log.", "code": r"""
import torch
x = torch.tensor([100.0, -100.0, 100.0, -100.0])
y = torch.tensor([0.0, 1.0, 1.0, 0.0])
out = {fn}(x, y)
assert torch.isfinite(out), out
assert abs(out.item() - 50.0) < 1e-4, out
"""},
        {"name": "Matches a seeded oracle with soft labels and tensor pos_weight", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Soft labels and a broadcast pos_weight must follow the stated formula exactly.", "code": r"""
import torch
for seed in (3, 17, 29):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 5, generator=g, dtype=torch.float64) * 4
    y = torch.rand(4, 5, generator=g, dtype=torch.float64)
    w = torch.rand(5, generator=g, dtype=torch.float64) * 3 + 0.2
    sp = torch.nn.functional.softplus
    want = (w * y * sp(-x) + (1 - y) * sp(x)).mean()
    out = {fn}(x, y, pos_weight=w)
    assert torch.allclose(out, want, atol=1e-10), (seed, out, want)
"""},
        {"name": "Gradient equals (sigmoid(x) - y) / N without pos_weight", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "The loss must stay differentiable with respect to logits and average over every element.", "code": r"""
import torch
x = torch.tensor([[0.3, -1.2], [2.0, 0.0]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[1.0, 0.0], [0.25, 1.0]], dtype=torch.float64)
{fn}(x, y).backward()
assert torch.allclose(x.grad, (torch.sigmoid(x.detach()) - y) / 4, atol=1e-10), x.grad
"""},
        {"name": "Implements the loss without the built-in", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Build the loss from softplus or log-sigmoid rather than calling PyTorch's BCE functions.", "code": r"""
import torch
F = torch.nn.functional
saved = (F.binary_cross_entropy_with_logits, F.binary_cross_entropy, torch.nn.BCEWithLogitsLoss)
def banned(*args, **kwargs):
    raise AssertionError("built-in BCE called")
F.binary_cross_entropy_with_logits = F.binary_cross_entropy = banned
torch.nn.BCEWithLogitsLoss = banned
try:
    {fn}(torch.tensor([0.1, -0.2]), torch.tensor([1.0, 0.0]))
finally:
    F.binary_cross_entropy_with_logits, F.binary_cross_entropy, torch.nn.BCEWithLogitsLoss = saved
"""},
        {"name": "Rejects mismatched shapes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Logits and targets must have the same shape; raise ValueError instead of broadcasting.", "code": r"""
import torch
try:
    {fn}(torch.zeros(3, 1), torch.zeros(3))
except ValueError:
    pass
else:
    raise AssertionError("broadcast targets accepted")
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def binary_cross_entropy(logits, targets, pos_weight=None):
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    weight = 1.0 if pos_weight is None else pos_weight
    per_element = weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)
    return per_element.mean()
''',
    "interview_questions": interview(
        concept=[
            "What is binary cross-entropy, and when do you use it instead of softmax cross-entropy?",
            "Why does a multi-label classifier use one sigmoid per class rather than a softmax over classes?",
        ],
        deep_dive=[
            "Why does computing sigmoid first and then log fail for large logits? Rewrite the loss in a stable form.",
            "What is the gradient of BCE with respect to the logit, and why is it so simple?",
            "How does pos_weight change the loss and the gradient, and how would you pick its value?",
        ],
        tradeoffs=[
            "pos_weight versus resampling versus focal loss for class imbalance: what does each change?",
            "What happens to calibration when you train with pos_weight far from 1?",
        ],
    ),
}
