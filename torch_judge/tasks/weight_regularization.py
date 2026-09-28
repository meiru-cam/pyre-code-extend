"""L1 and L2 weight penalties that skip biases and normalization scales."""

from ._interview import interview

TASK = {
    "title": "L1 and L2 Weight Regularization",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "regularized_loss",
    "description_en": r"""Add L1 and L2 penalties on a model's weight matrices to a data loss.

**Signature:** `regularized_loss(data_loss, named_parameters, l1=0.0, l2=0.0) -> Tensor`

**Parameters:**
- `data_loss` — scalar tensor. The unregularized loss.
- `named_parameters` — an iterable of `(name, tensor)` pairs, such as `model.named_parameters()`. It may be a one-shot generator.
- `l1` — non-negative float. Weight of the L1 penalty.
- `l2` — non-negative float. Weight of the L2 penalty.

**Returns:** scalar tensor:

    loss = data_loss + sum over penalized w of ( l1 * sum(|w|) + 0.5 * l2 * sum(w * w) )

**Constraints:**
- Penalize only tensors with 2 or more dimensions. Biases and normalization scales, which are 1-D, are not penalized.
- Sum over elements. Do not average.
- Keep the penalty differentiable, so each penalized weight receives gradient `l1 * sign(w) + l2 * w`.
- With `l1 == l2 == 0` the result must equal `data_loss`.
- Raise `ValueError` when `l1` or `l2` is negative.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the 0.5.** The gradient of `0.5 * l2 * w * w` is `l2 * w`, which is exactly what an optimizer's `weight_decay=l2` adds for plain SGD. With Adam the two are no longer equivalent, which is why AdamW decouples weight decay from the gradient.

**Why skip 1-D tensors.** Biases shift outputs and normalization scales set the magnitude that normalization just removed; shrinking either toward zero does not reduce model complexity and often hurts.""",
    "advisory_prerequisites": ["linear_regression"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which parameters should the penalty skip, and how can you tell them apart by shape? What gradient should the L2 term produce?"},
        {"level": 2, "kind": "analysis", "content": "Validate l1 and l2, start from data_loss, loop once over named_parameters, skip tensors with ndim < 2, and add l1 * w.abs().sum() + 0.5 * l2 * (w * w).sum(). Do not detach the weights."},
    ],
    "model_connections": [
        "Optimizer weight decay in PyTorch and the `no_decay` parameter groups in Hugging Face trainers apply the same rule of skipping biases and norm weights.",
    ],
    "pro_con_analysis": {
        "pros": ["L1 drives weights to exactly zero for sparsity; L2 shrinks all weights smoothly and keeps the problem strongly convex for linear models."],
        "cons": ["Penalty strength interacts with the learning rate and optimizer; with Adam an L2 loss term is not the same as decoupled weight decay."],
    },
    "sources": [{'kind': 'paper',
      'url': 'https://www.deeplearningbook.org/contents/regularization.html',
      'section': '7.1.1 L2 Parameter Regularization and 7.1.2 L1 Regularization',
      'note': "The 0.5 * l2 * ||w||^2 and l1 * ||w||_1 penalties; penalizing weights but not biases follows the chapter's "
              'discussion.'}],
    "tests": [
        {"name": "Adds both penalties to the data loss", "behavior": "state.invariant", "code": r"""
import torch
w = torch.tensor([[1.0, -2.0], [0.5, 0.0]], requires_grad=True)
b = torch.tensor([3.0, -4.0], requires_grad=True)
out = {fn}(torch.tensor(1.0), [("fc.weight", w), ("fc.bias", b)], l1=0.1, l2=0.2)
want = 1.0 + 0.1 * 3.5 + 0.5 * 0.2 * 5.25
assert abs(out.item() - want) < 1e-6, (out, want)
"""},
        {"name": "Skips every 1-D parameter", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Biases and normalization scales are 1-D and must not be penalized.", "code": r"""
import torch
model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
with torch.no_grad():
    for p in model.parameters():
        p.fill_(2.0)
out = {fn}(torch.tensor(0.0), model.named_parameters(), l1=1.0, l2=1.0)
weights = [p for p in model.parameters() if p.ndim >= 2]
count = sum(p.numel() for p in weights)
want = 1.0 * 2.0 * count + 0.5 * 1.0 * 4.0 * count
assert abs(out.item() - want) < 1e-5, (out, want)
"""},
        {"name": "Gradient is l1 * sign(w) + l2 * w on weights only", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "The penalty must stay differentiable and give each weight gradient l1 * sign(w) + l2 * w.", "code": r"""
import torch
g = torch.Generator().manual_seed(5)
w = torch.randn(3, 4, generator=g, dtype=torch.float64).requires_grad_()
b = torch.randn(4, generator=g, dtype=torch.float64).requires_grad_()
out = {fn}(torch.zeros((), dtype=torch.float64), iter([("w", w), ("b", b)]), l1=0.3, l2=0.7)
out.backward()
assert torch.allclose(w.grad, 0.3 * torch.sign(w.detach()) + 0.7 * w.detach(), atol=1e-12), w.grad
assert b.grad is None or torch.count_nonzero(b.grad) == 0, b.grad
"""},
        {"name": "Sums rather than averages over elements", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "The penalty is a sum over elements; averaging makes its strength depend on layer size.", "code": r"""
import torch
small = torch.ones(2, 2, requires_grad=True)
large = torch.ones(10, 10, requires_grad=True)
out = {fn}(torch.tensor(0.0), [("a", small), ("b", large)], l1=1.0)
assert abs(out.item() - 104.0) < 1e-5, out
"""},
        {"name": "Zero coefficients return the data loss", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "With l1 and l2 both zero the result must equal the data loss.", "code": r"""
import torch
w = torch.randn(3, 3, requires_grad=True)
loss = torch.tensor(2.5)
out = {fn}(loss, [("w", w)])
assert out.item() == 2.5, out
"""},
        {"name": "Rejects negative coefficients", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Negative regularization strengths are invalid; raise ValueError.", "code": r"""
import torch
for kwargs in ({"l1": -0.1}, {"l2": -1.0}):
    try:
        {fn}(torch.tensor(0.0), [("w", torch.ones(2, 2))], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
"""},
    ],
    "solution": '''def regularized_loss(data_loss, named_parameters, l1=0.0, l2=0.0):
    if l1 < 0 or l2 < 0:
        raise ValueError("regularization strengths must be non-negative")
    loss = data_loss
    for _name, weight in named_parameters:
        if weight.ndim < 2:
            continue
        loss = loss + l1 * weight.abs().sum() + 0.5 * l2 * (weight * weight).sum()
    return loss
''',
    "interview_questions": interview(
        concept=[
            "What do L1 and L2 regularization do to the weights, and why does L1 produce exact zeros while L2 does not?",
            "How do L1 and L2 relate to Laplace and Gaussian priors in a MAP view?",
        ],
        deep_dive=[
            "Why is the L2 term usually written with a 0.5 factor, and how does it relate to weight_decay in SGD?",
            "L1 is not differentiable at zero. What gradient does autograd use there, and how do proximal methods handle it instead?",
            "Why exclude biases and normalization scales from the penalty?",
        ],
        tradeoffs=[
            "Why is L2 in the loss not equivalent to weight decay under Adam, and what does AdamW change?",
            "L1 versus L2 versus elastic net: when would you pick each?",
        ],
    ),
}
