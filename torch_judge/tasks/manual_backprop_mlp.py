"""Hand-derived forward and backward pass of a two-layer MLP classifier."""

from ._interview import interview

TASK = {
    "title": "Manual Backprop Through an MLP",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "mlp_forward_backward",
    "description_en": r"""Compute the loss and all parameter gradients of a two-layer ReLU classifier by hand, without autograd.

**Signature:** `mlp_forward_backward(x, W1, b1, W2, b2, y) -> tuple[Tensor, dict]`

**Parameters:**
- `x` — float tensor `(N, D)`.
- `W1` — `(H, D)`, `b1` — `(H,)`, `W2` — `(C, H)`, `b2` — `(C,)`.
- `y` — int64 tensor `(N,)` of class ids in `[0, C)`.

**Forward pass:**

    z1 = x @ W1.T + b1
    a1 = relu(z1)
    logits = a1 @ W2.T + b2
    loss = mean over n of cross_entropy(logits[n], y[n])

**Returns:** `(loss, grads)` where `loss` is a scalar tensor and `grads` has keys `"W1"`, `"b1"`, `"W2"`, `"b2"`, each a tensor with the same shape as its parameter.

**Constraints:**
- Do not use autograd: no `backward`, `torch.autograd.grad` or `requires_grad`.
- Use the convention that the ReLU derivative at exactly 0 is 0.
- Compute the softmax stably; logits as large as 1000 must not produce NaN.
- Raise `ValueError` for shapes that do not line up or labels outside `[0, C)`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The key identity.** For mean cross-entropy, the gradient with respect to the logits is `(softmax(logits) - onehot(y)) / N`. Everything else is the chain rule through two matmuls and an elementwise ReLU.

**Shapes as a check.** Each weight gradient is an outer product summed over the batch, so `dW2 = dlogits.T @ a1` has shape `(C, H)`. Each bias gradient sums its upstream gradient over the batch dimension.""",
    "advisory_prerequisites": ["mlp", "cross_entropy"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the gradient of mean cross-entropy with respect to the logits? How do you push a gradient back through y = a @ W.T + b to W, b and a? What does ReLU do to the gradient?"},
        {"level": 2, "kind": "analysis", "content": "Forward with a max-subtracted softmax. dlogits = (probs - onehot) / N. dW2 = dlogits.T @ a1, db2 = dlogits.sum(0), da1 = dlogits @ W2. dz1 = da1 * (z1 > 0). dW1 = dz1.T @ x, db1 = dz1.sum(0)."},
    ],
    "model_connections": [
        "Karpathy's micrograd and makemore lectures and CS231n assignments derive exactly these gradients; autograd engines implement the same rules per operator.",
    ],
    "pro_con_analysis": {
        "pros": ["Deriving gradients by hand makes shape bugs, missing batch averaging and dead-ReLU behaviour obvious."],
        "cons": ["Manual gradients do not scale to large graphs and are easy to get subtly wrong; autograd plus a gradient check is the practical workflow."],
    },
    "sources": [{'kind': 'paper',
      'url': 'https://www.deeplearningbook.org/contents/mlp.html',
      'section': '6.5 Back-Propagation and Other Differentiation Algorithms',
      'note': 'Chain-rule gradients for affine layers, ReLU and softmax cross-entropy.'}],
    "tests": [
        {"name": "Matches autograd on a small batch", "behavior": "gradient.flow", "code": r"""
import torch, torch.nn.functional as F
g = torch.Generator().manual_seed(0)
x = torch.randn(4, 3, generator=g, dtype=torch.float64)
W1 = torch.randn(5, 3, generator=g, dtype=torch.float64); b1 = torch.randn(5, generator=g, dtype=torch.float64)
W2 = torch.randn(2, 5, generator=g, dtype=torch.float64); b2 = torch.randn(2, generator=g, dtype=torch.float64)
y = torch.tensor([0, 1, 1, 0])
loss, grads = {fn}(x, W1, b1, W2, b2, y)
params = {k: v.clone().requires_grad_() for k, v in dict(W1=W1, b1=b1, W2=W2, b2=b2).items()}
ref = F.cross_entropy(F.relu(x @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"], y)
ref.backward()
assert abs(loss.item() - ref.item()) < 1e-10, (loss, ref)
for k in params:
    assert grads[k].shape == params[k].shape and torch.allclose(grads[k], params[k].grad, atol=1e-10), k
"""},
        {"name": "Matches autograd on seeded shapes without using autograd", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Every gradient must match autograd, and the solution itself must not call autograd.", "code": r"""
import torch, torch.nn.functional as F
for seed in (5, 18, 60):
    g = torch.Generator().manual_seed(seed)
    N, D, H, C = (int(v) for v in torch.randint(2, 9, (4,), generator=g))
    x = torch.randn(N, D, generator=g, dtype=torch.float64)
    W1 = torch.randn(H, D, generator=g, dtype=torch.float64); b1 = torch.randn(H, generator=g, dtype=torch.float64)
    W2 = torch.randn(C, H, generator=g, dtype=torch.float64); b2 = torch.randn(C, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (N,), generator=g)
    real_backward, real_grad = torch.Tensor.backward, torch.autograd.grad
    def banned(*args, **kwargs):
        raise AssertionError("autograd is not allowed")
    torch.Tensor.backward = banned; torch.autograd.grad = banned
    try:
        loss, grads = {fn}(x, W1, b1, W2, b2, y)
    finally:
        torch.Tensor.backward, torch.autograd.grad = real_backward, real_grad
    assert not loss.requires_grad and not any(v.requires_grad for v in grads.values())
    params = {k: v.clone().requires_grad_() for k, v in dict(W1=W1, b1=b1, W2=W2, b2=b2).items()}
    ref = F.cross_entropy(F.relu(x @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"], y)
    ref.backward()
    assert abs(loss.item() - ref.item()) < 1e-10, (seed, loss, ref)
    for k in params:
        assert grads[k].shape == params[k].shape and torch.allclose(grads[k], params[k].grad, atol=1e-10), (seed, k)
"""},
        {"name": "ReLU derivative at zero and large logits", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Use derivative 0 at z1 == 0 and a max-subtracted softmax.", "code": r"""
import torch
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
W1 = torch.tensor([[1.0, -1.0], [0.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
b1 = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
W2 = torch.tensor([[1.0, 5.0, 400.0], [-1.0, 5.0, -400.0]], dtype=torch.float64)
b2 = torch.zeros(2, dtype=torch.float64)
y = torch.tensor([1, 0])
loss, grads = {fn}(x, W1, b1, W2, b2, y)
assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in grads.values()), (loss, grads)
assert abs(loss.item() - (1600.0 + 0.0) / 2) < 1e-6, loss
assert torch.all(grads["W1"][0] == 0) and grads["b1"][0] == 0, grads["W1"]
assert torch.all(grads["W1"][1] == 0) and grads["b1"][1] == 0, grads["W1"]
"""},
        {"name": "Rejects mismatched shapes and bad labels", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Parameter shapes must chain together and labels must lie in [0, C).", "code": r"""
import torch
x = torch.randn(3, 4); W1 = torch.randn(5, 4); b1 = torch.randn(5); W2 = torch.randn(2, 5); b2 = torch.randn(2)
y = torch.tensor([0, 1, 1])
cases = [
    (x, torch.randn(5, 3), b1, W2, b2, y),
    (x, W1, torch.randn(4), W2, b2, y),
    (x, W1, b1, torch.randn(2, 4), b2, y),
    (x, W1, b1, W2, torch.randn(3), y),
    (x, W1, b1, W2, b2, torch.tensor([0, 1])),
    (x, W1, b1, W2, b2, torch.tensor([0, 2, 1])),
]
for args in cases:
    try:
        {fn}(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted invalid input")
"""},
    ],
    "solution": '''import torch

def mlp_forward_backward(x, W1, b1, W2, b2, y):
    N, D = x.shape
    H, C = W1.shape[0], W2.shape[0]
    if W1.shape != (H, D) or b1.shape != (H,) or W2.shape != (C, H) or b2.shape != (C,) or y.shape != (N,):
        raise ValueError("parameter shapes do not line up")
    if bool((y < 0).any()) or bool((y >= C).any()):
        raise ValueError("labels must lie in [0, C)")

    z1 = x @ W1.T + b1
    a1 = z1.clamp(min=0)
    logits = a1 @ W2.T + b2
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    rows = torch.arange(N)
    loss = -log_probs[rows, y].mean()

    dlogits = torch.exp(log_probs)
    dlogits[rows, y] -= 1
    dlogits /= N
    dW2 = dlogits.T @ a1
    db2 = dlogits.sum(dim=0)
    da1 = dlogits @ W2
    dz1 = da1 * (z1 > 0).to(da1.dtype)
    dW1 = dz1.T @ x
    db1 = dz1.sum(dim=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
''',
    "interview_questions": interview(
        concept=[
            "What is backpropagation, and why is it much cheaper than computing each gradient separately?",
            "Derive the gradient of softmax cross-entropy with respect to the logits.",
        ],
        deep_dive=[
            "Walk through the backward pass of y = x @ W.T + b. Why is dW an outer product summed over the batch?",
            "What does the ReLU do to the gradient, what is a dead ReLU, and how does the derivative convention at 0 matter?",
            "How would you verify hand-written gradients with a finite-difference check, and why use float64 for it?",
        ],
        tradeoffs=[
            "Reverse-mode versus forward-mode automatic differentiation: when is each cheaper?",
            "Which intermediate activations must be stored for this backward pass, and how does activation checkpointing trade compute for that memory?",
        ],
    ),
}
