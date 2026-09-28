"""SGD with momentum, Nesterov and coupled weight decay, matching torch.optim.SGD."""

from ._interview import interview

TASK = {
    "title": "SGD with Momentum",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "SGDMomentum",
    "description_en": r"""Implement stochastic gradient descent with momentum as a small optimizer class.

**Signature:** `SGDMomentum(params, lr, momentum=0.0, weight_decay=0.0, nesterov=False)`

**Methods:**
- `step()` — update every parameter that has a gradient, in place.
- `zero_grad()` — clear every parameter's gradient, either to `None` or to zeros.

**Update rule** for each parameter `p` with gradient `g`:

    d = g + weight_decay * p
    buf = d                         on the first step for p
    buf = momentum * buf + d        on later steps
    d = d + momentum * buf          if nesterov, otherwise d = buf
    p = p - lr * d

With `momentum == 0` keep no buffer and use `d` directly.

**Constraints:**
- Match `torch.optim.SGD` with the same arguments and `dampening=0`.
- Update parameters in place, so a model that holds them sees the new values.
- Skip parameters whose `.grad` is `None`. Do not create state for them.
- Keep one momentum buffer per parameter, created on that parameter's first update.
- Do not use `torch.optim.SGD` or any other `torch.optim` optimizer.
- Raise `ValueError` for negative `lr`, `momentum` or `weight_decay`, and for `nesterov=True` with `momentum == 0`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why momentum.** The buffer is an exponentially weighted sum of past gradients. Along a direction where gradients agree it grows toward `g / (1 - momentum)`, so progress speeds up; along a direction where gradients alternate in sign they cancel, which damps oscillation in narrow valleys.

**Why the first buffer is the gradient.** Initializing `buf = d` rather than zero avoids a slow first few steps. PyTorch does this; the Sutskever et al. formulation initializes to zero and scales the learning rate differently, so the two are not interchangeable.

**Nesterov.** Adding `momentum * buf` again looks one step ahead along the momentum direction, which corrects overshoot earlier.""",
    "advisory_prerequisites": ["adam"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where must the buffer live so it survives between step() calls? What is the buffer on the very first step? Why must the update run without autograd tracking?"},
        {"level": 2, "kind": "analysis", "content": "Store params as a list and a dict from parameter id or index to buffer. In step(), under torch.no_grad(), skip p.grad is None, form d = p.grad + wd * p, create or update the buffer, apply Nesterov if asked, then p.add_(d, alpha=-lr)."},
    ],
    "model_connections": [
        "torch.optim.SGD with momentum is still the default optimizer for training ResNets and many vision models.",
    ],
    "pro_con_analysis": {
        "pros": ["One buffer per parameter, half of Adam's optimizer memory, and often better generalization on vision tasks."],
        "cons": ["Needs careful learning-rate tuning and scheduling; one global step size handles badly scaled parameters worse than adaptive methods."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/pytorch/pytorch',
      'commit': '2166a71ef9393436f33138e6d0d50a3fde30b699',
      'path': 'torch/optim/sgd.py',
      'symbol': '_single_tensor_sgd',
      'license': 'BSD-3-Clause',
      'adapted': 'Weight decay added to the gradient, momentum buffer initialized to the first gradient, dampening-free '
                 'update and the Nesterov variant.',
      'simplifications': 'No dampening, maximize flag, foreach or fused paths; one parameter list only.'}],
    "tests": [
        {"name": "One momentum step matches the rule", "behavior": "optim.state", "code": r"""
import torch
w = torch.tensor([1.0, -2.0], requires_grad=True)
opt = {fn}([w], lr=0.1, momentum=0.9)
w.grad = torch.tensor([0.5, 1.0])
opt.step()
assert torch.allclose(w.detach(), torch.tensor([0.95, -2.1])), w
w.grad = torch.tensor([0.5, 1.0])
opt.step()
assert torch.allclose(w.detach(), torch.tensor([0.855, -2.29])), w
"""},
        {"name": "zero_grad clears gradients", "behavior": "optim.state", "code": r"""
import torch
w = torch.randn(3, requires_grad=True)
opt = {fn}([w], lr=0.1)
(w ** 2).sum().backward()
opt.zero_grad()
assert w.grad is None or torch.count_nonzero(w.grad) == 0
"""},
        {"name": "Matches torch.optim.SGD across variants", "visibility": "unshown", "behavior": "optim.state", "failure_message": "Several steps must match torch.optim.SGD for plain, momentum, Nesterov and weight-decay settings.", "code": r"""
import torch
configs = [
    dict(lr=0.1),
    dict(lr=0.05, momentum=0.9),
    dict(lr=0.05, momentum=0.9, nesterov=True),
    dict(lr=0.02, momentum=0.8, weight_decay=0.1),
    dict(lr=0.02, momentum=0.9, weight_decay=0.05, nesterov=True),
]
reference_cls = torch.optim.SGD
for config in configs:
    torch.manual_seed(7)
    a = [torch.randn(4, 3, dtype=torch.float64, requires_grad=True), torch.randn(3, dtype=torch.float64, requires_grad=True)]
    b = [p.detach().clone().requires_grad_() for p in a]
    mine = {fn}(a, **config)
    ref = reference_cls(b, **config)
    for step in range(6):
        grads = [torch.randn_like(p) for p in a]
        for p, g in zip(a, grads):
            p.grad = g.clone()
        for p, g in zip(b, grads):
            p.grad = g.clone()
        mine.step()
        ref.step()
    for p, q in zip(a, b):
        assert torch.allclose(p.detach(), q.detach(), atol=1e-12), (config, (p - q).abs().max())
"""},
        {"name": "Updates in place so a module sees new weights", "visibility": "unshown", "behavior": "optim.state", "failure_message": "Update the existing parameter tensors in place instead of rebinding them.", "code": r"""
import torch
model = torch.nn.Linear(2, 1)
before = model.weight.detach().clone()
opt = {fn}(model.parameters(), lr=0.5, momentum=0.9)
model(torch.ones(1, 2)).sum().backward()
opt.step()
assert not torch.equal(model.weight.detach(), before)
assert model.weight.grad is not None
"""},
        {"name": "Skips parameters without gradients", "visibility": "unshown", "behavior": "optim.state", "failure_message": "A parameter with no gradient must stay unchanged and must not start a momentum buffer.", "code": r"""
import torch
a = torch.ones(2, dtype=torch.float64, requires_grad=True)
b = torch.ones(2, dtype=torch.float64, requires_grad=True)
opt = {fn}([a, b], lr=0.1, momentum=0.9, weight_decay=0.5)
a.grad = torch.ones(2, dtype=torch.float64)
opt.step()
assert torch.equal(b.detach(), torch.ones(2, dtype=torch.float64)), b
b.grad = torch.ones(2, dtype=torch.float64)
opt.step()
assert torch.allclose(b.detach(), torch.full((2,), 1 - 0.1 * 1.5, dtype=torch.float64)), b
"""},
        {"name": "Does not delegate to torch.optim", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Implement the update yourself instead of wrapping a torch.optim optimizer.", "code": r"""
import torch
saved = torch.optim.SGD
class Banned:
    def __init__(self, *args, **kwargs):
        raise AssertionError("torch.optim.SGD used")
torch.optim.SGD = Banned
try:
    w = torch.zeros(2, requires_grad=True)
    opt = {fn}([w], lr=0.1, momentum=0.9)
    w.grad = torch.ones(2)
    opt.step()
    assert torch.allclose(w.detach(), torch.full((2,), -0.1)), w
finally:
    torch.optim.SGD = saved
"""},
        {"name": "Rejects invalid hyperparameters", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Negative lr, momentum or weight decay, and Nesterov without momentum, must raise ValueError.", "code": r"""
import torch
for kwargs in (dict(lr=-0.1), dict(lr=0.1, momentum=-0.5), dict(lr=0.1, weight_decay=-1.0), dict(lr=0.1, nesterov=True)):
    try:
        {fn}([torch.zeros(1, requires_grad=True)], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
"""},
    ],
    "solution": '''import torch

class SGDMomentum:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0, nesterov=False):
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum and weight_decay must be non-negative")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buffers = {}

    @torch.no_grad()
    def step(self):
        for index, p in enumerate(self.params):
            if p.grad is None:
                continue
            d = p.grad
            if self.weight_decay:
                d = d + self.weight_decay * p
            if self.momentum:
                buf = self.buffers.get(index)
                if buf is None:
                    buf = d.clone()
                    self.buffers[index] = buf
                else:
                    buf.mul_(self.momentum).add_(d)
                d = d + self.momentum * buf if self.nesterov else buf
            p.add_(d, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
''',
    "interview_questions": interview(
        concept=[
            "What problem does momentum solve for plain SGD? Describe it for a long narrow valley.",
            "What is the effective step size along a direction with a constant gradient, in terms of momentum?",
        ],
        deep_dive=[
            "Walk through the update. Why does PyTorch initialize the buffer to the first gradient rather than zero?",
            "How does Nesterov momentum differ from classical momentum, both in the math and in this implementation?",
            "Why must step() run under torch.no_grad() and update parameters in place?",
        ],
        tradeoffs=[
            "SGD with momentum versus Adam: memory, tuning effort and generalization?",
            "How do momentum and learning rate interact, and why do schedules sometimes change both?",
        ],
    ),
}
