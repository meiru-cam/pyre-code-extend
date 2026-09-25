"""Masked REINFORCE policy gradient with a detached baseline."""

TASK = {
    "title": "REINFORCE Policy Loss",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "reinforce_policy_loss",
    "description_en": r"""Turn sampled log probabilities and reward-to-go into a policy-gradient loss.

**Signature:** `reinforce_policy_loss(log_probs, returns, baseline, mask) -> Tensor`

All inputs have the same one-dimensional shape. `log_probs`, `returns`, and
`baseline` are floating tensors; `mask` is boolean. Return the scalar
negative masked mean of `log_probs * (returns - baseline)`. Detach returns
and baseline, so this actor loss sends gradients only into `log_probs`.
A masked position contributes neither value nor gradient. Require at least
one True mask entry, matching shapes/devices, and finite inputs; otherwise
raise ValueError.

────────────────────────────────

**Background — context only.** REINFORCE uses sampled actions' log
probabilities to estimate a gradient. Subtracting an action-independent
baseline reduces variance; the baseline is trained with a separate value
loss. This actor-only exercise deliberately omits PPO ratios and clipping.""",
    "advisory_prerequisites": ["reinforce_discounted_returns", "response_token_mask"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which tensor should receive the actor gradient? Why is the denominator the number of valid actions rather than the sequence length?"},
        {"level": 2, "kind": "analysis", "content": "Compute advantage = returns.detach() - baseline.detach(), multiply by log_probs and the float-cast mask, sum, divide by mask.sum(), and negate. Validate before reducing so all-masked batches fail clearly."},
    ],
    "model_connections": [
        "VERL's compute_policy_loss_reinforce uses log probabilities directly rather than PPO importance ratios.",
        "The previous exercise computes Monte Carlo returns; a separately trained critic can supply the baseline.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/verl-project/verl", "commit": "12ebe0cb4d300c58449fb6c675379e8700015c51", "path": "verl/trainer/ppo/core_algos.py", "symbol": "compute_policy_loss_reinforce", "license": "Apache-2.0", "adapted": "The masked log-probability policy gradient without PPO clipping.", "simplifications": "Uses a single flat tensor and a detached baseline, without rollout importance sampling or framework-specific aggregation."},
        {"kind": "paper", "url": "https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf", "section": "2. The REINFORCE Algorithms"}
    ],
    "tests": [
        {"name": "Computes masked mean policy gradient", "behavior": "rl.masking", "code": r"""
import torch
p = torch.tensor([-0.2, -0.4, -0.1], requires_grad=True)
r = torch.tensor([2., 99., 3.])
b = torch.tensor([0.5, 0., 0.5])
loss = {fn}(p, r, b, torch.tensor([True, False, True]))
assert torch.allclose(loss, torch.tensor(0.275)), loss
loss.backward()
assert torch.allclose(p.grad, torch.tensor([-0.75, 0., -1.25])), p.grad
""" },
        {"name": "Detaches returns and baseline", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "The actor loss must not update reward or baseline tensors.", "code": r"""
import torch
p = torch.tensor([-0.7, -0.3], requires_grad=True)
r = torch.tensor([2., 1.], requires_grad=True)
b = torch.tensor([0.5, 0.5], requires_grad=True)
loss = {fn}(p, r, b, torch.tensor([True, True]))
loss.backward()
assert p.grad is not None and p.grad.abs().sum() > 0
assert r.grad is None and b.grad is None
""" },
        {"name": "Matches a seeded valid-action oracle and gradient", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Only valid actions contribute to the actor objective and its gradient.", "code": r"""
import random, torch
for seed in (11, 23, 37):
    rng = random.Random(seed)
    logp = [rng.uniform(-2, 0) for _ in range(8)]
    returns = [rng.uniform(-3, 4) for _ in logp]
    baseline = [rng.uniform(-1, 1) for _ in logp]
    mask = [rng.random() < 0.6 for _ in logp]
    mask[0] = True
    p = torch.tensor(logp, dtype=torch.float64, requires_grad=True)
    loss = {fn}(p, torch.tensor(returns, dtype=torch.float64),
                torch.tensor(baseline, dtype=torch.float64), torch.tensor(mask))
    valid = [i for i, yes in enumerate(mask) if yes]
    want = -sum(logp[i] * (returns[i] - baseline[i]) for i in valid) / len(valid)
    assert abs(loss.item() - want) < 1e-10, (seed, loss, want)
    loss.backward()
    grad = [-(returns[i] - baseline[i]) / len(valid) if mask[i] else 0.0 for i in range(8)]
    assert torch.allclose(p.grad, torch.tensor(grad, dtype=torch.float64), atol=1e-10)
""" },
        {"name": "Rejects an all-masked batch", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "An all-masked batch has no valid actor gradient and must raise ValueError.", "code": r"""
import torch
try: {fn}(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.tensor([False, False]))
except ValueError: pass
else: raise AssertionError("empty policy mask accepted")
""" },
        {"name": "Rejects mismatched shapes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "All policy-loss inputs must be aligned one-dimensional tensors.", "code": r"""
import torch
try: {fn}(torch.zeros(2), torch.ones(3), torch.zeros(2), torch.ones(2,dtype=torch.bool))
except ValueError: pass
else: raise AssertionError("mismatched returns accepted")
""" },
    ],
    "solution": '''def reinforce_policy_loss(log_probs, returns, baseline, mask):
    import torch
    tensors = (log_probs, returns, baseline, mask)
    if (any(x.ndim != 1 for x in tensors)
            or any(x.shape != log_probs.shape or x.device != log_probs.device for x in tensors)
            or any(not x.is_floating_point() or not torch.isfinite(x).all()
                   for x in (log_probs, returns, baseline))
            or mask.dtype != torch.bool or not bool(mask.any())):
        raise ValueError("invalid policy-gradient inputs")
    advantage = returns.detach() - baseline.detach()
    return -(log_probs * advantage * mask).sum() / mask.sum()
''',
}
