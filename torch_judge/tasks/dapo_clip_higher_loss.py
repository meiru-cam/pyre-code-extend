"""DAPO asymmetric clipping over policy importance ratios."""

from ._interview import interview

TASK = {
    "title": "DAPO Clip-Higher Token Loss",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "dapo_clip_higher_loss",
    "description_en": r"""Implement DAPO's asymmetric PPO-style token objective.

**Signature:** `dapo_clip_higher_loss(log_probs, old_log_probs, advantages, mask, clip_low=0.2, clip_high=0.28) -> Tensor`

The first four tensors have identical one-dimensional shapes. `mask` is
boolean; the others are floating. For valid tokens compute
`ratio = exp(log_probs - old_log_probs.detach())`, then the negative of
`min(ratio * advantage, clamp(ratio, 1-clip_low, 1+clip_high) * advantage)`.
Return the mean over valid tokens only. Detach advantages and old log
probabilities. Require `0 <= clip_low < 1`, `clip_high >= clip_low`,
at least one valid token, and aligned tensor shapes/devices.

────────────────────────────────

**Background — context only.** DAPO's Clip-Higher gives positive-advantage
tokens a wider upper ratio bound, so useful sampled actions retain gradient
over a wider range. It still limits low-ratio negative-advantage tokens.
The dynamic sampling exercise handles DAPO's separate group filter; this
exercise isolates the clipping branch and token-level reduction.""",
    "advisory_prerequisites": ["ppo_clipped_policy_loss", "dapo_dynamic_sampling"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How does a wider upper clamp change the positive-advantage branch? Why does the minimum objective become a maximum after negation?"},
        {"level": 2, "kind": "analysis", "content": "Select valid positions before computing ratios, so masked overflow cannot poison the loss. Form the raw and clamped surrogates, take their minimum, then negate and average. Detach old log probabilities and advantages; retain gradient through current log_probs."},
    ],
    "model_connections": [
        "VERL's compute_policy_loss_vanilla reads separate clip_ratio_low and clip_ratio_high configuration values.",
        "DAPO uses Clip-Higher alongside dynamic sampling, token-level loss aggregation and overlong reward shaping.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/verl-project/verl", "commit": "12ebe0cb4d300c58449fb6c675379e8700015c51", "path": "verl/trainer/ppo/core_algos.py", "symbol": "compute_policy_loss_vanilla", "license": "Apache-2.0", "adapted": "Separate lower and upper clipping bounds around the policy ratio.", "simplifications": "Omits dual-clip, optional importance weights and global distributed reduction."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2503.14476", "section": "3.1 Clip-Higher"}
    ],
    "tests": [
        {"name": "Positive token keeps gradient up to the higher cap", "behavior": "rl.clipping", "code": r"""
import math, torch
p = torch.tensor([math.log(1.4)], requires_grad=True)
loss = {fn}(p, torch.zeros(1), torch.ones(1), torch.tensor([True]), 0.2, 0.5)
assert torch.allclose(loss, torch.tensor(-1.4)), loss
loss.backward()
assert torch.allclose(p.grad, torch.tensor([-1.4])), p.grad
""" },
        {"name": "Upper cap stops positive gradient beyond clip_high", "visibility": "unshown", "behavior": "rl.clipping", "failure_message": "The positive-advantage upper clip must use clip_high, not clip_low.", "code": r"""
import math, torch
p = torch.tensor([math.log(1.7)], requires_grad=True)
loss = {fn}(p, torch.zeros(1), torch.ones(1), torch.tensor([True]), 0.2, 0.5)
assert torch.allclose(loss, torch.tensor(-1.5))
loss.backward()
assert torch.allclose(p.grad, torch.zeros(1))
""" },
        {"name": "Lower cap clips a negative-advantage token", "visibility": "unshown", "behavior": "rl.clipping", "failure_message": "Negative advantages use the lower clipping bound and the pessimistic surrogate.", "code": r"""
import math, torch
p = torch.tensor([math.log(0.6)], requires_grad=True)
loss = {fn}(p, torch.zeros(1), -torch.ones(1), torch.tensor([True]), 0.2, 0.5)
assert torch.allclose(loss, torch.tensor(0.8))
loss.backward()
assert torch.allclose(p.grad, torch.zeros(1))
""" },
        {"name": "Masked tokens and old policy have no gradient", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Only unmasked current-policy log probabilities should receive gradients.", "code": r"""
import torch
p = torch.tensor([0., 0.], requires_grad=True)
old = torch.zeros(2, requires_grad=True)
a = torch.tensor([1., 9.], requires_grad=True)
loss = {fn}(p, old, a, torch.tensor([True, False]))
assert torch.allclose(loss, torch.tensor(-1.))
loss.backward()
assert torch.allclose(p.grad, torch.tensor([-1., 0.]))
assert old.grad is None and a.grad is None
""" },
        {"name": "Masked overflow cannot poison the loss", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Exclude masked positions before exponentiation; inf times zero becomes NaN.", "code": r"""
import torch
p = torch.tensor([0., 1000.], requires_grad=True)
loss = {fn}(p, torch.zeros(2), torch.tensor([1., -1.]), torch.tensor([True, False]))
assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(-1.)), loss
loss.backward()
assert torch.allclose(p.grad, torch.tensor([-1., 0.])), p.grad
""" },
        {"name": "Matches an independent seeded asymmetric-clip oracle", "visibility": "unshown", "behavior": "rl.clipping", "failure_message": "The clipped surrogate must average valid tokens with separate lower and upper bounds.", "code": r"""
import math, random, torch
for seed in (11, 23, 37):
    rng = random.Random(seed)
    old = [rng.uniform(-2, -0.2) for _ in range(9)]
    ratios = [rng.uniform(0.5, 1.8) for _ in old]
    current = [o + math.log(r) for o, r in zip(old, ratios)]
    adv = [rng.uniform(-3, 3) for _ in old]
    mask = [rng.random() < 0.7 for _ in old]
    mask[0] = True
    valid = [i for i, yes in enumerate(mask) if yes]
    terms = []
    for i in valid:
        ratio = ratios[i]
        clipped = max(0.8, min(1.35, ratio))
        terms.append(min(ratio * adv[i], clipped * adv[i]))
    want = -sum(terms) / len(terms)
    got = {fn}(torch.tensor(current, dtype=torch.float64), torch.tensor(old, dtype=torch.float64),
               torch.tensor(adv, dtype=torch.float64), torch.tensor(mask), 0.2, 0.35)
    assert abs(got.item() - want) < 1e-10, (seed, got, want)
""" },
        {"name": "Rejects invalid bounds or an empty mask", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Clip bounds and token mask must define a valid nonempty reduction.", "code": r"""
import torch
x=torch.zeros(1); m=torch.tensor([True])
for low,high,mask in [(0.5,0.2,m),(1.,1.,m),(0.2,0.3,torch.tensor([False]))]:
    try: {fn}(x,x,x,mask,low,high)
    except ValueError: pass
    else: raise AssertionError("invalid input accepted")
""" },
    ],
    "solution": '''def dapo_clip_higher_loss(log_probs, old_log_probs, advantages, mask, clip_low=0.2, clip_high=0.28):
    import torch
    values = (log_probs, old_log_probs, advantages, mask)
    if (any(x.ndim != 1 or x.shape != log_probs.shape or x.device != log_probs.device for x in values)
            or any(not x.is_floating_point() for x in values[:3])
            or mask.dtype != torch.bool or not bool(mask.any())
            or not (0 <= clip_low < 1 and clip_high >= clip_low)):
        raise ValueError("invalid DAPO inputs")
    valid = mask
    ratio = torch.exp(log_probs[valid] - old_log_probs[valid].detach())
    advantage = advantages[valid].detach()
    raw = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * advantage
    return -torch.minimum(raw, clipped).mean()
''',
    "interview_questions": interview(
        concept=[
            'What is Clip-Higher in DAPO, and why use an asymmetric clip range?',
            'How does clip-higher relate to entropy collapse?',
        ],
        deep_dive=[
            'Walk through which tokens hit the upper bound and which hit the lower bound for positive and negative advantages.',
            'Why detach both old log-probs and advantages?',
            'How do you reduce the per-token objective to a scalar, and why only over valid tokens?',
        ],
        tradeoffs=[
            'What risk does a larger upper clip bound bring, and how would you choose clip_high?',
            'Compare clip-higher with adding an entropy bonus to prevent collapse.',
        ],
    ),
}
