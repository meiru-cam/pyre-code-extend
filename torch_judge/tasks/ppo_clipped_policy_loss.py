"""PPO clipped policy loss — the token-level, masked, dual-clipped form used in LLM RL."""

TASK = {
    "title": "PPO Clipped Policy Loss (Token-Level)",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ppo_clipped_policy_loss",
    "description_en": r"""Implement the clipped surrogate objective in the form an LLM post-training stack actually uses: per token, masked, and with the dual-clip lower bound.

**Signature:** `ppo_clipped_policy_loss(logprobs, old_logprobs, advantages, mask, clip_eps=0.2, dual_clip=None) -> Tensor`

**Parameters:**
- `logprobs` — float tensor of shape `(B, T)`. Per-token log-probabilities under the policy being updated.
- `old_logprobs` — float tensor of shape `(B, T)`. The same tokens under the policy that generated the rollout. A constant.
- `advantages` — float tensor of shape `(B, T)`. One advantage per token. A constant.
- `mask` — boolean tensor of shape `(B, T)`. True exactly on generated response tokens.
- `clip_eps` — positive float. The trust-region half-width.
- `dual_clip` — `None`, or a float strictly greater than 1.0.

**Returns:** scalar tensor.

Per token:

    ratio = exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    per_token = -min(unclipped, clipped)

When `dual_clip` is not None, apply the lower bound only where the advantage is negative:

    per_token = min(per_token, -dual_clip * advantages)   where advantages < 0

Then reduce: `loss = sum(per_token * mask) / sum(mask)`.

**Constraints:**
- Take the elementwise minimum of the clipped and unclipped terms, then negate. Not the maximum, and not the clipped term alone.
- Clamp the ratio, not the product of ratio and advantage.
- Apply `dual_clip` only where the advantage is negative.
- Do not clamp the log-ratio before exponentiating.
- Divide by the number of unmasked tokens. Return a zero scalar when the mask is empty.
- Gradients reach only `logprobs`. Treat `old_logprobs` and `advantages` as constants.
- Raise a `ValueError` when the four tensors do not share a shape, when `clip_eps` is not positive, or when `dual_clip` is not None and not greater than 1.0.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**How this differs from the sequence-level exercise.** The existing `ppo_loss` exercise takes one log-probability per sequence and averages over the batch. This contract is per token and masked, which is what a language-model rollout requires: the prompt and the padding must not enter the objective, and the denominator must be the number of generated tokens.

**Why `min` and not `clamp`.** Clipping the ratio alone does not create a trust region. The minimum of the clipped and unclipped terms is what makes the objective pessimistic: when the advantage is positive the update is capped above, and when it is negative the ratio is allowed to grow without bound in the direction that reduces the probability. Taking the maximum, or using only the clipped term, produces an objective that still trains and still descends but has no trust region at all.

**What dual-clip fixes.** For a strongly negative advantage and a ratio that has drifted far above one, `-min(...)` becomes a large positive loss whose gradient can dominate the batch. The lower bound caps how much a single badly-off-policy token can contribute. Applying it everywhere would silently truncate legitimate positive updates.

**One declared difference from production code.** OpenRLHF clamps the log-ratio to the band from -20 to +20 before exponentiating, as a defensive bound against overflow. This contract does not, and the evaluator will not accept a version that does. The clamp is a numerical guard rather than part of the objective: inside the band it changes nothing, and outside it the ratio has already told you the policy has run far away from the rollout — a fact worth surfacing rather than silently bounding. A production implementation should add the clamp; an exercise about the objective should not hide it.""",
    "advisory_prerequisites": [
        "ppo_loss",
        "per_token_logprobs",
        "response_token_mask",
        "grpo_token_loss",
    ],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "The ratio is an exponential of a difference — which of the two log-probability tensors carries the gradient, and what does that mean for the other one? Write out the four combinations of advantage sign and ratio position (inside the clip band, above it, below it) and ask, for each, whether the minimum picks the clipped or the unclipped term; that table is the whole trust region. For dual-clip, ask why the bound is stated as a minimum against a negative number, and what would happen to a positive-advantage token if you applied it there too. Last, the reduction: which tokens are in the denominator?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Build it in the order of the formula and do not fuse steps. `ratio = torch.exp(logprobs - old_logprobs)`; no detach on `logprobs` and nothing needed on `old_logprobs` since the caller passes it detached. Form both products, then `torch.min(unclipped, clipped)` elementwise, then negate — the negation is what turns a maximized objective into a minimized loss, and doing it before the min inverts the trust region. For dual-clip, compute `torch.min(per_token, -dual_clip * advantages)` as a candidate and select it only where `advantages < 0`, using `torch.where`. Note the bound is on the loss, which is why it reads as a minimum even though it is called a lower bound on the objective. Reduce exactly as in `grpo_token_loss`: cast the mask, multiply, sum, divide by `mask.sum()`, and return a zero scalar built from the input when that sum is zero. The mistakes that pass every shape check are using `max` instead of `min`, clamping the product instead of the ratio, and applying dual-clip to every token.",
        },
    ],
    "model_connections": [
        "OpenRLHF's PolicyLoss.forward builds surr1 and surr2, takes their negative minimum, and exposes the dual-clip variant behind a coefficient, taking the action mask as an explicit argument.",
        "verl implements the same token-level clipped objective and reports a clip fraction as a training diagnostic, which is the fraction of tokens where the clipped branch was selected.",
        "DAPO's clip-higher modification uses an asymmetric band, raising only the upper clip bound to give low-probability tokens room to grow; this contract keeps the symmetric band.",
        "GRPO drops the ratio entirely for on-policy updates, which is why grpo_token_loss has no clipping; this exercise is the off-policy counterpart.",
    ],
    "pro_con_analysis": {
        "pros": [
            "The pessimistic minimum gives a real trust region, so a batch can be reused for several gradient steps without the policy running away.",
            "Per-token masking means prompt and padding never enter the objective, which sequence-level forms cannot express.",
            "Dual-clip bounds the damage a single badly-off-policy token with a negative advantage can do to a batch.",
        ],
        "cons": [
            "The clipped branch has zero gradient, so tokens outside the band contribute nothing and a heavily off-policy batch wastes most of its samples.",
            "The symmetric band treats low-probability and high-probability tokens identically, which is the asymmetry DAPO's clip-higher addresses.",
            "It needs old_logprobs stored for every token of every rollout, which GRPO's on-policy form avoids entirely.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "compute_pg_loss",
            "license": "MIT",
            "adapted": "The masked per-token policy-gradient reduction that this loss shares, normalized by the total response token count.",
            "simplifications": "The clipped surrogate and dual-clip branches are specified by this contract rather than taken from the file, which computes an unclipped on-policy objective.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1707.06347",
            "section": "3 Clipped Surrogate Objective, equation 7",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated inside the clip band",
            "behavior": "rl.clipping",
            "code": r"""
import torch
# ratio = exp(0) = 1, which is inside [0.8, 1.2], so clipping does nothing.
logprobs = torch.tensor([[-1.0, -1.0]])
advantages = torch.tensor([[2.0, -3.0]])
mask = torch.ones(1, 2, dtype=torch.bool)
out = {fn}(logprobs, logprobs.clone(), advantages, mask, clip_eps=0.2)
assert out.ndim == 0, f'expected a scalar, got shape {tuple(out.shape)}'
# loss = -(2.0 + -3.0) / 2 = 0.5
assert torch.allclose(out, torch.tensor(0.5), atol=1e-6), out
""",
        },
        {
            "name": "A positive advantage is capped above",
            "behavior": "rl.clipping",
            "code": r"""
import torch
import math
# ratio = exp(1.0) ~ 2.718, far above 1 + eps = 1.2. With A > 0 the clipped branch wins.
logprobs = torch.tensor([[0.0]])
old = torch.tensor([[-1.0]])
advantages = torch.tensor([[1.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(logprobs, old, advantages, mask, clip_eps=0.2)
assert torch.allclose(out, torch.tensor(-1.2), atol=1e-5), f'expected -1.2, got {out.item()}'
""",
        },
        {
            "name": "Uses the minimum, not the maximum",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The objective is not pessimistic. Take the elementwise minimum of the clipped and unclipped terms before negating.",
            "code": r"""
import torch
logprobs = torch.tensor([[0.0, 0.0]])
old = torch.tensor([[-1.0, 1.0]])
advantages = torch.tensor([[1.0, 1.0]])
mask = torch.ones(1, 2, dtype=torch.bool)
out = {fn}(logprobs, old, advantages, mask, clip_eps=0.2)
ratio = torch.exp(logprobs - old)
unclipped = ratio * advantages
clipped = ratio.clamp(0.8, 1.2) * advantages
expected_min = -(torch.min(unclipped, clipped)).mean()
expected_max = -(torch.max(unclipped, clipped)).mean()
assert not torch.allclose(out, expected_max, atol=1e-4), 'this is the maximum, not the minimum'
assert torch.allclose(out, expected_min, atol=1e-5), f'{out.item()} vs {expected_min.item()}'
""",
        },
        {
            "name": "Clipping is asymmetric in the advantage sign",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "A negative advantage was clipped in the same direction as a positive one. The minimum makes the active bound depend on the advantage sign.",
            "code": r"""
import torch
mask = torch.ones(1, 1, dtype=torch.bool)
# Ratio well above the band.
high = {fn}(torch.tensor([[0.0]]), torch.tensor([[-1.0]]), torch.tensor([[-1.0]]), mask, clip_eps=0.2)
# With A < 0 and ratio > 1 + eps, the unclipped branch is the smaller one, so it is selected.
ratio = float(torch.exp(torch.tensor(1.0)))
assert torch.allclose(high, torch.tensor(ratio), atol=1e-4), f'expected {ratio}, got {high.item()}'
# Ratio well below the band with A < 0: the clipped branch binds at 1 - eps.
low = {fn}(torch.tensor([[-2.0]]), torch.tensor([[0.0]]), torch.tensor([[-1.0]]), mask, clip_eps=0.2)
assert torch.allclose(low, torch.tensor(0.8), atol=1e-4), f'expected 0.8, got {low.item()}'
""",
        },
        {
            "name": "Clips the ratio, not the product",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The clamp was applied to ratio times advantage rather than to the ratio alone. The trust region is on the policy ratio.",
            "code": r"""
import torch
logprobs = torch.tensor([[0.0]])
old = torch.tensor([[-0.5]])
advantages = torch.tensor([[4.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(logprobs, old, advantages, mask, clip_eps=0.2)
ratio = torch.exp(logprobs - old)
correct = -torch.min(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
wrong = -torch.min(ratio * advantages, (ratio * advantages).clamp(0.8, 1.2))
assert not torch.allclose(out, wrong.mean(), atol=1e-4), 'the product was clamped instead of the ratio'
assert torch.allclose(out, correct.mean(), atol=1e-5), f'{out.item()} vs {correct.mean().item()}'
""",
        },
        {
            "name": "Dual-clip bounds only negative advantages",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The dual-clip bound was applied to positive-advantage tokens, or not applied to negative ones. It binds only where the advantage is negative.",
            "code": r"""
import torch
mask = torch.ones(1, 1, dtype=torch.bool)
# Strongly off-policy, negative advantage: without dual-clip the loss is ~ 2.0 * e^2.
lp, old, adv = torch.tensor([[0.0]]), torch.tensor([[-2.0]]), torch.tensor([[-2.0]])
plain = {fn}(lp, old, adv, mask, clip_eps=0.2)
bounded = {fn}(lp, old, adv, mask, clip_eps=0.2, dual_clip=3.0)
assert bounded < plain, f'dual-clip should reduce the loss: {bounded.item()} vs {plain.item()}'
assert torch.allclose(bounded, torch.tensor(6.0), atol=1e-4), f'expected 6.0, got {bounded.item()}'
# A positive advantage must be untouched by dual_clip.
adv_pos = torch.tensor([[2.0]])
a = {fn}(lp, old, adv_pos, mask, clip_eps=0.2)
b = {fn}(lp, old, adv_pos, mask, clip_eps=0.2, dual_clip=3.0)
assert torch.allclose(a, b, atol=1e-6), f'dual-clip changed a positive-advantage token: {a.item()} vs {b.item()}'
""",
        },
        {
            "name": "Masked positions and the denominator",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Masked-out positions changed the loss, or the denominator is not the masked token count.",
            "code": r"""
import torch
torch.manual_seed(0)
lp = torch.randn(2, 5)
old = torch.randn(2, 5)
adv = torch.randn(2, 5)
mask = torch.tensor([[True, True, False, False, False],
                     [True, False, False, False, False]])
base = {fn}(lp, old, adv, mask, clip_eps=0.2)
polluted = lp.clone(); polluted[~mask] = 30.0
adv_polluted = adv.clone(); adv_polluted[~mask] = -99.0
after = {fn}(polluted, old, adv_polluted, mask, clip_eps=0.2)
assert torch.allclose(base, after, atol=1e-5), f'{base.item()} vs {after.item()}'
# Three unmasked tokens, all on-policy with advantage 1 -> loss is exactly -1.
lp2 = torch.zeros(2, 5)
out = {fn}(lp2, lp2.clone(), torch.ones(2, 5), mask, clip_eps=0.2)
assert torch.allclose(out, torch.tensor(-1.0), atol=1e-6), out
""",
        },
        {
            "name": "Matches an elementwise oracle",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The scalar disagrees with an independent elementwise computation of the clipped objective.",
            "code": r"""
import torch
torch.manual_seed(1)
for B, T in ((3, 4), (2, 7)):
    lp = torch.randn(B, T) * 0.5
    old = torch.randn(B, T) * 0.5
    adv = torch.randn(B, T)
    mask = torch.rand(B, T) > 0.25
    mask[:, 0] = True
    eps = 0.2
    ratio = torch.exp(lp - old)
    per_token = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    keep = mask.float()
    expected = (per_token * keep).sum() / keep.sum()
    out = {fn}(lp, old, adv, mask, clip_eps=eps)
    assert torch.allclose(out, expected, atol=1e-5), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Gradients reach only the current policy",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "Gradients did not reach logprobs, or the clipped branch was not detached from its own gradient path.",
            "code": r"""
import torch
torch.manual_seed(2)
lp = torch.randn(2, 4, requires_grad=True)
old = torch.randn(2, 4)
adv = torch.randn(2, 4)
mask = torch.ones(2, 4, dtype=torch.bool)
out = {fn}(lp, old, adv, mask, clip_eps=0.2)
assert out.requires_grad, 'loss is detached from the graph'
out.backward()
assert lp.grad is not None, 'logprobs received no gradient'
assert torch.isfinite(lp.grad).all(), 'gradient contains inf or nan'
# A token deep inside the clipped branch has zero gradient: ratio far above 1+eps with A>0.
lp2 = torch.tensor([[5.0]], requires_grad=True)
{fn}(lp2, torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.ones(1, 1, dtype=torch.bool), clip_eps=0.2).backward()
assert torch.allclose(lp2.grad, torch.zeros(1, 1), atol=1e-7), f'clipped branch should have no gradient, got {lp2.grad}'
""",
        },
        {
            "name": "Empty mask returns zero, and invalid arguments raise",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An empty mask produced NaN, or an invalid clip_eps or dual_clip was accepted.",
            "code": r"""
import torch
lp, old, adv = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
empty = torch.zeros(2, 3, dtype=torch.bool)
out = {fn}(lp, old, adv, empty, clip_eps=0.2)
assert torch.isfinite(out).all() and torch.allclose(out, torch.tensor(0.0), atol=1e-7), out
mask = torch.ones(2, 3, dtype=torch.bool)
bad = ((lp, torch.randn(2, 4), adv, mask, 0.2, None),
       (lp, old, torch.randn(3, 3), mask, 0.2, None),
       (lp, old, adv, mask, 0.0, None),
       (lp, old, adv, mask, -0.1, None),
       (lp, old, adv, mask, 0.2, 1.0),
       (lp, old, adv, mask, 0.2, 0.5))
for a, b, c, m, eps, dc in bad:
    try:
        {fn}(a, b, c, m, clip_eps=eps, dual_clip=dc)
    except ValueError:
        continue
    raise AssertionError(f'clip_eps={eps} dual_clip={dc} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def ppo_clipped_policy_loss(
    logprobs, old_logprobs, advantages, mask, clip_eps=0.2, dual_clip=None
):
    shapes = (logprobs.shape, old_logprobs.shape, advantages.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"all tensors must share a shape, got {shapes}")
    if clip_eps <= 0:
        raise ValueError(f"clip_eps must be positive, got {clip_eps}")
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError(f"dual_clip must be greater than 1.0, got {dual_clip}")

    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)

    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)

    if dual_clip is not None:
        bounded = torch.min(per_token, -dual_clip * advantages)
        per_token = torch.where(advantages < 0, bounded, per_token)

    return (per_token * keep).sum() / denominator
''',
}
