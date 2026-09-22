"""PPO value loss — the clipped critic objective, and the one place max is correct."""

TASK = {
    "title": "PPO Clipped Value Loss",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ppo_value_loss",
    "description_en": r"""Implement the clipped value-function loss that trains PPO's critic.

**Signature:** `ppo_value_loss(values, old_values, returns, mask, clip_eps=0.2) -> Tensor`

**Parameters:**
- `values` — float tensor of shape `(B, T)`. Current critic predictions.
- `old_values` — float tensor of shape `(B, T)`. Predictions of the critic that generated the rollout. A constant.
- `returns` — float tensor of shape `(B, T)`. The regression targets. A constant.
- `mask` — boolean tensor of shape `(B, T)`. True exactly on generated response tokens.
- `clip_eps` — positive float. How far the prediction may move from `old_values` before the loss stops rewarding the move.

**Returns:** scalar tensor.

Per token:

    clipped_value = old_values + clamp(values - old_values, -clip_eps, clip_eps)
    unclipped_error = (values - returns) ** 2
    clipped_error = (clipped_value - returns) ** 2
    per_token = 0.5 * max(unclipped_error, clipped_error)

Then reduce: `loss = sum(per_token * mask) / sum(mask)`.

**Constraints:**
- Take the elementwise **maximum** of the two errors. The policy loss takes a minimum; this one does not.
- Clamp the *change* in prediction, not the prediction itself and not the error.
- Keep the factor of one half.
- Divide by the number of unmasked tokens. Return a zero scalar when the mask is empty.
- Gradients reach only `values`. Treat `old_values` and `returns` as constants.
- Raise a `ValueError` when the four tensors do not share a shape or when `clip_eps` is not positive.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why maximum here and minimum there.** Both choices make the objective pessimistic, but they point in opposite directions because one is an objective to maximize and the other is an error to minimize. Taking the minimum here inverts the intent: it would let the critic escape a large error simply by moving far from `old_values`, which is the opposite of a trust region. The two losses sitting side by side with opposite reductions is the single most common confusion in a PPO implementation, and nothing about the shapes or the loss curve reveals the mistake.

**What the clip is doing.** A critic that has moved less than `clip_eps` from its old value sees the two errors coincide, so the clip is inert. Once it moves further, the clipped branch freezes the error at the boundary, and the maximum ensures the frozen value is used only when it is the larger of the two — that is, only when freezing is the conservative choice.

**Why the one half.** It is conventional, it cancels the two from the derivative of the square, and its absence doubles the critic's effective learning rate relative to the policy's.""",
    "advisory_prerequisites": ["ppo_clipped_policy_loss", "response_token_mask"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Three quantities get clamped in different PPO implementations: the prediction, the change in prediction, and the squared error. Which one does this contract name, and what would the other two do differently when the critic has barely moved? Now the reduction: you are minimizing an error rather than maximizing an objective — if being pessimistic means assuming the worse case, which of min and max picks the worse case for an error? Sanity-check your answer by asking what happens to a critic that moves far from old_values and happens to land close to returns.",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Build the clipped prediction first as its own named tensor: `old_values + (values - old_values).clamp(-clip_eps, clip_eps)`. Gradients still flow through it, because `values` appears inside the clamp and the clamp passes gradient through in its linear region. Square both errors, take `torch.max` elementwise, multiply by 0.5. Then reduce with the same three lines as `grpo_token_loss`: cast the mask to the value dtype, multiply, sum, divide by `mask.sum()`, guarding the zero case. Two mutations survive every shape check here: writing `torch.min`, which silently removes the trust region, and clamping `values` directly to `[-clip_eps, clip_eps]` instead of clamping the difference, which destroys any prediction whose magnitude exceeds the epsilon. Verify the first by constructing a case where the critic has moved far and landed near the target, and the second by passing values well outside the epsilon band.",
        },
    ],
    "model_connections": [
        "OpenRLHF's ValueLoss.forward clamps values to a band around old_values, squares both errors, takes the maximum, and reduces with the action mask and an explicit token count.",
        "verl implements the same clipped value objective and reports a value clip fraction next to the policy clip fraction as a diagnostic.",
        "The original PPO implementation in OpenAI baselines introduced this clipped critic; the PPO paper itself specifies only a squared-error value term.",
        "GRPO removes this loss entirely by removing the critic, which is why the RL path treats it as the PPO-family branch rather than a shared primitive.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Bounds how far the critic can move per update, which keeps the advantage estimates that depend on it from shifting underneath the policy.",
            "Shares the mask and denominator convention with the policy loss, so both terms are weighted consistently.",
            "Inert when the critic has barely moved, so it costs nothing in the common case.",
        ],
        "cons": [
            "A slow-moving critic lags a fast-moving reward scale, and the clip makes that lag worse rather than better.",
            "The clip band is in raw value units, so it interacts badly with an unnormalized or drifting return scale.",
            "It is a second loss with its own coefficient and its own clip constant, all of which GRPO avoids by having no critic at all.",
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
            "adapted": "The masked token-level reduction and denominator convention that this loss shares.",
            "simplifications": "The clipped value objective is specified by this contract; the referenced file trains a critic-free objective and has no value head.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1707.06347",
            "section": "5 Algorithm, squared-error value loss term",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated with no movement",
            "behavior": "rl.clipping",
            "code": r"""
import torch
# values equal old_values, so both branches coincide: 0.5 * (2 - 5)^2 = 4.5
values = torch.tensor([[2.0]])
returns = torch.tensor([[5.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(values, values.clone(), returns, mask, clip_eps=0.2)
assert out.ndim == 0, f'expected a scalar, got shape {tuple(out.shape)}'
assert torch.allclose(out, torch.tensor(4.5), atol=1e-6), out
""",
        },
        {
            "name": "The clip is inert inside the band",
            "behavior": "rl.clipping",
            "code": r"""
import torch
# The critic moved 0.1, which is inside clip_eps = 0.2, so the clip changes nothing.
values = torch.tensor([[3.1]])
old = torch.tensor([[3.0]])
returns = torch.tensor([[1.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(values, old, returns, mask, clip_eps=0.2)
assert torch.allclose(out, torch.tensor(0.5 * (3.1 - 1.0) ** 2), atol=1e-5), out
""",
        },
        {
            "name": "Uses the maximum, not the minimum",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The reduction takes the minimum. A value loss is an error to minimize, so the pessimistic choice is the larger of the two errors.",
            "code": r"""
import torch
# The critic jumped far from old_values and landed exactly on the target.
# The unclipped error is 0; the clipped error is large. max must pick the large one.
values = torch.tensor([[5.0]])
old = torch.tensor([[0.0]])
returns = torch.tensor([[5.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(values, old, returns, mask, clip_eps=0.2)
# clipped_value = 0 + clamp(5, -0.2, 0.2) = 0.2 ; error = (0.2 - 5)^2 = 23.04
assert torch.allclose(out, torch.tensor(0.5 * 23.04), atol=1e-4), f'expected 11.52, got {out.item()}'
assert out > 1.0, 'taking the minimum would give 0 here'
""",
        },
        {
            "name": "Clamps the change, not the prediction",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "values was clamped directly to the epsilon band instead of the change from old_values. A prediction larger than clip_eps must survive.",
            "code": r"""
import torch
# Large predictions that barely moved. Clamping values directly would destroy them.
values = torch.tensor([[10.05]])
old = torch.tensor([[10.0]])
returns = torch.tensor([[10.0]])
mask = torch.ones(1, 1, dtype=torch.bool)
out = {fn}(values, old, returns, mask, clip_eps=0.2)
expected = torch.tensor(0.5 * (10.05 - 10.0) ** 2)
assert torch.allclose(out, expected, atol=1e-6), f'expected {expected.item()}, got {out.item()}'
assert out < 0.01, 'a near-perfect critic should have a near-zero loss'
""",
        },
        {
            "name": "Includes the one-half factor",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "The loss is off by a factor of two. The contract specifies one half times the squared error.",
            "code": r"""
import torch
values = torch.tensor([[0.0, 0.0]])
returns = torch.tensor([[2.0, 4.0]])
mask = torch.ones(1, 2, dtype=torch.bool)
out = {fn}(values, values.clone(), returns, mask, clip_eps=0.2)
# 0.5 * (4 + 16) / 2 = 5.0
assert torch.allclose(out, torch.tensor(5.0), atol=1e-6), f'expected 5.0, got {out.item()}'
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
values = torch.randn(2, 5)
old = torch.randn(2, 5)
returns = torch.randn(2, 5)
mask = torch.tensor([[True, True, False, False, False],
                     [True, False, False, False, False]])
base = {fn}(values, old, returns, mask, clip_eps=0.2)
polluted = values.clone(); polluted[~mask] = 99.0
after = {fn}(polluted, old, returns, mask, clip_eps=0.2)
assert torch.allclose(base, after, atol=1e-5), f'{base.item()} vs {after.item()}'
# Three unmasked tokens, each with a squared error of 4 -> 0.5 * 4 = 2.0
v = torch.zeros(2, 5)
out = {fn}(v, v.clone(), torch.full((2, 5), 2.0), mask, clip_eps=0.2)
assert torch.allclose(out, torch.tensor(2.0), atol=1e-6), out
""",
        },
        {
            "name": "Matches an elementwise oracle",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The scalar disagrees with an independent elementwise computation of the clipped value objective.",
            "code": r"""
import torch
torch.manual_seed(1)
for B, T in ((3, 4), (2, 6)):
    values = torch.randn(B, T) * 2.0
    old = torch.randn(B, T) * 2.0
    returns = torch.randn(B, T) * 2.0
    mask = torch.rand(B, T) > 0.25
    mask[:, 0] = True
    eps = 0.3
    clipped = old + (values - old).clamp(-eps, eps)
    per_token = 0.5 * torch.max((values - returns) ** 2, (clipped - returns) ** 2)
    keep = mask.float()
    expected = (per_token * keep).sum() / keep.sum()
    out = {fn}(values, old, returns, mask, clip_eps=eps)
    assert torch.allclose(out, expected, atol=1e-5), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Gradients reach only the current values",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "Gradients did not reach values, or the clamp cut the gradient inside its linear region.",
            "code": r"""
import torch
torch.manual_seed(2)
values = torch.randn(2, 4, requires_grad=True)
old = torch.randn(2, 4)
returns = torch.randn(2, 4)
mask = torch.ones(2, 4, dtype=torch.bool)
out = {fn}(values, old, returns, mask, clip_eps=0.5)
assert out.requires_grad, 'loss is detached from the graph'
out.backward()
assert values.grad is not None, 'values received no gradient'
assert torch.isfinite(values.grad).all(), 'gradient contains inf or nan'
# Inside the band both branches coincide, so the gradient is the plain residual.
v2 = torch.tensor([[1.0]], requires_grad=True)
{fn}(v2, torch.tensor([[1.0]]), torch.tensor([[4.0]]), torch.ones(1, 1, dtype=torch.bool), clip_eps=0.5).backward()
assert torch.allclose(v2.grad, torch.tensor([[-3.0]]), atol=1e-5), v2.grad
""",
        },
        {
            "name": "Empty mask returns zero, and invalid arguments raise",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "An empty mask produced NaN, or an inconsistent shape or non-positive clip_eps was accepted.",
            "code": r"""
import torch
values, old, returns = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
empty = torch.zeros(2, 3, dtype=torch.bool)
out = {fn}(values, old, returns, empty, clip_eps=0.2)
assert torch.isfinite(out).all() and torch.allclose(out, torch.tensor(0.0), atol=1e-7), out
mask = torch.ones(2, 3, dtype=torch.bool)
bad = ((values, torch.randn(2, 4), returns, mask, 0.2),
       (values, old, torch.randn(3, 3), mask, 0.2),
       (values, old, returns, mask, 0.0),
       (values, old, returns, mask, -1.0))
for v, o, r, m, eps in bad:
    try:
        {fn}(v, o, r, m, clip_eps=eps)
    except ValueError:
        continue
    raise AssertionError(f'clip_eps={eps} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def ppo_value_loss(values, old_values, returns, mask, clip_eps=0.2):
    shapes = (values.shape, old_values.shape, returns.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"all tensors must share a shape, got {shapes}")
    if clip_eps <= 0:
        raise ValueError(f"clip_eps must be positive, got {clip_eps}")

    keep = mask.to(values.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)

    clipped_value = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    unclipped_error = (values - returns) ** 2
    clipped_error = (clipped_value - returns) ** 2
    per_token = 0.5 * torch.max(unclipped_error, clipped_error)
    return (per_token * keep).sum() / denominator
''',
}
