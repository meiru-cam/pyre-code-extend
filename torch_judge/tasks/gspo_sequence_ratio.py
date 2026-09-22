"""GSPO sequence ratio — importance weighting at the sequence level, not the token level."""

TASK = {
    "title": "GSPO Sequence-Level Importance Ratio",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "gspo_sequence_ratio",
    "description_en": r"""Implement the length-normalized, sequence-level importance ratio that GSPO substitutes for PPO's per-token ratio.

**Signature:** `gspo_sequence_ratio(logprobs, old_logprobs, mask) -> Tensor`

**Parameters:**
- `logprobs` — float tensor of shape `(B, T)`. Per-token log-probabilities under the policy being updated.
- `old_logprobs` — float tensor of shape `(B, T)`. The same tokens under the policy that generated the rollout. A constant.
- `mask` — boolean tensor of shape `(B, T)`. True exactly on generated response tokens.

**Returns:** float tensor of shape `(B,)`. One ratio per sequence:

    mean_logratio[b] = sum_t (logprobs[b, t] - old_logprobs[b, t]) * mask[b, t] / sum_t mask[b, t]
    ratio[b]         = exp(mean_logratio[b])

**Constraints:**
- Aggregate the log-ratio first, then exponentiate **once**. Do not exponentiate per token and then average.
- Divide by each sequence's own unmasked token count, not by `T` and not by the batch total.
- Exclude masked positions from both the sum and the count.
- A sequence with no unmasked tokens has a ratio of 1.0, the neutral weight. Do not return NaN.
- Gradients reach only `logprobs`.
- Raise a `ValueError` when the three tensors do not share a shape.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the order of operations is the whole exercise.** Averaging in log space then exponentiating is the geometric mean of the per-token ratios; exponentiating per token then averaging is their arithmetic mean. The arithmetic mean is dominated by whichever single token has the largest ratio, so one token whose probability happened to rise sharply can drag the whole sequence's weight with it. The geometric mean gives every token equal influence in log space, which is exactly the property GSPO wants.

**Why length normalization.** Without the division, the log-ratio is a sum whose magnitude grows with the response length, so `exp` of it underflows or overflows as soon as responses get long, and a long response is weighted incomparably against a short one. Dividing by the token count makes the quantity a per-token average, so responses of different lengths land on the same scale and a single clipping band applies to all of them.

**Where this fits.** PPO and GRPO clip a ratio computed per token; this is the GSPO alternative, where one weight applies to the whole sequence. Clipping becomes an all-or-nothing decision about a response rather than a per-token decision, which removes a source of gradient noise in long generations and removes the ability to accept part of a response.""",
    "advisory_prerequisites": ["ppo_clipped_policy_loss", "per_token_logprobs", "response_token_mask"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Write down the two candidate orderings — exponentiate then average, or average then exponentiate — and evaluate both by hand on a two-token sequence whose log-ratios are +3 and -3. Do they agree? Which one is unchanged if you swap the two tokens, and which is dominated by the larger one? For the denominator, each row may have a different number of real tokens: where does that count come from, and what does your expression return for a row where it is zero? Last, which tensor needs to reach the sum with its gradient intact?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Compute the per-token log-ratio as a single tensor, then reduce along the time axis only: `keep = mask.to(logprobs.dtype)`, `numerator = (log_ratio * keep).sum(dim=-1)`, `count = keep.sum(dim=-1)`. Guard the count before dividing — `count.clamp(min=1)` keeps the division finite, and because the numerator is already zero for an all-masked row, the quotient is zero and `exp(0)` is the 1.0 the contract asks for. Exponentiate last, once, on the `(B,)` tensor. Two mutations survive every shape check: exponentiating the per-token log-ratio before reducing, which silently computes the arithmetic mean of ratios, and summing without dividing, which is a different estimator whose scale grows with length. Build a fixture with two rows of very different lengths to separate them.",
        },
    ],
    "model_connections": [
        "OpenRLHF's PolicyLoss carries a GSPO branch that aggregates the log-ratio over the action mask and normalizes by the sequence length before exponentiating.",
        "GSPO was introduced for Qwen's RL training as a response to instability in token-level importance weighting on long generations, particularly in mixture-of-experts models where per-token routing adds its own variance.",
        "verl exposes the same choice as a configuration switch between token-level and sequence-level importance weighting, keeping the rest of the clipped objective identical.",
        "The exercise pairs with ppo_clipped_policy_loss: the same clipped surrogate, with this ratio substituted, is the GSPO objective.",
    ],
    "pro_con_analysis": {
        "pros": [
            "The geometric mean gives every token equal influence, so one outlier token cannot dominate a sequence's weight.",
            "Length normalization puts short and long responses on one scale, so a single clip band applies to both.",
            "Removes a large source of gradient variance in long generations, which is the instability it was designed to fix.",
        ],
        "cons": [
            "Clipping becomes all-or-nothing per response, so a mostly-good response with one bad token is accepted or rejected as a unit.",
            "Credit assignment is coarser than the token-level ratio, which matters when a long response is only partly correct.",
            "The whole sequence's weight collapses to one number, so per-token diagnostics such as a clip fraction lose their meaning.",
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
            "adapted": "The masked per-token log-ratio between the policy and a stored reference, which this contract aggregates per sequence instead of per token.",
            "simplifications": "Reduces to one ratio per sequence with length normalization; no clipping, no advantage and no loss.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2507.18071",
            "section": "3 Group Sequence Policy Optimization, sequence-level importance ratio",
        },
    ],
    "tests": [
        {
            "name": "On-policy sequences have a ratio of one",
            "behavior": "rl.clipping",
            "code": r"""
import torch
logprobs = torch.tensor([[-1.0, -2.0, -3.0],
                         [-0.5, -0.5, -0.5]])
mask = torch.ones(2, 3, dtype=torch.bool)
out = {fn}(logprobs, logprobs.clone(), mask)
assert out.shape == (2,), f'expected (2,), got {tuple(out.shape)}'
assert torch.allclose(out, torch.ones(2), atol=1e-6), out
""",
        },
        {
            "name": "Hand-calculated geometric mean",
            "behavior": "rl.clipping",
            "code": r"""
import torch
import math
# Log-ratios of +1 and -1 average to 0, so the sequence ratio is exp(0) = 1.
logprobs = torch.tensor([[0.0, -1.0]])
old = torch.tensor([[-1.0, 0.0]])
mask = torch.ones(1, 2, dtype=torch.bool)
assert torch.allclose({fn}(logprobs, old, mask), torch.ones(1), atol=1e-6)
# Log-ratios of +2 and 0 average to 1, so the ratio is e.
logprobs = torch.tensor([[0.0, 0.0]])
old = torch.tensor([[-2.0, 0.0]])
assert torch.allclose({fn}(logprobs, old, mask), torch.tensor([math.e]), atol=1e-5)
""",
        },
        {
            "name": "Averages in log space, not in ratio space",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "The per-token ratios were exponentiated before averaging. Aggregate the log-ratio first, then exponentiate once.",
            "code": r"""
import torch
# Log-ratios of +3 and -3. Geometric mean is 1; arithmetic mean of ratios is ~10.07.
logprobs = torch.tensor([[0.0, -3.0]])
old = torch.tensor([[-3.0, 0.0]])
mask = torch.ones(1, 2, dtype=torch.bool)
out = {fn}(logprobs, old, mask)
arithmetic = (torch.exp(torch.tensor(3.0)) + torch.exp(torch.tensor(-3.0))) / 2
assert not torch.allclose(out, arithmetic.reshape(1), atol=1e-2), 'this is the arithmetic mean of ratios'
assert torch.allclose(out, torch.ones(1), atol=1e-5), f'expected 1.0, got {out.item()}'
""",
        },
        {
            "name": "Normalizes by each sequence's own token count",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "The denominator is not each row's unmasked token count. Two rows with the same per-token log-ratio but different lengths must get the same ratio.",
            "code": r"""
import torch
import math
# Both rows have a per-token log-ratio of exactly 1, but different lengths.
logprobs = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])
old = torch.tensor([[-1.0, -1.0, -1.0, -1.0],
                    [-1.0, 0.0, 0.0, 0.0]])
mask = torch.tensor([[True, True, True, True],
                     [True, False, False, False]])
out = {fn}(logprobs, old, mask)
assert torch.allclose(out, torch.full((2,), math.e), atol=1e-5), f'expected both to be e, got {out}'
""",
        },
        {
            "name": "Masked positions are excluded from sum and count",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Masked-out positions changed the result. Exclude them from both the numerator and the denominator.",
            "code": r"""
import torch
torch.manual_seed(0)
logprobs = torch.randn(3, 6)
old = torch.randn(3, 6)
mask = torch.tensor([[True, True, False, False, False, False],
                     [True, True, True, False, False, False],
                     [True, False, False, False, False, False]])
base = {fn}(logprobs, old, mask)
polluted_lp = logprobs.clone(); polluted_lp[~mask] = 40.0
polluted_old = old.clone(); polluted_old[~mask] = -40.0
after = {fn}(polluted_lp, polluted_old, mask)
assert torch.allclose(base, after, atol=1e-5), f'{base} vs {after}'
""",
        },
        {
            "name": "Matches a row-by-row oracle",
            "visibility": "unshown",
            "behavior": "rl.clipping",
            "failure_message": "Values disagree with an independent per-row computation of the length-normalized log-ratio.",
            "code": r"""
import torch
import math
torch.manual_seed(1)
for B, T in ((4, 5), (2, 9)):
    logprobs = torch.randn(B, T) * 0.4
    old = torch.randn(B, T) * 0.4
    mask = torch.rand(B, T) > 0.3
    mask[:, 0] = True
    expected = torch.zeros(B)
    for b in range(B):
        total = 0.0
        count = 0
        for t in range(T):
            if bool(mask[b, t]):
                total += float(logprobs[b, t]) - float(old[b, t])
                count += 1
        expected[b] = math.exp(total / count)
    out = {fn}(logprobs, old, mask)
    assert out.shape == (B,), f'expected ({B},), got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-4), (out - expected).abs().max()
""",
        },
        {
            "name": "An all-masked row is neutral",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A row with no unmasked tokens produced NaN or inf. Its ratio is the neutral weight 1.0.",
            "code": r"""
import torch
logprobs = torch.randn(2, 4)
old = torch.randn(2, 4)
mask = torch.tensor([[True, True, False, False],
                     [False, False, False, False]])
out = {fn}(logprobs, old, mask)
assert torch.isfinite(out).all(), f'expected finite values, got {out}'
assert torch.allclose(out[1], torch.tensor(1.0), atol=1e-6), f'expected 1.0, got {out[1].item()}'
""",
        },
        {
            "name": "Gradients reach only the current policy",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "No gradient reached logprobs, or the gradient does not follow the length-normalized form.",
            "code": r"""
import torch
torch.manual_seed(2)
logprobs = torch.randn(2, 4, requires_grad=True)
old = torch.randn(2, 4)
mask = torch.ones(2, 4, dtype=torch.bool)
out = {fn}(logprobs, old, mask)
assert out.requires_grad, 'output is detached from the graph'
out.sum().backward()
assert logprobs.grad is not None, 'logprobs received no gradient'
assert torch.isfinite(logprobs.grad).all(), 'gradient contains inf or nan'
# d ratio / d logprob[b, t] = ratio[b] / count[b]; every token of a row shares it.
ratios = out.detach()
for b in range(2):
    assert torch.allclose(logprobs.grad[b], torch.full((4,), float(ratios[b]) / 4), atol=1e-4), logprobs.grad[b]
""",
        },
        {
            "name": "Rejects mismatched shapes",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "Mismatched shapes were broadcast instead of rejected. Raise ValueError when the three tensors differ.",
            "code": r"""
import torch
lp = torch.randn(2, 3)
bad = ((lp, torch.randn(2, 4), torch.ones(2, 3, dtype=torch.bool)),
       (lp, torch.randn(2, 3), torch.ones(3, 3, dtype=torch.bool)),
       (lp, torch.randn(3, 3), torch.ones(2, 3, dtype=torch.bool)))
for a, b, m in bad:
    try:
        {fn}(a, b, m)
    except ValueError:
        continue
    raise AssertionError(f'shapes {tuple(a.shape)}, {tuple(b.shape)}, {tuple(m.shape)} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def gspo_sequence_ratio(logprobs, old_logprobs, mask):
    shapes = (logprobs.shape, old_logprobs.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"all tensors must share a shape, got {shapes}")

    keep = mask.to(logprobs.dtype)
    log_ratio = (logprobs - old_logprobs) * keep
    count = keep.sum(dim=-1)
    mean_log_ratio = log_ratio.sum(dim=-1) / count.clamp(min=1.0)
    return torch.exp(mean_log_ratio)
''',
}
