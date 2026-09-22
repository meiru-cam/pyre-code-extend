"""GRPO token loss — where advantages, the KL penalty, the mask and the denominator meet."""

TASK = {
    "title": "GRPO Token Loss",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "grpo_token_loss",
    "description_en": r"""Assemble the four primitives of this path into the loss GRPO actually optimizes.

**Signature:** `grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04) -> Tensor`

**Parameters:**
- `logprobs` — float tensor of shape `(B, T)`. Per-token log-probabilities under the policy being trained.
- `ref_logprobs` — float tensor of shape `(B, T)`. The same tokens under the frozen reference model.
- `advantages` — float tensor of shape `(B,)`. One group-relative advantage per response, already standardized and already detached.
- `mask` — boolean tensor of shape `(B, T)`. True exactly on generated response tokens.
- `beta` — non-negative float. The weight on the KL penalty.

**Returns:** scalar tensor.

    kl[b, t]        = exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    objective[b, t] = advantages[b] * logprobs[b, t] - beta * kl[b, t]
    loss            = -sum(objective * mask) / sum(mask)

**Constraints:**
- Broadcast each response's scalar advantage across all of its tokens.
- Subtract the KL penalty from the objective, so it is added to the loss.
- Divide by the number of unmasked tokens in the whole batch. Not by `B * T`, and not by averaging within each response first.
- Return a scalar differentiable with respect to `logprobs`. Treat `advantages` and `ref_logprobs` as constants.
- Return a zero scalar when the mask is entirely False.
- Raise a `ValueError` when the shapes are inconsistent or when `beta` is negative.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the KL sign matters.** A sign error rewards divergence from the reference. The policy degenerates within a few steps while the loss curve still looks like it is descending.

**Why this denominator.** The original GRPO objective normalizes within each response and then averages over responses. DAPO showed that this weights a short response's tokens more heavily than a long one's, biasing the policy toward brevity independently of reward. Summing over the whole batch and dividing by the total unmasked token count removes that bias, and it is what the reference implementations do. Both forms are defensible; this contract specifies the token-level one and the evaluator enforces it.

**Why there is no ratio.** GRPO's published objective multiplies the advantage by an importance ratio between the current policy and the policy that generated the rollout. When the update is on-policy those are the same distribution and the ratio is exactly one, which is the case this contract covers. The reference implementations express it as `exp(logprobs - logprobs.detach())` — a quantity whose value is always one but whose gradient is the same as the direct product used here. Clipping that ratio, for the off-policy case, is a separate exercise.""",
    "advisory_prerequisites": [
        "per_token_logprobs",
        "group_relative_advantage",
        "k3_kl_penalty",
        "response_token_mask",
    ],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "The advantages tensor has one entry per row but the objective has one entry per token — what is the smallest change to the advantages tensor that lets it multiply a (B, T) tensor? For the KL term, check the sign twice: the loss is the negative of the objective, so if the penalty should increase the loss, what sign does it carry inside the objective? For the reduction, ask what number of tokens actually contributed: is it B times T, or something you can read off the mask itself? And what does your expression do when that number is zero — does it return zero, or NaN?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Four steps. First, the KL term: reuse the k3 form on the raw tensors, unmasked, since masked positions are discarded by the reduction anyway. Second, broadcast the advantage with `advantages.unsqueeze(-1)` so it becomes `(B, 1)` and multiplies against `(B, T)`. Third, form `advantages_b * logprobs - beta * kl`; note the minus, which becomes a plus once the whole thing is negated. Fourth, reduce: convert the boolean mask with a dtype cast so it can multiply the float objective, sum the product, and divide by `mask.sum()`. Guard that denominator: compute it first, and if it is zero return a zero scalar built from the input so it still carries dtype and device. Two mistakes survive every shape assertion — using `mean()` on the masked product, which divides by `B * T`, and computing a per-row mean followed by a mean over rows, which is the length-biased form DAPO argues against. Both give a plausible-looking number, so verify your denominator against `mask.sum()` explicitly.",
        },
    ],
    "model_connections": [
        "simple_GRPO's GRPO_step builds the per-token objective, subtracts beta times the k3 KL term, applies the completion mask, and normalizes by the masked token count.",
        "nano-aha-moment's compute_pg_loss takes total_response_len as an explicit argument and divides the masked sum by it, which is the token-level denominator this contract specifies.",
        "DAPO identified the per-response normalization as a source of length bias and proposed the token-level denominator used here.",
        "OpenRLHF's PolicyLoss takes the action mask and the batch token count as arguments for the same reason: the loss cannot infer its own denominator.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One scalar per response keeps the credit assignment simple and needs no value network.",
            "The token-level denominator removes the length bias of per-response normalization.",
            "The KL term anchors the policy to the reference, which is what keeps RLVR from drifting into degenerate text.",
        ],
        "cons": [
            "Every token of a response receives the same advantage, so the loss cannot tell a good reasoning step from a bad one inside a correct answer.",
            "With no ratio clipping, an off-policy batch produces unbounded updates; this form is only safe when the rollout and the update use the same policy.",
            "The KL penalty is one more coefficient to tune, and its correct value depends on the reward scale that the group normalization has already changed.",
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
            "adapted": "The masked per-token policy-gradient objective with a k3 KL penalty, normalized by the total response token count.",
            "simplifications": "Takes precomputed log-probabilities instead of running the policy and reference forward passes; no temperature division, no entropy term and no distributed reduction.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO, objective with KL regularization",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated single token",
            "behavior": "rl.advantage",
            "code": r"""
import torch
# One row, one token, policy equals reference so the KL term is exactly zero.
logprobs = torch.tensor([[-2.0]])
advantages = torch.tensor([3.0])
mask = torch.tensor([[True]])
out = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.04)
assert out.ndim == 0, f'expected a scalar, got shape {tuple(out.shape)}'
# loss = -(3.0 * -2.0) / 1 = 6.0
assert torch.allclose(out, torch.tensor(6.0), atol=1e-6), out
""",
        },
        {
            "name": "The advantage is shared across a response's tokens",
            "behavior": "rl.advantage",
            "code": r"""
import torch
logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
advantages = torch.tensor([2.0])
mask = torch.ones(1, 3, dtype=torch.bool)
out = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.0)
# loss = -(2 * (-1 - 2 - 3)) / 3 = 4.0
assert torch.allclose(out, torch.tensor(4.0), atol=1e-6), out
""",
        },
        {
            "name": "Masked positions do not affect the loss",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Changing a masked-out position changed the loss. Multiply by the mask before summing.",
            "code": r"""
import torch
torch.manual_seed(0)
logprobs = torch.randn(3, 6)
ref = torch.randn(3, 6)
advantages = torch.randn(3)
mask = torch.tensor([[True, True, False, False, False, False],
                     [True, True, True, True, False, False],
                     [True, False, False, False, False, False]])
base = {fn}(logprobs, ref, advantages, mask, beta=0.05)
polluted_lp = logprobs.clone()
polluted_ref = ref.clone()
polluted_lp[~mask] = 50.0
polluted_ref[~mask] = -50.0
after = {fn}(polluted_lp, polluted_ref, advantages, mask, beta=0.05)
assert torch.allclose(base, after, atol=1e-5), f'{base.item()} vs {after.item()}'
""",
        },
        {
            "name": "Denominator is the masked token count",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "The loss scales with padding. Divide by mask.sum(), not by the padded size and not by a per-response mean.",
            "code": r"""
import torch
logprobs = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])
advantages = torch.tensor([1.0])
mask = torch.tensor([[True, True, False, False]])
narrow = {fn}(logprobs[:, :2], logprobs[:, :2].clone(), advantages, mask[:, :2], beta=0.0)
padded = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.0)
assert torch.allclose(narrow, padded, atol=1e-6), f'padding changed the loss: {narrow.item()} vs {padded.item()}'
assert torch.allclose(padded, torch.tensor(1.0), atol=1e-6), padded
""",
        },
        {
            "name": "Token-level rather than per-response normalization",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "The reduction averages within each response first. Sum over the whole batch and divide by the total masked token count.",
            "code": r"""
import torch
# Row 0 has one token, row 1 has three. The two reductions differ.
logprobs = torch.tensor([[-1.0, 0.0, 0.0],
                         [-1.0, -1.0, -1.0]])
advantages = torch.tensor([1.0, 1.0])
mask = torch.tensor([[True, False, False],
                     [True, True, True]])
out = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.0)
token_level = torch.tensor(1.0)          # -(-1 - 1 - 1 - 1) / 4
per_response = torch.tensor(1.0)         # same here by construction
assert torch.allclose(out, token_level, atol=1e-6), out
# Now make them differ: row 0's single token carries a much larger magnitude.
logprobs = torch.tensor([[-9.0, 0.0, 0.0],
                         [-1.0, -1.0, -1.0]])
out = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.0)
assert torch.allclose(out, torch.tensor(3.0), atol=1e-6), f'expected token-level 3.0, got {out.item()}'
per_response_value = ((9.0 / 1.0) + (3.0 / 3.0)) / 2.0
assert not torch.allclose(out, torch.tensor(per_response_value), atol=1e-3), 'this is the per-response reduction'
""",
        },
        {
            "name": "The KL penalty increases the loss",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "A larger divergence from the reference did not increase the loss. The KL term is subtracted from the objective, so it is added to the loss.",
            "code": r"""
import torch
logprobs = torch.tensor([[-1.0, -1.0]])
advantages = torch.tensor([1.0])
mask = torch.ones(1, 2, dtype=torch.bool)
same = {fn}(logprobs, logprobs.clone(), advantages, mask, beta=0.5)
diverged = {fn}(logprobs, torch.tensor([[-4.0, -4.0]]), advantages, mask, beta=0.5)
assert diverged > same, f'divergence should cost more: {diverged.item()} vs {same.item()}'
# With beta = 0 the reference is irrelevant.
zero_beta = {fn}(logprobs, torch.tensor([[-4.0, -4.0]]), advantages, mask, beta=0.0)
assert torch.allclose(zero_beta, torch.tensor(1.0), atol=1e-6), zero_beta
""",
        },
        {
            "name": "Matches a loop-based oracle",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "The scalar disagrees with an independent element-by-element computation of the same objective.",
            "code": r"""
import torch
torch.manual_seed(1)
for B, T in ((3, 5), (2, 8), (4, 2)):
    logprobs = torch.randn(B, T)
    ref = torch.randn(B, T)
    advantages = torch.randn(B)
    mask = torch.rand(B, T) > 0.3
    mask[:, 0] = True
    beta = 0.07
    total = 0.0
    count = 0
    for b in range(B):
        for t in range(T):
            if not bool(mask[b, t]):
                continue
            r = float(ref[b, t] - logprobs[b, t])
            kl = torch.exp(torch.tensor(r)).item() - r - 1.0
            total += float(advantages[b]) * float(logprobs[b, t]) - beta * kl
            count += 1
    expected = torch.tensor(-total / count)
    out = {fn}(logprobs, ref, advantages, mask, beta=beta)
    assert torch.allclose(out, expected, atol=1e-4), f'{out.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Gradients reach only the policy log-probabilities",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "Gradients did not reach logprobs, or they leaked into advantages or ref_logprobs, which the caller supplies as constants.",
            "code": r"""
import torch
torch.manual_seed(2)
logprobs = torch.randn(2, 4, requires_grad=True)
ref = torch.randn(2, 4, requires_grad=True)
advantages = torch.randn(2, requires_grad=True)
mask = torch.ones(2, 4, dtype=torch.bool)
out = {fn}(logprobs, ref, advantages, mask, beta=0.1)
assert out.requires_grad, 'loss is detached from the graph'
out.backward()
assert logprobs.grad is not None, 'logprobs received no gradient'
assert torch.isfinite(logprobs.grad).all(), 'gradient contains inf or nan'
assert not torch.allclose(logprobs.grad, torch.zeros_like(logprobs.grad)), 'gradient is identically zero'
# Masked-out rows must contribute nothing.
half = torch.tensor([[True, True, False, False], [False, False, False, False]])
lp2 = logprobs.detach().clone().requires_grad_(True)
{fn}(lp2, ref.detach(), advantages.detach(), half, beta=0.1).backward()
assert torch.allclose(lp2.grad[1], torch.zeros(4), atol=1e-7), lp2.grad[1]
assert torch.allclose(lp2.grad[0, 2:], torch.zeros(2), atol=1e-7), lp2.grad[0]
""",
        },
        {
            "name": "Empty mask returns zero, not NaN",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "An all-False mask produced NaN or inf. Guard the denominator and return a zero scalar.",
            "code": r"""
import torch
logprobs = torch.randn(2, 3)
mask = torch.zeros(2, 3, dtype=torch.bool)
out = {fn}(logprobs, torch.randn(2, 3), torch.randn(2), mask, beta=0.04)
assert torch.isfinite(out).all(), f'expected a finite value, got {out}'
assert torch.allclose(out, torch.tensor(0.0), atol=1e-7), out
assert out.ndim == 0, f'expected a scalar, got shape {tuple(out.shape)}'
""",
        },
        {
            "name": "Rejects inconsistent shapes and a negative beta",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An inconsistent input was accepted. Validate that logprobs, ref_logprobs and mask share a shape, that advantages has one entry per row, and that beta is non-negative.",
            "code": r"""
import torch
lp = torch.randn(2, 3)
ref = torch.randn(2, 3)
adv = torch.randn(2)
mask = torch.ones(2, 3, dtype=torch.bool)
bad = ((lp, torch.randn(2, 4), adv, mask, 0.1),
       (lp, ref, torch.randn(3), mask, 0.1),
       (lp, ref, adv, torch.ones(2, 4, dtype=torch.bool), 0.1),
       (lp, ref, adv, mask, -0.1))
for args in bad:
    try:
        {fn}(*args[:4], beta=args[4])
    except ValueError:
        continue
    raise AssertionError(f'should raise ValueError: beta={args[4]}')
""",
        },
    ],
    "solution": '''import torch


def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
    if logprobs.shape != ref_logprobs.shape or logprobs.shape != mask.shape:
        raise ValueError(
            f"logprobs {tuple(logprobs.shape)}, ref_logprobs {tuple(ref_logprobs.shape)} "
            f"and mask {tuple(mask.shape)} must share a shape"
        )
    if advantages.shape != logprobs.shape[:1]:
        raise ValueError(
            f"advantages {tuple(advantages.shape)} must have one entry per row of "
            f"{tuple(logprobs.shape)}"
        )
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)

    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    return -(objective * keep).sum() / denominator
''',
}
