"""Truncated importance sampling to correct the rollout-engine versus trainer probability mismatch."""

from ._interview import interview

TASK = {
    "title": "Truncated Importance Sampling",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "truncated_is_weights",
    "description_en": r"""Compute truncated importance-sampling (TIS) weights that correct for the gap between the rollout engine's log-probabilities and the trainer's.

**Signature:** `truncated_is_weights(train_logprobs, rollout_logprobs, mask, cap=2.0, level="token") -> Tensor`

**Parameters:**
- `train_logprobs` — float tensor `(B, T)`: log-probabilities of the sampled tokens under the trainer's forward pass (for example FSDP).
- `rollout_logprobs` — float tensor `(B, T)`: log-probabilities of the same tokens reported by the rollout engine (for example vLLM).
- `mask` — boolean tensor `(B, T)`, True on response tokens.
- `cap` — positive float, the truncation threshold.
- `level` — `"token"` or `"sequence"`.

**Returns:** float tensor `(B, T)` of weights, detached from autograd.
- `"token"`: `w[b, t] = min(exp(train[b, t] - rollout[b, t]), cap)` on valid tokens.
- `"sequence"`: `w_seq[b] = min(exp(sum over valid t of (train[b, t] - rollout[b, t])), cap)`, written to every valid token of sequence `b`.
- Masked positions are exactly 0.

**Constraints:**
- The weights must not carry gradient, even when `train_logprobs` requires grad.
- Masked positions must not affect any output, even when they hold NaN or infinity.
- Stay finite when a log-ratio is very large: truncate in log space before exponentiating.
- Raise `ValueError` for `cap <= 0`, an unknown `level`, or shapes that differ or are not 2-D.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the mismatch exists.** Rollout engines use different kernels, precisions and batching from the trainer, so the same token gets a slightly different probability in each. Training on vLLM samples as if they came from the trainer's policy makes on-policy RL quietly off-policy.

**Why truncate.** The unbiased correction multiplies the loss by the ratio of trainer to rollout probability, but rare tokens can produce huge ratios and explode the variance. Capping the ratio trades a little bias for bounded variance. Sequence-level weights are more faithful to the true ratio of whole trajectories but hit the cap far more often.

**How the weight is used.** The loss becomes `w * per_token_policy_loss`. The weight is a constant with respect to the parameters, which is why it is detached.""",
    "advisory_prerequisites": ["per_token_logprobs", "ppo_clipped_policy_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which direction is the ratio: trainer over rollout, or the reverse? Why must the weight be detached? What happens to exp(1000) if you clamp after exponentiating?"},
        {"level": 2, "kind": "analysis", "content": "Compute delta = train - rollout, replace masked entries with 0 via torch.where, and detach. For token level use exp(clamp(delta, max=log(cap))). For sequence level sum delta over T first, clamp, exponentiate, and broadcast. Finally zero the masked positions with torch.where."},
    ],
    "model_connections": [
        "verl and slime expose TIS for vLLM or SGLang rollouts; the Fengyao blog post on off-policy RL from the inference-training mismatch popularized the token-level cap.",
    ],
    "pro_con_analysis": {
        "pros": ["Cheap correction for a real source of off-policy drift that needs no extra forward pass beyond the logprobs you already compute."],
        "cons": ["Truncation biases the gradient, and a cap that is too low hides a broken rollout engine instead of fixing it."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/volcengine/verl',
      'commit': '00094bd9cd3fef9cf8903daf60ea4a2bcf832efc',
      'path': 'verl/trainer/ppo/rollout_corr_helper.py',
      'symbol': 'compute_rollout_correction_weights',
      'license': 'Apache-2.0',
      'adapted': 'Token-level and sequence-level weights exp(log ratio), truncated at an upper threshold and zeroed on '
                 'padding.',
      'simplifications': 'TIS only: no IcePop lower bound, batch normalization or metrics; truncation is applied in log '
                         "space instead of verl's fixed safety bound of 20 before exponentiating."}],
    "tests": [
        {"name": "Token-level weights with a cap", "behavior": "rl.clipping", "code": r"""
import math, torch
train = torch.tensor([[0.0, -1.0, -0.5]])
rollout = torch.tensor([[-0.5, -1.0, -2.0]])
mask = torch.tensor([[True, True, True]])
out = {fn}(train, rollout, mask, cap=2.0, level="token")
want = torch.tensor([[math.exp(0.5), 1.0, 2.0]])
assert out.shape == (1, 3) and torch.allclose(out, want), out
"""},
        {"name": "Matches a seeded oracle at both levels", "visibility": "unshown", "behavior": "rl.clipping", "failure_message": "Use the trainer-over-rollout ratio, sum log-ratios over valid tokens for the sequence level, and cap the result.", "code": r"""
import math, random, torch
for seed in (3, 17, 58):
    rng = random.Random(seed)
    B, T = rng.randint(2, 4), rng.randint(3, 6)
    cap = rng.choice([1.5, 2.0, 5.0])
    train = [[rng.uniform(-3, 0) for _ in range(T)] for _ in range(B)]
    rollout = [[x + rng.gauss(0, 0.6) for x in row] for row in train]
    mask = [[rng.random() < 0.7 for _ in range(T)] for _ in range(B)]
    mask[0][0] = True
    for level in ("token", "sequence"):
        out = {fn}(torch.tensor(train, dtype=torch.float64), torch.tensor(rollout, dtype=torch.float64), torch.tensor(mask), cap=cap, level=level)
        for b in range(B):
            seq_delta = sum(train[b][t] - rollout[b][t] for t in range(T) if mask[b][t])
            for t in range(T):
                if not mask[b][t]:
                    want = 0.0
                elif level == "token":
                    want = min(math.exp(train[b][t] - rollout[b][t]), cap)
                else:
                    want = min(math.exp(seq_delta), cap)
                assert abs(out[b, t].item() - want) < 1e-9, (seed, level, b, t, out[b, t], want)
"""},
        {"name": "Weights are detached from autograd", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "The importance weight is a constant multiplier; detach it so it does not receive gradient.", "code": r"""
import torch
train = torch.tensor([[-1.0, -2.0]], requires_grad=True)
rollout = torch.tensor([[-1.2, -1.5]])
for level in ("token", "sequence"):
    out = {fn}(train, rollout, torch.tensor([[True, True]]), cap=3.0, level=level)
    assert not out.requires_grad, level
"""},
        {"name": "Stays finite for huge ratios and ignores masked NaN", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Truncate in log space and exclude masked positions before any arithmetic.", "code": r"""
import torch
train = torch.tensor([[0.0, 0.0, float("nan")], [1000.0, 0.0, 0.0]])
rollout = torch.tensor([[-0.1, 0.0, 0.0], [0.0, 0.0, float("inf")]])
mask = torch.tensor([[True, True, False], [True, True, False]])
for level in ("token", "sequence"):
    out = {fn}(train, rollout, mask, cap=2.0, level=level)
    assert torch.isfinite(out).all(), (level, out)
    assert out[0, 2] == 0 and out[1, 2] == 0, (level, out)
    assert out[1, 0] == 2.0, (level, out)
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Non-positive caps, unknown levels and misaligned shapes must raise ValueError.", "code": r"""
import torch
x = torch.zeros(2, 3); m = torch.ones(2, 3, dtype=torch.bool)
cases = [
    dict(train_logprobs=x, rollout_logprobs=x, mask=m, cap=0.0),
    dict(train_logprobs=x, rollout_logprobs=x, mask=m, level="batch"),
    dict(train_logprobs=x, rollout_logprobs=torch.zeros(2, 2), mask=m),
    dict(train_logprobs=torch.zeros(3), rollout_logprobs=torch.zeros(3), mask=torch.ones(3, dtype=torch.bool)),
]
for kwargs in cases:
    try:
        {fn}(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs.keys()}")
"""},
    ],
    "solution": '''import math
import torch

def truncated_is_weights(train_logprobs, rollout_logprobs, mask, cap=2.0, level="token"):
    if cap <= 0:
        raise ValueError("cap must be positive")
    if level not in ("token", "sequence"):
        raise ValueError(f"unknown level {level!r}")
    if train_logprobs.ndim != 2 or train_logprobs.shape != rollout_logprobs.shape or train_logprobs.shape != mask.shape:
        raise ValueError("inputs must share one (B, T) shape")
    mask = mask.bool()
    delta = torch.where(mask, train_logprobs.detach() - rollout_logprobs.detach(), torch.zeros_like(train_logprobs))
    if level == "sequence":
        delta = delta.sum(dim=1, keepdim=True).expand_as(delta)
    weights = torch.exp(torch.clamp(delta, max=math.log(cap)))
    return torch.where(mask, weights, torch.zeros_like(weights))
''',
    "interview_questions": interview(
        concept=[
            "Why do the rollout engine and the trainer assign different probabilities to the same token, and why does that make on-policy RL off-policy?",
            "What does importance sampling correct, and why truncate the ratio instead of using it directly?",
        ],
        deep_dive=[
            "Write the token-level and sequence-level weights. Which one is closer to the true trajectory ratio, and which one hits the cap more often?",
            "Why must the weight be detached, and where does it enter the loss?",
            "How do you keep the weight finite when a log-ratio is huge, and why must masked positions be excluded before arithmetic?",
        ],
        tradeoffs=[
            "How do you choose the cap? What do you monitor to detect that the rollout engine itself is broken?",
            "Truncated IS versus fixing the mismatch at the source (same kernels, fp32 logits, batch-invariant ops): costs and benefits?",
        ],
    ),
}
