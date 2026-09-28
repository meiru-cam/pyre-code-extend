"""Policy-loss aggregation modes and the length bias they introduce."""

from ._interview import interview

TASK = {
    "title": "Policy Loss Aggregation Modes",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "aggregate_loss",
    "description_en": r"""Reduce a per-token policy loss to a scalar under four aggregation modes.

**Signature:** `aggregate_loss(per_token_loss, mask, mode) -> Tensor`

**Parameters:**
- `per_token_loss` — float tensor of shape `(B, T)`.
- `mask` — boolean tensor of shape `(B, T)`. True on response tokens.
- `mode` — one of the four strings below.

**Returns:** scalar tensor. Let `L[b] = sum_t per_token_loss[b, t] * mask[b, t]` and `n[b] = sum_t mask[b, t]`. A sequence is *valid* when `n[b] > 0`; the seq means below run over valid sequences only.

    "token-mean"               sum_b L[b] / sum_b n[b]
    "seq-mean-token-mean"      mean over valid b of  L[b] / n[b]
    "seq-mean-token-sum"       mean over valid b of  L[b]
    "seq-mean-token-sum-norm"  mean over valid b of  L[b] / T

**Constraints:**
- Masked positions contribute neither value nor gradient, even when they hold NaN or infinity.
- Keep the result differentiable with respect to `per_token_loss`.
- Raise `ValueError` for an unknown mode, mismatched shapes, a non-2-D input, or a mask with no True entry.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the choice matters.** With `seq-mean-token-mean` every sequence gets equal total weight, so each token of a long response gets less gradient than each token of a short one. For a negative advantage that makes long wrong answers cheap, which is one reason GRPO responses grow in length. `token-mean` weights every token equally across the batch, as DAPO recommends. `seq-mean-token-sum-norm` divides by a constant horizon `T` instead of the response length, the unbiased form proposed by Dr. GRPO.

**Masked NaN.** Multiplying a NaN by zero still gives NaN, so select valid tokens with `torch.where` or boolean indexing rather than multiplying by the mask.""",
    "advisory_prerequisites": ["grpo_token_loss", "response_token_mask"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which quantity does each mode divide by: total tokens, each sequence's own length, or a constant? Which sequences take part in a seq mean? Why can loss * mask still be NaN?"},
        {"level": 2, "kind": "analysis", "content": "Replace masked entries with zero via torch.where(mask, loss, 0). Compute per-sequence sums L and counts n, keep rows with n > 0, then apply the mode's formula. Validate shapes, mode and an all-false mask first."},
    ],
    "model_connections": [
        "verl exposes exactly these choices as `loss_agg_mode` in `agg_loss`; DAPO uses token-mean and Dr. GRPO argues for a constant normalizer.",
    ],
    "pro_con_analysis": {
        "pros": ["Making aggregation explicit exposes a hidden length bias that changes what the policy learns."],
        "cons": ["No mode is best everywhere; token-mean lets long responses dominate the batch and constant normalization needs a fixed horizon."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/volcengine/verl',
      'commit': '00094bd9cd3fef9cf8903daf60ea4a2bcf832efc',
      'path': 'verl/trainer/ppo/core_algos.py',
      'symbol': 'agg_loss',
      'license': 'Apache-2.0',
      'adapted': 'The four loss_agg_mode reductions, excluding fully masked sequences from sequence means and using the '
                 'horizon T for seq-mean-token-sum-norm.',
      'simplifications': 'Single process: no dp_size, batch_num_tokens, global_batch_size, loss_scale_factor or token-sum '
                         'mode; an all-false mask raises instead of dividing by zero.'},
     {'kind': 'paper',
      'url': 'https://arxiv.org/abs/2503.20783',
      'section': 'Liu et al. 2025 (Dr. GRPO), Section 3.1 length bias',
      'note': 'Motivation for a constant normaliser.'}],
    "tests": [
        {"name": "All four modes on a ragged batch", "behavior": "rl.masking", "code": r"""
import torch
loss = torch.tensor([[1.0, 3.0, 0.0, 0.0], [2.0, 2.0, 2.0, 6.0]])
mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
want = {
    "token-mean": 16.0 / 6,
    "seq-mean-token-mean": (2.0 + 3.0) / 2,
    "seq-mean-token-sum": (4.0 + 12.0) / 2,
    "seq-mean-token-sum-norm": (4.0 / 4 + 12.0 / 4) / 2,
}
for mode, value in want.items():
    out = {fn}(loss, mask, mode)
    assert out.ndim == 0 and abs(out.item() - value) < 1e-6, (mode, out, value)
"""},
        {"name": "Matches a seeded oracle and gradient for every mode", "visibility": "unshown", "behavior": "rl.masking", "failure_message": "Each mode must use its stated numerator, denominator and set of valid sequences.", "code": r"""
import random, torch
for seed in (6, 21, 40):
    rng = random.Random(seed)
    B, T = rng.randint(3, 5), rng.randint(3, 7)
    values = [[rng.uniform(-2, 2) for _ in range(T)] for _ in range(B)]
    mask = [[rng.random() < 0.6 for _ in range(T)] for _ in range(B)]
    mask[0] = [True] + [False] * (T - 1)
    mask[1] = [True] * T
    mask[-1] = [False] * T
    valid = [b for b in range(B) if any(mask[b])]
    L = {b: sum(v for v, m in zip(values[b], mask[b]) if m) for b in valid}
    n = {b: sum(mask[b]) for b in valid}
    oracle = {
        "token-mean": sum(L.values()) / sum(n.values()),
        "seq-mean-token-mean": sum(L[b] / n[b] for b in valid) / len(valid),
        "seq-mean-token-sum": sum(L.values()) / len(valid),
        "seq-mean-token-sum-norm": sum(L[b] / T for b in valid) / len(valid),
    }
    grads = {
        "token-mean": lambda b: 1 / sum(n.values()),
        "seq-mean-token-mean": lambda b: 1 / (n[b] * len(valid)),
        "seq-mean-token-sum": lambda b: 1 / len(valid),
        "seq-mean-token-sum-norm": lambda b: 1 / (T * len(valid)),
    }
    for mode, want in oracle.items():
        x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        out = {fn}(x, torch.tensor(mask), mode)
        assert abs(out.item() - want) < 1e-10, (seed, mode, out, want)
        out.backward()
        expected = [[grads[mode](b) if mask[b][t] else 0.0 for t in range(T)] for b in range(B)]
        assert torch.allclose(x.grad, torch.tensor(expected, dtype=torch.float64), atol=1e-12), (seed, mode)
"""},
        {"name": "Ignores NaN and infinity at masked positions", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Masked positions must not leak NaN or infinity; select valid tokens instead of multiplying by the mask.", "code": r"""
import torch
loss = torch.tensor([[1.0, float("nan")], [2.0, float("inf")]], requires_grad=True)
mask = torch.tensor([[True, False], [True, False]])
for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"):
    out = {fn}(loss, mask, mode)
    assert torch.isfinite(out), (mode, out)
    out.backward()
    assert torch.isfinite(loss.grad).all() and loss.grad[:, 1].abs().sum() == 0, (mode, loss.grad)
    loss.grad = None
"""},
        {"name": "Rejects invalid modes, shapes and empty masks", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Unknown modes, misaligned or non-2-D inputs, and an all-false mask must raise ValueError.", "code": r"""
import torch
cases = [
    (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), "seq-mean"),
    (torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool), "token-mean"),
    (torch.ones(3), torch.ones(3, dtype=torch.bool), "token-mean"),
    (torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.bool), "token-mean"),
]
for loss, mask, mode in cases:
    try:
        {fn}(loss, mask, mode)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {tuple(loss.shape)} {tuple(mask.shape)} {mode}")
"""},
    ],
    "solution": '''import torch

MODES = ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm")

def aggregate_loss(per_token_loss, mask, mode):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if per_token_loss.ndim != 2 or per_token_loss.shape != mask.shape:
        raise ValueError("per_token_loss and mask must share one (B, T) shape")
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("mask has no valid tokens")
    masked = torch.where(mask, per_token_loss, torch.zeros_like(per_token_loss))
    seq_sum = masked.sum(dim=1)
    seq_len = mask.sum(dim=1)
    if mode == "token-mean":
        return seq_sum.sum() / seq_len.sum()
    valid = seq_len > 0
    if mode == "seq-mean-token-mean":
        per_seq = seq_sum[valid] / seq_len[valid]
    elif mode == "seq-mean-token-sum":
        per_seq = seq_sum[valid]
    else:
        per_seq = seq_sum[valid] / mask.shape[1]
    return per_seq.mean()
''',
    "interview_questions": interview(
        concept=[
            "In GRPO-style training, how can the way you average the per-token loss change what the policy learns?",
            "What is the length bias that Dr. GRPO points out in the original GRPO objective?",
        ],
        deep_dive=[
            "Write the four aggregation modes. For each, how much gradient does one token of a long response get compared with one token of a short response?",
            "Which sequences should take part in a sequence mean, and what goes wrong with sequences that have no valid tokens?",
            "Why can loss * mask still produce NaN, and how do you avoid it?",
        ],
        tradeoffs=[
            "Token-mean versus sequence-mean: which one would you use for long chain-of-thought RL, and why?",
            "How does the aggregation mode interact with gradient accumulation and data-parallel batch splits?",
        ],
    ),
}
