"""REINFORCE leave-one-out advantage: each sample's baseline is the mean of its group peers."""

from ._interview import interview

TASK = {
    "title": "RLOO Leave-One-Out Advantage",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "rloo_advantage",
    "description_en": r"""Compute REINFORCE leave-one-out (RLOO) advantages for grouped completions.

**Signature:** `rloo_advantage(rewards, group_size) -> Tensor`

**Parameters:**
- `rewards` — 1-D float tensor of shape `(N,)`, laid out group-major: the first `group_size` entries are the completions of prompt 0, the next `group_size` belong to prompt 1, and so on.
- `group_size` — integer `k >= 2` that divides `N`.

**Returns:** float tensor of shape `(N,)`. Each reward minus the mean reward of the *other* completions in its group:

    A_i = r_i - (sum of rewards in the group of i  -  r_i) / (k - 1)

**Constraints:**
- The baseline for sample `i` must exclude `r_i` itself.
- Do not divide by a standard deviation.
- Compute groups independently; never mix rewards across prompts.
- Raise `ValueError` when `rewards` is not 1-D, when `group_size < 2`, or when `group_size` does not divide `N`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why leave one out.** A baseline that includes the sample's own reward is correlated with that sample, which biases the policy gradient. Excluding it keeps the estimator unbiased while still reducing variance.

**Relation to GRPO.** Algebraically `A_i = k / (k - 1) * (r_i - mean(group))`, so RLOO is a rescaled mean-centering. GRPO additionally divides by the group standard deviation, which reweights prompts by how spread out their rewards are.""",
    "advisory_prerequisites": ["group_relative_advantage"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How can you get every sample's leave-one-out mean from one group sum instead of a loop? Why must k be at least 2?"},
        {"level": 2, "kind": "analysis", "content": "Reshape to (N // k, k), take the row sum, compute (row_sum - r) / (k - 1) as the baseline, subtract it from r, and flatten back. Validate the shape and group size first."},
    ],
    "model_connections": [
        "TRL's RLOOTrainer and OpenRLHF's rloo advantage estimator both use this leave-one-out baseline over k samples per prompt.",
    ],
    "pro_con_analysis": {
        "pros": ["Unbiased baseline without a value network, using samples you already drew."],
        "cons": ["Needs at least two samples per prompt, and groups where every reward is equal give zero advantage everywhere."],
    },
    "tests": [
        {"name": "Leave-one-out baseline on two groups", "behavior": "rl.advantage", "code": r"""
import torch
rewards = torch.tensor([1.0, 0.0, 0.0, 2.0, 2.0, 5.0])
out = {fn}(rewards, 3)
want = torch.tensor([1.0, -0.5, -0.5, -1.5, -1.5, 3.0])
assert out.shape == (6,) and torch.allclose(out, want), out
"""},
        {"name": "Matches a seeded per-sample oracle", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "Each baseline must be the mean of the other completions of the same prompt.", "code": r"""
import random, torch
for seed in (4, 19, 33):
    rng = random.Random(seed)
    k = rng.randint(2, 6)
    groups = rng.randint(2, 5)
    rewards = [rng.uniform(-2, 3) for _ in range(k * groups)]
    out = {fn}(torch.tensor(rewards, dtype=torch.float64), k)
    for i, r in enumerate(rewards):
        start = (i // k) * k
        peers = [rewards[j] for j in range(start, start + k) if j != i]
        want = r - sum(peers) / len(peers)
        assert abs(out[i].item() - want) < 1e-10, (seed, i, out[i], want)
"""},
        {"name": "Group advantages sum to zero and are not standardized", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "RLOO advantages are mean-centred per group but not divided by a standard deviation.", "code": r"""
import torch
rewards = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
out = {fn}(rewards, 4)
assert abs(out.sum().item()) < 1e-10, out
assert torch.allclose(out, torch.tensor([-20.0, -20.0 / 3, 20.0 / 3, 20.0], dtype=torch.float64)), out
"""},
        {"name": "Keeps groups independent in group-major layout", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "Consecutive blocks of group_size entries form one group; do not group by stride.", "code": r"""
import torch
rewards = torch.tensor([0.0, 0.0, 100.0, 100.0])
out = {fn}(rewards, 2)
assert torch.allclose(out, torch.zeros(4)), out
"""},
        {"name": "Rejects invalid group sizes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "group_size must be at least 2 and divide the number of rewards.", "code": r"""
import torch
for rewards, k in ((torch.ones(4), 1), (torch.ones(5), 2), (torch.ones(2, 2), 2)):
    try:
        {fn}(rewards, k)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted k={k} for shape {tuple(rewards.shape)}")
"""},
    ],
    "solution": '''import torch

def rloo_advantage(rewards, group_size):
    if rewards.ndim != 1 or group_size < 2 or rewards.numel() % group_size != 0:
        raise ValueError("rewards must be 1-D and group_size >= 2 must divide its length")
    grouped = rewards.reshape(-1, group_size)
    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)
    return (grouped - baseline).reshape(-1)
''',
    "interview_questions": interview(
        concept=[
            "What is RLOO, and what baseline does it use for each sample?",
            "Why does including a sample's own reward in its baseline bias the policy gradient?",
        ],
        deep_dive=[
            "How do you compute every leave-one-out mean in one vectorized step?",
            "Show that the RLOO advantage equals k / (k - 1) times the mean-centred reward.",
            "Why must the group size be at least 2, and what happens when all rewards in a group are equal?",
        ],
        tradeoffs=[
            "RLOO versus GRPO: what does dividing by the group standard deviation add, and what bias does it introduce?",
            "RLOO versus PPO with a learned critic: sample cost, variance and stability?",
        ],
    ),
}
