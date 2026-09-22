"""Group-relative advantage — the critic-free advantage estimate at the heart of GRPO."""

TASK = {
    "title": "Group-Relative Advantage",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "group_relative_advantage",
    "description_en": r"""Implement the group-relative advantage estimate that lets GRPO train a policy without a value network.

**Signature:** `group_relative_advantage(rewards, group_size, eps=1e-4) -> Tensor`

**Parameters:**
- `rewards` — float tensor of shape `(N,)`. One scalar reward per sampled response, laid out group-major: the first `group_size` entries are the completions of prompt 0, the next `group_size` are the completions of prompt 1, and so on.
- `group_size` — positive integer. The number of completions sampled per prompt. Must divide `N` exactly.
- `eps` — small positive float added to the standard deviation before dividing.

**Returns:** float tensor of shape `(N,)`. Each entry is its reward standardized against the other completions of the same prompt:

    A_i = (r_i - mean(group of i)) / (std(group of i) + eps)

**Constraints:**
- Standardize within each group independently. Do not use a batch-wide mean or standard deviation.
- Use the population standard deviation: divide by `group_size`, not by `group_size - 1`.
- Add `eps` to the standard deviation *before* dividing, so a group of identical rewards returns zeros rather than NaN.
- Raise a `ValueError` when `group_size` is not positive or does not divide the number of rewards.
- The result is a constant to the caller. It does not need to carry gradients.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why batch-wide normalization is wrong.** It mixes easy prompts with hard prompts and reintroduces exactly the prompt-difficulty bias this estimator exists to remove.

**Why the population standard deviation.** It is what the reference implementations use, and it keeps a group of size 1 well defined instead of returning NaN.

**Why a group replaces a critic.** PPO learns a value function to predict the expected return of a state, then subtracts it as a baseline. Sampling several completions for the same prompt gives the same variance reduction for free, because the other completions in the group are themselves samples of the return from that prompt. The cost is `group_size` times more generation; the saving is an entire value network, its optimizer state, and the instability of training it alongside the policy.""",
    "advisory_prerequisites": ["grpo_loss", "per_token_logprobs"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "If the rewards arrive as a flat vector laid out group-major, what single reshape puts every group on its own row, and which axis do the mean and standard deviation then run along? Once you have one mean per group, how do you subtract it from every member of that group without a loop — what does keepdim do for you? Think about what a group of all-identical rewards means: what is the numerator for every member, what is the denominator, and what should the answer be? Finally, is the sample standard deviation or the population standard deviation the right one when a group can have as few as one member?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Reshape the flat rewards to `(num_groups, group_size)`. Because the layout is group-major, a plain view is enough; no permutation is needed. Compute `mean(dim=-1, keepdim=True)` and `std(dim=-1, keepdim=True, unbiased=False)`. The unbiased flag matters: `torch.std` defaults to the sample standard deviation, which divides by `group_size - 1` and returns NaN for a group of size 1. Subtract the mean, divide by `std + eps`, then flatten back to `(N,)`. The constant-group case falls out on its own: the numerator is exactly zero and the denominator is `eps`, so the quotient is zero rather than NaN — but only if you add `eps` before dividing, never after. Validate `group_size` before reshaping so the error message names the real problem instead of surfacing as a shape error.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's create_training_episodes groups generations per prompt, computes the group mean and standard deviation, and normalizes each reward against its own group — the same estimator with the rollout bookkeeping attached.",
        "simple_GRPO applies the identical per-group z-score, then broadcasts one scalar advantage across every token of the corresponding response.",
        "DeepSeekMath introduced GRPO precisely to drop PPO's value network; the group of sampled completions supplies the baseline the critic used to provide.",
        "DAPO later showed that groups whose rewards are all-correct or all-wrong contribute no gradient, which is why production stacks filter those groups before this step rather than relying on the eps guard.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Removes the value network entirely, along with its parameters, optimizer state, and the instability of co-training it with the policy.",
            "The baseline is unbiased for the prompt it belongs to, so easy and hard prompts contribute comparable gradient magnitudes.",
            "Scale normalization makes one learning rate work across reward functions with very different ranges.",
        ],
        "cons": [
            "Costs group_size generations per prompt, so rollout compute grows linearly with the group size.",
            "Standardizing by the group standard deviation inflates the advantage of near-tied groups, which is the reason DAPO filters degenerate groups instead.",
            "A scalar advantage per response gives every token in that response the same credit, so it cannot localize which step of a long reasoning chain was good.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "create_training_episodes",
            "license": "MIT",
            "adapted": "The per-group reward mean and standard deviation normalization that produces one advantage per sampled response.",
            "simplifications": "Isolated as a pure tensor function over a flat group-major reward vector: no tokenizer, no generation grouping, no episode dictionary and no token broadcast.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated two-member group",
            "behavior": "rl.advantage",
            "code": r"""
import torch
# One group of two: mean 0.5, population std 0.5. Advantages are -1 and +1 (up to eps).
rewards = torch.tensor([0.0, 1.0])
out = {fn}(rewards, 2)
assert out.shape == (2,), f'expected (2,), got {tuple(out.shape)}'
assert torch.allclose(out, torch.tensor([-1.0, 1.0]), atol=1e-3), out
""",
        },
        {
            "name": "Groups are standardized independently",
            "behavior": "rl.advantage",
            "code": r"""
import torch
# Group A is centred at 0, group B at 100. Both must map to the same advantages.
rewards = torch.tensor([0.0, 1.0, 100.0, 101.0])
out = {fn}(rewards, 2)
assert torch.allclose(out[:2], out[2:], atol=1e-5), out
# A global normalization would make the two halves very different.
assert torch.allclose(out, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-3), out
""",
        },
        {
            "name": "Constant group returns zeros, not NaN",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A group with identical rewards produced NaN or inf. Add eps to the standard deviation before dividing, not after.",
            "code": r"""
import torch
rewards = torch.tensor([0.7, 0.7, 0.7, 0.7])
out = {fn}(rewards, 4)
assert torch.isfinite(out).all(), f'expected finite values, got {out}'
assert torch.allclose(out, torch.zeros(4), atol=1e-6), out
# Mixed batch: one degenerate group and one normal group.
mixed = torch.tensor([2.0, 2.0, 0.0, 1.0])
out = {fn}(mixed, 2)
assert torch.isfinite(out).all(), f'expected finite values, got {out}'
assert torch.allclose(out[:2], torch.zeros(2), atol=1e-6), out
assert torch.allclose(out[2:], torch.tensor([-1.0, 1.0]), atol=1e-3), out
""",
        },
        {
            "name": "Matches a per-group oracle on random batches",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "Values disagree with an independent per-group standardization. Check that you reshape group-major and reduce along the within-group axis.",
            "code": r"""
import torch
torch.manual_seed(0)
eps = 1e-4
for num_groups, group_size in ((5, 4), (3, 8), (7, 2)):
    rewards = torch.randn(num_groups * group_size) * 3.0
    out = {fn}(rewards, group_size)
    expected = torch.empty_like(rewards)
    for g in range(num_groups):
        chunk = rewards[g * group_size:(g + 1) * group_size]
        expected[g * group_size:(g + 1) * group_size] = (
            chunk - chunk.mean()
        ) / (chunk.std(unbiased=False) + eps)
    assert out.shape == expected.shape, f'expected {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-4), (out - expected).abs().max()
""",
        },
        {
            "name": "Uses the population standard deviation",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "The scale is off by the Bessel correction. Use unbiased=False so the divisor is group_size, not group_size - 1.",
            "code": r"""
import torch
rewards = torch.tensor([0.0, 0.0, 3.0])
out = {fn}(rewards, 3)
chunk = rewards
pop = (chunk - chunk.mean()) / (chunk.std(unbiased=False) + 1e-4)
sample = (chunk - chunk.mean()) / (chunk.std(unbiased=True) + 1e-4)
assert not torch.allclose(pop, sample, atol=1e-2), 'test fixture is degenerate'
assert torch.allclose(out, pop, atol=1e-3), f'expected {pop}, got {out}'
""",
        },
        {
            "name": "Each group sums to approximately zero",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "Per-group advantages do not sum to zero. The mean subtracted must be the group's own mean.",
            "code": r"""
import torch
torch.manual_seed(3)
group_size = 6
rewards = torch.randn(4 * group_size)
out = {fn}(rewards, group_size)
sums = out.reshape(4, group_size).sum(dim=-1)
assert torch.allclose(sums, torch.zeros(4), atol=1e-3), sums
""",
        },
        {
            "name": "Invariant to a per-group shift, sensitive to a per-group scale",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "Shifting one group's rewards by a constant changed another group's advantages, or the standardization is not scale-normalizing.",
            "code": r"""
import torch
torch.manual_seed(4)
group_size = 4
rewards = torch.randn(3 * group_size)
base = {fn}(rewards, group_size)
bumped = rewards.clone()
bumped[:group_size] += 50.0
after = {fn}(bumped, group_size)
assert torch.allclose(base[group_size:], after[group_size:], atol=1e-5), 'a shift leaked across groups'
assert torch.allclose(base[:group_size], after[:group_size], atol=1e-3), 'shift invariance broken'
scaled = rewards.clone()
scaled[:group_size] *= 10.0
assert torch.allclose({fn}(scaled, group_size)[:group_size], base[:group_size], atol=1e-3), 'scale normalization broken'
""",
        },
        {
            "name": "Rejects an invalid group size",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An invalid group_size was accepted. Raise ValueError when it is not positive or does not divide the number of rewards.",
            "code": r"""
import torch
rewards = torch.randn(6)
for bad in (0, -2, 4, 5):
    try:
        {fn}(rewards, bad)
    except ValueError:
        continue
    raise AssertionError(f'group_size={bad} should raise ValueError')
""",
        },
        {
            "name": "Single-member groups and dtype preservation",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "group_size=1 failed or the dtype changed. A group of one has zero deviation, so every advantage is zero.",
            "code": r"""
import torch
out = {fn}(torch.tensor([5.0, -3.0, 0.0]), 1)
assert torch.isfinite(out).all(), f'expected finite values, got {out}'
assert torch.allclose(out, torch.zeros(3), atol=1e-6), out
rewards = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
out = {fn}(rewards, 2)
assert out.dtype == torch.float64, f'expected float64, got {out.dtype}'
assert out.shape == (4,), f'expected (4,), got {tuple(out.shape)}'
""",
        },
    ],
    "solution": '''import torch


def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"group_size {group_size} does not divide {rewards.numel()} rewards"
        )
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)
''',
}
