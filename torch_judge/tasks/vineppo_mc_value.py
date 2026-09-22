"""VinePPO Monte Carlo values — per-step credit assignment without a critic."""

TASK = {
    "title": "VinePPO Monte Carlo Value Estimation",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "vineppo_mc_value",
    "description_en": r"""Estimate the value of each intermediate state by Monte Carlo rollout, then turn those values into per-step advantages.

**Signature:** `vineppo_mc_value(rollout_rewards, final_reward) -> tuple`

**Parameters:**
- `rollout_rewards` — float tensor of shape `(S, K)`. For each of `S` intermediate states along one trajectory, the terminal rewards of `K` independent continuations sampled from that state.
- `final_reward` — float scalar tensor. The terminal reward of the trajectory itself, which serves as the value of the state after the last step.

**Returns:** a tuple `(values, advantages)`, both float tensors of shape `(S,)`.

    values[s]     = mean over the K continuations of state s
    next_value[s] = values[s + 1] for s < S - 1, and final_reward for s = S - 1
    advantages[s] = next_value[s] - values[s]

**Constraints:**
- Average across the `K` continuations of one state, never across states.
- The advantage is `next_value - value`, in that order.
- `final_reward` is the bootstrap for the last state. Do not leave `advantages[S - 1]` at zero.
- Both returned tensors have shape `(S,)`. Do not reduce to a scalar and do not broadcast one advantage across all states.
- Preserve the dtype and device of `rollout_rewards`.
- Raise a `ValueError` when `rollout_rewards` is not two-dimensional or has zero states or zero continuations.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the axis matters.** Averaging the wrong way produces a tensor of the right shape whose entries are the average difficulty of the `k`-th continuation across the whole trajectory, which is meaningless.

**What this buys.** GRPO assigns one scalar advantage to an entire response, so every token of a correct answer is rewarded equally, including the tokens of a wrong turn the model later recovered from. VinePPO instead asks, at each step, whether taking that step raised or lowered the probability of eventually succeeding. A step that moved the trajectory from a state worth 0.3 to one worth 0.8 gets a large positive advantage; a step that changed nothing gets approximately zero. That is genuine per-step credit assignment, obtained without training a critic.

**What it costs.** `S` times `K` extra generations per trajectory — the most expensive advantage estimator in this path by a wide margin, which is why it is usually applied to a subset of states rather than to every token.

**Why the estimates are noisy.** Each value is an average of `K` samples of a Bernoulli-like outcome, so its standard error falls only as the square root of `K`. With a small `K`, two adjacent states can easily produce an advantage of the wrong sign purely by sampling noise, and the advantage is a difference of two noisy estimates, so its variance is roughly twice that of a single value.""",
    "advisory_prerequisites": ["group_relative_advantage", "gae_advantage"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "The input is a matrix and you need a vector, so one axis disappears — which one, and what does an entry of the surviving vector mean in words? For the advantages you need each state paired with the one after it: what slice gives you 'every value except the first', and what do you attach to its end so the lengths still match? Check the sign by imagining a step that moves the trajectory from a hopeless state to a promising one — should its advantage be positive or negative, and does your expression agree? Finally, what is the advantage of the last state when the trajectory succeeded but its continuations mostly failed?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Two lines of real work. `values = rollout_rewards.mean(dim=-1)` reduces across continuations; using `dim=0` instead is the mutation that keeps the shape only when S and K happen to be equal, and is wrong in every case. For the next-state values, build the shifted tensor explicitly: `torch.cat([values[1:], final_reward.reshape(1)])` gives a `(S,)` tensor whose last entry is the bootstrap. Reshaping the scalar is what lets the concatenation work regardless of whether the caller passed a zero-dimensional tensor or a one-element one. Then `advantages = next_value - values`; the order matters and the reverse is a sign flip that trains the policy to avoid progress. Validate the shape before reducing, since a one-dimensional input would otherwise average to a single number and fail much later with a confusing error.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's create_vineppo_training_episodes samples VINEPPO_K continuations from each intermediate state, averages their rewards into a value, and differences adjacent values into token-level advantages.",
        "VinePPO was proposed as a critic-free alternative to PPO for LLM reasoning, arguing that a learned value head is a poor estimator on reasoning traces and that Monte Carlo estimates are better despite their cost.",
        "GRPO's group-relative advantage is the degenerate case where S is 1: one value for the whole response, and therefore one advantage for every token in it.",
        "gae_advantage solves the same problem with a learned critic and a recursion, trading Monte Carlo variance for critic bias.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Real per-step credit assignment, so a recovery from a wrong turn is not rewarded as if the wrong turn had been correct.",
            "No critic to train, so no critic bias and no second network to tune or keep in memory.",
            "The estimator is unbiased by construction; only its variance, not its centre, depends on K.",
        ],
        "cons": [
            "Costs S times K extra generations per trajectory, by far the most expensive estimator in this path.",
            "Standard error falls only as the square root of K, so adjacent advantages can take the wrong sign under sampling noise.",
            "The advantage is a difference of two noisy estimates, roughly doubling the variance relative to a single value.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "create_vineppo_training_episodes",
            "license": "MIT",
            "adapted": "The Monte Carlo value estimate per intermediate state and the adjacent-value difference that turns it into a per-step advantage.",
            "simplifications": "Takes the rollout rewards as a matrix instead of generating them: no inference engine, no state extraction from a trajectory and no token-level broadcast.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2410.01679",
            "section": "3 VinePPO, Monte Carlo advantage estimation",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated two-state trajectory",
            "behavior": "rl.trajectory",
            "code": r"""
import torch
# State 0 succeeds in half its continuations, state 1 in all of them.
rollout_rewards = torch.tensor([[1.0, 0.0],
                                [1.0, 1.0]])
values, advantages = {fn}(rollout_rewards, torch.tensor(1.0))
assert values.shape == (2,) and advantages.shape == (2,), (values.shape, advantages.shape)
assert torch.allclose(values, torch.tensor([0.5, 1.0]), atol=1e-6), values
# A[0] = 1.0 - 0.5 = 0.5 ; A[1] = final_reward - 1.0 = 0.0
assert torch.allclose(advantages, torch.tensor([0.5, 0.0]), atol=1e-6), advantages
""",
        },
        {
            "name": "A step that improves the state earns a positive advantage",
            "behavior": "rl.trajectory",
            "code": r"""
import torch
# The trajectory moves from a hopeless state to a promising one, then stalls.
rollout_rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0]])
values, advantages = {fn}(rollout_rewards, torch.tensor(1.0))
assert advantages[0] > 0, f'the improving step should be positive, got {advantages[0].item()}'
assert advantages[1] < 0, f'the regressing step should be negative, got {advantages[1].item()}'
""",
        },
        {
            "name": "Averages across continuations, not across states",
            "visibility": "unshown",
            "behavior": "rl.trajectory",
            "failure_message": "The mean was taken along the wrong axis. Each value is the average over the K continuations of one state.",
            "code": r"""
import torch
# A square input, so a wrong axis still produces the right shape.
rollout_rewards = torch.tensor([[1.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0]])
values, _ = {fn}(rollout_rewards, torch.tensor(0.0))
assert torch.allclose(values, torch.tensor([1.0, 0.0, 1.0 / 3.0]), atol=1e-5), values
wrong_axis = rollout_rewards.mean(dim=0)
assert not torch.allclose(values, wrong_axis, atol=1e-3), 'this is the mean across states'
""",
        },
        {
            "name": "The final reward bootstraps the last state",
            "visibility": "unshown",
            "behavior": "rl.trajectory",
            "failure_message": "final_reward did not enter the last advantage. It is the value of the state after the last step.",
            "code": r"""
import torch
rollout_rewards = torch.tensor([[0.5, 0.5],
                                [0.25, 0.25]])
_, high = {fn}(rollout_rewards, torch.tensor(1.0))
_, low = {fn}(rollout_rewards, torch.tensor(0.0))
assert torch.allclose(high[-1], torch.tensor(0.75), atol=1e-6), f'expected 0.75, got {high[-1].item()}'
assert torch.allclose(low[-1], torch.tensor(-0.25), atol=1e-6), f'expected -0.25, got {low[-1].item()}'
# Earlier advantages must not depend on the bootstrap.
assert torch.allclose(high[:-1], low[:-1], atol=1e-6), 'the bootstrap leaked into earlier states'
""",
        },
        {
            "name": "Advantages are differences of adjacent values",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "The advantage is not next_value minus value. A sign flip here trains the policy to avoid progress.",
            "code": r"""
import torch
torch.manual_seed(0)
for S, K in ((5, 3), (2, 8), (7, 4)):
    rollout_rewards = torch.rand(S, K)
    final = torch.tensor(0.6)
    values, advantages = {fn}(rollout_rewards, final)
    expected_values = rollout_rewards.mean(dim=-1)
    next_value = torch.cat([expected_values[1:], final.reshape(1)])
    expected_adv = next_value - expected_values
    assert torch.allclose(values, expected_values, atol=1e-6), (values - expected_values).abs().max()
    assert torch.allclose(advantages, expected_adv, atol=1e-6), (advantages - expected_adv).abs().max()
""",
        },
        {
            "name": "Per-state advantages, not one broadcast value",
            "visibility": "unshown",
            "behavior": "rl.trajectory",
            "failure_message": "Every state received the same advantage. VinePPO's whole point is that different steps get different credit.",
            "code": r"""
import torch
rollout_rewards = torch.tensor([[0.0, 0.0],
                                [0.5, 0.5],
                                [1.0, 1.0]])
_, advantages = {fn}(rollout_rewards, torch.tensor(1.0))
assert advantages.shape == (3,), f'expected (3,), got {tuple(advantages.shape)}'
assert len(set(round(float(a), 5) for a in advantages)) > 1, f'all advantages are identical: {advantages}'
assert torch.allclose(advantages, torch.tensor([0.5, 0.5, 0.0]), atol=1e-6), advantages
""",
        },
        {
            "name": "Telescopes to the overall improvement",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The advantages do not sum to final_reward minus the first value. Adjacent differences must telescope.",
            "code": r"""
import torch
torch.manual_seed(1)
for S, K in ((6, 5), (3, 2)):
    rollout_rewards = torch.rand(S, K)
    final = torch.tensor(0.9)
    values, advantages = {fn}(rollout_rewards, final)
    total = advantages.sum()
    expected = final - values[0]
    assert torch.allclose(total, expected, atol=1e-5), f'{total.item()} vs {expected.item()}'
""",
        },
        {
            "name": "Single state and single continuation",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A boundary shape failed. S = 1 and K = 1 must still return two tensors of shape (S,).",
            "code": r"""
import torch
values, advantages = {fn}(torch.tensor([[0.25]]), torch.tensor(1.0))
assert values.shape == (1,) and advantages.shape == (1,), (values.shape, advantages.shape)
assert torch.allclose(values, torch.tensor([0.25]), atol=1e-6), values
assert torch.allclose(advantages, torch.tensor([0.75]), atol=1e-6), advantages
values, advantages = {fn}(torch.tensor([[1.0], [0.0], [1.0]]), torch.tensor(0.0))
assert values.shape == (3,) and advantages.shape == (3,), (values.shape, advantages.shape)
assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0]), atol=1e-6), advantages
""",
        },
        {
            "name": "Rejects a malformed rollout matrix",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "A malformed input was accepted. rollout_rewards must be two-dimensional with at least one state and one continuation.",
            "code": r"""
import torch
bad = (torch.rand(4), torch.rand(2, 3, 4), torch.rand(0, 3), torch.rand(3, 0))
for rollout_rewards in bad:
    try:
        {fn}(rollout_rewards, torch.tensor(0.5))
    except ValueError:
        continue
    raise AssertionError(f'shape {tuple(rollout_rewards.shape)} should raise ValueError')
""",
        },
        {
            "name": "Preserves dtype and device",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The output dtype or device does not follow rollout_rewards.",
            "code": r"""
import torch
rollout_rewards = torch.rand(3, 4, dtype=torch.float64)
values, advantages = {fn}(rollout_rewards, torch.tensor(0.5, dtype=torch.float64))
assert values.dtype == torch.float64, f'expected float64, got {values.dtype}'
assert advantages.dtype == torch.float64, f'expected float64, got {advantages.dtype}'
assert values.device == rollout_rewards.device, values.device
""",
        },
    ],
    "solution": '''import torch


def vineppo_mc_value(rollout_rewards, final_reward):
    if rollout_rewards.ndim != 2:
        raise ValueError(
            f"rollout_rewards must be two-dimensional, got {tuple(rollout_rewards.shape)}"
        )
    states, continuations = rollout_rewards.shape
    if states == 0 or continuations == 0:
        raise ValueError(
            f"need at least one state and one continuation, got {tuple(rollout_rewards.shape)}"
        )

    values = rollout_rewards.mean(dim=-1)
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, next_value - values
''',
}
