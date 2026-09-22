"""Generalized advantage estimation — the backward recursion that feeds PPO's policy loss."""

TASK = {
    "title": "Generalized Advantage Estimation",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "gae_advantage",
    "description_en": r"""Implement generalized advantage estimation, the interpolation between a one-step TD residual and a full Monte Carlo return.

**Signature:** `gae_advantage(rewards, values, dones, gamma=0.99, lam=0.95) -> Tensor`

**Parameters:**
- `rewards` — float tensor of shape `(B, T)`. The reward received at each step.
- `values` — float tensor of shape `(B, T + 1)`. Critic predictions. The extra final column is the bootstrap value of the state after the last step.
- `dones` — float or boolean tensor of shape `(B, T)`. 1 or True when step `t` ended the episode.
- `gamma` — discount factor in the closed interval from 0 to 1.
- `lam` — the GAE parameter in the closed interval from 0 to 1.

**Returns:** float tensor of shape `(B, T)`.

Compute the TD residual at each step, then accumulate it backwards, with `A[T]` taken to be zero:

    not_done[t] = 1 - dones[t]
    delta[t]    = rewards[t] + gamma * values[t + 1] * not_done[t] - values[t]
    A[t]        = delta[t] + gamma * lam * not_done[t] * A[t + 1]

**Constraints:**
- Iterate from `t = T - 1` down to 0. The recursion runs backwards.
- Apply `not_done` in **both** places: inside `delta` and inside the recursion.
- Keep the `gamma * lam` factor on the recursion term.
- `values` must have exactly one more column than `rewards`.
- Preserve the dtype and device of `rewards`. Do not convert through Python floats.
- Raise a `ValueError` when the shapes disagree or when `gamma` or `lam` falls outside the closed unit interval.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**A note on the name.** The id `gae` is already taken by the Graph Autoencoder exercise, which is unrelated. This contract is the reinforcement-learning estimator.

**Why `not_done` appears twice.** The two uses do different jobs and both are required. In `delta` it stops the estimator from bootstrapping across an episode boundary: once an episode has ended, the value of the next state belongs to a different trajectory and must not leak in. In the recursion it stops the accumulated advantage from flowing backwards past that boundary. Dropping either one produces an estimator that looks correct on a single uninterrupted trajectory and is silently wrong on any batch that contains a boundary.

**What `lam` controls.** At `lam = 0` the recursion collapses to `A[t] = delta[t]`, a one-step TD residual: low variance, but biased by whatever error the critic carries. At `lam = 1` it becomes the discounted sum of all future residuals, which telescopes to the Monte Carlo return minus the baseline: unbiased, but with variance that grows with the horizon. Values near 0.95 are the usual compromise. The estimator is an interpolation, not an approximation of one.""",
    "advisory_prerequisites": ["ppo_value_loss", "ppo_clipped_policy_loss"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Look at the recursion and ask which index it needs before it can compute index t — that answer fixes the loop direction, and there is no way to vectorize it away along the time axis. Why does `values` have T + 1 columns rather than T; what would you have to invent if it had only T? Take a two-step episode where the first step is terminal and write out both deltas by hand: which term must vanish, and what would leak in if it did not? Finally, check your recursion at lam = 1 against the plain discounted sum of residuals — do they agree?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Compute the whole delta tensor first, vectorized: `not_done = 1 - dones` cast to the reward dtype, then `rewards + gamma * values[:, 1:] * not_done - values[:, :-1]`. The slicing is what uses the extra column, and it is also where an off-by-one hides, so check that both slices have width T. Then allocate an output with `torch.zeros_like(rewards)` and run a Python loop over `t in reversed(range(T))`, carrying a running tensor of shape `(B,)` that starts at zero: `running = delta[:, t] + gamma * lam * not_done[:, t] * running`, and store it. The loop over time is unavoidable because each step depends on the next; the batch dimension stays vectorized inside it. Two mutations pass every shape check: iterating forwards, and dropping `not_done` from the recursion term while keeping it in delta. Test the first with any multi-step trajectory and the second with a batch that contains a mid-sequence done flag.",
        },
    ],
    "model_connections": [
        "OpenRLHF computes GAE with this exact backward recursion over the rollout buffer and feeds the result to PolicyLoss, while the matching returns go to ValueLoss.",
        "verl implements the same recursion and exposes gamma and lam as trainer configuration, since they are the main knobs of the PPO branch.",
        "GRPO replaces this entire computation with a group of sampled completions, which is why the critic-free branch of this path needs neither values nor dones.",
        "VinePPO estimates the same per-step advantage with Monte Carlo rollouts from intermediate states rather than with a learned critic, trading compute for critic bias.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One parameter interpolates the whole bias-variance range between a TD residual and a Monte Carlo return.",
            "The batch dimension stays vectorized, so the unavoidable time loop costs T steps regardless of batch size.",
            "Handling episode boundaries explicitly lets one batch hold trajectories of different lengths.",
        ],
        "cons": [
            "It needs a trained critic, and the advantage inherits whatever bias that critic carries.",
            "The time loop cannot be parallelized, so a long horizon serializes the computation.",
            "Two more constants to tune, and their right values interact with the reward scale and the episode length.",
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
            "adapted": "The per-step advantage framing that this estimator produces; the referenced function obtains it by Monte Carlo rollout instead of by a critic recursion.",
            "simplifications": "Pure tensor recursion over rewards, values and done flags: no critic, no rollout and no episode bookkeeping.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1506.02438",
            "section": "3 Advantage Function Estimation, equation 16",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated two-step trajectory",
            "behavior": "rl.advantage",
            "code": r"""
import torch
# gamma = 1, lam = 1, no terminal states, zero critic: A[t] is the remaining reward sum.
rewards = torch.tensor([[1.0, 2.0]])
values = torch.zeros(1, 3)
dones = torch.zeros(1, 2)
out = {fn}(rewards, values, dones, gamma=1.0, lam=1.0)
assert out.shape == (1, 2), f'expected (1, 2), got {tuple(out.shape)}'
assert torch.allclose(out, torch.tensor([[3.0, 2.0]]), atol=1e-6), out
""",
        },
        {
            "name": "lam = 0 collapses to the TD residual",
            "behavior": "rl.advantage",
            "code": r"""
import torch
rewards = torch.tensor([[1.0, 1.0, 1.0]])
values = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
dones = torch.zeros(1, 3)
gamma = 0.9
out = {fn}(rewards, values, dones, gamma=gamma, lam=0.0)
delta = rewards + gamma * values[:, 1:] - values[:, :-1]
assert torch.allclose(out, delta, atol=1e-6), f'{out} vs {delta}'
""",
        },
        {
            "name": "Runs the recursion backwards",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "The accumulation runs forwards. Each advantage depends on the next step, so iterate from the last index down to the first.",
            "code": r"""
import torch
# A reward only at the final step. With gamma = lam = 1 every earlier advantage sees it.
rewards = torch.tensor([[0.0, 0.0, 5.0]])
values = torch.zeros(1, 4)
dones = torch.zeros(1, 3)
out = {fn}(rewards, values, dones, gamma=1.0, lam=1.0)
assert torch.allclose(out, torch.tensor([[5.0, 5.0, 5.0]]), atol=1e-6), out
# A forward accumulation would give [0, 0, 5].
assert out[0, 0] != 0.0, 'the final reward did not propagate backwards'
""",
        },
        {
            "name": "Does not bootstrap across a terminal step",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The next state's value leaked across an episode boundary. Multiply the bootstrap term by 1 - dones.",
            "code": r"""
import torch
# Step 0 is terminal. Its delta must ignore values[1] entirely.
rewards = torch.tensor([[1.0, 0.0]])
values = torch.tensor([[0.0, 100.0, 0.0]])
dones = torch.tensor([[1.0, 0.0]])
out = {fn}(rewards, values, dones, gamma=0.9, lam=0.95)
assert torch.allclose(out[0, 0], torch.tensor(1.0), atol=1e-5), f'expected 1.0, got {out[0, 0].item()}'
""",
        },
        {
            "name": "Does not accumulate across a terminal step",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The advantage flowed backwards past an episode boundary. The recursion term also needs the 1 - dones factor.",
            "code": r"""
import torch
# Step 0 is terminal, so a large reward at step 1 belongs to a different episode
# and must not appear in A[0].
rewards = torch.tensor([[0.0, 50.0]])
values = torch.zeros(1, 3)
dones = torch.tensor([[1.0, 0.0]])
out = {fn}(rewards, values, dones, gamma=1.0, lam=1.0)
assert torch.allclose(out[0, 0], torch.tensor(0.0), atol=1e-5), f'expected 0.0, got {out[0, 0].item()}'
assert torch.allclose(out[0, 1], torch.tensor(50.0), atol=1e-5), out
""",
        },
        {
            "name": "Matches a loop-based oracle",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "Values disagree with an independent step-by-step evaluation of the same recursion.",
            "code": r"""
import torch
torch.manual_seed(0)
for B, T in ((3, 6), (2, 9)):
    rewards = torch.randn(B, T)
    values = torch.randn(B, T + 1)
    dones = (torch.rand(B, T) > 0.75).float()
    gamma, lam = 0.97, 0.9
    expected = torch.zeros(B, T)
    for b in range(B):
        running = 0.0
        for t in reversed(range(T)):
            nd = 1.0 - float(dones[b, t])
            delta = float(rewards[b, t]) + gamma * float(values[b, t + 1]) * nd - float(values[b, t])
            running = delta + gamma * lam * nd * running
            expected[b, t] = running
    out = {fn}(rewards, values, dones, gamma=gamma, lam=lam)
    assert out.shape == expected.shape, f'expected {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-4), (out - expected).abs().max()
""",
        },
        {
            "name": "lam = 1 telescopes to the Monte Carlo advantage",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "At lam = 1 the estimator does not equal the discounted return minus the baseline. Check the gamma factor inside the recursion.",
            "code": r"""
import torch
torch.manual_seed(1)
B, T = 2, 5
rewards = torch.randn(B, T)
values = torch.randn(B, T + 1)
dones = torch.zeros(B, T)
gamma = 0.9
out = {fn}(rewards, values, dones, gamma=gamma, lam=1.0)
# Discounted return with bootstrap, minus the baseline.
expected = torch.zeros(B, T)
for b in range(B):
    running = float(values[b, T])
    for t in reversed(range(T)):
        running = float(rewards[b, t]) + gamma * running
        expected[b, t] = running - float(values[b, t])
assert torch.allclose(out, expected, atol=1e-4), (out - expected).abs().max()
""",
        },
        {
            "name": "Accepts boolean done flags and a single step",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A boolean dones tensor or a single-step trajectory failed. Cast dones to the reward dtype and handle T = 1.",
            "code": r"""
import torch
out = {fn}(torch.tensor([[2.0]]), torch.tensor([[0.0, 7.0]]), torch.tensor([[False]]), gamma=0.5, lam=0.9)
assert out.shape == (1, 1), f'expected (1, 1), got {tuple(out.shape)}'
assert torch.allclose(out, torch.tensor([[2.0 + 0.5 * 7.0]]), atol=1e-6), out
terminal = {fn}(torch.tensor([[2.0]]), torch.tensor([[0.0, 7.0]]), torch.tensor([[True]]), gamma=0.5, lam=0.9)
assert torch.allclose(terminal, torch.tensor([[2.0]]), atol=1e-6), terminal
""",
        },
        {
            "name": "Rejects a mismatched value horizon and bad constants",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An inconsistent shape or an out-of-range gamma or lam was accepted. values must have exactly one more column than rewards.",
            "code": r"""
import torch
rewards = torch.randn(2, 4)
values = torch.randn(2, 5)
dones = torch.zeros(2, 4)
bad = ((rewards, torch.randn(2, 4), dones, 0.99, 0.95),
       (rewards, torch.randn(2, 7), dones, 0.99, 0.95),
       (rewards, values, torch.zeros(2, 3), 0.99, 0.95),
       (rewards, values, dones, 1.5, 0.95),
       (rewards, values, dones, -0.1, 0.95),
       (rewards, values, dones, 0.99, 1.5),
       (rewards, values, dones, 0.99, -0.1))
for r, v, d, g, l in bad:
    try:
        {fn}(r, v, d, gamma=g, lam=l)
    except ValueError:
        continue
    raise AssertionError(f'gamma={g} lam={l} shapes {tuple(v.shape)}/{tuple(d.shape)} should raise ValueError')
""",
        },
        {
            "name": "Preserves dtype and device",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The output dtype or device does not follow rewards. Do not convert through Python floats or numpy.",
            "code": r"""
import torch
rewards = torch.randn(2, 3, dtype=torch.float64)
values = torch.randn(2, 4, dtype=torch.float64)
dones = torch.zeros(2, 3, dtype=torch.float64)
out = {fn}(rewards, values, dones, gamma=0.9, lam=0.9)
assert out.dtype == torch.float64, f'expected float64, got {out.dtype}'
assert out.device == rewards.device, f'expected {rewards.device}, got {out.device}'
assert out.shape == (2, 3), f'expected (2, 3), got {tuple(out.shape)}'
""",
        },
    ],
    "solution": '''import torch


def gae_advantage(rewards, values, dones, gamma=0.99, lam=0.95):
    if rewards.shape != dones.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)} and dones {tuple(dones.shape)} must match"
        )
    horizon = rewards.shape[-1]
    if values.shape[:-1] != rewards.shape[:-1] or values.shape[-1] != horizon + 1:
        raise ValueError(
            f"values {tuple(values.shape)} must have exactly one more column than "
            f"rewards {tuple(rewards.shape)}"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1], got {lam}")

    not_done = 1.0 - dones.to(rewards.dtype)
    deltas = rewards + gamma * values[..., 1:] * not_done - values[..., :-1]

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + gamma * lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
''',
}
