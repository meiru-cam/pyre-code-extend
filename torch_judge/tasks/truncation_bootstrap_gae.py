"""GAE for robot rollouts: bootstrap through truncation from the true final observation, and run over action chunks."""

from ._interview import interview

# An independent per-element evaluation with Python floats, used by several unshown cases.
_LOOP_ORACLE = r"""
import torch

def loop_gae(rewards, values, next_value, terminated, truncated, final_values, gamma, lam):
    T, B = rewards.shape
    adv = torch.zeros(T, B, dtype=torch.float64)
    for b in range(B):
        running = 0.0
        for t in range(T - 1, -1, -1):
            term = bool(terminated[t, b])
            trunc = bool(truncated[t, b])
            if term:
                boot = 0.0
            elif trunc:
                boot = float(final_values[t, b])
            elif t == T - 1:
                boot = float(next_value[b])
            else:
                boot = float(values[t + 1, b])
            cont = 0.0 if (term or trunc) else 1.0
            delta = float(rewards[t, b]) + gamma * boot - float(values[t, b])
            running = delta + gamma * lam * cont * running
            adv[t, b] = running
    return adv, adv + values.double()

def random_rollout(seed, T, B):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(T, B, generator=g, dtype=torch.float64)
    values = torch.randn(T, B, generator=g, dtype=torch.float64)
    next_value = torch.randn(B, generator=g, dtype=torch.float64)
    terminated = torch.rand(T, B, generator=g) < 0.2
    truncated = torch.rand(T, B, generator=g) < 0.2
    # Real final values only where the episode was truncated; large junk elsewhere.
    final_values = torch.randn(T, B, generator=g, dtype=torch.float64)
    junk = 1000.0 + 100.0 * torch.rand(T, B, generator=g, dtype=torch.float64)
    final_values = torch.where(truncated & ~terminated, final_values, junk)
    return rewards, values, next_value, terminated, truncated, final_values
"""

TASK = {
    "title": "GAE with Truncation Bootstrapping for Robot Rollouts",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "robot_gae",
    "description_en": r"""Implement generalized advantage estimation for a vectorized robot environment that auto-resets, where a time-limit truncation must be bootstrapped from the value of the true final observation, and optionally run it over action chunks.

**Signature:** `robot_gae(rewards, values, next_value, terminated, truncated, final_values, gamma=0.99, lam=0.95, valid_mask=None) -> (advantages, returns)`

The layout is time first, then environments. Row `t` is one call to a vector env's `step` (or one executed action chunk).

**Parameters:**
- `rewards` — float tensor `(T, B)` of per-step rewards, or `(T, B, H)` of per-env-step rewards inside each chunk of `H` actions.
- `values` — float tensor `(T, B)`. `values[t]` is the critic's value of the observation the action at row `t` was chosen from.
- `next_value` — float tensor `(B,)`. The value of the observation returned by the last row.
- `terminated`, `truncated` — boolean (or 0/1 float) tensors `(T, B)`: the flags returned at row `t`. Both can be set at once.
- `final_values` — float tensor `(T, B)`. Where `truncated[t]` is set, the value of the true last observation of the episode that ended at row `t`. Every other entry is an arbitrary finite number and must be ignored.
- `gamma`, `lam` — numbers in the closed interval from 0 to 1.
- `valid_mask` — boolean tensor `(T, B, H)`, required exactly when `rewards` is 3-D.

**Returns:** `advantages` and `returns`, both float `(T, B)` in the dtype of `rewards`, with `returns = advantages + values`.

The environments auto-reset: after an episode ends at row `t`, `values[t + 1]` is the value of the **reset** observation of a new episode. Compute, with `A[T]` taken to be zero:

    next_v[t] = next_value     if t == T - 1
                values[t + 1]  otherwise
    boot[t]   = 0              if terminated[t]
                final_values[t] if truncated[t] and not terminated[t]
                next_v[t]      otherwise
    cont[t]   = 0 if terminated[t] or truncated[t], else 1
    delta[t]  = r[t] + gamma * boot[t] - values[t]
    A[t]      = delta[t] + gamma * lam * cont[t] * A[t + 1]

**Chunk level.** When `rewards` is `(T, B, H)`, first reduce it to `r[t] = sum over h of rewards[t, :, h] * valid_mask[t, :, h]`, then run the same recursion over rows. Row `t` is one chunk: `gamma` is applied once per chunk, not once per env step, and the rewards inside a chunk are summed without discounting. `terminated` and `truncated` are already folded to chunk level. Rewards where `valid_mask` is False are arbitrary finite numbers.

**Constraints:**
- Termination wins: when both flags are set, the bootstrap is 0.
- A truncation bootstraps from `final_values[t]`, never from `values[t + 1]` and never from 0.
- The recursion is cut at both termination and truncation.
- Iterate from the last row to the first; keep the env axis vectorized.
- Do not modify the inputs.
- Raise a `ValueError` when a shape disagrees, when `valid_mask` is missing for 3-D rewards or given for 2-D rewards, or when `gamma` or `lam` is outside the closed unit interval.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**How this extends `gae_advantage`.** That exercise has a single `dones` flag that zeroes both the bootstrap and the recursion. That is right for a true terminal state and wrong for a time limit. Gymnasium defines `terminated` as reaching a terminal state of the task's MDP and `truncated` as a condition outside the MDP, typically a time limit. After a truncation the state still has a future, so the TD target should still include `gamma * V(s_next)`. Pardo et al. call this partial-episode bootstrapping (Time Limits in Reinforcement Learning, section 3). Their section 2 covers the other case: when the time limit is part of the task, they argue for time-awareness, adding the remaining time to the input, instead.

**Why a separate `final_values` tensor.** A vector env that auto-resets returns the reset observation of the next episode in place of the true last one; Gymnasium's same-step mode puts the true one in `info["final_obs"]`, and ManiSkill calls it `final_observation`. So `values[t + 1]` is the wrong state to bootstrap from, and the true final state's value has to be computed separately. ManiSkill's PPO baseline stores it as `final_values[t]` and forms `real_next_values = next_not_done * nextvalues + final_values[t]`, cutting the recursion at every done flag. The trace has to be cut at a truncation even though the bootstrap is kept: the next row belongs to a different episode.

**Where this contract differs from that baseline.** At the pinned commit ManiSkill fills `final_values` for every env that finished on a step, terminated or truncated, so its baseline also bootstraps through terminations; a comment refers to a `bootstrap_at_done` option that is no longer an argument. This exercise follows the Gymnasium definitions and bootstraps only through truncation.

**Chunk level.** A VLA that outputs `H` actions at a time can be trained with one decision per chunk. RLinf's `chunk_level` reward type sums the chunk's rewards along the chunk axis, takes the max of the done flags, and then runs GAE with one `gamma` per chunk. RLinf's GAE takes a single done flag and documents that it does not support auto-reset, which is why the step-level form here follows ManiSkill.""",
    "advisory_prerequisites": ["gae_advantage", "vector_autoreset", "chunk_step_executor"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "After an auto-reset, which episode does the observation behind values[t + 1] belong to? If a robot is stopped by a time limit, is the value of its final state zero? Does a truncation change whether A[t + 1] may flow into A[t]? When both flags are set at once, which one decides the bootstrap? At chunk level, how many discount factors separate two consecutive rows?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "If rewards are 3-D, multiply by the valid mask cast to the reward dtype and sum over the last axis. Build the next-value tensor by shifting values up one row and appending next_value. Select the bootstrap with two torch.where calls: final_values where truncated, then zero where terminated, so termination wins. Build cont as one minus the OR of the two flags. Compute every delta at once, then loop backwards over rows with a running tensor of shape (B,), multiplying the carried term by gamma, lam and cont. Return the advantages and advantages plus values.",
        },
    ],
    "model_connections": [
        "ManiSkill's PPO baseline computes V(final_observation) for every env that finished on a step and adds it as final_values[t] in its GAE loop.",
        "Gymnasium's vector envs default to next-step autoreset; in same-step mode the true last observation is returned in info['final_obs'], the input this exercise's final_values is computed from.",
        "RLinf reduces per-step rewards to one reward per action chunk for its chunk_level reward type and runs GAE over chunks.",
        "gae_advantage is the single-flag special case: with no truncations the two estimators compute the same numbers, though gae_advantage is batch-first (B, T) with values (B, T + 1) while this exercise is time-first (T, B) with a separate next_value.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Time limits stop acting as fake failures, so the value of states near the limit is not pulled toward zero.",
            "Keeping the recursion cut at every boundary lets an auto-resetting vector env fill one rollout buffer with many episodes.",
            "Chunk-level GAE matches the number of credit-assignment steps to the number of policy decisions.",
        ],
        "cons": [
            "It costs an extra critic call on every final observation.",
            "If the time limit is part of the task, bootstrapping through it estimates a different objective; Pardo et al. suggest time-aware inputs for that case.",
            "Chunk-level rewards drop within-chunk discounting and timing, so two chunks that reach the goal at different steps can get the same reward.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "examples/baselines/ppo/ppo.py",
            "symbol": "GAE block of the training loop (final_values, real_next_values, lastgaelam)",
            "license": "Apache-2.0",
            "adapted": "The step-level recursion: a separate final_values term for finished envs, next values from the following row or the bootstrap value, and the trace cut at every done flag.",
            "simplifications": "Upstream fills final_values for every finished env, terminated or truncated, so it bootstraps through terminations too; the exercise bootstraps only through truncation and ignores final_values elsewhere. "
                               "Upstream stores the done flag of step t - 1 at row t; the exercise takes the flags returned at row t. "
                               "The finite_horizon_gae branch and reward scaling are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/algorithms/utils.py",
            "symbol": "preprocess_embodied_advantages_inputs",
            "license": "Apache-2.0",
            "adapted": "The chunk_level reduction: sum rewards along the chunk axis and run GAE with one row per chunk.",
            "simplifications": "Upstream sums every reward slot in the chunk with no mask; the exercise multiplies by an explicit valid_mask. "
                               "Done flags arrive already folded to chunk level instead of being max-reduced, and terminated and truncated stay separate.",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/algorithms/advantages.py",
            "symbol": "compute_gae_advantages_and_returns",
            "license": "Apache-2.0",
            "adapted": "The single-flag GAE the chunk-level rewards feed, used as a second oracle when there are no truncations.",
            "simplifications": "Advantage and return normalization and the critic-free branch are dropped.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1712.00378",
            "section": "2 Time-awareness for time-limited tasks; 3 Partial-episode bootstrapping for time-unlimited tasks",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1506.02438",
            "section": "3 Advantage Function Estimation, equation 16",
        },
    ],
    "tests": [
        {
            "name": "A truncation bootstraps from final_values and cuts the trace",
            "behavior": "rl.advantage",
            "code": r"""
import torch
rewards = torch.tensor([[1.0], [1.0], [1.0]])
values = torch.tensor([[2.0], [3.0], [8.0]])
next_value = torch.tensor([10.0])
terminated = torch.tensor([[False], [False], [False]])
truncated = torch.tensor([[False], [True], [False]])
final_values = torch.tensor([[99.0], [6.0], [99.0]])  # only row 1 is read
adv, ret = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=0.5, lam=1.0)
# Row 2: 1 + 0.5 * 10 - 8 = -2.
# Row 1: 1 + 0.5 * 6 - 3 = 1, and row 2 belongs to the next episode, so A[1] = 1.
# Row 0: 1 + 0.5 * 3 - 2 = 0.5, plus 0.5 * 1 * A[1] = 1.0.
assert torch.allclose(adv, torch.tensor([[1.0], [1.0], [-2.0]])), adv
assert torch.allclose(ret, torch.tensor([[3.0], [4.0], [6.0]])), ret
""",
        },
        {
            "name": "Chunk level sums the valid rewards of each chunk",
            "behavior": "rl.advantage",
            "code": r"""
import torch
rewards = torch.tensor([[[1.0, 2.0, 5.0]], [[1.0, 1.0, 1.0]]])  # (T=2, B=1, H=3)
valid = torch.tensor([[[True, True, False]], [[True, True, True]]])
values = torch.zeros(2, 1)
terminated = torch.tensor([[False], [True]])
truncated = torch.zeros(2, 1, dtype=torch.bool)
adv, ret = {fn}(rewards, values, torch.tensor([7.0]), terminated, truncated, torch.zeros(2, 1),
                gamma=1.0, lam=1.0, valid_mask=valid)
# Chunk rewards are 3 and 3; the second chunk terminates.
assert adv.shape == (2, 1), tuple(adv.shape)
assert torch.allclose(adv, torch.tensor([[6.0], [3.0]])), adv
""",
        },
        {
            "name": "Termination bootstraps zero, even when truncated is also set",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A terminated step bootstrapped a nonzero value. A terminal state has no future: use 0, and let termination win over truncation.",
            "code": r"""
import torch
rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
values = torch.tensor([[0.5, 0.5], [50.0, 50.0]])
next_value = torch.tensor([40.0, 40.0])
terminated = torch.tensor([[True, True], [True, True]])
truncated = torch.tensor([[False, True], [False, True]])
final_values = torch.full((2, 2), 30.0)
adv, ret = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=0.9, lam=0.95)
expected = torch.tensor([[0.5, 0.5], [-48.0, -48.0]])
assert torch.allclose(adv, expected, atol=1e-5), adv
""",
        },
        {
            "name": "A truncation reads final_values, not the reset value and not zero",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A truncated step did not bootstrap from final_values[t]. After auto-reset values[t + 1] belongs to a new episode, and a time limit is not a terminal state.",
            "code": r"""
import torch
rewards = torch.tensor([[0.0], [0.0]])
values = torch.tensor([[0.0], [-20.0]])
next_value = torch.tensor([0.0])
terminated = torch.zeros(2, 1, dtype=torch.bool)
truncated = torch.tensor([[True], [False]])
final_values = torch.tensor([[5.0], [123.0]])
adv, _ = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=0.8, lam=0.5)
assert torch.allclose(adv[0], torch.tensor([4.0]), atol=1e-5), f'expected 0.8 * 5 = 4.0 at the truncated row, got {adv[0].item()}'
assert torch.allclose(adv[1], torch.tensor([20.0]), atol=1e-5), adv
""",
        },
        {
            "name": "The trace is cut at a truncation",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "Advantage from the next episode flowed backwards through a truncation. Cut the recursion term at truncation as well as at termination.",
            "code": r"""
import torch
rewards = torch.tensor([[0.0], [100.0]])
values = torch.zeros(2, 1)
next_value = torch.tensor([0.0])
terminated = torch.zeros(2, 1, dtype=torch.bool)
truncated = torch.tensor([[True], [False]])
final_values = torch.tensor([[0.0], [0.0]])
adv, _ = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=1.0, lam=1.0)
assert torch.allclose(adv[0], torch.tensor([0.0]), atol=1e-5), f'A[0] should be 0, got {adv[0].item()}'
assert torch.allclose(adv[1], torch.tensor([100.0]), atol=1e-5), adv
""",
        },
        {
            "name": "The last row bootstraps from next_value only when it did not end",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "The last row used the wrong bootstrap. Use next_value when it continues, final_values when it was truncated, and 0 when it terminated.",
            "code": r"""
import torch
rewards = torch.tensor([[1.0, 1.0, 1.0]])
values = torch.zeros(1, 3)
next_value = torch.tensor([10.0, 10.0, 10.0])
terminated = torch.tensor([[False, False, True]])
truncated = torch.tensor([[False, True, False]])
final_values = torch.tensor([[-7.0, 4.0, -7.0]])
adv, ret = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=0.5, lam=0.9)
assert torch.allclose(adv, torch.tensor([[6.0, 3.0, 1.0]]), atol=1e-5), adv
assert torch.allclose(ret, adv), ret
""",
        },
        {
            "name": "Matches a loop-based oracle on random rollouts",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "Values disagree with an independent per-element evaluation of the recursion on rollouts that mix terminations and truncations.",
            "code": _LOOP_ORACLE + r"""
for seed, T, B, gamma, lam in ((0, 7, 3, 0.97, 0.9), (1, 12, 5, 0.8, 0.6), (2, 1, 4, 0.99, 0.95), (3, 9, 2, 1.0, 1.0)):
    rollout = random_rollout(seed, T, B)
    adv, ret = {fn}(*rollout, gamma=gamma, lam=lam)
    want_adv, want_ret = loop_gae(*rollout, gamma, lam)
    assert adv.shape == (T, B) and ret.shape == (T, B), (tuple(adv.shape), tuple(ret.shape))
    assert torch.allclose(adv, want_adv, atol=1e-8), (seed, (adv - want_adv).abs().max())
    assert torch.allclose(ret, want_ret, atol=1e-8), (seed, (ret - want_ret).abs().max())
""",
        },
        {
            "name": "lam = 1 telescopes to the bootstrapped discounted return",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "At lam = 1 the returns are not the discounted reward sum up to the episode end, bootstrapped from final_values at a truncation and from next_value at the rollout end.",
            "code": r"""
import torch
g = torch.Generator().manual_seed(11)
T, B, gamma = 8, 3, 0.9
rewards = torch.randn(T, B, generator=g, dtype=torch.float64)
values = torch.randn(T, B, generator=g, dtype=torch.float64)
next_value = torch.randn(B, generator=g, dtype=torch.float64)
terminated = torch.zeros(T, B, dtype=torch.bool)
truncated = torch.zeros(T, B, dtype=torch.bool)
terminated[2, 0] = True
truncated[4, 1] = True
truncated[1, 2] = True
terminated[5, 2] = True
final_values = torch.full((T, B), 500.0, dtype=torch.float64)
final_values[4, 1] = 2.5
final_values[1, 2] = -1.5
_, ret = {fn}(rewards, values, next_value, terminated, truncated, final_values, gamma=gamma, lam=1.0)
want = torch.zeros(T, B, dtype=torch.float64)
for b in range(B):
    G = float(next_value[b])
    for t in range(T - 1, -1, -1):
        if terminated[t, b]:
            G = 0.0
        elif truncated[t, b]:
            G = float(final_values[t, b])
        G = float(rewards[t, b]) + gamma * G
        want[t, b] = G
assert torch.allclose(ret, want, atol=1e-8), (ret - want).abs().max()
""",
        },
        {
            "name": "Chunk level masks, sums without discount and discounts once per chunk",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Chunk-level rewards are wrong. Sum only the rewards where valid_mask is True, without discounting inside the chunk, and apply gamma once per chunk row.",
            "code": _LOOP_ORACLE + r"""
g = torch.Generator().manual_seed(5)
T, B, H = 6, 3, 4
step_rewards = torch.randn(T, B, H, generator=g, dtype=torch.float64)
counts = torch.randint(1, H + 1, (T, B), generator=g)
valid = torch.arange(H).view(1, 1, H) < counts.unsqueeze(-1)
step_rewards = torch.where(valid, step_rewards, torch.full_like(step_rewards, 77.0))
_, values, next_value, terminated, truncated, final_values = random_rollout(5, T, B)
gamma, lam = 0.5, 0.8
adv, ret = {fn}(step_rewards, values, next_value, terminated, truncated, final_values,
                gamma=gamma, lam=lam, valid_mask=valid)
chunk_rewards = torch.zeros(T, B, dtype=torch.float64)
for t in range(T):
    for b in range(B):
        chunk_rewards[t, b] = sum(float(step_rewards[t, b, h]) for h in range(int(counts[t, b])))
want_adv, want_ret = loop_gae(chunk_rewards, values, next_value, terminated, truncated, final_values, gamma, lam)
assert adv.shape == (T, B), tuple(adv.shape)
assert torch.allclose(adv, want_adv, atol=1e-8), (adv - want_adv).abs().max()
assert torch.allclose(ret, want_ret, atol=1e-8), (ret - want_ret).abs().max()
""",
        },
        {
            "name": "Returns equal advantages plus values",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "returns must be advantages + values, the critic's regression target.",
            "code": _LOOP_ORACLE + r"""
rollout = random_rollout(21, 6, 4)
adv, ret = {fn}(*rollout, gamma=0.95, lam=0.7)
assert torch.allclose(ret, adv + rollout[1], atol=1e-10), (ret - adv - rollout[1]).abs().max()
assert not torch.allclose(ret, adv), 'returns equal advantages; add the values back'
""",
        },
        {
            "name": "Rejects bad shapes, a misused valid_mask and bad constants",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An inconsistent input was accepted. Check every shape against rewards, require valid_mask exactly for 3-D rewards, and keep gamma and lam in [0, 1].",
            "code": r"""
import torch
T, B, H = 3, 2, 4
r = torch.randn(T, B)
v = torch.randn(T, B)
nv = torch.randn(B)
f = torch.zeros(T, B, dtype=torch.bool)
fv = torch.randn(T, B)
r3 = torch.randn(T, B, H)
m3 = torch.ones(T, B, H, dtype=torch.bool)
cases = [
    ((r, torch.randn(T + 1, B), nv, f, f, fv), {}),
    ((r, v, torch.randn(B + 1), f, f, fv), {}),
    ((r, v, nv, torch.zeros(T, B + 1, dtype=torch.bool), f, fv), {}),
    ((r, v, nv, f, torch.zeros(T - 1, B, dtype=torch.bool), fv), {}),
    ((r, v, nv, f, f, torch.randn(T, 1)), {}),
    ((r3, v, nv, f, f, fv), {}),
    ((r, v, nv, f, f, fv), {'valid_mask': m3}),
    ((r3, v, nv, f, f, fv), {'valid_mask': torch.ones(T, B, H + 1, dtype=torch.bool)}),
    ((r, v, nv, f, f, fv), {'gamma': 1.5}),
    ((r, v, nv, f, f, fv), {'gamma': -0.1}),
    ((r, v, nv, f, f, fv), {'lam': 1.1}),
    ((r, v, nv, f, f, fv), {'lam': -0.5}),
]
for i, (args, kwargs) in enumerate(cases):
    try:
        {fn}(*args, **kwargs)
    except ValueError:
        continue
    raise AssertionError(f'case {i} should raise ValueError')
""",
        },
        {
            "name": "Keeps the dtype, accepts float flags and leaves inputs unchanged",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The output dtype does not follow rewards, 0/1 float flags were mishandled, or an input was modified in place.",
            "code": _LOOP_ORACLE + r"""
rollout = random_rollout(8, 5, 3)
rewards, values, next_value, terminated, truncated, final_values = rollout
copies = [x.clone() for x in rollout]
adv, ret = {fn}(rewards, values, next_value, terminated.double(), truncated.double(), final_values, gamma=0.9, lam=0.9)
assert adv.dtype == torch.float64 and ret.dtype == torch.float64, (adv.dtype, ret.dtype)
want_adv, _ = loop_gae(*rollout, 0.9, 0.9)
assert torch.allclose(adv, want_adv, atol=1e-8), (adv - want_adv).abs().max()
for before, after in zip(copies, rollout):
    assert torch.equal(before, after), 'an input tensor was modified in place'
""",
        },
    ],
    "solution": '''import torch


def robot_gae(rewards, values, next_value, terminated, truncated, final_values,
              gamma=0.99, lam=0.95, valid_mask=None):
    if rewards.dim() == 3:
        if valid_mask is None or valid_mask.shape != rewards.shape:
            raise ValueError("3-D rewards need a valid_mask of the same shape")
        rewards = (rewards * valid_mask.to(rewards.dtype)).sum(dim=-1)
    elif rewards.dim() == 2:
        if valid_mask is not None:
            raise ValueError("valid_mask is only accepted with 3-D rewards")
    else:
        raise ValueError(f"rewards must be 2-D or 3-D, got {tuple(rewards.shape)}")
    T, B = rewards.shape
    for name, x in (("values", values), ("terminated", terminated),
                    ("truncated", truncated), ("final_values", final_values)):
        if tuple(x.shape) != (T, B):
            raise ValueError(f"{name} must have shape {(T, B)}, got {tuple(x.shape)}")
    if tuple(next_value.shape) != (B,):
        raise ValueError(f"next_value must have shape {(B,)}, got {tuple(next_value.shape)}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1], got {lam}")

    dtype = rewards.dtype
    values = values.to(dtype)
    term = terminated.bool()
    trunc = truncated.bool()

    next_values = torch.cat([values[1:], next_value.to(dtype).unsqueeze(0)], dim=0)
    boot = torch.where(trunc, final_values.to(dtype), next_values)
    boot = torch.where(term, torch.zeros_like(boot), boot)
    cont = (~(term | trunc)).to(dtype)

    deltas = rewards + gamma * boot - values
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        running = deltas[t] + gamma * lam * cont[t] * running
        advantages[t] = running
    return advantages, advantages + values
''',
    "interview_questions": interview(
        concept=[
            "What is the difference between terminated and truncated, and which one should a value target bootstrap through?",
            "Why does an auto-resetting vector environment force you to compute a separate value for the final observation?",
        ],
        deep_dive=[
            "Walk through the bootstrap value and the continuation factor at a step that is truncated, one that is terminated, and one where both flags are set.",
            "Why must the GAE trace still be cut at a truncation even though the TD residual bootstraps through it?",
            "How do you turn per-step rewards inside an action chunk into a chunk-level GAE, and what happens to gamma?",
        ],
        tradeoffs=[
            "When is bootstrapping through a time limit the wrong choice, and what is the alternative?",
            "Step-level versus chunk-level advantages for a VLA that outputs action chunks: what do you gain and lose?",
        ],
    ),
}
