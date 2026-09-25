"""Monte Carlo reward-to-go targets for REINFORCE."""

TASK = {
    "title": "REINFORCE Discounted Returns",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "reinforce_discounted_returns",
    "description_en": r"""Compute a Monte Carlo reward-to-go target for every action in a flat rollout.

**Signature:** `reinforce_discounted_returns(rewards, dones, gamma=1.0) -> Tensor`

`rewards` is a one-dimensional floating tensor of length T. `dones` is a
same-length boolean tensor; True means that transition ends an episode.
For each time step t, return the discounted sum from t through the first
terminal transition, or through T-1 if the final episode is unfinished.
The terminal transition's own reward is included. Return a new tensor with
the same shape, dtype and device as `rewards`. Require `0 <= gamma <= 1`
and aligned, non-empty inputs; raise ValueError otherwise.

────────────────────────────────

**Background — context only.** REINFORCE weights each sampled action's log
probability by its future return. A flat rollout may contain several episodes,
so a return must reset at each done boundary. This computes targets only;
policy gradients appear in the next exercise.""",
    "advisory_prerequisites": ["rlvr_format_reward"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Can you scan backward while carrying one running return? What happens to that carry immediately after a terminal step?"},
        {"level": 2, "kind": "analysis", "content": "Scan from T-1 to zero. At each step replace the carry by reward[t] + gamma * carry * (not dones[t]). Write into a fresh tensor; the terminal step includes its own reward but excludes all later episodes."},
    ],
    "model_connections": [
        "This is the return target used by the classical Monte Carlo REINFORCE estimator.",
        "VERL's REINFORCE loss consumes an advantage tensor; subtracting a baseline from these returns supplies that advantage.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/ShangtongZhang/reinforcement-learning-an-introduction", "commit": "96bc203617a70eba6b7784a8f876aaf218b2c914", "path": "chapter13/short_corridor.py", "symbol": "ReinforceAgent.episode_end", "license": "MIT", "adapted": "Backward reward-to-go accumulation for Monte Carlo policy gradients.", "simplifications": "Accepts a flat tensor of multiple episodes and explicit done markers rather than running a grid-world agent."},
        {"kind": "paper", "url": "https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf", "section": "2. The REINFORCE Algorithms"}
    ],
    "tests": [
        {"name": "Includes terminal reward but not the next episode", "behavior": "rl.advantage", "code": r"""
import torch
r = torch.tensor([1., 2., 3., 4.])
d = torch.tensor([False, True, False, True])
out = {fn}(r, d, 0.5)
assert torch.equal(out, torch.tensor([2., 2., 5., 4.])), out
assert out.data_ptr() != r.data_ptr()
""" },
        {"name": "Handles an unfinished final episode", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "A nonterminal final step still contributes its reward to earlier returns.", "code": r"""
import torch
out = {fn}(torch.tensor([2., -1., 4.], dtype=torch.float64), torch.tensor([False, False, False]), 1.)
assert out.dtype == torch.float64
assert torch.equal(out, torch.tensor([5., 3., 4.], dtype=torch.float64))
""" },
        {"name": "Matches a forward Monte Carlo oracle over seeded episodes", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "Reward-to-go must stop at each sampled terminal boundary.", "code": r"""
import random, torch
for seed in (11, 23, 37):
    rng = random.Random(seed)
    rewards = [rng.randint(-4, 5) for _ in range(9)]
    dones = [rng.random() < 0.3 for _ in rewards]
    dones[3] = True
    for gamma in (0.0, 0.35, 1.0):
        expected = []
        for start in range(len(rewards)):
            total = 0.0
            for step in range(start, len(rewards)):
                total += rewards[step] * gamma ** (step - start)
                if dones[step]: break
            expected.append(total)
        got = {fn}(torch.tensor(rewards, dtype=torch.float64), torch.tensor(dones), gamma)
        assert torch.allclose(got, torch.tensor(expected, dtype=torch.float64), atol=1e-12), (seed, gamma, got)
""" },
        {"name": "Zero discount uses immediate rewards", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Gamma zero must leave only each step's immediate reward.", "code": r"""
import torch
r = torch.tensor([-2., 5., 3.])
assert torch.equal({fn}(r, torch.tensor([False, True, True]), 0.), r)
""" },
        {"name": "Rejects malformed rollout inputs", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Rewards and done markers must be nonempty aligned 1D tensors with valid gamma.", "code": r"""
import torch
for r,d,g in [(torch.empty(0),torch.empty(0,dtype=torch.bool),1.),
               (torch.ones(2),torch.zeros(1,dtype=torch.bool),1.),
               (torch.ones(2),torch.zeros(2,dtype=torch.bool),1.1)]:
    try: {fn}(r,d,g)
    except ValueError: pass
    else: raise AssertionError("invalid input accepted")
""" },
    ],
    "solution": '''def reinforce_discounted_returns(rewards, dones, gamma=1.0):
    import torch
    if (rewards.ndim != 1 or not rewards.is_floating_point() or rewards.numel() == 0
            or dones.ndim != 1 or dones.dtype != torch.bool
            or dones.shape != rewards.shape or dones.device != rewards.device
            or not 0 <= gamma <= 1):
        raise ValueError("invalid rewards, dones, or gamma")
    out = torch.empty_like(rewards)
    carry = rewards.new_zeros(())
    for index in range(rewards.numel() - 1, -1, -1):
        carry = rewards[index] + gamma * carry * (~dones[index])
        out[index] = carry
    return out
''',
}
