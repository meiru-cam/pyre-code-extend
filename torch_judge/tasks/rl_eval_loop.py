"""Evaluate a policy in an episodic environment without updating it."""

TASK = {
    "title": "RL Evaluation Loop",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "rl_eval_loop",
    "description_en": r"""Run a policy for a fixed number of episodes and report evaluation metrics.

**Signature:** `rl_eval_loop(env, policy, episodes, seed=0) -> dict`

`env.reset(seed=seed)` returns the first observation. `env.step(action)` returns
`(observation, reward, terminated, truncated, info)`. `policy(observation)`
returns the next action. `info["success"]` is the authoritative success flag.

Return exactly four keys: `average_return`, `average_length`, `success_rate`,
and `episodes`. Each episode must reset with seed `seed + episode_index`, stop
on either `terminated` or `truncated`, and call the policy exactly once per
environment step. This is evaluation: never update a policy or environment
state outside the normal reset/step calls.

────────────────

**Background — context only.** An RL eval is not a training loop. It collects
episodes with a fixed policy; there is no advantage calculation, optimizer
step, or weight synchronization. Keeping `terminated` separate from
`truncated` matters because a timeout is not the same as task success, and
deterministic episode seeds make regressions reproducible.""",
    "advisory_prerequisites": ["agentic_rollout_loop", "rollout_batch_assembly"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What are the two independent episode-ending flags? Which object owns the authoritative success bit? How can you prove each episode used a different reproducible seed?"},
        {"level": 2, "kind": "analysis", "content": "For each episode call reset(seed + index), then loop: policy(obs), env.step(action), accumulate reward and length, and break when terminated or truncated. Count success from info.get(\"success\", False), not from a timeout or from the final reward."},
    ],
    "model_connections": [
        "A rollout evaluator runs the same environment boundary as an agent loop, but omits reward-to-advantage and optimizer steps.",
        "The loop is the small, framework-independent core that VERL or Slime can parallelize across workers.",
    ],
    "sources": [{
        "kind": "code",
        "url": "https://github.com/verl-project/verl",
        "commit": "12ebe0cb4d300c58449fb6c675379e8700015c51",
        "path": "verl/experimental/agent_loop/agent_loop.py",
        "symbol": "AgentLoopBase.run",
        "license": "Apache-2.0",
        "adapted": "The evaluator reuses the agent/environment episode boundary as a framework-independent loop.",
        "simplifications": "Removes model-server calls, tokenization, and training; the policy is a synchronous callable and metrics are returned directly."
    }],
    "tests": [
        {"name": "Collects returns, lengths, and success rate", "behavior": "rl.reward_verifiable", "code": r"""
class Env:
    def __init__(self): self.episodes = []
    def reset(self, seed=None): self.episodes.append([seed, 0]); return 0
    def step(self, action):
        self.episodes[-1][1] += 1
        if self.episodes[-1][1] == 2:
            return 0, 2.0, True, False, {"success": action == 1}
        return 0, 1.0, False, False, {"success": False}
env = Env()
out = {fn}(env, lambda obs: 1, 3, seed=10)
assert set(out) == {"average_return", "average_length", "success_rate", "episodes"}
assert out["episodes"] == 3
assert out["average_return"] == 3.0
assert out["average_length"] == 2.0
assert out["success_rate"] == 1.0
assert env.episodes == [[10, 2], [11, 2], [12, 2]]
"""},
        {"name": "Truncation stops without being success", "visibility": "unshown", "behavior": "budget.enforcement", "failure_message": "A timeout must stop the episode, but it is not a successful completion and must not trigger another step.", "code": r"""
class Env:
    def __init__(self): self.steps = 0
    def reset(self, seed=None): return 0
    def step(self, action):
        self.steps += 1
        return 0, 4.0, False, True, {"success": False}
env = Env()
out = {fn}(env, lambda obs: 0, 2)
assert out["average_return"] == 4.0 and out["average_length"] == 1.0
assert out["success_rate"] == 0.0 and env.steps == 2
"""},
        {"name": "Seeded episode table independently determines evaluation metrics", "visibility": "unshown", "behavior": "rl.reward_verifiable", "failure_message": "Metrics must be derived from each completed episode, not a shared running counter.", "code": r"""
import random
for seed in (11, 23, 37):
    rng = random.Random(seed)
    episodes = []
    for i in range(5):
        rewards = [rng.randint(-2, 4) for _ in range(rng.randrange(1, 5))]
        success = bool(rng.randrange(2))
        episodes.append((rewards, success))
    class Env:
        def __init__(self): self.reset_seeds=[]; self.index=-1; self.step_index=0
        def reset(self, seed=None):
            self.reset_seeds.append(seed); self.index += 1; self.step_index=0
            return 0
        def step(self, action):
            assert action == self.step_index
            rewards, success = episodes[self.index]
            reward = rewards[self.step_index]
            self.step_index += 1
            done = self.step_index == len(rewards)
            return self.step_index, reward, done and success, done and not success, {"success": done and success}
    env=Env()
    got={fn}(env, lambda observation: observation, 5, seed=seed)
    assert got == {
        "average_return": sum(sum(row[0]) for row in episodes) / 5,
        "average_length": sum(len(row[0]) for row in episodes) / 5,
        "success_rate": sum(row[1] for row in episodes) / 5,
        "episodes": 5,
    }, (seed, got)
    assert env.reset_seeds == [seed + i for i in range(5)]
"""},
        {"name": "Uses the policy once per step", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "The evaluator skipped a policy call or stepped after the episode had ended.", "code": r"""
class Env:
    def __init__(self): self.n = 0
    def reset(self, seed=None): self.n = 0; return 3
    def step(self, action):
        assert action == self.n
        self.n += 1
        return self.n, 0.0, self.n == 3, False, {"success": self.n == 3}
env = Env(); calls = []
out = {fn}(env, lambda obs: calls.append(obs) or len(calls)-1, 1)
assert calls == [3, 1, 2]
assert out["average_length"] == 3.0
"""},
        {"name": "Rejects a non-positive episode count", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "The evaluator accepted zero or a negative number of episodes.", "code": r"""
for bad in (0, -1):
    try: {fn}(None, None, bad)
    except ValueError: pass
    else: raise AssertionError("episodes must be positive")
"""},
    ],
    "solution": '''def rl_eval_loop(env, policy, episodes, seed=0):
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    returns, lengths, successes = [], [], []
    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        total, length, success = 0.0, 0, False
        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            success = bool(info.get("success", False))
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(length)
        successes.append(success)
    return {
        "average_return": sum(returns) / episodes,
        "average_length": sum(lengths) / episodes,
        "success_rate": sum(successes) / episodes,
        "episodes": episodes,
    }
''',
}
