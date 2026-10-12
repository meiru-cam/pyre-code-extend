"""A 2-D point-mass reaching task with the Gymnasium reset/step contract."""

from ._interview import interview

# A step-by-step model of the contract, shared by the unshown cases.
_MODEL = r"""
import math
import torch

def px(c, S):
    return min(int(math.floor((float(c) + 1.0) / 2.0 * S)), S - 1)

def model_image(pos, goal, S):
    img = torch.zeros(1, S, S)
    img[0, px(goal[1], S), px(goal[0], S)] = 0.5
    img[0, px(pos[1], S), px(pos[0], S)] = 1.0
    return img

def seeded_draws(seed, episodes):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(episodes):
        pos = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
        goal = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
        out.append((pos, goal))
    return out

def model_step(pos, goal, action, radius, max_action):
    delta = action.to(torch.float32).clamp(-max_action, max_action)
    pos = (pos + delta).clamp(-1.0, 1.0)
    return pos, float(torch.linalg.norm(pos - goal)) <= radius

def same_obs(obs, pos, goal, S):
    assert set(obs) == {"image", "state"}, sorted(obs)
    assert obs["state"].shape == (2,) and obs["state"].dtype == torch.float32, obs["state"]
    assert torch.allclose(obs["state"], pos, atol=1e-6), (obs["state"], pos)
    assert obs["image"].shape == (1, S, S) and obs["image"].dtype == torch.float32, obs["image"].shape
    assert torch.equal(obs["image"], model_image(pos, goal, S)), (obs["image"], pos, goal)

def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False
"""

TASK = {
    "title": "Point Reach Environment",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "PointReachEnv",
    "description_en": r"""Build a tiny robot reaching task behind the Gymnasium environment contract. A point mass moves in the square [-1, 1] x [-1, 1] and must reach a goal. It sees a one-channel image of the scene and its own position.

**Signature:** `PointReachEnv(max_steps, goal_radius, image_size, max_action=0.1)` with two methods:
- `reset(*, seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

**Constructor.** Raise `ValueError` unless `max_steps` and `image_size` are positive integers and `goal_radius` and `max_action` are positive numbers.

**reset:**
- The environment owns one `torch.Generator`. When `seed` is an integer, replace it with `torch.Generator().manual_seed(seed)`. When `seed` is `None`, keep drawing from the existing generator without re-seeding it. Only when no generator exists yet, create one seeded from entropy.
- Without an initial state, draw the start position first and then the goal, each as `torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1`.
- `options["init_state"]`, when present, is a tensor `(x, y, goal_x, goal_y)` of shape `(4,)`. It sets the start and the goal directly and draws nothing from the generator. Raise `ValueError` if its shape is not `(4,)` or a value lies outside [-1, 1].
- Set the step counter to 0 and return the first observation and info.

**step:**
- Raise `RuntimeError` if `reset` was never called, or if the previous step ended the episode (terminated or truncated). Raise `ValueError` if `action` does not have shape `(2,)`.
- Clip each action component to [-max_action, max_action] and add it to the position. Then clip the position to [-1, 1] and add 1 to the step counter.
- `success` is `euclidean_distance(position, goal) <= goal_radius`.
- `terminated = success`, and `reward = 1.0` on success, else `0.0`.
- `truncated = step_counter >= max_steps`, regardless of `terminated`. Success on the last allowed step sets both flags.

**Observation and info:**
- `obs = {"image": ..., "state": ...}`. `image` is a float32 tensor of shape `(1, S, S)` with `S = image_size`. `state` is the float32 position, shape `(2,)`.
- A coordinate `c` maps to the pixel index `min(floor((c + 1) / 2 * S), S - 1)`. The row index comes from y and the column index from x.
- The image is 0 everywhere, 0.5 at the goal pixel and 1.0 at the agent pixel. When the two share a pixel, it holds 1.0.
- `info = {"success": ..., "elapsed_steps": ...}`. `reset` uses the same success rule and reports 0 elapsed steps.
- `reward` is a Python `float`. `terminated`, `truncated` and `info["success"]` are Python `bool`s, and `info["elapsed_steps"]` is an `int`.
- Every call returns new tensors. Changing a returned observation, or the `init_state` tensor after `reset`, never changes the environment or an earlier observation.

**Example:**

    env = PointReachEnv(max_steps=3, goal_radius=0.25, image_size=4)
    obs, info = env.reset(options={"init_state": torch.tensor([-0.9, -0.9, -0.4, -0.9])})
    # state [-0.9, -0.9]; image[0, 0, 0] = 1.0 (agent), image[0, 0, 1] = 0.5 (goal)
    env.step(torch.tensor([0.5, 0.0]))  # clipped to 0.1: x = -0.8, distance 0.4 -> 0.0, False, False
    env.step(torch.tensor([0.1, 0.0]))  # x = -0.7, distance 0.3 -> 0.0, False, False
    env.step(torch.tensor([0.1, 0.0]))  # x = -0.6, distance 0.2 -> 1.0, True, True (last allowed step)
    env.step(torch.tensor([0.0, 0.0]))  # raises RuntimeError: call reset first

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Two ways to end an episode.** Gymnasium splits the old `done` flag in two. `terminated` means the agent reached a terminal state of the task's MDP, such as success or failure. `truncated` means the episode was cut off for a reason outside the MDP, usually a time limit. The split matters for value estimation: after termination no more reward can arrive, but after truncation the agent would have kept acting, so a value target bootstraps from the next state after a truncation but not after a termination.

**Where the flags come from.** Gymnasium's `TimeLimit` wrapper counts steps and sets `truncated = True` once `elapsed >= max_episode_steps`, without looking at `terminated`. So both flags can be true on the same step. ManiSkill's base `step` takes `terminated` from the task's `success` (or `fail`) and always returns `truncated` as False, so the time limit has to come from outside the task.

**Seeding.** Gymnasium's `Env.reset` re-creates its generator only when a seed is passed (and creates one from entropy the first time it is needed without one). The documented pattern is to seed once right after creating the environment and then reset without a seed, so each episode gets a fresh but reproducible start. For environments registered as deterministic, Gymnasium's environment checker tests this: two `reset(seed=123)` calls must give the same observation, and so must the unseeded resets that follow each of them.

**The task shape.** ManiSkill's PickCube samples the cube and goal positions in `_initialize_episode`, and its `evaluate` reports `success` when the cube lies within `goal_thresh` of the goal (inclusive) and the robot is static. PickCube has no `fail` flag, so ManiSkill's base sparse reward (`success - fail` when both exist) reduces to the success indicator. Resetting to a stored initial state (ManiSkill's `reset_to_env_states` option) skips the sampling step.

**Fresh buffers.** Rollout buffers keep every observation, so Gymnasium asks `step` and `reset` to return new objects rather than mutating and returning a buffer from an earlier call.

**Stepping after the end.** Gymnasium only says that `reset` must be called after an episode ends. Its `OrderEnforcing` wrapper raises when `step` comes before the first `reset`. This exercise also raises after the episode ends, so rollout bugs fail loudly.""",
    "advisory_prerequisites": ["rl_eval_loop", "agent_env_adapter"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which reset calls replace the generator, and which ones keep drawing from it? Does an init_state reset draw anything? Is truncation decided before or after you know whether the step succeeded, and does success suppress it? What state must change so that the next step after the end raises?"},
        {"level": 2, "kind": "analysis", "content": "Store the generator, position, goal, step counter and an episode-over flag. In reset, re-seed only for an integer seed, then either validate and clone init_state or draw position then goal. In step, check the flag, validate the action shape, clamp the action, add, clamp the position, count the step, compute the distance, set both flags independently and remember whether the episode ended. Build the image and clone the state for each observation."},
    ],
    "model_connections": [
        "ManiSkill's BaseEnv.step takes terminated from evaluate()[\"success\"] (or fail) and returns truncated as all False; PickCube's evaluate is the success detector.",
        "Gymnasium's TimeLimit wrapper sets truncated = True once elapsed >= max_episode_steps, whatever terminated says.",
        "The rl_eval_loop exercise is the consumer side of this contract: it resets with seed + episode index and stops on either flag.",
        "Value targets bootstrap through truncation but not through termination, which is why the two flags stay separate all the way to the advantage computation.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One reset/step contract lets the same evaluator, vectorizer and trainer drive any task, from a point mass to a simulated arm.",
            "Seeding once and then continuing the generator makes a whole run of episodes reproducible from one integer while each episode still starts somewhere new.",
        ],
        "cons": [
            "Raising on a step after the end is stricter than Gymnasium, which leaves that case to the caller; vectorized auto-reset needs its own handling.",
            "A sparse success reward is zero until the goal is reached; tasks such as PickCube also ship a shaped dense reward.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/core.py",
            "symbol": "Env.step, Env.reset",
            "license": "MIT",
            "adapted": "The five-tuple step result, the terminated/truncated meaning, re-seeding only when a seed is given, and returning new observation objects.",
            "simplifications": "A torch.Generator replaces np_random; there are no observation or action spaces; info always holds success and elapsed_steps.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/wrappers/common.py",
            "symbol": "TimeLimit.step, TimeLimit.reset, OrderEnforcing.step",
            "license": "MIT",
            "adapted": "The step counter and truncated = elapsed >= max_steps are folded into the environment; step before the first reset raises.",
            "simplifications": "Step after the episode ends also raises (Gymnasium leaves it undefined); errors are RuntimeError rather than ResetNeeded, and a non-integer max_steps raises ValueError rather than TypeError.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/utils/env_checker.py",
            "symbol": "check_reset_seed_determinism, data_equivalence",
            "license": "MIT",
            "adapted": "The seeded/unseeded reset sequence is used as an evaluator case and an oracle check.",
            "simplifications": "Generator states are compared through torch.Generator.get_state instead of numpy bit-generator state.",
        },
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "mani_skill/envs/tasks/tabletop/pick_cube.py",
            "symbol": "PickCubeEnv._initialize_episode, PickCubeEnv.evaluate",
            "license": "Apache-2.0",
            "adapted": "Per-episode sampling of the object and goal positions and an inclusive distance threshold for success.",
            "simplifications": "A 2-D point mass replaces the arm and cube; there is no static-robot condition, no fail flag and no dense reward.",
        },
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "mani_skill/envs/sapien_env.py",
            "symbol": "BaseEnv.step, BaseEnv.compute_sparse_reward",
            "license": "Apache-2.0",
            "adapted": "terminated is taken from success, the sparse reward is the success indicator, and info carries elapsed_steps.",
            "simplifications": "Single environment instead of a batch; truncation is computed inside the environment instead of by a wrapper.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2407.17032",
            "section": "Towers et al. 2024, Gymnasium, Section 4.2 (Termination and Truncation), Equations 1-2 and Figure 2",
            "note": "Why value estimates must treat termination and truncation differently.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/1712.00378",
            "section": "Pardo et al. 2018, Time Limits in Reinforcement Learning, Abstract",
            "note": "Bootstrapping from the last state of a partial episode when the time limit is not part of the task.",
        },
    ],
    "tests": [
        {"name": "Reset to an initial state, step to the goal on the last allowed step", "behavior": "state.invariant", "code": r"""
import torch
env = {fn}(max_steps=3, goal_radius=0.25, image_size=4)
obs, info = env.reset(options={"init_state": torch.tensor([-0.9, -0.9, -0.4, -0.9])})
assert torch.allclose(obs["state"], torch.tensor([-0.9, -0.9])), obs["state"]
expected = torch.zeros(1, 4, 4); expected[0, 0, 1] = 0.5; expected[0, 0, 0] = 1.0
assert torch.equal(obs["image"], expected), obs["image"]
assert info == {"success": False, "elapsed_steps": 0}, info
_, r, term, trunc, info = env.step(torch.tensor([0.5, 0.0]))
assert (r, term, trunc) == (0.0, False, False) and info["elapsed_steps"] == 1, (r, term, trunc, info)
_, r, term, trunc, _ = env.step(torch.tensor([0.1, 0.0]))
assert (r, term, trunc) == (0.0, False, False), (r, term, trunc)
obs, r, term, trunc, info = env.step(torch.tensor([0.1, 0.0]))
assert torch.allclose(obs["state"], torch.tensor([-0.6, -0.9])), obs["state"]
assert (r, term, trunc) == (1.0, True, True), (r, term, trunc)
assert info == {"success": True, "elapsed_steps": 3}, info
try:
    env.step(torch.tensor([0.0, 0.0]))
except RuntimeError:
    pass
else:
    raise AssertionError("step after the episode ended must raise RuntimeError")
"""},
        {"name": "Seeded reset draws the start, then the goal", "behavior": "state.invariant", "code": r"""
import torch
env = {fn}(max_steps=5, goal_radius=0.1, image_size=8)
obs, info = env.reset(seed=7)
g = torch.Generator().manual_seed(7)
start = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
goal = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
assert torch.allclose(obs["state"], start), (obs["state"], start)
assert obs["image"].shape == (1, 8, 8) and obs["image"].dtype == torch.float32
assert obs["image"].sum().item() in (1.0, 1.5), obs["image"]
"""},
        {"name": "Seed once, then continue the generator", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Re-seed the generator only when reset gets an integer seed; an unseeded reset must keep drawing from the existing generator.",
         "code": _MODEL + r"""
for seed in (0, 3, 123):
    env = {fn}(max_steps=4, goal_radius=0.05, image_size=6)
    draws = seeded_draws(seed, 3)
    for k in range(3):
        obs, _ = env.reset(seed=seed) if k == 0 else env.reset()
        same_obs(obs, draws[k][0], draws[k][1], 6)
    # check_reset_seed_determinism: seeded, unseeded, seeded, unseeded
    o1, _ = env.reset(seed=123); o2, _ = env.reset(); o3, _ = env.reset(seed=123); o4, _ = env.reset()
    assert torch.equal(o1["state"], o3["state"]) and torch.equal(o1["image"], o3["image"])
    assert torch.equal(o2["state"], o4["state"]) and torch.equal(o2["image"], o4["image"])
    assert not torch.equal(o1["state"], o2["state"])
    o5, _ = env.reset(seed=456)
    assert not torch.equal(o5["state"], o1["state"])
"""},
        {"name": "An initial-state reset draws nothing from the generator", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "options['init_state'] must set the start and goal without consuming random draws, while a seed passed with it still re-seeds.",
         "code": _MODEL + r"""
draws = seeded_draws(11, 2)
env = {fn}(max_steps=4, goal_radius=0.05, image_size=5)
env.reset(seed=11)
fixed = torch.tensor([0.2, -0.3, 0.9, 0.9])
obs, _ = env.reset(options={"init_state": fixed})
same_obs(obs, fixed[:2], fixed[2:], 5)
obs, _ = env.reset()
same_obs(obs, draws[1][0], draws[1][1], 5)
obs, _ = env.reset(seed=11, options={"init_state": fixed})
same_obs(obs, fixed[:2], fixed[2:], 5)
obs, _ = env.reset(options={})
same_obs(obs, draws[0][0], draws[0][1], 5)
"""},
        {"name": "Seeded rollouts match a step-by-step model", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "Clip the action, add it, clip the position to [-1, 1], then compute success, reward, both flags and the image from the new position.",
         "code": _MODEL + r"""
for seed in range(5):
    gen = torch.Generator().manual_seed(100 + seed)
    radius, max_action, S, T = 0.3, 0.15, 7, 12
    env = {fn}(max_steps=T, goal_radius=radius, image_size=S, max_action=max_action)
    pos, goal = seeded_draws(seed, 1)[0]
    obs, info = env.reset(seed=seed)
    same_obs(obs, pos, goal, S)
    for t in range(1, T + 1):
        action = (torch.rand(2, generator=gen) * 2 - 1) * 0.4
        pos, success = model_step(pos, goal, action, radius, max_action)
        obs, r, term, trunc, info = env.step(action)
        same_obs(obs, pos, goal, S)
        assert r == (1.0 if success else 0.0), (seed, t, r)
        assert term is success and trunc is (t >= T), (seed, t, term, trunc)
        assert info == {"success": success, "elapsed_steps": t}, (seed, t, info)
        if term or trunc:
            break
"""},
        {"name": "Truncation fires exactly at max_steps", "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "truncated must become True on the step where the counter reaches max_steps, not one step earlier or later.",
         "code": _MODEL + r"""
for T in (1, 2, 5):
    env = {fn}(max_steps=T, goal_radius=0.01, image_size=3)
    for episode in range(2):
        env.reset(options={"init_state": torch.tensor([-1.0, -1.0, 1.0, 1.0])})
        for t in range(1, T + 1):
            _, r, term, trunc, info = env.step(torch.tensor([0.0, 0.0]))
            assert term is False and r == 0.0
            assert trunc is (t == T), (T, episode, t, trunc)
            assert info["elapsed_steps"] == t, (T, episode, info)
        assert raises(RuntimeError, env.step, torch.tensor([0.0, 0.0]))
"""},
        {"name": "Success is termination, never truncation", "visibility": "unshown", "behavior": "rl.reward_verifiable",
         "failure_message": "Reaching the goal before the time limit must set terminated=True and truncated=False with reward 1.0.",
         "code": _MODEL + r"""
env = {fn}(max_steps=10, goal_radius=0.25, image_size=4)
env.reset(options={"init_state": torch.tensor([0.0, 0.0, 0.3, 0.0])})
_, r, term, trunc, info = env.step(torch.tensor([0.0, 0.0]))
assert (r, term, trunc, info["success"]) == (0.0, False, False, False)
_, r, term, trunc, info = env.step(torch.tensor([0.1, 0.0]))
assert (r, term, trunc, info["success"]) == (1.0, True, False, True), (r, term, trunc, info)
assert type(r) is float and type(term) is bool and type(trunc) is bool and type(info["success"]) is bool
assert type(info["elapsed_steps"]) is int
assert raises(RuntimeError, env.step, torch.tensor([0.0, 0.0]))
env.reset(options={"init_state": torch.tensor([0.0, 0.0, 0.3, 0.0])})
_, r, term, trunc, info = env.step(torch.tensor([0.0, 0.0]))
assert (term, trunc, info["elapsed_steps"]) == (False, False, 1), "reset must clear the step counter and the end flag"
"""},
        {"name": "Success on the last allowed step sets both flags", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "truncated depends only on the step counter; success on step max_steps must report terminated=True and truncated=True.",
         "code": _MODEL + r"""
for T in (1, 2, 4):
    env = {fn}(max_steps=T, goal_radius=0.25, image_size=4)
    env.reset(options={"init_state": torch.tensor([-0.5, 0.5, -0.5 + 0.1 * T + 0.2, 0.5])})
    for t in range(1, T + 1):
        _, r, term, trunc, info = env.step(torch.tensor([0.1, 0.0]))
        if t < T:
            assert (term, trunc) == (False, False), (T, t, term, trunc)
    assert (r, term, trunc, info["success"]) == (1.0, True, True, True), (T, r, term, trunc)
    assert raises(RuntimeError, env.step, torch.tensor([0.0, 0.0]))
"""},
        {"name": "Distance equal to goal_radius counts as success", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Success is distance <= goal_radius; a distance exactly on the radius succeeds.",
         "code": _MODEL + r"""
env = {fn}(max_steps=4, goal_radius=0.5, image_size=4)
obs, info = env.reset(options={"init_state": torch.tensor([0.0, 0.0, 0.5, 0.0])})
assert info["success"] is True, info
_, r, term, trunc, info = env.step(torch.tensor([0.0, 0.0]))
assert (r, term, trunc) == (1.0, True, False), (r, term, trunc)
"""},
        {"name": "Actions are clipped and the position stays in the square", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Clip each action component to [-max_action, max_action] before adding it, then clip the position to [-1, 1].",
         "code": _MODEL + r"""
env = {fn}(max_steps=20, goal_radius=0.01, image_size=4, max_action=0.25)
env.reset(options={"init_state": torch.tensor([0.9, -0.9, -1.0, 1.0])})
obs, *_ = env.step(torch.tensor([5.0, -0.1]))
assert torch.allclose(obs["state"], torch.tensor([1.0, -1.0])), obs["state"]
obs, *_ = env.step(torch.tensor([-3.0, 0.1], dtype=torch.float64))
assert obs["state"].dtype == torch.float32
assert torch.allclose(obs["state"], torch.tensor([0.75, -0.9])), obs["state"]
obs, *_ = env.step(torch.tensor([-0.2, 0.3]))
assert torch.allclose(obs["state"], torch.tensor([0.55, -0.65])), obs["state"]
"""},
        {"name": "Image pixels at the corners and on overlap", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Map c to min(floor((c + 1) / 2 * S), S - 1), index rows by y and columns by x, and draw the agent after the goal.",
         "code": _MODEL + r"""
env = {fn}(max_steps=3, goal_radius=0.01, image_size=5)
obs, _ = env.reset(options={"init_state": torch.tensor([1.0, -1.0, -1.0, 1.0])})
expected = torch.zeros(1, 5, 5); expected[0, 4, 0] = 0.5; expected[0, 0, 4] = 1.0
assert torch.equal(obs["image"], expected), obs["image"]
obs, _ = env.reset(options={"init_state": torch.tensor([0.05, 0.1, 0.1, 0.05])})
expected = torch.zeros(1, 5, 5); expected[0, 2, 2] = 1.0
assert torch.equal(obs["image"], expected), obs["image"]
env1 = {fn}(max_steps=3, goal_radius=0.01, image_size=1)
obs, _ = env1.reset(options={"init_state": torch.tensor([-1.0, 1.0, 0.0, 0.0])})
assert torch.equal(obs["image"], torch.ones(1, 1, 1)), obs["image"]
"""},
        {"name": "Observations are fresh tensors", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Return new image and state tensors and copy init_state, so callers can keep or modify observations without touching the environment.",
         "code": _MODEL + r"""
init = torch.tensor([0.0, 0.0, 0.8, 0.8])
env = {fn}(max_steps=5, goal_radius=0.01, image_size=4)
obs0, _ = env.reset(options={"init_state": init})
init.fill_(-0.5)
first_state, first_image = obs0["state"].clone(), obs0["image"].clone()
obs0["state"].add_(0.5); obs0["image"].fill_(9.0)
obs1, *_ = env.step(torch.tensor([0.1, 0.0]))
same_obs(obs1, torch.tensor([0.1, 0.0]), torch.tensor([0.8, 0.8]), 4)
snapshot = obs1["state"].clone()
obs2, *_ = env.step(torch.tensor([0.1, 0.0]))
assert torch.equal(obs1["state"], snapshot), "a later step changed an earlier observation"
assert obs2["state"] is not obs1["state"] and obs2["image"] is not obs1["image"]
"""},
        {"name": "Step before reset and bad arguments raise", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise RuntimeError for a step before reset, and ValueError for bad constructor arguments, a bad action shape or a bad init_state.",
         "code": _MODEL + r"""
env = {fn}(max_steps=3, goal_radius=0.1, image_size=4)
assert raises(RuntimeError, env.step, torch.tensor([0.0, 0.0]))
for args in [(0, 0.1, 4), (-1, 0.1, 4), (2.5, 0.1, 4), (3, 0.0, 4), (3, -0.1, 4), (3, 0.1, 0), (3, 0.1, 2.0)]:
    assert raises(ValueError, {fn}, *args), args
assert raises(ValueError, {fn}, 3, 0.1, 4, max_action=0.0)
env.reset(seed=0)
for bad in (torch.zeros(3), torch.zeros(1, 2), torch.zeros(())):
    assert raises(ValueError, env.step, bad), bad.shape
_, _, _, _, info = env.step(torch.zeros(2))
assert info["elapsed_steps"] == 1, "a rejected action must not count as a step"
for bad in (torch.zeros(3), torch.zeros(2, 2), torch.tensor([0.0, 0.0, 1.5, 0.0]), torch.tensor([-1.01, 0.0, 0.0, 0.0])):
    assert raises(ValueError, env.reset, options={"init_state": bad}), bad
"""},
    ],
    "solution": '''import math
import torch


class PointReachEnv:
    def __init__(self, max_steps, goal_radius, image_size, max_action=0.1):
        for name, value in (("max_steps", max_steps), ("image_size", image_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("goal_radius", goal_radius), ("max_action", max_action)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"{name} must be a positive number")
        self.max_steps = max_steps
        self.goal_radius = float(goal_radius)
        self.image_size = image_size
        self.max_action = float(max_action)
        self._generator = None
        self._pos = None
        self._goal = None
        self._elapsed = 0
        self._needs_reset = True

    def _pixel(self, c):
        return min(int(math.floor((float(c) + 1.0) / 2.0 * self.image_size)), self.image_size - 1)

    def _observation(self):
        image = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        image[0, self._pixel(self._goal[1]), self._pixel(self._goal[0])] = 0.5
        image[0, self._pixel(self._pos[1]), self._pixel(self._pos[0])] = 1.0
        return {"image": image, "state": self._pos.clone()}

    def _success(self):
        return bool(float(torch.linalg.norm(self._pos - self._goal)) <= self.goal_radius)

    def reset(self, *, seed=None, options=None):
        init_state = None if options is None else options.get("init_state")
        if init_state is not None:
            init_state = torch.as_tensor(init_state, dtype=torch.float32).clone()
            if init_state.shape != (4,) or bool((init_state.abs() > 1.0).any()):
                raise ValueError("init_state must be (x, y, goal_x, goal_y) inside [-1, 1]")
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        elif self._generator is None:
            self._generator = torch.Generator()
            self._generator.seed()
        if init_state is not None:
            self._pos, self._goal = init_state[:2].clone(), init_state[2:].clone()
        else:
            g = self._generator
            self._pos = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
            self._goal = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1
        self._elapsed = 0
        self._needs_reset = False
        return self._observation(), {"success": self._success(), "elapsed_steps": 0}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before step(): the episode has not started or has ended")
        action = torch.as_tensor(action)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        delta = action.to(torch.float32).clamp(-self.max_action, self.max_action)
        self._pos = (self._pos + delta).clamp(-1.0, 1.0)
        self._elapsed += 1
        success = self._success()
        terminated = success
        truncated = self._elapsed >= self.max_steps
        self._needs_reset = terminated or truncated
        reward = 1.0 if success else 0.0
        info = {"success": success, "elapsed_steps": self._elapsed}
        return self._observation(), reward, terminated, truncated, info
''',
    "interview_questions": interview(
        concept=[
            "What is the difference between terminated and truncated in Gymnasium, and which one does a time limit set?",
            "Why should reset be seeded once and then called without a seed?",
        ],
        deep_dive=[
            "Why can one step set both terminated and truncated, and how should a value target treat that step?",
            "Why should a reset to a stored initial state draw nothing from the environment's random generator?",
            "Why must step return new observation tensors instead of a shared buffer?",
        ],
        tradeoffs=[
            "Raising on a step after the episode ends versus resetting automatically: what does each choice cost a rollout worker?",
            "Sparse success reward versus a shaped dense reward for a reaching task: what do you gain and what do you risk?",
        ],
    ),
}
