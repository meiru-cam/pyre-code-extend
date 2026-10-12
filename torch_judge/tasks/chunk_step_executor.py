"""Execute one action chunk in a batched robot environment, stopping each env at its own episode end."""

from ._interview import interview

# A batched fake simulator. Env b ends at slot ends[b] (None: never) with kind
# "term", "trunc" or "both". Like a GPU simulator it keeps stepping an env after
# its episode ended, and what it returns then is garbage: huge rewards, random
# flags and an observation of -999.
_SCRIPTED_ENV = r"""
import torch

class ScriptedEnv:
    def __init__(self, ends, kinds, H, seed=0, reward_dtype=torch.float32, reuse=False):
        self.B, self.H = len(ends), H
        self.ends, self.kinds = list(ends), list(kinds)
        g = torch.Generator().manual_seed(seed)
        self.r = (torch.randn(self.B, H, generator=g) * 2).to(reward_dtype)
        self.junk_flags = torch.rand(self.B, H, 2, generator=g) < 0.5
        self.reuse = reuse
        self.buf = torch.zeros(self.B, 3)
        self.t = 0
        self.calls = []

    def __call__(self, actions):
        assert actions.shape[0] == self.B, "env_step must receive the full batch"
        assert self.t < self.H, "env_step was called more than H times"
        self.calls.append(actions.clone())
        t = self.t
        self.t += 1
        B = self.B
        obs = torch.stack([torch.full((B,), float(t)), torch.arange(B, dtype=torch.float32),
                           actions.sum(-1).float()], dim=1)
        reward = self.r[:, t].clone()
        term = torch.zeros(B, dtype=torch.bool)
        trunc = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            e = self.ends[b]
            if e is not None and t == e:
                term[b] = self.kinds[b] in ("term", "both")
                trunc[b] = self.kinds[b] in ("trunc", "both")
            elif e is not None and t > e:
                reward[b] = 1000.0 + t
                term[b] = bool(self.junk_flags[b, t, 0])
                trunc[b] = bool(self.junk_flags[b, t, 1])
                obs[b] = -999.0
        if self.reuse:
            self.buf.copy_(obs)
            return self.buf, reward, term, trunc
        return obs, reward, term, trunc

    def expected(self, chunk):
        B, H = self.B, self.H
        n = torch.tensor([H if e is None else e + 1 for e in self.ends], dtype=torch.long)
        mask = torch.zeros(B, H, dtype=torch.bool)
        rewards = torch.zeros(B, H, dtype=self.r.dtype)
        final_obs = torch.zeros(B, 3)
        term = torch.zeros(B, dtype=torch.bool)
        trunc = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            k = int(n[b])
            for h in range(k):
                mask[b, h] = True
                rewards[b, h] = self.r[b, h]
            final_obs[b] = torch.tensor([float(k - 1), float(b), float(chunk[b, k - 1].sum())])
            if self.ends[b] is not None:
                term[b] = self.kinds[b] in ("term", "both")
                trunc[b] = self.kinds[b] in ("trunc", "both")
        calls = H if any(e is None for e in self.ends) else max(self.ends) + 1
        return {"num_executed": n, "valid_mask": mask, "rewards": rewards,
                "chunk_reward": rewards.sum(dim=1), "terminated": term, "truncated": trunc,
                "final_obs": final_obs, "calls": calls}

def check(out, exp, tag=""):
    assert set(out) >= {"num_executed", "valid_mask", "rewards", "chunk_reward", "terminated", "truncated", "final_obs"}, (tag, sorted(out))
    assert torch.equal(out["num_executed"].long(), exp["num_executed"]), (tag, "num_executed", out["num_executed"], exp["num_executed"])
    assert torch.equal(out["valid_mask"].bool(), exp["valid_mask"]), (tag, "valid_mask", out["valid_mask"])
    assert torch.allclose(out["rewards"].to(exp["rewards"].dtype), exp["rewards"], atol=1e-6), (tag, "rewards", out["rewards"], exp["rewards"])
    assert torch.allclose(out["chunk_reward"].to(exp["rewards"].dtype), exp["chunk_reward"], atol=1e-5), (tag, "chunk_reward", out["chunk_reward"], exp["chunk_reward"])
    assert torch.equal(out["terminated"].bool(), exp["terminated"]), (tag, "terminated", out["terminated"], exp["terminated"])
    assert torch.equal(out["truncated"].bool(), exp["truncated"]), (tag, "truncated", out["truncated"], exp["truncated"])
    assert torch.allclose(out["final_obs"].float(), exp["final_obs"], atol=1e-5), (tag, "final_obs", out["final_obs"], exp["final_obs"])
"""

TASK = {
    "title": "Chunk Step Executor",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "chunk_step",
    "description_en": r"""A VLA policy emits a chunk of `H` actions per env. Execute one chunk in a batched simulator where each env's episode can end partway through the chunk, and return per-action masks and chunk-level results.

**Signature:** `chunk_step(env_step, chunk) -> dict`

**Inputs:**
- `chunk` is a float tensor of shape `(B, H, D)`: `H` actions for each of `B` envs.
- `env_step(actions)` takes a `(B, D)` tensor and returns `(obs, reward, terminated, truncated)`. `obs` is a tensor whose first dimension is `B`. `reward` is a float tensor of shape `(B,)`. `terminated` and `truncated` are bool tensors of shape `(B,)`.

**Stepping rules:**
- Call `env_step(chunk[:, 0])`, then `env_step(chunk[:, 1])`, and so on. Every call passes the full batch, including envs whose episode has ended.
- An env's episode ends at the first slot where `terminated` or `truncated` is True. The action at that slot (the ending action) counts as executed.
- After an env's episode ends, ignore everything `env_step` returns for that env.
- Stop calling `env_step` as soon as every env's episode has ended. Never call it more than `H` times.
- `env_step` may overwrite and return the same tensors on every call. Copy what you keep.
- Do not modify `chunk`.

**Return** a dict with these keys:
- `num_executed`: int64 `(B,)`. Executed actions, including the ending action. `H` for an env that did not end.
- `valid_mask`: bool `(B, H)`. True at slot `h` exactly when `h < num_executed[b]`.
- `rewards`: `(B, H)` in the dtype of `env_step`'s reward. The env's reward at executed slots, 0 at every other slot.
- `chunk_reward`: `(B,)`, same dtype. The sum of `rewards` over the chunk.
- `terminated`, `truncated`: bool `(B,)`. The flags from the env's ending step. Both are False for an env that did not end. Both can be True.
- `final_obs`: shaped like `obs`. Each env's observation from its ending step, or from the last call if it did not end.

Raise `ValueError` when `chunk` is not a 3-D floating-point tensor, or when `B` or `H` is 0.

**Example** (`B = 2`, `H = 3`, `D = 1`; env 0 terminates at slot 1, env 1 never ends; the simulator returns reward 7 for env 0 at slot 2, after its episode ended):

    rewards from env_step:  env 0: 1, 5, 7    env 1: 2, 2, 2
    num_executed = [2, 3]
    valid_mask   = [[True, True, False], [True, True, True]]
    rewards      = [[1, 5, 0], [2, 2, 2]]
    chunk_reward = [6, 6]
    terminated   = [True, False]; truncated = [False, False]
    final_obs    = [obs of env 0 at slot 1, obs of env 1 at slot 2]

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why chunks end early.** A chunked policy picks `H` actions from one observation and the environment executes them before the policy is queried again. Episodes do not line up with chunk boundaries: a task can succeed, fail or hit its time limit at any slot. A batched simulator cannot pause one env, so every call steps every env, and the rows of an env whose episode has ended carry data that belongs to no episode.

**Why the ending action counts.** The transition that ends the episode is a real one: with a sparse success reward it is the only one that carries any reward. RLinf's `valid_action_mask_from_counts` documents this: "The count includes the episode-ending action. Later padded slots are false." Dropping that slot drops the success signal from the update.

**Three upstream choices.**
- ManiSkill's `ActionRepeatWrapper` repeats one action up to `repeat` times. After the first step it updates the observation, flags and infos, and adds the reward, only for envs that are not done, and it breaks out of the loop once every env is done. The returned observation is the true last one.
- RLinf's LIBERO `chunk_step` executes all `H` actions with auto-reset off. It returns the raw per-slot rewards and records a prefix `valid_action_mask` from the count of executed actions. It folds termination and truncation onto the chunk's last index with `any` over the whole chunk, then auto-resets done envs after the chunk. RLinf's ManiSkill `chunk_step` folds the flags the same way but records no mask.
- DSRL's `ActionChunkWrapper` (single env) steps all actions, sums every reward, takes the maximum of the done flags and returns the last observation. If the environment keeps paying reward after the end, that sum includes it. Its repository has no license, so it is cited here and not vendored.

This exercise takes ManiSkill's rule for the observation and the flags (both from the ending step) and RLinf's per-action mask.

**Where the outputs go.** The valid mask zeroes the log-probability and loss terms of actions that were never executed. Chunk-level rewards and flags feed a chunk-level advantage estimate, and `final_obs` is what a value function should bootstrap from when the episode was truncated.""",
    "advisory_prerequisites": ["embodied_action_stream", "act_temporal_ensemble", "rl_eval_loop"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which slot of an env's chunk is its last executed one, and is that slot inside or outside the mask? After an env has ended, which of the values the simulator still returns for it may you read? When can the loop stop before slot H - 1?"},
        {"level": 2, "kind": "analysis", "content": "Validate the chunk. Keep a boolean active vector, all True at the start. At each slot, call env_step on the full slice, then write reward, flags and a copy of the observation only into the rows that are still active, and add one to their executed count. An env stops being active once either flag is True. Break when no env is active. Build the mask by comparing a slot index range with the counts."},
    ],
    "model_connections": [
        "RLinf's LIBERO env executes a VLA's action chunk with chunk_step and, outside evaluation, attaches valid_action_mask to the last slot's infos.",
        "ManiSkill's ActionRepeatWrapper applies the same not-done update when one action is held for several simulator steps.",
        "The truncation flag and final_obs produced here are what a value target needs to bootstrap through a time limit instead of treating it as a terminal state.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Stepping the full batch on every call keeps a GPU-parallel simulator synchronous, and masking afterwards costs nothing.",
            "A per-action mask lets the same rollout feed action-level, chunk-level or token-level losses.",
            "Stopping once every env has ended skips simulator calls whose results would all be ignored.",
        ],
        "cons": [
            "Envs that end early still burn simulator steps until the slowest env in the batch ends or the chunk runs out.",
            "Without an auto-reset inside the chunk, an env that ends at slot 0 idles for the rest of the chunk.",
            "Upstream frameworks fold flags differently (ending step versus any over the chunk), so data from two frameworks is not interchangeable without checking.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/envs/sim/libero/libero_env.py",
            "symbol": "LiberoEnv.chunk_step",
            "license": "Apache-2.0",
            "adapted": "Stepping a (B, H, D) chunk slot by slot on the full batch, the executed-action count that includes the ending action, and the prefix valid mask.",
            "simplifications": "Flags come from each env's ending step instead of any() over all H slots, and final_obs is the ending-step observation instead of the last slot's; "
                               "rewards outside the mask are zeroed instead of returned raw; "
                               "the loop stops once every env has ended instead of always running H steps; "
                               "auto-reset, eval-count masks, rendering control, infos and the per-slot obs list are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/envs/utils.py",
            "symbol": "valid_action_mask_from_counts",
            "license": "Apache-2.0",
            "adapted": "The mask arange(H) < count, with the count including the episode-ending action.",
            "simplifications": "Counts are always an int64 tensor; numpy and scalar inputs are not accepted.",
        },
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "mani_skill/utils/wrappers/action_repeat.py",
            "symbol": "ActionRepeatWrapper.step",
            "license": "Apache-2.0",
            "adapted": "Updating observation, reward and flags only for envs that are not done, and breaking once every env is done.",
            "simplifications": "Each slot has its own action instead of one repeated action; per-slot rewards and a mask are returned, not only the sum; "
                               "observations are tensors (no nested dicts) and infos are dropped; results are copied instead of written into the first step's tensors.",
        },
        {
            "kind": "paper",
            "url": "https://github.com/ajwagen/dsrl/blob/8d21b9cf55459f022c0e769bca1343edd9d30e5a/env_utils.py",
            "section": "env_utils.py, ActionChunkWrapper.step (DSRL, arXiv 2506.15799). The repository has no license: cited as the counter-example that sums every reward and returns the last observation, not vendored.",
        },
    ],
    "tests": [
        {"name": "One env ends mid-chunk, one never ends", "behavior": "rl.masking", "code": r"""
import torch
r = torch.tensor([[1.0, 5.0, 7.0], [2.0, 2.0, 2.0]])
ends = [1, None]
calls = []
def env_step(actions):
    t = len(calls)
    calls.append(actions.clone())
    term = torch.tensor([t == 1, False])
    return actions * 10, r[:, t].clone(), term, torch.zeros(2, dtype=torch.bool)
chunk = torch.tensor([[[0.1], [0.2], [0.3]], [[1.0], [2.0], [3.0]]])
out = {fn}(env_step, chunk)
assert out["num_executed"].tolist() == [2, 3], out["num_executed"]
assert out["num_executed"].dtype == torch.int64
assert out["valid_mask"].tolist() == [[True, True, False], [True, True, True]], out["valid_mask"]
assert torch.allclose(out["rewards"], torch.tensor([[1.0, 5.0, 0.0], [2.0, 2.0, 2.0]])), out["rewards"]
assert torch.allclose(out["chunk_reward"], torch.tensor([6.0, 6.0])), out["chunk_reward"]
assert out["terminated"].tolist() == [True, False] and out["truncated"].tolist() == [False, False]
assert torch.allclose(out["final_obs"], torch.tensor([[2.0], [30.0]])), out["final_obs"]
assert len(calls) == 3
"""},
        {"name": "Stops calling the simulator once every env has ended", "behavior": "budget.enforcement", "code": r"""
import torch
calls = []
def env_step(actions):
    t = len(calls)
    calls.append(t)
    term = torch.tensor([False, t == 1])
    trunc = torch.tensor([t == 0, False])
    return torch.zeros(2, 1), torch.ones(2), term, trunc
out = {fn}(env_step, torch.zeros(2, 4, 3))
assert calls == [0, 1], calls
assert out["num_executed"].tolist() == [1, 2]
assert out["terminated"].tolist() == [False, True] and out["truncated"].tolist() == [True, False]
assert out["valid_mask"].tolist() == [[True, False, False, False], [True, True, False, False]]
assert out["chunk_reward"].tolist() == [1.0, 2.0]
"""},
        {"name": "Seeded scripted batches match a per-env replay", "visibility": "unshown", "behavior": "rl.masking",
         "failure_message": "Each env's mask, rewards, flags and final_obs must stop at its own ending action and ignore every later row the simulator returns.",
         "code": _SCRIPTED_ENV + r"""
import random
for seed in range(6):
    rng = random.Random(seed)
    B, H = rng.randint(2, 6), rng.randint(2, 7)
    ends = [rng.choice([None] + list(range(H))) for _ in range(B)]
    ends[0] = None if seed % 2 == 0 else ends[0]
    kinds = [rng.choice(["term", "trunc", "both"]) for _ in range(B)]
    env = ScriptedEnv(ends, kinds, H, seed=seed)
    g = torch.Generator().manual_seed(100 + seed)
    chunk = torch.randn(B, H, 2, generator=g)
    exp = env.expected(chunk)
    out = {fn}(env, chunk)
    check(out, exp, (seed, ends, kinds))
"""},
        {"name": "An action that ends the episode at slot 0 is still executed", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "The ending action counts: an env that ends at slot 0 has num_executed 1, a mask with one True and that slot's reward.",
         "code": _SCRIPTED_ENV + r"""
env = ScriptedEnv([0, 2, 0], ["term", "trunc", "both"], 4, seed=3)
chunk = torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(1))
exp = env.expected(chunk)
out = {fn}(env, chunk)
check(out, exp)
assert out["num_executed"].tolist() == [1, 3, 1]
assert len(env.calls) == 3, len(env.calls)
"""},
        {"name": "Flags after the end do not change the chunk's flags", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "terminated and truncated come from each env's ending step only; flags the simulator reports after that belong to no episode.",
         "code": r"""
import torch
# env 0 truncates at slot 1 and then reports terminated at slot 2 (after the end); env 1 never ends.
term = torch.tensor([[False, False, True, True], [False, False, False, False]])
trunc = torch.tensor([[False, True, False, True], [False, False, False, False]])
n = [0]
def env_step(actions):
    t = n[0]; n[0] += 1
    return torch.full((2, 1), float(t)), torch.ones(2), term[:, t].clone(), trunc[:, t].clone()
out = {fn}(env_step, torch.zeros(2, 4, 1))
assert out["terminated"].tolist() == [False, False], out["terminated"]
assert out["truncated"].tolist() == [True, False], out["truncated"]
assert out["num_executed"].tolist() == [2, 4]
assert out["final_obs"][:, 0].tolist() == [1.0, 3.0], out["final_obs"]
assert out["chunk_reward"].tolist() == [2.0, 4.0]
"""},
        {"name": "Truncation alone ends the episode", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "A time-limit truncation ends the episode exactly like a termination: the env is masked after it and reports truncated, not terminated.",
         "code": _SCRIPTED_ENV + r"""
env = ScriptedEnv([1, 3, None], ["trunc", "trunc", "term"], 5, seed=11)
chunk = torch.randn(3, 5, 1, generator=torch.Generator().manual_seed(2))
exp = env.expected(chunk)
out = {fn}(env, chunk)
check(out, exp)
assert out["truncated"].tolist() == [True, True, False] and out["terminated"].tolist() == [False, False, False]
"""},
        {"name": "Call budget and the actions passed each call", "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "Call env_step once per slot with the full chunk[:, h] slice, keep going while any env is still running, and stop as soon as all have ended.",
         "code": _SCRIPTED_ENV + r"""
for ends, H in (([1, 4, 2], 7), ([0, 0], 5), ([2, None, 0], 4), ([5, 1, 3, 2], 6)):
    env = ScriptedEnv(ends, ["term"] * len(ends), H, seed=H)
    chunk = torch.randn(len(ends), H, 3, generator=torch.Generator().manual_seed(H))
    before = chunk.clone()
    exp = env.expected(chunk)
    out = {fn}(env, chunk)
    check(out, exp, ends)
    assert len(env.calls) == exp["calls"], (ends, len(env.calls), exp["calls"])
    for h, a in enumerate(env.calls):
        assert torch.equal(a, chunk[:, h]), (ends, h)
    assert torch.equal(chunk, before), "chunk was modified in place"
"""},
        {"name": "Simulator buffers reused across calls", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "env_step may overwrite the tensors it returned earlier; copy the observation (and anything else you keep) when you store it.",
         "code": _SCRIPTED_ENV + r"""
for seed in (0, 1, 2):
    env = ScriptedEnv([0, 2, None, 1], ["term", "both", "term", "trunc"], 4, seed=seed, reuse=True)
    chunk = torch.randn(4, 4, 2, generator=torch.Generator().manual_seed(seed))
    exp = env.expected(chunk)
    out = {fn}(env, chunk)
    check(out, exp, seed)
"""},
        {"name": "Output dtypes and shapes", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "rewards and chunk_reward keep env_step's reward dtype, num_executed is int64, valid_mask and the flags are bool.",
         "code": _SCRIPTED_ENV + r"""
env = ScriptedEnv([1, None], ["term", "term"], 3, seed=5, reward_dtype=torch.float64)
chunk = torch.randn(2, 3, 2, generator=torch.Generator().manual_seed(5))
exp = env.expected(chunk)
out = {fn}(env, chunk)
check(out, exp)
assert out["rewards"].dtype == torch.float64 and out["rewards"].shape == (2, 3), out["rewards"].dtype
assert out["chunk_reward"].dtype == torch.float64 and out["chunk_reward"].shape == (2,)
assert torch.allclose(out["chunk_reward"], exp["rewards"].sum(dim=1), atol=0, rtol=1e-12)
assert out["num_executed"].dtype == torch.int64 and out["num_executed"].shape == (2,)
assert out["valid_mask"].dtype == torch.bool and out["valid_mask"].shape == (2, 3)
assert out["terminated"].dtype == torch.bool and out["truncated"].dtype == torch.bool
assert out["final_obs"].shape == (2, 3)
"""},
        {"name": "Rejects malformed chunks", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a chunk that is not a 3-D floating-point tensor or has B or H equal to 0.",
         "code": r"""
import torch
def env_step(actions):
    raise AssertionError("env_step must not be called for a malformed chunk")
for bad in (torch.zeros(2, 3), torch.zeros(2, 3, 1, dtype=torch.long), torch.zeros(0, 3, 1), torch.zeros(2, 0, 1), [[[0.0]]]):
    try:
        {fn}(env_step, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted a malformed chunk: {bad!r}")
"""},
    ],
    "solution": '''import torch


def chunk_step(env_step, chunk):
    if not torch.is_tensor(chunk) or chunk.ndim != 3 or not chunk.is_floating_point():
        raise ValueError("chunk must be a floating-point tensor of shape (B, H, D)")
    B, H, _ = chunk.shape
    if B == 0 or H == 0:
        raise ValueError("chunk needs at least one env and one action")

    device = chunk.device
    active = torch.ones(B, dtype=torch.bool, device=device)
    num_executed = torch.zeros(B, dtype=torch.long, device=device)
    terminated = torch.zeros(B, dtype=torch.bool, device=device)
    truncated = torch.zeros(B, dtype=torch.bool, device=device)
    rewards = None
    final_obs = None

    for h in range(H):
        obs, reward, term, trunc = env_step(chunk[:, h])
        term = term.to(torch.bool)
        trunc = trunc.to(torch.bool)
        if rewards is None:
            rewards = torch.zeros(B, H, dtype=reward.dtype, device=device)
            final_obs = obs.clone()
        # Only envs whose episode is still running take this slot's results.
        rewards[active, h] = reward[active]
        final_obs[active] = obs[active]
        terminated[active] = term[active]
        truncated[active] = trunc[active]
        num_executed[active] += 1
        active = active & ~(term | trunc)
        if not active.any():
            break

    valid_mask = torch.arange(H, device=device).unsqueeze(0) < num_executed.unsqueeze(1)
    return {
        "num_executed": num_executed,
        "valid_mask": valid_mask,
        "rewards": rewards,
        "chunk_reward": rewards.sum(dim=1),
        "terminated": terminated,
        "truncated": truncated,
        "final_obs": final_obs,
    }
''',
    "interview_questions": interview(
        concept=[
            "Why does a VLA RL rollout need a per-action valid mask when it executes action chunks?",
            "Why must the action that ends an episode be counted as executed?",
        ],
        deep_dive=[
            "A batched GPU simulator cannot pause one env. What does it return for an env whose episode ended mid-chunk, and how do you keep that out of training data?",
            "RLinf folds termination with any() over the whole chunk, while ManiSkill's ActionRepeatWrapper keeps the flags of the ending step. When do the two disagree?",
            "Which observation should a value function bootstrap from when an env is truncated at slot 2 of an 8-action chunk, and why not the observation after slot 7?",
        ],
        tradeoffs=[
            "Auto-resetting envs inside a chunk versus only after it: what do you gain and what gets harder?",
            "Larger chunks mean fewer policy calls. What do they cost in wasted simulator steps and in credit assignment?",
        ],
    ),
}
