"""A synchronous vector environment with Gymnasium's NextStep and SameStep autoreset modes."""

from ._interview import interview

# Scripted sub-environments shared by the cases: episode k of an env lasts
# lengths[k % len(lengths)] steps and ends by termination ("term") or truncation ("trunc").
# The observation is [env_id, episode, t, u], where u comes from the env's own generator,
# so a re-seeded reset is visible in the observation.
_ENVS = r"""
import torch

class ScriptedEnv:
    def __init__(self, env_id, lengths, kinds, extra_info=None):
        self.env_id, self.lengths, self.kinds = env_id, lengths, kinds
        self.extra_info = extra_info
        self.episode, self.t, self.done = -1, 0, True
        self.log = []
        self.gen = torch.Generator().manual_seed(1000 + env_id)

    def _obs(self):
        u = torch.rand((), generator=self.gen, dtype=torch.float64).item()
        return torch.tensor([float(self.env_id), float(self.episode), float(self.t), u], dtype=torch.float64)

    def reset(self, seed=None):
        self.log.append(("reset", seed))
        if seed is not None:
            self.gen.manual_seed(seed)
        self.episode += 1
        self.t, self.done = 0, False
        return self._obs(), {"episode": self.episode}

    def step(self, action):
        assert not self.done, f"env {self.env_id}: a finished episode was stepped"
        a = float(action)
        self.log.append(("step", a))
        self.t += 1
        k = self.episode % len(self.lengths)
        end = self.t >= self.lengths[k]
        self.done = end
        term = end and self.kinds[k] == "term"
        trunc = end and self.kinds[k] == "trunc"
        reward = 100.0 * self.episode + 10.0 * self.t + a
        info = {"t": self.t, "success": term}
        if self.extra_info is not None:
            info.update(self.extra_info)
        return self._obs(), reward, term, trunc, info


def random_envs(seed, n):
    g = torch.Generator().manual_seed(seed)
    envs = []
    for i in range(n):
        lengths = torch.randint(1, 5, (4,), generator=g).tolist()
        kinds = ["term" if b else "trunc" for b in (torch.rand(4, generator=g) < 0.5).tolist()]
        envs.append(ScriptedEnv(i, lengths, kinds))
    return envs


def random_actions(seed, steps, n):
    g = torch.Generator().manual_seed(seed + 500)
    return torch.randint(0, 9, (steps, n), generator=g).to(torch.float64)


def batch_infos(per_env, n):
    # The batching rule from the description: lists with None for missing envs, bool masks.
    out = {}
    for i, info in enumerate(per_env):
        _put(out, info, i, n)
    return out


def _put(out, info, i, n):
    for key, value in info.items():
        if isinstance(value, dict):
            out[key] = _put(out.get(key, {}), value, i, n)
        else:
            out.setdefault(key, [None] * n)[i] = value
        out.setdefault("_" + key, torch.zeros(n, dtype=torch.bool))[i] = True
    return out


def same(got, want, where="info"):
    if isinstance(want, dict):
        assert isinstance(got, dict), (where, type(got))
        assert set(got) == set(want), (where, sorted(got), sorted(want))
        for key in want:
            same(got[key], want[key], f"{where}[{key!r}]")
    elif isinstance(want, torch.Tensor):
        assert isinstance(got, torch.Tensor) and torch.equal(got.cpu(), want), (where, got, want)
    elif isinstance(want, list):
        assert len(got) == len(want), (where, got, want)
        for j, (g, w) in enumerate(zip(got, want)):
            same(g, w, f"{where}[{j}]")
    else:
        assert got == want and (got is None) == (want is None), (where, got, want)

"""

TASK = {
    "title": "Vector Env Autoreset: NextStep vs SameStep",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "SyncVectorEnv",
    "description_en": r"""Run `N` environments in lock-step behind one batched `reset`/`step` API, and reset an env automatically when its episode ends. Implement both of Gymnasium's autoreset modes, `"NextStep"` and `"SameStep"`.

**Signature:** `SyncVectorEnv(envs, autoreset_mode="NextStep")` with two methods:
- `reset(seed=None) -> (obs, info)`
- `step(actions) -> (obs, rewards, terminated, truncated, info)`

**Sub-environments.** `envs` is a list of `N` objects. Each has `reset(seed=None) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`. `obs` is a tensor (the same shape for every env), `reward` a float, the flags bools and `info` a dict.

**Construction.** Raise `ValueError` when `autoreset_mode` is not `"NextStep"` or `"SameStep"`, or when `envs` is empty.

**reset(seed=None):**
- Reset every env. With an int `seed`, env `i` gets `seed + i`. With `None`, every env gets `None`. With a list, env `i` gets `seed[i]`; raise `ValueError` if its length is not `N`.
- Forget any pending autoreset, so the next `step` steps every env.
- Return the stacked observations, shape `(N, *obs_shape)`, and the batched reset infos.

**step(actions):**
- `actions` has length `N`; env `i` receives `actions[i]`. Raise `ValueError` for any other length.
- Return `obs` stacked to `(N, *obs_shape)`, `rewards` as a float64 tensor `(N,)`, and `terminated` and `truncated` as bool tensors `(N,)`.
- An env's episode has ended when its `terminated` or `truncated` is True.
- An autoreset calls the env's `reset()` with no seed.

**"NextStep" mode:**
- An env whose episode ended in the previous `step` call is not stepped. It is reset instead and its action is ignored.
- For that env, return the reset observation, reward `0.0`, both flags False and the reset info.
- Every other env is stepped normally. An env whose episode ends returns its true last observation.

**"SameStep" mode:**
- Every env is stepped with its action. When its episode ends, reset it in the same call.
- For a reset env, return the reset observation, the reward and flags of the step that ended the episode, and the reset info as its info.
- Put that step's observation in `info["final_obs"]` and that step's info in `info["final_info"]`, batched like any other key.
- When no env's episode ends, `info` has no `final_obs` or `final_info` key.

**Info batching.** Build `info` from the per-env info dicts:
- For every key `k` that any env reports, `info[k]` is a list of length `N` with that env's value at its index and `None` for envs that did not report `k`.
- `info["_" + k]` is a bool tensor `(N,)`, True exactly where an env reported `k`.
- A value that is a dict is batched the same way, recursively, so `info[k]` is then a dict of lists and masks.

**Example:** env 0 ends after 1 step, env 1 after 3 steps.

    vec = SyncVectorEnv(envs, "SameStep"); vec.reset(seed=5)  # env seeds 5, 6
    obs, rew, term, trunc, info = vec.step(torch.tensor([1.0, 2.0]))
    # term = [True, False]; obs[0] is env 0's reset observation
    # info["final_obs"] = [last obs of env 0, None], info["_final_obs"] = [True, False]
    # info["final_info"]["t"] = [1, None]; info["episode"] = [1, None], info["t"] = [None, 1]

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why autoreset.** A vectorized rollout steps all envs together, but episodes end at different times. Waiting for the slowest env idles the others, so the vector env resets a finished env by itself and keeps it running. The question is when the reset happens and how the caller still sees the true end of the episode.

**NextStep.** Gymnasium's current `SyncVectorEnv` (1.x) defaults to this mode. A step that ends an episode returns the true last observation; the following call resets that env and returns the reset observation with reward 0 and both flags False. That call is not a real transition: no action was executed. A training loop that stores it must mask it out, for example by keeping the previous call's `terminated | truncated` and dropping the next row for those envs.

**SameStep.** This was the behavior of Gymnasium 0.29, where the keys were `final_observation` and `final_info`. Every call does useful work, but for a reset env the returned observation is the start of a new episode, not the successor of the action. Bootstrapping a truncated episode needs the value of the true last observation, which only `info["final_obs"]` holds.

**ManiSkill's variant.** `ManiSkillVectorEnv` hard-codes SameStep and keeps the old names: `info["final_observation"]` holds a clone of the whole batched observation, and `_final_observation`, `_final_info` and `_elapsed_steps` are all set to the done mask. Its PPO baseline evaluates the value of `final_observation[done_mask]` for those envs.

**Gymnasium details left out.** Gymnasium also has a `"Disabled"` mode (the caller resets envs through `reset(options={"reset_mask": ...})`). Its info batching stores numbers in zero-filled numpy arrays typed by the first value instead of lists with `None`, and keeps `final_obs` as an object array.""",
    "advisory_prerequisites": ["rl_eval_loop", "gae_advantage"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does the vector env have to remember between two step calls in NextStep mode, and when does it update that memory? In SameStep mode, which observation and which info does a finished env return at the top level, and where do the step's own observation and info go? What happens to a pending autoreset when the caller resets the whole vector env?"},
        {"level": 2, "kind": "analysis", "content": "Keep a per-env pending-reset flag, cleared in reset() and set at the end of step() from terminated or truncated. In step(), branch per env on the mode: NextStep with a pending flag calls env.reset() and writes reward 0 and False flags; otherwise call env.step(action). In SameStep, after a step that ends the episode, first add final_obs and final_info for that env, then call env.reset() and use its observation and info. Write one helper that adds one env's info dict at index i: create a list of N Nones and a False mask the first time a key appears, recurse for dict values."},
    ],
    "model_connections": [
        "Gymnasium's SyncVectorEnv (1.x, at the pinned commit) branches on its AutoresetMode enum (NEXT_STEP, SAME_STEP, DISABLED); the first two branches are the ones built here, and NEXT_STEP is the constructor default.",
        "ManiSkill3's ManiSkillVectorEnv runs GPU-batched robot tasks with SameStep autoreset and the older final_observation key; its PPO baseline uses that observation to compute final_values for GAE.",
        "The truncation bootstrap in GAE needs the true last observation: under SameStep it lives only in info['final_obs'], under NextStep it is the observation returned by the step that ended the episode.",
    ],
    "pro_con_analysis": {
        "pros": [
            "NextStep: the call that ends an episode returns its true last observation in obs itself, so no extra info key is needed to see it.",
            "SameStep: every call executes an action in every env, and the true final observation is available in the same call for value bootstrapping.",
        ],
        "cons": [
            "NextStep: one call per finished episode per env is not a real transition; it costs an env slot and must be masked out of training data.",
            "SameStep: for a finished env the returned observation does not follow from the action, so code that bootstraps from it, or ignores final_obs, silently trains on the wrong value.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/vector/sync_vector_env.py",
            "symbol": "SyncVectorEnv.reset, SyncVectorEnv.step",
            "license": "MIT",
            "adapted": "The seed expansion of reset and the NEXT_STEP and SAME_STEP branches of step, including where final_obs and final_info are added and which info becomes the top-level info.",
            "simplifications": "Takes built envs instead of env_fns; modes are the strings 'NextStep' and 'SameStep' (DISABLED and the reset_mask partial reset are dropped); "
                               "observations are stacked tensors instead of space-concatenated numpy arrays; rewards and flags are torch tensors; "
                               "an empty env list raises ValueError; reset takes no options.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/vector/vector_env.py",
            "symbol": "AutoresetMode, VectorEnv._add_info",
            "license": "MIT",
            "adapted": "The mode names and the recursive per-key info batching with _key masks.",
            "simplifications": "Leaf values are batched into Python lists with None for missing envs, instead of numpy arrays zero-filled and typed by the first value; "
                               "masks are torch bool tensors; final_obs needs no special case because observations are tensors, not dicts.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "7cf49527c945297f0bfeb89edab417d87819ab00",
            "path": "gymnasium/vector/sync_vector_env.py",
            "symbol": "SyncVectorEnv.step_wait",
            "license": "MIT",
            "adapted": "Cited for history only (tag v0.29.1): a finished env was reset inside the same step and its last observation and info were stored as info['final_observation'] and info['final_info'].",
            "simplifications": "Not implemented; the exercise follows the current SameStep key names final_obs and final_info.",
        },
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "mani_skill/vector/wrappers/gymnasium.py",
            "symbol": "ManiSkillVectorEnv.__init__, ManiSkillVectorEnv.step",
            "license": "Apache-2.0",
            "adapted": "Cited for the naming variant only: SameStep hard-coded, info['final_observation'] holding the whole batch, and _final_observation, _final_info and _elapsed_steps set to the done mask.",
            "simplifications": "Not implemented; the exercise uses current Gymnasium names and per-env final_obs entries.",
        },
    ],
    "tests": [
        {"name": "NextStep resets on the call after the episode ends", "behavior": "rl.trajectory", "code": _ENVS + r"""
envs = [ScriptedEnv(0, [1], ["term"]), ScriptedEnv(1, [3], ["trunc"])]
vec = {fn}(envs)
obs, info = vec.reset(seed=5)
assert obs.shape == (2, 4), obs.shape
assert [e.log for e in envs] == [[("reset", 5)], [("reset", 6)]], [e.log for e in envs]
assert info["episode"] == [0, 0] and info["_episode"].tolist() == [True, True], info
obs, rew, term, trunc, info = vec.step(torch.tensor([1.0, 2.0]))
assert rew.tolist() == [11.0, 12.0] and term.tolist() == [True, False] and trunc.tolist() == [False, False]
assert obs[0, 1:3].tolist() == [0.0, 1.0], "env 0 returns its true last observation (episode 0, t = 1)"
obs, rew, term, trunc, info = vec.step(torch.tensor([7.0, 3.0]))
assert envs[0].log[-1] == ("reset", None), "env 0 is reset with no seed and its action is ignored"
assert rew.tolist() == [0.0, 23.0] and term.tolist() == [False, False] and trunc.tolist() == [False, False]
assert obs[0, 1:3].tolist() == [1.0, 0.0], "env 0 returns the first observation of episode 1"
assert info["episode"] == [1, None] and info["t"] == [None, 2], info
assert info["_episode"].tolist() == [True, False] and info["_t"].tolist() == [False, True], info
"""},
        {"name": "SameStep resets in the same call and reports final_obs", "behavior": "rl.trajectory", "code": _ENVS + r"""
envs = [ScriptedEnv(0, [1], ["term"]), ScriptedEnv(1, [3], ["trunc"])]
vec = {fn}(envs, "SameStep")
vec.reset(seed=5)
obs, rew, term, trunc, info = vec.step(torch.tensor([1.0, 2.0]))
assert rew.tolist() == [11.0, 12.0] and term.tolist() == [True, False] and trunc.tolist() == [False, False]
assert obs[0, 1:3].tolist() == [1.0, 0.0], "obs[0] is env 0's reset observation"
assert info["final_obs"][0][1:3].tolist() == [0.0, 1.0] and info["final_obs"][1] is None, info["final_obs"]
assert info["_final_obs"].tolist() == [True, False] and info["_final_info"].tolist() == [True, False]
assert info["final_info"]["t"] == [1, None] and info["final_info"]["success"] == [True, None], info["final_info"]
assert info["final_info"]["_t"].tolist() == [True, False]
assert info["episode"] == [1, None] and info["t"] == [None, 1], info
obs, rew, term, trunc, info = vec.step(torch.tensor([7.0, 3.0]))
assert envs[0].log[-2] == ("step", 7.0), "in SameStep the action after a reset is executed"
assert rew.tolist() == [117.0, 23.0], rew
"""},
        {"name": "Seeded NextStep rollouts match a per-env simulation", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "In NextStep, an env that ended on the previous call is reset instead of stepped, with reward 0, False flags and the reset info; every other env is stepped.",
         "code": _ENVS + r"""
for seed in (0, 3, 11, 29):
    n = 2 + seed % 3
    acts = random_actions(seed, 14, n)
    want_envs, envs = random_envs(seed, n), random_envs(seed, n)
    vec = {fn}(envs, "NextStep")
    vec.reset(seed=seed)
    for e in want_envs:
        e.reset(seed=seed + e.env_id)
    pending = [False] * n
    for step, a in enumerate(acts):
        obs, rew, term, trunc, info = vec.step(a)
        rows = []
        for i, e in enumerate(want_envs):
            if pending[i]:
                o, inf = e.reset(); r, te, tr = 0.0, False, False
            else:
                o, r, te, tr, inf = e.step(a[i])
            pending[i] = te or tr
            rows.append((o, r, te, tr, inf))
        where = (seed, step)
        assert torch.equal(obs, torch.stack([row[0] for row in rows])), where
        assert rew.tolist() == [row[1] for row in rows], (where, rew)
        assert term.tolist() == [row[2] for row in rows] and trunc.tolist() == [row[3] for row in rows], where
        same(info, batch_infos([row[4] for row in rows], n), f"info at {where}")
    assert [e.log for e in envs] == [e.log for e in want_envs], seed
"""},
        {"name": "Seeded SameStep rollouts match a per-env simulation", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "In SameStep, a finished env keeps its step reward and flags, returns the reset observation and reset info, and its step observation and info go into final_obs and final_info.",
         "code": _ENVS + r"""
for seed in (1, 4, 12, 31):
    n = 2 + seed % 3
    acts = random_actions(seed, 14, n)
    want_envs, envs = random_envs(seed, n), random_envs(seed, n)
    vec = {fn}(envs, "SameStep")
    vec.reset(seed=seed)
    for e in want_envs:
        e.reset(seed=seed + e.env_id)
    for step, a in enumerate(acts):
        obs, rew, term, trunc, info = vec.step(a)
        rows = []
        for i, e in enumerate(want_envs):
            o, r, te, tr, inf = e.step(a[i])
            if te or tr:
                last_o, last_inf = o, inf
                o, inf = e.reset()
                inf = {**inf, "final_obs": last_o, "final_info": last_inf}
            rows.append((o, r, te, tr, inf))
        where = (seed, step)
        assert torch.equal(obs, torch.stack([row[0] for row in rows])), where
        assert rew.tolist() == [row[1] for row in rows], (where, rew)
        assert term.tolist() == [row[2] for row in rows] and trunc.tolist() == [row[3] for row in rows], where
        same(info, batch_infos([row[4] for row in rows], n), f"info at {where}")
    assert [e.log for e in envs] == [e.log for e in want_envs], seed
"""},
        {"name": "SameStep final_obs is the observation before the reset", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "info['final_obs'][i] must be the observation returned by the step that ended the episode, not the reset observation.",
         "code": _ENVS + r"""
envs = [ScriptedEnv(0, [2], ["trunc"]), ScriptedEnv(1, [3], ["term"]), ScriptedEnv(2, [9], ["term"])]
vec = {fn}(envs, "SameStep")
vec.reset(seed=40)
vec.step(torch.zeros(3))
obs, rew, term, trunc, info = vec.step(torch.zeros(3))
assert trunc.tolist() == [True, False, False] and term.tolist() == [False, False, False]
assert info["final_obs"][0][1:3].tolist() == [0.0, 2.0], "final_obs holds episode 0 at t = 2"
assert obs[0, 1:3].tolist() == [1.0, 0.0], "the returned observation is episode 1 at t = 0"
assert not torch.equal(info["final_obs"][0], obs[0])
obs, rew, term, trunc, info = vec.step(torch.zeros(3))
assert term.tolist() == [False, True, False]
assert info["final_obs"][1][1:3].tolist() == [0.0, 3.0] and info["final_obs"][0] is None and info["final_obs"][2] is None
assert info["_final_obs"].tolist() == [False, True, False]
"""},
        {"name": "NextStep reset call has zero reward and cleared flags", "visibility": "unshown", "behavior": "rl.masking",
         "failure_message": "On the call that resets a finished env in NextStep, return reward 0.0 and terminated = truncated = False, not the values from the ending step.",
         "code": _ENVS + r"""
envs = [ScriptedEnv(0, [2], ["term"]), ScriptedEnv(1, [2], ["trunc"])]
vec = {fn}(envs, "NextStep")
vec.reset(seed=1)
vec.step(torch.tensor([5.0, 6.0]))
obs, rew, term, trunc, info = vec.step(torch.tensor([5.0, 6.0]))
assert rew.tolist() == [25.0, 26.0] and term.tolist() == [True, False] and trunc.tolist() == [False, True]
obs, rew, term, trunc, info = vec.step(torch.tensor([8.0, 8.0]))
assert rew.tolist() == [0.0, 0.0], rew
assert term.tolist() == [False, False] and trunc.tolist() == [False, False], (term, trunc)
assert obs[:, 1:3].tolist() == [[1.0, 0.0], [1.0, 0.0]], obs
assert "final_obs" not in info and "t" not in info, info
obs, rew, term, trunc, info = vec.step(torch.tensor([1.0, 2.0]))
assert rew.tolist() == [111.0, 112.0], "after the reset call both envs step normally"
"""},
        {"name": "Autoresets call reset without a seed", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "An autoreset must call env.reset() with no seed; re-using the initial seed replays the same episode.",
         "code": _ENVS + r"""
for mode in ("NextStep", "SameStep"):
    envs = [ScriptedEnv(0, [1], ["term"]), ScriptedEnv(1, [2], ["trunc"])]
    vec = {fn}(envs, mode)
    vec.reset(seed=9)
    for _ in range(4):
        vec.step(torch.zeros(2))
    for i, e in enumerate(envs):
        resets = [entry for entry in e.log if entry[0] == "reset"]
        assert resets[0] == ("reset", 9 + i), (mode, e.log)
        assert len(resets) >= 2 and all(entry == ("reset", None) for entry in resets[1:]), (mode, e.log)
"""},
        {"name": "Reset seeds: int, None and list", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "reset(seed=s) gives env i the seed s + i, reset() gives every env None, and a list gives env i seed[i].",
         "code": _ENVS + r"""
envs = [ScriptedEnv(i, [3], ["term"]) for i in range(4)]
vec = {fn}(envs)
obs, info = vec.reset(seed=100)
assert [e.log[-1] for e in envs] == [("reset", 100 + i) for i in range(4)], [e.log for e in envs]
assert len(set(obs[:, 3].tolist())) == 4, "different seeds give different observations"
vec.reset()
assert [e.log[-1] for e in envs] == [("reset", None)] * 4
vec.reset(seed=[7, 3, None, 7])
assert [e.log[-1][1] for e in envs] == [7, 3, None, 7]
assert info["episode"] == [0, 0, 0, 0] and info["_episode"].tolist() == [True] * 4
"""},
        {"name": "reset forgets a pending autoreset", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "reset() must clear the pending-autoreset state, so the next step steps every env instead of resetting it again.",
         "code": _ENVS + r"""
envs = [ScriptedEnv(0, [1, 3], ["term", "term"]), ScriptedEnv(1, [3], ["trunc"])]
vec = {fn}(envs, "NextStep")
vec.reset(seed=0)
obs, rew, term, trunc, info = vec.step(torch.tensor([1.0, 1.0]))
assert term.tolist() == [True, False]
vec.reset(seed=50)
obs, rew, term, trunc, info = vec.step(torch.tensor([4.0, 4.0]))
assert envs[0].log[-1] == ("step", 4.0), envs[0].log
assert rew.tolist() == [114.0, 114.0], rew
"""},
        {"name": "final keys appear only when an episode ends", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Add final_obs and final_info only for envs whose episode ended in this SameStep call; with none ended, info has neither key, and NextStep never adds them.",
         "code": _ENVS + r"""
envs = [ScriptedEnv(0, [3], ["term"]), ScriptedEnv(1, [3], ["trunc"])]
vec = {fn}(envs, "SameStep")
vec.reset(seed=2)
obs, rew, term, trunc, info = vec.step(torch.zeros(2))
assert "final_obs" not in info and "final_info" not in info and "_final_obs" not in info, sorted(info)
assert info["t"] == [1, 1] and info["_t"].tolist() == [True, True]
vec.step(torch.zeros(2))
obs, rew, term, trunc, info = vec.step(torch.zeros(2))
assert info["_final_obs"].tolist() == [True, True] and info["_final_info"].tolist() == [True, True]
assert info["final_info"]["success"] == [True, False], info["final_info"]
envs = [ScriptedEnv(0, [1], ["term"])]
vec = {fn}(envs, "NextStep")
vec.reset(seed=2)
for _ in range(3):
    obs, rew, term, trunc, info = vec.step(torch.zeros(1))
    assert "final_obs" not in info and "final_info" not in info, sorted(info)
"""},
        {"name": "SameStep top-level info for a reset env is the reset info", "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "In SameStep, add the step info under final_info first, then reset and use the reset info as the env's top-level info.",
         "code": _ENVS + r"""
envs = [ScriptedEnv(0, [1], ["trunc"]), ScriptedEnv(1, [5], ["term"]), ScriptedEnv(2, [1], ["term"])]
vec = {fn}(envs, "SameStep")
vec.reset(seed=3)
obs, rew, term, trunc, info = vec.step(torch.zeros(3))
assert info["t"] == [None, 1, None] and info["_t"].tolist() == [False, True, False], info
assert info["success"] == [None, False, None] and info["_success"].tolist() == [False, True, False], info
assert info["episode"] == [1, None, 1] and info["_episode"].tolist() == [True, False, True], info
assert info["final_info"]["t"] == [1, None, 1] and info["final_info"]["success"] == [False, None, True], info["final_info"]
assert info["final_info"]["_success"].tolist() == [True, False, True]
"""},
        {"name": "Info batching: None fill, masks and nested dicts", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Batch each info key into a length-N list with None where an env lacks it plus a bool '_key' mask, and batch dict values recursively.",
         "code": _ENVS + r"""
envs = [
    ScriptedEnv(0, [4], ["term"], extra_info={"stats": {"dist": 0.5, "contacts": 2}}),
    ScriptedEnv(1, [4], ["term"]),
    ScriptedEnv(2, [4], ["term"], extra_info={"stats": {"dist": 1.5}, "tag": "x"}),
]
vec = {fn}(envs, "NextStep")
vec.reset(seed=8)
obs, rew, term, trunc, info = vec.step(torch.zeros(3))
assert sorted(info) == sorted(["t", "_t", "success", "_success", "stats", "_stats", "tag", "_tag"]), sorted(info)
assert info["tag"] == [None, None, "x"] and info["_tag"].tolist() == [False, False, True]
assert info["_stats"].tolist() == [True, False, True]
stats = info["stats"]
assert isinstance(stats, dict), type(stats)
assert stats["dist"] == [0.5, None, 1.5] and stats["_dist"].tolist() == [True, False, True], stats
assert stats["contacts"] == [2, None, None] and stats["_contacts"].tolist() == [True, False, False], stats
assert info["_t"].dtype == torch.bool and info["_t"].shape == (3,)
"""},
        {"name": "Output shapes and dtypes, fresh tensors each call", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Return obs stacked to (N, *obs_shape), rewards as float64 (N,), and terminated/truncated as bool (N,), as new tensors each call.",
         "code": _ENVS + r"""
for mode in ("NextStep", "SameStep"):
    envs = [ScriptedEnv(i, [1, 2], ["term", "trunc"]) for i in range(3)]
    vec = {fn}(envs, mode)
    obs, info = vec.reset(seed=0)
    assert obs.shape == (3, 4) and obs.dtype == torch.float64, (obs.shape, obs.dtype)
    out1 = vec.step(torch.ones(3))
    saved = [t.clone() for t in out1[:4]]
    obs, rew, term, trunc, info = out1
    assert obs.shape == (3, 4) and rew.shape == (3,) and term.shape == (3,) and trunc.shape == (3,)
    assert rew.dtype == torch.float64 and term.dtype == torch.bool and trunc.dtype == torch.bool, (rew.dtype, term.dtype)
    vec.step(torch.ones(3)); vec.step(torch.ones(3))
    for before, now in zip(saved, out1[:4]):
        assert torch.equal(before, now), (mode, "a returned tensor was changed by a later step")
"""},
        {"name": "Single env with one-step episodes", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With one env and one-step episodes, NextStep alternates step and reset calls while SameStep ends and resets the episode on every call.",
         "code": _ENVS + r"""
env = ScriptedEnv(0, [1], ["term"])
vec = {fn}([env], "NextStep")
vec.reset(seed=4)
got = []
for k in range(4):
    obs, rew, term, trunc, info = vec.step(torch.tensor([float(k)]))
    got.append((rew.item(), term.item()))
assert got == [(10.0, True), (0.0, False), (112.0, True), (0.0, False)], got
assert [entry[0] for entry in env.log] == ["reset", "step", "reset", "step", "reset"], env.log
env = ScriptedEnv(0, [1], ["term"])
vec = {fn}([env], "SameStep")
vec.reset(seed=4)
for k in range(3):
    obs, rew, term, trunc, info = vec.step(torch.tensor([float(k)]))
    assert rew.item() == 100.0 * k + 10.0 + k and term.item(), (k, rew)
    assert obs[0, 1:3].tolist() == [k + 1.0, 0.0] and info["final_obs"][0][1:3].tolist() == [float(k), 1.0]
"""},
        {"name": "Rejects a bad mode, no envs, wrong action count and bad seed list", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for an unknown autoreset_mode, an empty env list, len(actions) != N, or a seed list whose length is not N.",
         "code": _ENVS + r"""
def raises(call):
    try:
        call()
    except ValueError:
        return True
    return False
assert raises(lambda: {fn}([ScriptedEnv(0, [2], ["term"])], "Disabled"))
assert raises(lambda: {fn}([ScriptedEnv(0, [2], ["term"])], "next_step"))
assert raises(lambda: {fn}([], "NextStep"))
vec = {fn}([ScriptedEnv(0, [2], ["term"]), ScriptedEnv(1, [2], ["term"])])
vec.reset(seed=0)
assert raises(lambda: vec.step(torch.zeros(3)))
assert raises(lambda: vec.step(torch.zeros(1)))
assert raises(lambda: vec.reset(seed=[1, 2, 3]))
"""},
    ],
    "solution": '''import torch

_MODES = ("NextStep", "SameStep")


class SyncVectorEnv:
    def __init__(self, envs, autoreset_mode="NextStep"):
        if autoreset_mode not in _MODES:
            raise ValueError(f"autoreset_mode must be one of {_MODES}, got {autoreset_mode!r}")
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("envs must not be empty")
        self.num_envs = len(self.envs)
        self.autoreset_mode = autoreset_mode
        self._pending_reset = [False] * self.num_envs

    def _add_info(self, infos, env_info, i):
        """Write one env's info dict into the batched dict at index i."""
        n = self.num_envs
        for key, value in env_info.items():
            if isinstance(value, dict):
                infos[key] = self._add_info(infos.get(key, {}), value, i)
            else:
                if key not in infos:
                    infos[key] = [None] * n
                infos[key][i] = value
            mask = infos.get("_" + key)
            if mask is None:
                mask = torch.zeros(n, dtype=torch.bool)
            mask[i] = True
            infos["_" + key] = mask
        return infos

    def reset(self, seed=None):
        n = self.num_envs
        if seed is None:
            seeds = [None] * n
        elif isinstance(seed, int) and not isinstance(seed, bool):
            seeds = [seed + i for i in range(n)]
        else:
            seeds = list(seed)
            if len(seeds) != n:
                raise ValueError(f"expected {n} seeds, got {len(seeds)}")
        self._pending_reset = [False] * n
        observations, infos = [], {}
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds)):
            obs, env_info = env.reset(seed=env_seed)
            observations.append(obs)
            infos = self._add_info(infos, env_info, i)
        return torch.stack(observations), infos

    def step(self, actions):
        n = self.num_envs
        if len(actions) != n:
            raise ValueError(f"expected {n} actions, got {len(actions)}")
        observations, rewards, terminated, truncated, infos = [], [], [], [], {}
        for i, env in enumerate(self.envs):
            if self.autoreset_mode == "NextStep" and self._pending_reset[i]:
                # The episode ended on the previous call: reset, ignore the action.
                obs, env_info = env.reset()
                reward, term, trunc = 0.0, False, False
            else:
                obs, reward, term, trunc, env_info = env.step(actions[i])
                if self.autoreset_mode == "SameStep" and (term or trunc):
                    infos = self._add_info(infos, {"final_obs": obs, "final_info": env_info}, i)
                    obs, env_info = env.reset()
            infos = self._add_info(infos, env_info, i)
            observations.append(obs)
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]
        return (
            torch.stack(observations),
            torch.tensor(rewards, dtype=torch.float64),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
            infos,
        )
''',
    "interview_questions": interview(
        concept=[
            "Why does a vectorized environment reset finished sub-environments by itself instead of waiting for the caller to reset them?",
            "For an env whose episode just ended, what does one step call return under Gymnasium's NextStep autoreset mode, and what under SameStep?",
        ],
        deep_dive=[
            "In NextStep mode, which rows of a rollout buffer are not real transitions, and how do you find and mask them using only the returned flags?",
            "In SameStep mode, why must a PPO learner read info['final_obs'] to bootstrap a truncated episode, and what goes wrong if it uses the returned observation instead?",
            "Why does an autoreset call env.reset() without a seed, and what would go wrong if it reused the seed passed to the last vector reset?",
        ],
        tradeoffs=[
            "NextStep versus SameStep for a GPU-batched robot simulator: which wastes more environment steps, and which is easier to get right in a PPO training loop?",
            "Batching info as per-key arrays with masks versus keeping a list of per-env dicts: what does each make easy or hard for logging and for value bootstrapping?",
        ],
    ),
}
