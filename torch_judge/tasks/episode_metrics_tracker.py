"""Per-environment episode metrics for a vectorized robot env with partial resets."""

from ._interview import interview

# A plain-Python model of the contract, one environment at a time, used by the
# seeded unshown cases. It shares no structure with the reference solution.
_LOOP_MODEL = r"""
import random
import torch

def loop_model(n, schedule, ignore_terminations, auto_reset, use_success, use_fail):
    ret = [0.0] * n
    length = [0] * n
    s_once = [False] * n
    f_once = [False] * n
    out = []
    for rew, term, trunc, succ, fail in schedule:
        row = {"return": [], "episode_len": [], "reward": [], "success_once": [], "fail_once": [],
               "success_at_end": [], "fail_at_end": [], "dones": [], "terminations": []}
        for i in range(n):
            ret[i] = float(torch.tensor(ret[i], dtype=torch.float32) + torch.tensor(rew[i], dtype=torch.float32))
            length[i] += 1
            if use_success and succ[i]:
                s_once[i] = True
            if use_fail and fail[i]:
                f_once[i] = True
            t = False if ignore_terminations else term[i]
            d = t or trunc[i]
            row["return"].append(ret[i])
            row["episode_len"].append(length[i])
            row["reward"].append(ret[i] / length[i])
            row["success_once"].append(s_once[i])
            row["fail_once"].append(f_once[i])
            row["success_at_end"].append(succ[i])
            row["fail_at_end"].append(fail[i])
            row["dones"].append(d)
            row["terminations"].append(t)
            if d and auto_reset:
                ret[i], length[i], s_once[i], f_once[i] = 0.0, 0, False, False
        out.append(row)
    return out

def random_schedule(seed, n, steps, p_term=0.15, p_trunc=0.08):
    rng = random.Random(seed)
    schedule = []
    for _ in range(steps):
        rew = [round(rng.uniform(-1.0, 2.0), 3) for _ in range(n)]
        succ = [rng.random() < 0.2 for _ in range(n)]
        fail = [rng.random() < 0.1 for _ in range(n)]
        term = [s or f or rng.random() < p_term for s, f in zip(succ, fail)]
        trunc = [rng.random() < p_trunc for _ in range(n)]
        schedule.append((rew, term, trunc, succ, fail))
    return schedule

def as_tensors(rew, term, trunc, succ, fail):
    return (torch.tensor(rew, dtype=torch.float32), torch.tensor(term), torch.tensor(trunc),
            torch.tensor(succ), torch.tensor(fail))

def check_row(got, want, keys, label):
    ep = got["episode"]
    for k in keys:
        assert k in ep, (label, "missing key", k)
        g = ep[k].tolist()
        w = want[k]
        if k in ("return", "reward"):
            assert all(abs(a - b) <= 1e-4 for a, b in zip(g, w)), (label, k, g, w)
        else:
            assert g == w, (label, k, g, w)
    assert got["dones"].tolist() == want["dones"], (label, "dones", got["dones"].tolist(), want["dones"])
    assert got["terminations"].tolist() == want["terminations"], (label, "terminations")
"""

TASK = {
    "title": "Episode Metrics Tracker for Vectorized Robot Envs",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "EpisodeMetricsTracker",
    "description_en": r"""Track per-environment episode metrics for `N` robot environments that step together and reset one at a time, the way ManiSkill's vector env does with `record_metrics=True`.

**Signature:** `EpisodeMetricsTracker(num_envs, ignore_terminations=False, auto_reset=True)` with two methods:
- `step(rewards, terminations, truncations, success=None, fail=None) -> dict`
- `reset(env_idx=None) -> None`

**Inputs to step.** Every argument is a 1-D tensor of shape `(num_envs,)`. `rewards` is numeric; `terminations`, `truncations`, `success` and `fail` are `torch.bool`. `success` and `fail` are optional: pass `None` when the task has no such criterion.

**What step does, in order:**
- Add `rewards` to each environment's running return (kept in `torch.float32`) and add 1 to its episode length.
- `success_once` becomes `success_once OR success`; `fail_once` becomes `fail_once OR fail`. Once true, they stay true until that environment is reset.
- Effective terminations: all `False` when `ignore_terminations` is true, otherwise `terminations`. `dones = effective terminations OR truncations`.
- Build the `episode` snapshot (keys below) from the values after this step's update.
- If `auto_reset` is true, reset every environment where `dones` is true, after taking the snapshot. Other environments keep their running values.

**Return value of step:** a dict with exactly three keys.
- `"episode"`: a dict of `(num_envs,)` tensors. Always `"return"` (float32), `"episode_len"` (an integer dtype) and `"reward"` (float32, `return / episode_len`). `"success_once"` (bool) only when `success` was given, `"fail_once"` (bool) only when `fail` was given. When `ignore_terminations` is true, also `"success_at_end"` (a copy of this step's `success`) and `"fail_at_end"` (a copy of this step's `fail`), each only when that input was given.
- `"terminations"`: the effective terminations (bool).
- `"dones"`: the `dones` mask (bool).

**reset(env_idx=None):** set return, length, `success_once` and `fail_once` to zero or `False` for the listed environment indices (a list or a 1-D integer tensor), or for all environments when `env_idx` is `None`. Others are untouched.

**Constraints:**
- The snapshot holds copies: later `step` or `reset` calls never change a dict that was already returned.
- Never modify any input tensor in place.
- With `auto_reset=False`, nothing is reset automatically; the values keep accumulating until `reset` is called.
- Raise `ValueError` when `num_envs` is not a positive integer, when an input is not shaped `(num_envs,)`, or when a flag input is not `torch.bool`.

**Example:**

    tracker = EpisodeMetricsTracker(2)
    out = tracker.step(torch.tensor([1.0, 2.0]), torch.tensor([False, False]),
                       torch.tensor([False, False]), success=torch.tensor([False, True]))
    # episode: return [1, 2], episode_len [1, 1], reward [1, 2], success_once [False, True]
    out = tracker.step(torch.tensor([3.0, 4.0]), torch.tensor([True, False]),
                       torch.tensor([False, False]), success=torch.tensor([True, False]))
    # episode: return [4, 6], episode_len [2, 2], reward [2, 3], success_once [True, True]
    # dones [True, False]; env 0 is reset after the snapshot, env 1 keeps going

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why per-environment state.** A GPU simulator steps thousands of environments in one batch, and with partial resets each one finishes its episode at a different step. The tracker therefore keeps one running return, length and success flag per environment and restarts only the ones that just finished. ManiSkill's wrapper resets finished environments inside the same `step` call (Gymnasium calls this `AutoresetMode.SAME_STEP`) and hands the finished episode's numbers back in `infos["final_info"]["episode"]`, with `infos["_final_info"]` as the mask of finished environments. Because the reset zeroes the running buffers, the snapshot has to be a clone.

**success_once versus success_at_end.** `success_once` asks whether the robot ever reached the goal during the episode; `success_at_end` asks whether it is still in the goal state at the last step. They differ when the robot succeeds and then knocks the object away. ManiSkill only reports the `_at_end` metrics with `ignore_terminations=True`, because without it a success ends the episode at that step and the two would coincide. By default its PPO baseline trains with partial resets and evaluates with `ignore_terminations=True`, so every evaluation environment runs to the time limit and both metrics are logged.

**Truncation is not failure.** A truncated episode hit the time limit; `fail` is a task-defined failure state. `dones` covers both, but only a termination means the value of the next state is zero, which is why the two flags stay separate in the return value.

**Other frameworks.** Gymnasium's `RecordEpisodeStatistics` reports only `r`, `l` and `t` (return, length, wall time), fills them only for finished environments with zeros elsewhere, and in `NEXT_STEP` autoreset mode (Gymnasium's default) the step after a done is the reset step and is not counted in the next episode. RLinf's LIBERO env takes success from the termination flag, stops adding reward to the return after the first success and divides it by the length at that first success.""",
    "advisory_prerequisites": ["rl_eval_loop", "rollout_batch_assembly"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which values must survive a reset of the same environment in the same step, and how do you make sure the returned dict cannot see that reset? Which environments should a done in environment 0 reset? When terminations are ignored, what is the only way an episode can still end?"},
        {"level": 2, "kind": "analysis", "content": "Keep four (num_envs,) buffers: float32 return, integer length, bool success_once and fail_once. In step, validate shapes and dtypes, update the buffers, OR in the flags, compute effective terminations and dones, clone the buffers into the episode dict and add the _at_end copies only when terminations are ignored. Then, if auto_reset, call reset with the indices where dones is true. reset builds a mask from the indices and zeroes the buffers there."},
    ],
    "model_connections": [
        "ManiSkillVectorEnv(record_metrics=True) keeps exactly these buffers and returns them in infos['episode'], cloned into infos['final_info'] before a partial reset.",
        "ManiSkill's PPO baseline averages every key of final_info['episode'] and logs it as train/ and eval/ scalars; the training log masks it to the finished environments with _final_info.",
        "RLinf's LIBERO env records the same success_once, return, episode_len and reward keys, with success taken from terminations.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Per-environment buffers make the metrics correct under partial resets, where each environment finishes at a different step.",
            "success_once and success_at_end separate reaching the goal from holding it, which a single success flag hides.",
        ],
        "cons": [
            "Averaging the snapshot over finished environments at each step weights short episodes more heavily, because they finish more often in a fixed number of steps.",
            "The tracker trusts the environment's success and fail flags; a mis-specified success check is reported as a high success rate.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/haosulab/ManiSkill",
            "commit": "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3",
            "path": "mani_skill/vector/wrappers/gymnasium.py",
            "symbol": "ManiSkillVectorEnv.__init__, ManiSkillVectorEnv.reset, ManiSkillVectorEnv.step",
            "license": "Apache-2.0",
            "adapted": "Primary oracle: the record_metrics buffers, the ignore_terminations and auto_reset logic, the episode snapshot and the partial reset of finished environments.",
            "simplifications": "The tracker is fed rewards and flags instead of wrapping a simulator, and owns the episode length that ManiSkill reads from base_env.elapsed_steps (reset to 0 on reset, plus 1 per step). "
                               "It returns the episode snapshot directly instead of infos['episode'] or infos['final_info']['episode'], and returns no observations. "
                               "success and fail are optional arguments instead of optional info keys. "
                               "Upstream zeroes the caller's terminations tensor in place under ignore_terminations; the exercise forbids modifying inputs and returns a new tensor. "
                               "Upstream adds the reward into a float32 buffer in place; the exercise casts rewards to float32 first, which can differ in the last float32 bit for float64 rewards (the oracle test compares within 1e-6).",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/envs/sim/libero/libero_env.py",
            "symbol": "LiberoEnv._init_metrics, LiberoEnv._reset_metrics, LiberoEnv._record_metrics, LiberoEnv.step",
            "license": "Apache-2.0",
            "adapted": "Comparison only: the same metric keys for a VLA RL environment.",
            "simplifications": "Not followed. RLinf sets success_once from terminations, adds step_reward * (not success_once) so the return stops growing after the first success, records the length at first success and computes reward as return / max(length at first success or current length, 1); it does not report fail_once.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Farama-Foundation/Gymnasium",
            "commit": "d77197c284406e6094c9d1a74d4988a8749ce849",
            "path": "gymnasium/wrappers/vector/common.py",
            "symbol": "RecordEpisodeStatistics.reset, RecordEpisodeStatistics.step",
            "license": "MIT",
            "adapted": "Comparison only: the SAME_STEP branch restarts statistics of finished environments right after recording them, as the exercise does.",
            "simplifications": "Not followed. Gymnasium reports r, l and t only, fills them only where an episode finished (zeros elsewhere), adds no success metrics, keeps return_queue and length_queue of the last buffer_length episodes, and in NEXT_STEP mode zeroes the statistics on the step after a done instead of the same step.",
        },
    ],
    "tests": [
        {"name": "Accumulates, snapshots and resets only the finished env", "behavior": "rl.trajectory", "code": r"""
import torch
F, T = False, True
tr = {fn}(2)
out = tr.step(torch.tensor([1.0, 2.0]), torch.tensor([F, F]), torch.tensor([F, F]), success=torch.tensor([F, T]))
assert set(out) == {"episode", "terminations", "dones"}, set(out)
ep = out["episode"]
assert set(ep) == {"return", "episode_len", "reward", "success_once"}, set(ep)
assert ep["return"].tolist() == [1.0, 2.0] and ep["episode_len"].tolist() == [1, 1]
assert ep["success_once"].tolist() == [F, T]
out = tr.step(torch.tensor([3.0, 4.0]), torch.tensor([T, F]), torch.tensor([F, F]), success=torch.tensor([T, F]))
ep = out["episode"]
assert ep["return"].tolist() == [4.0, 6.0], ep["return"]
assert ep["episode_len"].tolist() == [2, 2], ep["episode_len"]
assert ep["reward"].tolist() == [2.0, 3.0], ep["reward"]
assert ep["success_once"].tolist() == [T, T], "success_once is sticky"
assert out["dones"].tolist() == [T, F]
out = tr.step(torch.tensor([5.0, 1.0]), torch.tensor([F, F]), torch.tensor([F, T]), success=torch.tensor([F, F]))
ep = out["episode"]
assert ep["return"].tolist() == [5.0, 7.0], "env 0 restarted, env 1 kept going"
assert ep["episode_len"].tolist() == [1, 3], ep["episode_len"]
assert ep["success_once"].tolist() == [F, T], ep["success_once"]
assert out["dones"].tolist() == [F, T]
"""},
        {"name": "Ignored terminations: only truncation ends the episode", "behavior": "rl.trajectory", "code": r"""
import torch
F, T = False, True
tr = {fn}(1, ignore_terminations=True)
succ = [T, F, F]
for k in range(3):
    out = tr.step(torch.tensor([1.0]), torch.tensor([succ[k]]), torch.tensor([k == 2]),
                  success=torch.tensor([succ[k]]), fail=torch.tensor([F]))
    ep = out["episode"]
    assert set(ep) == {"return", "episode_len", "reward", "success_once", "fail_once", "success_at_end", "fail_at_end"}, set(ep)
    assert out["terminations"].tolist() == [F], "terminations are ignored"
    assert ep["success_once"].tolist() == [T], "reached the goal at step 0"
    assert ep["success_at_end"].tolist() == [succ[k]]
    assert ep["episode_len"].tolist() == [k + 1]
assert out["dones"].tolist() == [T], "the time limit still ends the episode"
"""},
        {"name": "The returned snapshot is a copy", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Clone the buffers into the episode dict before the auto reset; the reset zeroed values you had already returned.",
         "code": r"""
import torch
F, T = False, True
tr = {fn}(3)
r = torch.tensor([1.5, 2.0, -1.0])
first = tr.step(r, torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([F, T, F]), fail=torch.tensor([F, F, T]))
done = tr.step(r, torch.tensor([T, F, F]), torch.tensor([F, F, T]), success=torch.tensor([T, F, F]), fail=torch.tensor([F, F, F]))
assert done["episode"]["return"].tolist() == [3.0, 4.0, -2.0], "snapshot must hold the finished episode's return, not the reset value"
assert done["episode"]["episode_len"].tolist() == [2, 2, 2]
assert done["episode"]["success_once"].tolist() == [T, T, F]
assert done["episode"]["fail_once"].tolist() == [F, F, T]
tr.step(r, torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([T, T, T]), fail=torch.tensor([T, T, T]))
tr.reset()
assert first["episode"]["return"].tolist() == [1.5, 2.0, -1.0], "an earlier snapshot changed after later calls"
assert first["episode"]["episode_len"].tolist() == [1, 1, 1]
assert first["episode"]["success_once"].tolist() == [F, T, F]
assert first["episode"]["fail_once"].tolist() == [F, F, T]
assert done["episode"]["return"].tolist() == [3.0, 4.0, -2.0]
assert done["episode"]["success_once"].tolist() == [T, T, F]
"""},
        {"name": "Seeded partial-reset trajectories match a per-env loop", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "Running values must accumulate per environment and restart only where dones is true, after the snapshot.",
         "code": _LOOP_MODEL + r"""
for seed in (3, 11, 29):
    n = 5
    sched = random_schedule(seed, n, 40)
    want = loop_model(n, sched, False, True, True, True)
    tr = {fn}(n)
    for k, row in enumerate(sched):
        got = tr.step(*as_tensors(*row))
        check_row(got, want[k], ("return", "episode_len", "reward", "success_once", "fail_once"), (seed, k))
        assert set(got["episode"]) == {"return", "episode_len", "reward", "success_once", "fail_once"}, (seed, k)
"""},
        {"name": "Seeded ignore-terminations trajectories match a per-env loop", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "With ignore_terminations, only truncations end an episode, terminations come back all False and the _at_end keys copy this step's flags.",
         "code": _LOOP_MODEL + r"""
for seed in (5, 17):
    n = 4
    sched = random_schedule(seed, n, 40, p_term=0.4, p_trunc=0.1)
    want = loop_model(n, sched, True, True, True, True)
    tr = {fn}(n, ignore_terminations=True)
    keys = ("return", "episode_len", "reward", "success_once", "fail_once", "success_at_end", "fail_at_end")
    for k, row in enumerate(sched):
        got = tr.step(*as_tensors(*row))
        check_row(got, want[k], keys, (seed, k))
"""},
        {"name": "Truncation alone ends an episode", "visibility": "unshown", "behavior": "rl.trajectory",
         "failure_message": "dones is terminations OR truncations; a time-limit truncation must also reset the environment.",
         "code": r"""
import torch
F, T = False, True
tr = {fn}(2)
out = tr.step(torch.tensor([1.0, 1.0]), torch.tensor([F, F]), torch.tensor([T, F]))
assert out["dones"].tolist() == [T, F] and out["terminations"].tolist() == [F, F]
out = tr.step(torch.tensor([1.0, 1.0]), torch.tensor([F, F]), torch.tensor([F, F]))
assert out["episode"]["episode_len"].tolist() == [1, 2], out["episode"]["episode_len"]
assert out["episode"]["return"].tolist() == [1.0, 2.0]
"""},
        {"name": "Without auto reset nothing restarts until reset is called", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With auto_reset=False the values keep accumulating past a done; reset(env_idx) restarts only the listed environments.",
         "code": r"""
import torch
F, T = False, True
tr = {fn}(3, auto_reset=False)
s = torch.tensor([T, F, F])
tr.step(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([T, T, F]), torch.tensor([F, F, T]), success=s)
out = tr.step(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([F, F, F]))
assert out["episode"]["return"].tolist() == [2.0, 3.0, 4.0], out["episode"]["return"]
assert out["episode"]["episode_len"].tolist() == [2, 2, 2]
assert out["episode"]["success_once"].tolist() == [T, F, F]
assert out["dones"].tolist() == [F, F, F]
tr.reset([0, 2])
out = tr.step(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([F, F, F]))
assert out["episode"]["return"].tolist() == [1.0, 4.0, 1.0], out["episode"]["return"]
assert out["episode"]["episode_len"].tolist() == [1, 3, 1]
assert out["episode"]["success_once"].tolist() == [F, F, F]
tr.reset(torch.tensor([1]))
out = tr.step(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([F, F, F]))
assert out["episode"]["episode_len"].tolist() == [2, 1, 2]
tr.reset()
out = tr.step(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([F, F, F]), torch.tensor([F, F, F]), success=torch.tensor([F, F, F]))
assert out["episode"]["return"].tolist() == [0.5, 0.5, 0.5] and out["episode"]["episode_len"].tolist() == [1, 1, 1]
"""},
        {"name": "Key set depends on the flags given", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Emit success_once / fail_once only when that flag was passed, and the _at_end keys only when ignore_terminations is true.",
         "code": r"""
import torch
F, T = False, True
z = torch.tensor([F, F])
r = torch.tensor([1.0, 1.0])
assert set({fn}(2).step(r, z, z)["episode"]) == {"return", "episode_len", "reward"}
assert set({fn}(2).step(r, z, z, fail=z)["episode"]) == {"return", "episode_len", "reward", "fail_once"}
assert set({fn}(2).step(r, z, z, success=z, fail=z)["episode"]) == {"return", "episode_len", "reward", "success_once", "fail_once"}
assert set({fn}(2, ignore_terminations=True).step(r, z, z)["episode"]) == {"return", "episode_len", "reward"}
assert set({fn}(2, ignore_terminations=True).step(r, z, z, success=z)["episode"]) == {"return", "episode_len", "reward", "success_once", "success_at_end"}
"""},
        {"name": "Inputs are never modified", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Build the effective terminations as a new tensor; do not zero or reuse the caller's inputs in place.",
         "code": r"""
import torch
F, T = False, True
tr = {fn}(3, ignore_terminations=True)
rew = torch.tensor([1.0, 2.0, 3.0]); term = torch.tensor([T, F, T]); trunc = torch.tensor([F, T, F])
succ = torch.tensor([T, F, T]); fail = torch.tensor([F, T, F])
copies = [x.clone() for x in (rew, term, trunc, succ, fail)]
out = tr.step(rew, term, trunc, success=succ, fail=fail)
for x, c in zip((rew, term, trunc, succ, fail), copies):
    assert torch.equal(x, c), "an input tensor was modified"
out["episode"]["success_at_end"][:] = False
out["terminations"][:] = True
assert torch.equal(succ, copies[3]), "success_at_end must be a copy of success"
assert torch.equal(term, copies[1]), "the returned terminations must not alias the input"
"""},
        {"name": "Output dtypes", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "return and reward are float32, episode_len is an integer dtype, and every flag (including dones and terminations) is torch.bool.",
         "code": r"""
import torch
F, T = False, True
tr = {fn}(2, ignore_terminations=True)
out = tr.step(torch.tensor([1, 2], dtype=torch.int64), torch.tensor([F, T]), torch.tensor([F, F]),
              success=torch.tensor([T, F]), fail=torch.tensor([F, F]))
out = tr.step(torch.tensor([0.1, 0.2], dtype=torch.float64), torch.tensor([F, T]), torch.tensor([F, F]),
              success=torch.tensor([T, F]), fail=torch.tensor([F, F]))
ep = out["episode"]
assert ep["return"].dtype == torch.float32 and ep["reward"].dtype == torch.float32, (ep["return"].dtype, ep["reward"].dtype)
assert not ep["episode_len"].dtype.is_floating_point and ep["episode_len"].dtype != torch.bool, ep["episode_len"].dtype
for k in ("success_once", "fail_once", "success_at_end", "fail_at_end"):
    assert ep[k].dtype == torch.bool, (k, ep[k].dtype)
assert out["dones"].dtype == torch.bool and out["terminations"].dtype == torch.bool
assert torch.allclose(ep["return"], torch.tensor([1.1, 2.2])), ep["return"]
assert torch.allclose(ep["reward"], torch.tensor([0.55, 1.1])), ep["reward"]
for v in ep.values():
    assert v.shape == (2,), v.shape
"""},
        {"name": "Rejects bad sizes, shapes and flag dtypes", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a non-positive or non-integer num_envs, an input not shaped (num_envs,), or a flag that is not torch.bool.",
         "code": r"""
import torch
for bad in (0, -2, 2.0, True):
    try: {fn}(bad)
    except ValueError: pass
    else: raise AssertionError(f"num_envs={bad!r} accepted")
tr = {fn}(2)
r = torch.tensor([1.0, 1.0]); z = torch.tensor([False, False])
bad_calls = [
    lambda: tr.step(torch.tensor([1.0, 1.0, 1.0]), z, z),
    lambda: tr.step(torch.tensor([[1.0, 1.0]]), z, z),
    lambda: tr.step(r, torch.tensor([False]), z),
    lambda: tr.step(r, z, torch.tensor([0, 1])),
    lambda: tr.step(r, torch.tensor([0.0, 1.0]), z),
    lambda: tr.step(r, z, z, success=torch.tensor([1, 0])),
    lambda: tr.step(r, z, z, fail=torch.tensor([False, False, False])),
]
for i, call in enumerate(bad_calls):
    try: call()
    except ValueError: pass
    else: raise AssertionError(f"bad call {i} accepted")
"""},
    ],
    "solution": '''import torch


class EpisodeMetricsTracker:
    def __init__(self, num_envs, ignore_terminations=False, auto_reset=True):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.ignore_terminations = bool(ignore_terminations)
        self.auto_reset = bool(auto_reset)
        self.returns = torch.zeros(num_envs, dtype=torch.float32)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64)
        self.success_once = torch.zeros(num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(num_envs, dtype=torch.bool)

    def _check(self, name, x, flag):
        if not isinstance(x, torch.Tensor) or x.shape != (self.num_envs,):
            raise ValueError(f"{name} must be a tensor of shape ({self.num_envs},)")
        if flag and x.dtype != torch.bool:
            raise ValueError(f"{name} must be torch.bool")

    def reset(self, env_idx=None):
        if env_idx is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool)
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool)
            mask[torch.as_tensor(env_idx, dtype=torch.long)] = True
        self.returns[mask] = 0.0
        self.lengths[mask] = 0
        self.success_once[mask] = False
        self.fail_once[mask] = False

    def step(self, rewards, terminations, truncations, success=None, fail=None):
        self._check("rewards", rewards, False)
        self._check("terminations", terminations, True)
        self._check("truncations", truncations, True)
        if success is not None:
            self._check("success", success, True)
        if fail is not None:
            self._check("fail", fail, True)

        self.returns += rewards.to(torch.float32)
        self.lengths += 1
        episode = {}
        if success is not None:
            self.success_once = self.success_once | success
            episode["success_once"] = self.success_once.clone()
        if fail is not None:
            self.fail_once = self.fail_once | fail
            episode["fail_once"] = self.fail_once.clone()
        episode["return"] = self.returns.clone()
        episode["episode_len"] = self.lengths.clone()
        episode["reward"] = episode["return"] / episode["episode_len"]

        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)
            if success is not None:
                episode["success_at_end"] = success.clone()
            if fail is not None:
                episode["fail_at_end"] = fail.clone()
        else:
            terminations = terminations.clone()
        dones = terminations | truncations

        if self.auto_reset and bool(dones.any()):
            self.reset(torch.nonzero(dones).flatten())
        return {"episode": episode, "terminations": terminations, "dones": dones}
''',
    "interview_questions": interview(
        concept=[
            "What is the difference between success_once and success_at_end, and when do they disagree?",
            "Why does a vectorized robot environment need per-environment metric buffers instead of one running total?",
        ],
        deep_dive=[
            "With same-step autoreset, why must the episode snapshot be cloned before the finished environments are reset?",
            "What changes in the metrics when ignore_terminations is true, and why does ManiSkill only report the at_end metrics in that mode?",
            "How do Gymnasium's NEXT_STEP and SAME_STEP autoreset modes change when episode statistics should be zeroed?",
        ],
        tradeoffs=[
            "Partial resets versus running every environment to the time limit: what does each do to throughput and to the success metrics you can report?",
            "Averaging finished-episode metrics per step versus per episode: which episodes get over-weighted, and how would you correct for it?",
        ],
    ),
}
