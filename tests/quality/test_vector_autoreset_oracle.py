"""vector_autoreset's reference solution, checked against Gymnasium's SyncVectorEnv.

Both vector envs drive identical scripted sub-environments with the same actions,
and every returned observation, reward, flag and info entry is compared. The
oracle is `torch_judge/harness/embodied_rl/vendored/gymnasium_sync_vector_env.py`.

One deliberate divergence is pinned below: Gymnasium batches a numeric info key
into a zero-filled typed numpy array, so an env that did not report the key reads
0 or False there; the exercise uses a list with None. Wherever the mask is True
the values must agree exactly, and the masks must agree everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.gymnasium_sync_vector_env import AutoresetMode, GymSyncVectorEnv
from torch_judge.tasks import get_task
from torch_judge.tasks.vector_autoreset import _ENVS

TASK_ID = "vector_autoreset"


def reference_class():
    task = get_task(TASK_ID)
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def fakes() -> dict:
    namespace: dict = {}
    exec(_ENVS, namespace)
    return namespace


def assert_value_equal(ours, theirs, where):
    if isinstance(theirs, torch.Tensor):
        assert isinstance(ours, torch.Tensor) and torch.equal(ours, theirs), where
    else:
        assert ours == theirs, (where, ours, theirs)


def assert_info_match(ours: dict, theirs: dict, n: int, where: str) -> None:
    assert set(ours) == set(theirs), (where, sorted(ours), sorted(theirs))
    for key, value in theirs.items():
        if key.startswith("_"):
            assert isinstance(value, np.ndarray) and value.dtype == np.bool_, (where, key)
            assert ours[key].tolist() == value.tolist(), (where, key, ours[key], value)
        elif isinstance(value, dict):
            assert_info_match(ours[key], value, n, f"{where}[{key!r}]")
        else:
            mask = theirs[f"_{key}"]
            for i in range(n):
                if mask[i]:
                    assert_value_equal(ours[key][i], value[i], (where, key, i))
                else:
                    assert ours[key][i] is None, (where, key, i, ours[key][i])


def run_pair(seed: int, n: int, mode: str, steps: int, reset_seed=None, resets_at=()):
    ns = fakes()
    ours_envs, gym_envs = ns["random_envs"](seed, n), ns["random_envs"](seed, n)
    actions = ns["random_actions"](seed, steps, n)
    ours = reference_class()(ours_envs, mode)
    gym = GymSyncVectorEnv(gym_envs, AutoresetMode(mode))
    reset_seed = seed if reset_seed is None else reset_seed

    obs, info = ours.reset(seed=reset_seed)
    gym_obs, gym_info = gym.reset(seed=reset_seed)
    assert torch.equal(obs, torch.stack(gym_obs))
    assert_info_match(info, gym_info, n, "reset")

    for t, a in enumerate(actions):
        if t in resets_at:
            obs, info = ours.reset(seed=reset_seed + t)
            gym_obs, gym_info = gym.reset(seed=reset_seed + t)
            assert torch.equal(obs, torch.stack(gym_obs)), (seed, t)
            assert_info_match(info, gym_info, n, f"reset at {t}")
        obs, rew, term, trunc, info = ours.step(a)
        g_obs, g_rew, g_term, g_trunc, g_info = gym.step(a)
        where = (seed, mode, t)
        assert torch.equal(obs, torch.stack(g_obs)), where
        assert rew.tolist() == g_rew.tolist(), (where, rew, g_rew)
        assert term.tolist() == g_term.tolist() and trunc.tolist() == g_trunc.tolist(), where
        assert_info_match(info, g_info, n, str(where))
    assert [e.log for e in ours_envs] == [e.log for e in gym_envs], (seed, mode)


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13])
def test_seeded_rollouts_match_gymnasium(seed, mode):
    run_pair(seed, n=1 + seed % 4, mode=mode, steps=20)


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_full_reset_mid_rollout_matches_gymnasium(mode):
    for seed in (4, 9):
        run_pair(seed, n=3, mode=mode, steps=16, resets_at=(3, 7, 8))


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_one_step_episodes_match_gymnasium(mode):
    ns = fakes()
    ours_envs = [ns["ScriptedEnv"](i, [1], ["term" if i else "trunc"]) for i in range(2)]
    gym_envs = [ns["ScriptedEnv"](i, [1], ["term" if i else "trunc"]) for i in range(2)]
    ours, gym = reference_class()(ours_envs, mode), GymSyncVectorEnv(gym_envs, mode)
    ours.reset(seed=[3, None])
    gym.reset(seed=[3, None])
    for k in range(5):
        a = torch.tensor([float(k), 2.0 * k])
        obs, rew, term, trunc, info = ours.step(a)
        g_obs, g_rew, g_term, g_trunc, g_info = gym.step(a)
        assert torch.equal(obs, torch.stack(g_obs)) and rew.tolist() == g_rew.tolist()
        assert term.tolist() == g_term.tolist() and trunc.tolist() == g_trunc.tolist()
        assert_info_match(info, g_info, 2, f"{mode} step {k}")
    assert [e.log for e in ours_envs] == [e.log for e in gym_envs]


def test_nested_info_matches_gymnasium():
    ns = fakes()

    def make():
        return [
            ns["ScriptedEnv"](0, [2], ["term"], extra_info={"stats": {"dist": 0.5, "contacts": 2}}),
            ns["ScriptedEnv"](1, [3], ["trunc"]),
            ns["ScriptedEnv"](2, [2], ["trunc"], extra_info={"stats": {"dist": 1.5}}),
        ]

    ours, gym = reference_class()(make(), "SameStep"), GymSyncVectorEnv(make(), "SameStep")
    ours.reset(seed=0)
    gym.reset(seed=0)
    for _ in range(4):
        a = torch.zeros(3)
        *_, info = ours.step(a)
        *_, g_info = gym.step(a)
        assert_info_match(info, g_info, 3, "nested")


def test_pinned_divergence_missing_info_entries():
    """Gymnasium zero-fills a numeric key for envs that did not report it; the exercise uses None."""
    ns = fakes()
    make = lambda: [ns["ScriptedEnv"](0, [1], ["term"]), ns["ScriptedEnv"](1, [3], ["term"])]  # noqa: E731
    ours, gym = reference_class()(make(), "SameStep"), GymSyncVectorEnv(make(), "SameStep")
    ours.reset(seed=0)
    gym.reset(seed=0)
    *_, info = ours.step(torch.zeros(2))
    *_, g_info = gym.step(torch.zeros(2))
    assert g_info["t"].tolist() == [0, 1] and g_info["_t"].tolist() == [False, True]
    assert info["t"] == [None, 1] and info["_t"].tolist() == [False, True]
    assert g_info["final_info"]["success"].tolist() == [True, False]
    assert info["final_info"]["success"] == [True, None]
    assert g_info["final_obs"][1] is None and info["final_obs"][1] is None


def test_next_step_is_the_upstream_default():
    ns = fakes()
    assert GymSyncVectorEnv([ns["ScriptedEnv"](0, [1], ["term"])]).autoreset_mode == AutoresetMode.NEXT_STEP
    env = ns["ScriptedEnv"](0, [1], ["term"])
    ours = reference_class()([env])
    ours.reset(seed=0)
    ours.step(torch.zeros(1))
    ours.step(torch.zeros(1))
    assert env.log[-1] == ("reset", None), "the exercise's default must also be NextStep"
