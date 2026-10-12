"""episode_metrics_tracker checked against ManiSkill's record_metrics wrapper.

The oracle in `torch_judge/harness/embodied_rl/vendored/maniskill_vector_env.py`
is ManiSkillVectorEnv's step/reset transcribed line for line. It wraps a tiny
fake base env that replays a seeded schedule of rewards and flags and keeps
`elapsed_steps` the way ManiSkill's BaseEnv does (+1 per step, 0 on reset).
The reference tracker is fed the same schedule directly.
"""

from __future__ import annotations

import random

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.maniskill_vector_env import (
    ManiSkillRecordMetricsVectorEnv,
)
from torch_judge.tasks import get_task

TASK_ID = "episode_metrics_tracker"


def reference():
    task = get_task(TASK_ID)
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


class FakeBaseEnv:
    """Replays a schedule; elapsed_steps follows ManiSkill's BaseEnv."""

    def __init__(self, num_envs, schedule, use_success, use_fail):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self.schedule = schedule
        self.use_success = use_success
        self.use_fail = use_fail
        self.t = 0

    def reset(self, seed=None, options=None):
        if options is not None and "env_idx" in options:
            self.elapsed_steps[options["env_idx"]] = 0
        else:
            self.elapsed_steps[:] = 0
        return torch.zeros(self.num_envs), {}

    def step(self, actions):
        rew, term, trunc, succ, fail = self.schedule[self.t]
        self.t += 1
        self.elapsed_steps += 1
        infos = {}
        if self.use_success:
            infos["success"] = succ.clone()
        if self.use_fail:
            infos["fail"] = fail.clone()
        return torch.zeros(self.num_envs), rew.clone(), term.clone(), trunc.clone(), infos


def make_schedule(seed, num_envs, steps, p_term, p_trunc, reward_dtype=torch.float32):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    schedule = []
    for _ in range(steps):
        rew = (torch.randn(num_envs, generator=g) * 2.0).to(reward_dtype)
        succ = torch.tensor([rng.random() < 0.2 for _ in range(num_envs)])
        fail = torch.tensor([rng.random() < 0.1 for _ in range(num_envs)])
        term = succ | fail | torch.tensor([rng.random() < p_term for _ in range(num_envs)])
        trunc = torch.tensor([rng.random() < p_trunc for _ in range(num_envs)])
        schedule.append((rew, term, trunc, succ, fail))
    return schedule


def run_both(num_envs, schedule, *, ignore_terminations, auto_reset, use_success, use_fail, resets=None):
    """Step oracle and reference in lockstep; compare every snapshot and mask."""
    resets = resets or {}
    oracle = ManiSkillRecordMetricsVectorEnv(
        FakeBaseEnv(num_envs, schedule, use_success, use_fail),
        auto_reset=auto_reset,
        ignore_terminations=ignore_terminations,
        record_metrics=True,
    )
    oracle.reset()
    tracker = reference()(num_envs, ignore_terminations=ignore_terminations, auto_reset=auto_reset)
    finished = 0
    for k, (rew, term, trunc, succ, fail) in enumerate(schedule):
        if k in resets:
            idx = resets[k]
            oracle.reset(options=dict(env_idx=torch.tensor(idx, dtype=torch.long)))
            tracker.reset(idx)
        _, _, o_term, o_trunc, infos = oracle.step(None)
        got = tracker.step(
            rew.clone(), term.clone(), trunc.clone(),
            success=succ.clone() if use_success else None,
            fail=fail.clone() if use_fail else None,
        )
        want = infos["final_info"]["episode"] if "final_info" in infos else infos["episode"]
        assert set(got["episode"]) == set(want), (k, set(got["episode"]), set(want))
        for key, value in want.items():
            mine = got["episode"][key]
            if value.dtype == torch.bool:
                assert torch.equal(mine, value), (k, key, mine, value)
            else:
                assert torch.allclose(mine.double(), value.double(), atol=1e-6, rtol=0), (k, key, mine, value)
        assert torch.equal(got["terminations"], o_term), (k, got["terminations"], o_term)
        assert torch.equal(got["dones"], o_term | o_trunc), k
        if "final_info" in infos:
            assert torch.equal(infos["_final_info"], got["dones"]), k
            finished += int(got["dones"].sum())
    return finished


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_matches_maniskill_with_partial_resets(seed):
    schedule = make_schedule(seed, num_envs=6, steps=50, p_term=0.1, p_trunc=0.05)
    finished = run_both(6, schedule, ignore_terminations=False, auto_reset=True, use_success=True, use_fail=True)
    assert finished > 10, "the schedule should exercise many partial resets"


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_matches_maniskill_with_ignored_terminations(seed):
    schedule = make_schedule(seed, num_envs=4, steps=60, p_term=0.3, p_trunc=0.06)
    run_both(4, schedule, ignore_terminations=True, auto_reset=True, use_success=True, use_fail=True)


@pytest.mark.parametrize("seed", [20, 21, 22, 23, 24])
def test_matches_maniskill_without_auto_reset_and_manual_resets(seed):
    schedule = make_schedule(seed, num_envs=5, steps=30, p_term=0.2, p_trunc=0.1)
    resets = {7: [0, 3], 15: [1], 22: [0, 1, 2, 3, 4]}
    run_both(5, schedule, ignore_terminations=False, auto_reset=False, use_success=True, use_fail=True, resets=resets)


@pytest.mark.parametrize("use_success,use_fail", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_matches_maniskill_when_success_or_fail_is_absent(use_success, use_fail, ignore_terminations):
    schedule = make_schedule(31, num_envs=3, steps=25, p_term=0.2, p_trunc=0.1)
    run_both(3, schedule, ignore_terminations=ignore_terminations, auto_reset=True,
             use_success=use_success, use_fail=use_fail)


def test_edge_single_env_and_all_envs_done_together():
    schedule = make_schedule(40, num_envs=1, steps=20, p_term=0.3, p_trunc=0.2)
    run_both(1, schedule, ignore_terminations=False, auto_reset=True, use_success=True, use_fail=True)
    n = 4
    all_done = []
    for k in range(9):
        rew = torch.full((n,), float(k + 1))
        flag = torch.full((n,), k % 3 == 2)
        all_done.append((rew, flag, torch.zeros(n, dtype=torch.bool), flag.clone(), torch.zeros(n, dtype=torch.bool)))
    run_both(n, all_done, ignore_terminations=False, auto_reset=True, use_success=True, use_fail=True)


def test_edge_float64_rewards_are_accumulated_in_float32():
    schedule = make_schedule(50, num_envs=3, steps=30, p_term=0.1, p_trunc=0.05, reward_dtype=torch.float64)
    run_both(3, schedule, ignore_terminations=False, auto_reset=True, use_success=True, use_fail=True)


def test_divergence_upstream_zeroes_terminations_in_place_the_exercise_does_not():
    """Pinned divergence: ManiSkill writes `terminations[:] = False` into the env's tensor."""
    term = torch.tensor([True, False])

    class Env(FakeBaseEnv):
        def step(self, actions):
            self.elapsed_steps += 1
            return None, torch.ones(2), term, torch.zeros(2, dtype=torch.bool), {"success": term.clone()}

    oracle = ManiSkillRecordMetricsVectorEnv(Env(2, [], True, False), ignore_terminations=True, record_metrics=True)
    oracle.reset()
    oracle.step(None)
    assert term.tolist() == [False, False], "upstream mutates the terminations tensor in place"

    mine = torch.tensor([True, False])
    reference()(2, ignore_terminations=True).step(torch.ones(2), mine, torch.zeros(2, dtype=torch.bool), success=mine.clone())
    assert mine.tolist() == [True, False], "the exercise must leave its inputs unchanged"


def test_divergence_episode_len_dtype():
    """Pinned divergence: upstream episode_len is BaseEnv's int32 counter; the exercise allows any integer dtype."""
    schedule = make_schedule(60, num_envs=2, steps=1, p_term=0.0, p_trunc=0.0)
    oracle = ManiSkillRecordMetricsVectorEnv(FakeBaseEnv(2, schedule, True, True), record_metrics=True)
    oracle.reset()
    _, _, _, _, infos = oracle.step(None)
    assert infos["episode"]["episode_len"].dtype == torch.int32
    rew, term, trunc, succ, fail = schedule[0]
    got = reference()(2).step(rew, term, trunc, success=succ, fail=fail)
    assert not got["episode"]["episode_len"].dtype.is_floating_point
    assert got["episode"]["reward"].dtype == infos["episode"]["reward"].dtype == torch.float32
