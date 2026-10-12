"""The point_reach_env reference, checked against Gymnasium's wrapper stack.

The mutation gate proves the evaluator rejects wrong implementations; it cannot
prove the contract matches upstream. Here the point-mass dynamics are written as
a bare ManiSkill-shaped task (sample in `_initialize_episode`, success from
`evaluate`, flags and sparse reward from `BaseEnv.step`), wrapped in Gymnasium's
`TimeLimit` and `OrderEnforcing`, all transcribed in
`torch_judge/harness/embodied_rl/vendored/gymnasium_core.py`. The single-class
reference must produce the same episodes.
"""

from __future__ import annotations

import math

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.gymnasium_core import (
    GymEnv,
    OrderEnforcing,
    ResetNeeded,
    TimeLimit,
    check_reset_seed_determinism,
    data_equivalence,
    maniskill_is_obj_placed,
    maniskill_sparse_reward,
    maniskill_step_flags,
)
from torch_judge.tasks import get_task

SEEDS = [0, 1, 2, 7, 42, 2022]


def reference():
    task = get_task("point_reach_env")
    assert task is not None
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


class PointMassTask(GymEnv):
    """Point-mass dynamics in the shape of a ManiSkill task; no time limit of its own."""

    def __init__(self, goal_radius: float, image_size: int, max_action: float):
        self.goal_radius = goal_radius
        self.image_size = image_size
        self.max_action = max_action
        self._elapsed_steps = 0

    def _initialize_episode(self, options: dict):
        if "init_state" in options:  # like reset_to_env_states: skip sampling
            state = options["init_state"].to(torch.float32).clone()
            self.pos, self.goal = state[:2], state[2:]
            return
        self.pos = torch.rand(2, generator=self.np_random, dtype=torch.float32) * 2 - 1
        self.goal = torch.rand(2, generator=self.np_random, dtype=torch.float32) * 2 - 1

    def evaluate(self) -> dict:
        return {"success": maniskill_is_obj_placed(self.pos, self.goal, self.goal_radius)}

    def get_info(self) -> dict:
        info = {"elapsed_steps": self._elapsed_steps}
        info.update(self.evaluate())
        return info

    def get_obs(self) -> dict:
        S = self.image_size

        def px(c):
            return min(int(math.floor((float(c) + 1.0) / 2.0 * S)), S - 1)

        image = torch.zeros(1, S, S)
        image[0, px(self.goal[1]), px(self.goal[0])] = 0.5
        image[0, px(self.pos[1]), px(self.pos[0])] = 1.0
        return {"image": image, "state": self.pos.clone()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_episode(options or {})
        self._elapsed_steps = 0
        return self.get_obs(), self.get_info()

    def step(self, action):
        action = action.to(torch.float32).clamp(-self.max_action, self.max_action)
        self.pos = (self.pos + action).clamp(-1.0, 1.0)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs()
        reward = maniskill_sparse_reward(info)
        terminated, truncated = maniskill_step_flags(info)
        return obs, reward, terminated, truncated, info


def oracle_env(max_steps, goal_radius, image_size, max_action=0.1):
    return OrderEnforcing(TimeLimit(PointMassTask(goal_radius, image_size, max_action), max_steps))


def assert_same(got, want, where):
    assert data_equivalence(got, want), (where, got, want)


def run_episodes(seed: int, max_steps: int, goal_radius: float, image_size: int, max_action: float, episodes: int):
    """Drive both environments with one seeded action stream; compare every transition."""
    actions = torch.Generator().manual_seed(1000 + seed)
    ref = reference()(max_steps, goal_radius, image_size, max_action)
    ora = oracle_env(max_steps, goal_radius, image_size, max_action)
    ends = {"terminated": 0, "truncated": 0, "both": 0}
    for episode in range(episodes):
        kwargs = {"seed": seed} if episode == 0 else {}
        assert_same(ref.reset(**kwargs), ora.reset(**kwargs), (seed, episode, "reset"))
        while True:
            action = (torch.rand(2, generator=actions) * 2 - 1) * 0.3
            got, want = ref.step(action), ora.step(action)
            assert_same(got, want, (seed, episode, "step"))
            terminated, truncated = want[2], want[3]
            if terminated or truncated:
                ends["both" if terminated and truncated else "terminated" if terminated else "truncated"] += 1
                break
    return ends


@pytest.mark.parametrize("seed", SEEDS)
def test_seeded_episodes_match_the_wrapper_stack(seed):
    ends = run_episodes(seed, max_steps=8, goal_radius=0.45, image_size=6, max_action=0.2, episodes=12)
    assert ends["terminated"] + ends["truncated"] + ends["both"] == 12


def test_both_end_reasons_are_exercised():
    totals = {"terminated": 0, "truncated": 0, "both": 0}
    for seed in SEEDS:
        for key, value in run_episodes(seed, 8, 0.45, 6, 0.2, 12).items():
            totals[key] += value
    assert totals["terminated"] > 0 and totals["truncated"] > 0, totals


@pytest.mark.parametrize("max_steps", [1, 2, 3])
def test_success_on_the_last_step_sets_both_flags_like_time_limit(max_steps):
    init = torch.tensor([-0.5, 0.0, -0.5 + 0.1 * max_steps + 0.2, 0.0])
    ref = reference()(max_steps, 0.25, 4)
    ora = oracle_env(max_steps, 0.25, 4)
    assert_same(ref.reset(options={"init_state": init}), ora.reset(options={"init_state": init}), "reset")
    for _ in range(max_steps):
        got, want = ref.step(torch.tensor([0.1, 0.0])), ora.step(torch.tensor([0.1, 0.0]))
        assert_same(got, want, "step")
    assert want[2] is True and want[3] is True


def test_init_state_resets_match_and_leave_the_generator_alone():
    ref, ora = reference()(5, 0.1, 5), oracle_env(5, 0.1, 5)
    for kwargs in (
        {"seed": 3},
        {"options": {"init_state": torch.tensor([0.1, 0.2, -0.3, 0.4])}},
        {},
        {"seed": 9, "options": {"init_state": torch.tensor([1.0, -1.0, -1.0, 1.0])}},
        {"options": {}},
        {},
    ):
        assert_same(ref.reset(**kwargs), ora.reset(**kwargs), kwargs)


@pytest.mark.parametrize("seed", SEEDS)
def test_reference_passes_check_reset_seed_determinism(seed):
    env = reference()(4, 0.1, 5)
    env.reset(seed=seed)
    check_reset_seed_determinism(env, rng_of=lambda e: e._generator)


def test_seed_determinism_check_rejects_an_env_that_ignores_the_seed():
    class IgnoresSeed(PointMassTask):
        def reset(self, *, seed=None, options=None):
            return super().reset(seed=None, options=options)

    with pytest.raises(AssertionError):
        check_reset_seed_determinism(IgnoresSeed(0.1, 5, 0.1), rng_of=lambda e: e.np_random)
    check_reset_seed_determinism(PointMassTask(0.1, 5, 0.1), rng_of=lambda e: e.np_random)


def test_unseeded_reset_continues_the_generator_as_env_reset_documents():
    """The checker alone would accept re-seeding on every reset; Env.reset's docstring does not."""
    ref, ora = reference()(4, 0.1, 5), oracle_env(4, 0.1, 5)
    first = [ref.reset(seed=5)] + [ref.reset() for _ in range(3)]
    want = [ora.reset(seed=5)] + [ora.reset() for _ in range(3)]
    assert_same(first, want, "unseeded sequence")
    assert not data_equivalence(first[1], first[2])


def test_step_before_reset_raises_in_both():
    with pytest.raises(ResetNeeded):
        oracle_env(3, 0.1, 4).step(torch.zeros(2))
    with pytest.raises(RuntimeError):
        reference()(3, 0.1, 4).step(torch.zeros(2))


def test_documented_divergences_from_upstream():
    # 1. Upstream leaves a step after the end to the caller; the exercise raises.
    ref, ora = reference()(1, 0.01, 4), oracle_env(1, 0.01, 4)
    init = {"init_state": torch.tensor([-1.0, -1.0, 1.0, 1.0])}
    ref.reset(options=init), ora.reset(options=init)
    ref.step(torch.zeros(2)), ora.step(torch.zeros(2))
    ora.step(torch.zeros(2))  # no error upstream
    with pytest.raises(RuntimeError):
        ref.step(torch.zeros(2))
    # 2. A non-integer limit is a TypeError in TimeLimit and a ValueError in the exercise.
    with pytest.raises(TypeError):
        TimeLimit(PointMassTask(0.1, 4, 0.1), 2.0)
    with pytest.raises(ValueError):
        reference()(2.0, 0.1, 4)
