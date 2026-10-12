"""Vendored Gymnasium environment-contract oracles, plus ManiSkill's step flags.

Upstream:   https://github.com/Farama-Foundation/Gymnasium
Commit:     d77197c284406e6094c9d1a74d4988a8749ce849
Files:      gymnasium/core.py, gymnasium/wrappers/common.py, gymnasium/utils/env_checker.py
Symbols:    Env.reset, Env.np_random, TimeLimit.step, TimeLimit.reset, OrderEnforcing.step,
            OrderEnforcing.reset, data_equivalence, check_reset_seed_determinism
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2016 OpenAI
            Copyright (c) 2022 Farama Foundation

Upstream:   https://github.com/haosulab/ManiSkill
Commit:     62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
Files:      mani_skill/envs/sapien_env.py, mani_skill/envs/tasks/tabletop/pick_cube.py
Symbols:    BaseEnv.step (the terminated/truncated block), BaseEnv.compute_sparse_reward,
            PickCubeEnv.evaluate (the is_obj_placed test)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for point_reach_env. The oracle test builds the point-mass dynamics
as a bare task, wraps it in these transcribed TimeLimit and OrderEnforcing
wrappers, and checks that the exercise's single-class reference behaves like the
upstream wrapper stack. It also runs the transcribed
check_reset_seed_determinism on the reference.

WHAT WAS CHANGED, AND WHY
-------------------------
The control flow is transcribed line for line. Changes:

  * numpy -> torch. ``seeding.np_random(seed)`` becomes
    ``torch.Generator().manual_seed(seed)``; an unseeded first use calls
    ``Generator.seed()`` (entropy), like ``seeding.np_random()`` with no seed.
    ``np.ndarray`` branches of ``data_equivalence`` become ``torch.Tensor``
    branches (``torch.equal`` / ``torch.allclose`` with the same 1e-5
    tolerances).
  * ``check_reset_seed_determinism`` compares generator states with
    ``Generator.get_state()`` instead of ``bit_generator.state``, and takes a
    ``rng_of(env)`` accessor because the exercise does not expose
    ``env.unwrapped._np_random``. Observation-space membership checks and the
    signature/TypeError handling are dropped (no spaces here). Upstream only
    asserts observation equality when ``env.spec.nondeterministic is False``;
    here the environment is deterministic, so those asserts always run.
    ``logger.warn`` calls for near-equal observations are dropped.
  * ``ResetNeeded`` is defined locally as a ``RuntimeError`` subclass (upstream:
    ``gymnasium.error.Error``) so the wrappers need no gymnasium import.
  * ``Wrapper``/``RecordConstructorArgs`` plumbing, ``spec`` and ``render`` are
    removed. ``TimeLimit.__init__`` keeps its TypeError/ValueError checks.
  * ManiSkill tensors are batched over parallel environments; the functions
    below take one environment's Python bools and floats, so
    ``torch.logical_or`` becomes ``or`` and ``.clone()`` disappears.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch

__all__ = [
    "GymEnv",
    "OrderEnforcing",
    "ResetNeeded",
    "TimeLimit",
    "check_reset_seed_determinism",
    "data_equivalence",
    "maniskill_is_obj_placed",
    "maniskill_sparse_reward",
    "maniskill_step_flags",
]


class ResetNeeded(RuntimeError):
    """gymnasium.error.ResetNeeded, re-based on RuntimeError."""


# --------------------------------------------------------------------------
# gymnasium/core.py
# --------------------------------------------------------------------------
class GymEnv:
    """The seeding part of gymnasium.core.Env."""

    _np_random: torch.Generator | None = None

    def step(self, action):
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random = torch.Generator().manual_seed(seed)

    @property
    def np_random(self) -> torch.Generator:
        if self._np_random is None:
            self._np_random = torch.Generator()
            self._np_random.seed()
        return self._np_random


# --------------------------------------------------------------------------
# gymnasium/wrappers/common.py
# --------------------------------------------------------------------------
class TimeLimit:
    def __init__(self, env, max_episode_steps: int):
        if not isinstance(max_episode_steps, int):
            raise TypeError(
                f"Expect the `max_episode_steps` to be an integer, actually: {type(max_episode_steps)}"
            )
        if max_episode_steps <= 0:
            raise ValueError(
                f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"
            )
        self.env = env
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)


class OrderEnforcing:
    def __init__(self, env):
        self.env = env
        self._has_reset: bool = False

    def step(self, action):
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._has_reset = True
        return self.env.reset(seed=seed, options=options)


# --------------------------------------------------------------------------
# gymnasium/utils/env_checker.py
# --------------------------------------------------------------------------
def data_equivalence(data_1: Any, data_2: Any, exact: bool = False) -> bool:
    if type(data_1) is not type(data_2):
        return False
    elif isinstance(data_1, dict):
        return data_1.keys() == data_2.keys() and all(
            data_equivalence(data_1[k], data_2[k], exact) for k in data_1.keys()
        )
    elif isinstance(data_1, (tuple, list)):
        return len(data_1) == len(data_2) and all(
            data_equivalence(o_1, o_2, exact) for o_1, o_2 in zip(data_1, data_2, strict=True)
        )
    elif isinstance(data_1, torch.Tensor):
        if data_1.shape == data_2.shape and data_1.dtype == data_2.dtype:
            if exact:
                return bool(torch.equal(data_1, data_2))
            else:
                return bool(torch.allclose(data_1, data_2, rtol=1e-5, atol=1e-5))
        else:
            return False
    else:
        return bool(data_1 == data_2)


def check_reset_seed_determinism(env, rng_of: Callable[[Any], torch.Generator]) -> None:
    obs_1, info = env.reset(seed=123)
    # Some environments reuse an internal observation buffer.  Keep a
    # snapshot before the next reset can mutate the object in place.
    obs_1 = deepcopy(obs_1)
    assert rng_of(env) is not None, (
        "Expects the random number generator to have been generated given a seed was passed to reset."
    )
    seed_123_rng_1 = rng_of(env).get_state().clone()

    obs_2, info = env.reset()
    obs_2 = deepcopy(obs_2)

    obs_3, info = env.reset(seed=123)
    seed_123_rng_3 = rng_of(env).get_state().clone()

    obs_4, info = env.reset()

    assert data_equivalence(obs_1, obs_3), (
        "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent."
    )
    assert data_equivalence(obs_2, obs_4), (
        "Using `env.reset(seed=123)` then `env.reset()` is non-deterministic as the observations are not equivalent."
    )

    assert torch.equal(seed_123_rng_1, seed_123_rng_3), (
        "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`."
    )

    obs_5, info = env.reset(seed=456)
    assert not torch.equal(rng_of(env).get_state(), seed_123_rng_1), (
        "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`."
    )


# --------------------------------------------------------------------------
# mani_skill/envs/sapien_env.py and tasks/tabletop/pick_cube.py
# --------------------------------------------------------------------------
def maniskill_step_flags(info: dict) -> tuple[bool, bool]:
    """BaseEnv.step: terminated from success/fail; truncated is always False."""
    if "success" in info:
        if "fail" in info:
            terminated = info["success"] or info["fail"]
        else:
            terminated = info["success"]
    else:
        if "fail" in info:
            terminated = info["fail"]
        else:
            terminated = False
    return terminated, False


def maniskill_sparse_reward(info: dict) -> float:
    """BaseEnv.compute_sparse_reward: success - fail, or success, or -fail, or 0."""
    if "success" in info:
        if "fail" in info:
            reward = info["success"] - info["fail"]
        else:
            reward = info["success"]
    else:
        if "fail" in info:
            reward = -info["fail"]
        else:
            reward = 0
    return float(reward)


def maniskill_is_obj_placed(obj_p: torch.Tensor, goal_p: torch.Tensor, goal_thresh: float) -> bool:
    """PickCubeEnv.evaluate: the object counts as placed within goal_thresh, inclusive."""
    return bool(torch.linalg.norm(goal_p - obj_p) <= goal_thresh)
