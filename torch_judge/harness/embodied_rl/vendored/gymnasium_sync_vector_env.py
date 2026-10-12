"""Vendored autoreset oracle from Gymnasium's synchronous vector environment.

Upstream:   https://github.com/Farama-Foundation/Gymnasium
Commit:     d77197c284406e6094c9d1a74d4988a8749ce849
Files:      gymnasium/vector/sync_vector_env.py, gymnasium/vector/vector_env.py
Symbols:    SyncVectorEnv.reset, SyncVectorEnv.step, VectorEnv._add_info, AutoresetMode
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2016 OpenAI
            Copyright (c) 2022 Farama Foundation

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in
            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
            THE SOFTWARE.

WHY THIS FILE EXISTS
--------------------
Ground truth for the vector_autoreset exercise, so its reference solution is not
checked only against an oracle written by the same author. The oracle test in
`tests/quality/test_vector_autoreset_oracle.py` drives both with identical fake
sub-environments and compares every returned value.

WHAT WAS CHANGED, AND WHY
-------------------------
The autoreset branches of ``step``, the seed handling of ``reset`` and the
``_add_info`` batching are transcribed line for line. Removed infrastructure:

  * Spaces are dropped. ``iterate(self.action_space, actions)`` becomes
    ``iter(actions)``, and ``concatenate(single_observation_space, ...)``
    becomes returning the list of per-env observations; the oracle test stacks
    it. Space-equality checks in ``__init__`` are dropped.
  * The constructor takes already-built environments instead of ``env_fns``.
  * ``copy`` is always on: rewards and flags are returned with ``np.copy`` as
    upstream; the observation list is a fresh list each call.
  * ``reset`` keeps the seed expansion (None, int, list) and drops ``options``
    and the ``reset_mask`` partial-reset branch.
  * ``render``, ``call``, ``get_attr``, ``set_attr`` and ``close_extras`` are
    dropped. ``AutoresetMode`` is copied verbatim.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

__all__ = ["AutoresetMode", "GymSyncVectorEnv"]


class AutoresetMode(Enum):
    """Enum representing the different autoreset modes, next step, same step and disabled."""

    NEXT_STEP = "NextStep"
    SAME_STEP = "SameStep"
    DISABLED = "Disabled"


class GymSyncVectorEnv:
    """SyncVectorEnv reduced to its reset, autoreset and info-batching logic."""

    def __init__(self, envs, autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP) -> None:
        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )
        self.envs = list(envs)
        self.num_envs = len(self.envs)

        self._env_obs = [None for _ in range(self.num_envs)]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(self, *, seed: int | list[int | None] | None = None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        if len(seed) != self.num_envs:
            raise ValueError(
                f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."
            )

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed, strict=True)):
            self._env_obs[i], env_info = env.reset(seed=single_seed)

            infos = self._add_info(infos, env_info, i)

        return list(self._env_obs), infos

    def step(self, actions):
        actions_iter = iter(actions)

        infos = {}
        for i, (action, _) in enumerate(zip(actions_iter, self.envs, strict=True)):
            if self.autoreset_mode == AutoresetMode.NEXT_STEP:
                if self._autoreset_envs[i]:
                    self._env_obs[i], env_info = self.envs[i].reset()

                    self._rewards[i] = 0.0
                    self._terminations[i] = False
                    self._truncations[i] = False
                else:
                    (
                        self._env_obs[i],
                        self._rewards[i],
                        self._terminations[i],
                        self._truncations[i],
                        env_info,
                    ) = self.envs[i].step(action)
            elif self.autoreset_mode == AutoresetMode.DISABLED:
                # assumes that the user has correctly autoreset
                assert not self._autoreset_envs[i], f"{self._autoreset_envs=}"
                (
                    self._env_obs[i],
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)
            elif self.autoreset_mode == AutoresetMode.SAME_STEP:
                (
                    self._env_obs[i],
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)

                if self._terminations[i] or self._truncations[i]:
                    infos = self._add_info(
                        infos,
                        {"final_obs": self._env_obs[i], "final_info": env_info},
                        i,
                    )

                    self._env_obs[i], env_info = self.envs[i].reset()
            else:
                raise ValueError(f"Unexpected autoreset mode, {self.autoreset_mode}")

            infos = self._add_info(infos, env_info, i)

        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            list(self._env_obs),
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def _add_info(self, vector_infos: dict[str, Any], env_info: dict[str, Any], env_num: int) -> dict[str, Any]:
        """Transcribed from vector_env.py::VectorEnv._add_info."""
        for key, value in env_info.items():
            # It is easier for users to access their `final_obs` in the unbatched array of `obs` objects
            if key == "final_obs":
                if "final_obs" in vector_infos:
                    array = vector_infos["final_obs"]
                else:
                    array = np.full(self.num_envs, fill_value=None, dtype=object)
                array[env_num] = value
            # If value is a dictionary, then we apply the `_add_info` recursively.
            elif isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            # Otherwise, we are a base case to group the data
            else:
                # If the key doesn't exist in the vector infos, then we can create an array of that batch type
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(type(value), np.number):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        # We assume that all instances of the np.array info are of the same shape
                        array = np.zeros((self.num_envs, *value.shape), dtype=value.dtype)
                    else:
                        # For unknown objects, we use a Numpy object array
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                # Otherwise, just use the array that already exists
                else:
                    array = vector_infos[key]

                # Assign the data in the `env_num` position
                #   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
                array[env_num] = value

            # Get the array mask and if it doesn't already exist then create a zero bool array
            array_mask = vector_infos.get(f"_{key}", np.zeros(self.num_envs, dtype=np.bool_))
            array_mask[env_num] = True

            # Update the vector info with the updated data and mask information
            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask
        return vector_infos
