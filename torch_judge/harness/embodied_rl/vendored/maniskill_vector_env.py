"""Vendored per-episode metric recording from ManiSkill's GPU vector env wrapper.

Upstream:   https://github.com/haosulab/ManiSkill
Commit:     62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
Files:      mani_skill/vector/wrappers/gymnasium.py
            (elapsed-step semantics from mani_skill/envs/sapien_env.py)
Symbols:    ManiSkillVectorEnv.__init__ (record_metrics buffers),
            ManiSkillVectorEnv.reset, ManiSkillVectorEnv.step
Retrieved:  2026-10-12
License:    Apache License 2.0
            The upstream LICENSE names no copyright holder; the code is by
            the ManiSkill authors. Licensed under the
            Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for the episode_metrics_tracker exercise, so its reference
solution is not checked only against an oracle written by the same author.
ManiSkill is the primary oracle: its wrapper records return, length,
success_once / fail_once and success_at_end / fail_at_end for every
sub-environment, snapshots them into ``final_info`` and then partially resets
only the finished sub-environments (Gymnasium's SAME_STEP autoreset).

WHAT WAS CHANGED, AND WHY
-------------------------
The record_metrics, ignore_terminations and auto_reset arithmetic of
``step`` and ``reset`` is transcribed line for line. Removed infrastructure:

  * The class no longer subclasses ``gymnasium.vector.VectorEnv`` and does not
    call ``gym.make``; it wraps any object with ``step(actions)``,
    ``reset(seed=, options=)``, ``num_envs``, ``device`` and an
    ``elapsed_steps`` tensor, which stands in for ``self.base_env``.
    Observation/action spaces, metadata, ``call``, ``get_attr``, ``render`` and
    ``close_extras`` are dropped.
  * ``torch_clone_dict`` (mani_skill.utils.common) is inlined as a recursive
    clone of tensors in nested dicts.
  * The ``isinstance(terminations, bool)`` branch for single CPU envs is kept.
  * The reconfiguration assertion is dropped; it guards simulator state only.

Upstream zeroes the caller's ``terminations`` tensor in place under
``ignore_terminations``; this is kept, so the oracle test passes it a copy.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["ManiSkillRecordMetricsVectorEnv"]


def _torch_clone_dict(data):
    """Inlined mani_skill.utils.common.torch_clone_dict."""
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, dict):
        return {k: _torch_clone_dict(v) for k, v in data.items()}
    return data


class ManiSkillRecordMetricsVectorEnv:
    """Transcribed from ManiSkillVectorEnv with record_metrics support."""

    def __init__(
        self,
        env,
        num_envs: int = 1,
        auto_reset: bool = True,
        ignore_terminations: bool = False,
        record_metrics: bool = False,
    ):
        self._env = env
        num_envs = self.base_env.num_envs
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics

        if self.record_metrics:
            self.success_once = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.bool
            )
            self.fail_once = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.bool
            )
            self.returns = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.float32
            )

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self):
        return self._env

    def reset(
        self,
        *,
        seed=None,
        options: Optional[dict] = None,
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        if options is not None and "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.base_env.device
            )
            mask[env_idx] = True
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0
        return obs, info

    def step(self, actions):
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        episode_info: Optional[dict] = None
        if self.record_metrics:
            episode_info = dict()
            self.returns += rew
            if "success" in infos:
                self.success_once = self.success_once | infos["success"]
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.base_env.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )

        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)

        if self.ignore_terminations:
            terminations[:] = False
            if episode_info:
                if "success" in infos:
                    episode_info["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    episode_info["fail_at_end"] = infos["fail"].clone()
        if self.record_metrics:
            infos["episode"] = episode_info

        dones = torch.logical_or(terminations, truncations)

        if dones.any() and self.auto_reset:
            final_obs = _torch_clone_dict(obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = _torch_clone_dict(infos)
            obs, infos = self.reset(options=dict(env_idx=env_idx))
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
        return (
            obs,
            rew,
            terminations,
            truncations,
            infos,
        )
