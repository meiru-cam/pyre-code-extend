"""Mutation gate for the episode_metrics_tracker exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "episode_metrics_tracker"


def _edit(*pairs: str) -> str:
    """Apply exact-substring edits (old, new, old, new, ...) to the reference."""
    solution = get_task(TASK_ID)["solution"]
    for old, new in zip(pairs[::2], pairs[1::2]):
        assert solution.count(old) == 1, f"edit anchor not unique: {old!r}"
        solution = solution.replace(old, new)
    return solution


MUTATIONS = [
    # The snapshot aliases the live buffers, so the auto reset zeroes it.
    Mutation("snapshot_not_cloned", _edit(
        '''        episode["return"] = self.returns.clone()
        episode["episode_len"] = self.lengths.clone()''',
        '''        episode["return"] = self.returns
        episode["episode_len"] = self.lengths''',
    )),
    # success_once reports this step's success instead of "ever succeeded".
    Mutation("success_not_sticky", _edit(
        "self.success_once = self.success_once | success",
        "self.success_once = success.clone()",
    )),
    # fail_once forgets an earlier failure.
    Mutation("fail_not_sticky", _edit(
        "self.fail_once = self.fail_once | fail",
        "self.fail_once = fail.clone()",
    )),
    # One done resets every environment.
    Mutation("resets_all_envs_on_any_done", _edit(
        "self.reset(torch.nonzero(dones).flatten())",
        "self.reset()",
    )),
    # Finished environments are never restarted.
    Mutation("never_auto_resets", _edit(
        "if self.auto_reset and bool(dones.any()):",
        "if False:",
    )),
    # auto_reset=False is ignored.
    Mutation("ignores_auto_reset_flag", _edit(
        "if self.auto_reset and bool(dones.any()):",
        "if bool(dones.any()):",
    )),
    # Resets before taking the snapshot (reports the restarted values).
    Mutation("resets_before_snapshot", _edit(
        '''        episode["return"] = self.returns.clone()''',
        '''        if self.auto_reset:
            _d = truncations | (torch.zeros_like(terminations) if self.ignore_terminations else terminations)
            if bool(_d.any()):
                self.reset(torch.nonzero(_d).flatten())
        episode["return"] = self.returns.clone()''',
    )),
    # ignore_terminations still lets a termination end the episode.
    Mutation("ignored_terminations_still_end_episode", _edit(
        "        self.returns += rewards",
        "        raw_terminations = terminations.clone()\n        self.returns += rewards",
        "dones = terminations | truncations",
        "dones = raw_terminations | truncations",
    )),
    # Truncations are not treated as episode ends.
    Mutation("truncation_does_not_end_episode", _edit(
        "dones = terminations | truncations",
        "dones = terminations.clone()",
    )),
    # Under ignore_terminations, the caller's tensor is zeroed in place (what upstream does).
    Mutation("zeroes_input_terminations_in_place", _edit(
        "terminations = torch.zeros_like(terminations)",
        "terminations[:] = False",
    )),
    # _at_end keys are emitted even when terminations are not ignored.
    Mutation("at_end_keys_always_emitted", _edit(
        '''        else:
            terminations = terminations.clone()''',
        '''        else:
            terminations = terminations.clone()
            if success is not None:
                episode["success_at_end"] = success.clone()''',
    )),
    # success_at_end reports the sticky flag instead of this step's success.
    Mutation("at_end_uses_sticky_flag", _edit(
        'episode["success_at_end"] = success.clone()',
        'episode["success_at_end"] = self.success_once.clone()',
    )),
    # success_once key emitted even when no success criterion was given.
    Mutation("success_key_without_success", _edit(
        '''        episode["return"] = self.returns.clone()''',
        '''        episode.setdefault("success_once", self.success_once.clone())
        episode["return"] = self.returns.clone()''',
    )),
    # Episode length counted from zero (incremented after the snapshot).
    Mutation("length_off_by_one", _edit(
        "        self.lengths += 1\n",
        "",
        '''        episode["reward"] = episode["return"] / episode["episode_len"]''',
        '''        episode["reward"] = episode["return"] / episode["episode_len"].clamp(min=1)
        self.lengths += 1''',
    )),
    # reward is the last step reward rather than return / length.
    Mutation("reward_is_last_step_reward", _edit(
        'episode["reward"] = episode["return"] / episode["episode_len"]',
        'episode["reward"] = rewards.to(torch.float32).clone()',
    )),
    # Partial reset forgets to clear the success flag.
    Mutation("reset_keeps_success_flag", _edit(
        "        self.success_once[mask] = False\n",
        "",
    )),
    # Returns accumulated in float64 and reported as float64.
    Mutation("returns_in_float64", _edit(
        "self.returns = torch.zeros(num_envs, dtype=torch.float32)",
        "self.returns = torch.zeros(num_envs, dtype=torch.float64)",
        "self.returns += rewards.to(torch.float32)",
        "self.returns += rewards.to(torch.float64)",
    )),
    # No input validation.
    Mutation("no_flag_dtype_check", _edit(
        "        if flag and x.dtype != torch.bool:\n            raise ValueError(f\"{name} must be torch.bool\")\n",
        "",
    )),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_episode_metrics_tracker_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {m.name for m in MUTATIONS}


def test_mutation_names_are_unique():
    names = [m.name for m in MUTATIONS]
    assert len(names) == len(set(names))
