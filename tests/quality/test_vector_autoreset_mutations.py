"""Mutation gate for vector_autoreset: each wrong autoreset rule must fail an unshown case."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "vector_autoreset"
REFERENCE = get_task(TASK_ID)["solution"]


def _edit(name: str, *pairs: str) -> Mutation:
    """Apply (old, new) exact-substring edits to the reference, each old occurring once."""
    code = REFERENCE
    for old, new in zip(pairs[::2], pairs[1::2], strict=True):
        assert code.count(old) == 1, f"{name}: {old!r} must occur exactly once"
        code = code.replace(old, new)
    return Mutation(name, code)


MUTATIONS = [
    # From the research sketch.
    _edit(
        "reset_obs_written_into_final_obs",
        """                    infos = self._add_info(infos, {"final_obs": obs, "final_info": env_info}, i)
                    obs, env_info = env.reset()""",
        """                    last_info = env_info
                    obs, env_info = env.reset()
                    infos = self._add_info(infos, {"final_obs": obs, "final_info": last_info}, i)""",
    ),
    _edit(
        "next_step_reset_keeps_stale_reward",
        "                reward, term, trunc = 0.0, False, False",
        "                reward, term, trunc = self._last_rewards[i], False, False",
        "        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]",
        "        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]\n"
        "        self._last_rewards = list(rewards)",
    ),
    _edit(
        "per_env_seeds_all_equal",
        "            seeds = [seed + i for i in range(n)]",
        "            seeds = [seed for i in range(n)]",
    ),
    # Further wrong rules.
    _edit(
        "next_step_reset_keeps_done_flags",
        "                reward, term, trunc = 0.0, False, False",
        "                reward, (term, trunc) = 0.0, self._last_flags[i]",
        "        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]",
        "        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]\n"
        "        self._last_flags = list(zip(terminated, truncated))",
    ),
    _edit(
        "next_step_resets_and_also_steps",
        """                obs, env_info = env.reset()
                reward, term, trunc = 0.0, False, False""",
        """                env.reset()
                obs, reward, term, trunc, env_info = env.step(actions[i])""",
    ),
    _edit(
        "autoreset_reuses_initial_seed",
        "        self._pending_reset = [False] * n\n        observations, infos = [], {}",
        "        self._pending_reset = [False] * n\n        self._seeds = seeds\n        observations, infos = [], {}",
        "                obs, env_info = env.reset()\n                reward",
        "                obs, env_info = env.reset(seed=self._seeds[i])\n                reward",
        "                    obs, env_info = env.reset()",
        "                    obs, env_info = env.reset(seed=self._seeds[i])",
    ),
    _edit(
        "reset_keeps_pending_autoreset",
        "        self._pending_reset = [False] * n\n        observations, infos = [], {}",
        "        observations, infos = [], {}",
    ),
    _edit(
        "same_step_top_level_info_from_the_ending_step",
        """                    obs, env_info = env.reset()
            infos = self._add_info(infos, env_info, i)""",
        """                    obs, _ = env.reset()
            infos = self._add_info(infos, env_info, i)""",
    ),
    _edit(
        "same_step_returns_last_obs_not_reset_obs",
        """                    obs, env_info = env.reset()
            infos = self._add_info(infos, env_info, i)""",
        """                    _, env_info = env.reset()
            infos = self._add_info(infos, env_info, i)""",
    ),
    _edit(
        "final_info_as_list_of_dicts",
        """            if isinstance(value, dict):
                infos[key] = self._add_info(infos.get(key, {}), value, i)
            else:""",
        """            if False:
                pass
            else:""",
    ),
    _edit(
        "final_keys_added_for_every_env",
        """                if self.autoreset_mode == "SameStep" and (term or trunc):
                    infos = self._add_info(infos, {"final_obs": obs, "final_info": env_info}, i)
                    obs, env_info = env.reset()""",
        """                if self.autoreset_mode == "SameStep":
                    infos = self._add_info(infos, {"final_obs": obs if (term or trunc) else None, "final_info": env_info}, i)
                if self.autoreset_mode == "SameStep" and (term or trunc):
                    obs, env_info = env.reset()""",
    ),
    _edit(
        "missing_info_filled_with_zero",
        "                    infos[key] = [None] * n",
        "                    infos[key] = [0] * n",
    ),
    _edit(
        "pending_reset_only_on_termination",
        "        self._pending_reset = [te or tr for te, tr in zip(terminated, truncated)]",
        "        self._pending_reset = list(terminated)",
    ),
]


def test_mutation_names_are_unique():
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


@pytest.mark.parametrize("_repeat", range(3))
def test_vector_autoreset_mutations_rejected(_repeat):
    assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
