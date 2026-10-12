"""Mutation gate for init_state_group_advantage."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "init_state_group_advantage"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(name: str, old: str, new: str) -> Mutation:
    """A mutation that replaces one exact substring of the reference solution."""
    assert REFERENCE.count(old) == 1, f"{name}: {old!r} must occur exactly once in the reference"
    return Mutation(name, REFERENCE.replace(old, new))


MUTATIONS = [
    edit("groups_by_instruction_only",
         "keys = torch.stack([task_ids.long(), init_state_ids.long()], dim=1)",
         "keys = torch.stack([task_ids.long(), task_ids.long()], dim=1)"),
    edit("groups_by_init_state_number_only",
         "keys = torch.stack([task_ids.long(), init_state_ids.long()], dim=1)",
         "keys = torch.stack([init_state_ids.long(), init_state_ids.long()], dim=1)"),
    edit("mask_counts_whole_chunks",
         "(finish_steps.long() * tokens_per_action)",
         "((finish_steps.long() + actions_per_chunk - 1) // actions_per_chunk * actions_per_chunk * tokens_per_action)"),
    edit("mask_ignores_tokens_per_action",
         "(finish_steps.long() * tokens_per_action)",
         "finish_steps.long()"),
    edit("mask_includes_one_past_the_end",
         "positions < (finish_steps.long() * tokens_per_action)",
         "positions <= (finish_steps.long() * tokens_per_action)"),
    edit("dropped_groups_stay_in_mask",
         ".unsqueeze(1)) & keep.unsqueeze(1)",
         ".unsqueeze(1))"),
    edit("drops_only_all_success_groups",
         "if drop_uniform and bool((r == r[0]).all()):",
         "if drop_uniform and bool((r == 1).all()):"),
    edit("population_std",
         "r.std(unbiased=True)",
         "r.std(unbiased=False)"),
    edit("eps_ignored",
         "(r.std(unbiased=True) + eps)",
         "(r.std(unbiased=True) + 1e-6)"),
    edit("singleton_gets_zero",
         "rollout_adv[members] = r / (1.0 + eps) if mode == \"grpo\" else r",
         "rollout_adv[members] = 0.0"),
    edit("rloo_includes_own_reward",
         "rollout_adv[members] = r - (r.sum() - r) / (size - 1)",
         "rollout_adv[members] = r - r.mean()"),
    edit("contiguous_blocks_assumed",
         "_, group = torch.unique(keys, dim=0, return_inverse=True)",
         "change = torch.ones(n, dtype=torch.long)\n"
         "    change[1:] = (keys[1:] != keys[:-1]).any(dim=1).long()\n"
         "    group = torch.cumsum(change, 0)"),
    edit("dropped_rollouts_keep_rollout_advantage",
         "        if drop_uniform and bool((r == r[0]).all()):\n            continue\n        keep[members] = True",
         "        keep[members] = True\n        if drop_uniform and bool((r == r[0]).all()):\n            keep[members] = False"),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_init_state_group_advantage_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {m.name for m in MUTATIONS}
