"""Mutation gate for the truncation-bootstrapping robot GAE exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "truncation_bootstrap_gae"

TRUNC_LINE = "boot = torch.where(trunc, final_values.to(dtype), next_values)"
TERM_LINE = "boot = torch.where(term, torch.zeros_like(boot), boot)"
CONT_LINE = "cont = (~(term | trunc)).to(dtype)"
CHUNK_LINE = "rewards = (rewards * valid_mask.to(rewards.dtype)).sum(dim=-1)"
NEXT_LINE = "next_values = torch.cat([values[1:], next_value.to(dtype).unsqueeze(0)], dim=0)"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    # ManiSkill's baseline at the pinned commit: final_values for every finished env.
    _mutation(
        "bootstrap_on_termination",
        (TRUNC_LINE, "boot = torch.where(trunc | term, final_values.to(dtype), next_values)"),
        (TERM_LINE, "boot = boot"),
    ),
    # gae_advantage's single done flag: a time limit treated as a terminal state.
    _mutation("zero_on_truncation", (TRUNC_LINE, "boot = torch.where(trunc, torch.zeros_like(next_values), next_values)")),
    _mutation("trace_not_cut_at_truncation", (CONT_LINE, "cont = (~term).to(dtype)")),
    _mutation("reset_value_at_truncation", (TRUNC_LINE, "boot = next_values")),
    _mutation(
        "truncation_overrides_termination",
        (TRUNC_LINE, "boot = torch.where(term, torch.zeros_like(next_values), next_values)"),
        (TERM_LINE, "boot = torch.where(trunc, final_values.to(dtype), boot)"),
    ),
    _mutation(
        "final_values_from_next_row",
        (TRUNC_LINE, "boot = torch.where(trunc, torch.cat([final_values[1:], final_values[-1:]]).to(dtype), next_values)"),
    ),
    _mutation(
        "last_row_bootstraps_zero",
        (NEXT_LINE, "next_values = torch.cat([values[1:], torch.zeros_like(next_value).to(dtype).unsqueeze(0)], dim=0)"),
    ),
    _mutation("chunk_sum_ignores_mask", (CHUNK_LINE, "rewards = rewards.sum(dim=-1)")),
    _mutation(
        "chunk_discounts_inside_chunk",
        (CHUNK_LINE, "rewards = (rewards * valid_mask.to(rewards.dtype)"
                     " * gamma ** torch.arange(rewards.shape[-1], dtype=rewards.dtype)).sum(dim=-1)"),
    ),
    _mutation(
        "chunk_gamma_per_env_step",
        (CHUNK_LINE, CHUNK_LINE + "\n        gamma = gamma ** valid_mask.shape[-1]"),
    ),
    _mutation("returns_without_values", ("return advantages, advantages + values", "return advantages, advantages")),
    _mutation("forward_recursion", ("for t in reversed(range(T)):", "for t in range(T):")),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_truncation_bootstrap_gae_rejects_mutations(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {m.name for m in MUTATIONS}


def test_truncation_bootstrap_gae_metadata():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    visible = [t for t in task["tests"] if t.get("visibility", "visible") == "visible"]
    assert 1 <= len(visible) <= 3
    assert task["advisory_prerequisites"] == ["gae_advantage", "vector_autoreset", "chunk_step_executor"]
