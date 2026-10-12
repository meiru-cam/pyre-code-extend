"""Mutation gate for the Diffusion Forcing noise-schedule exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "diffusion_forcing_schedule"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


PYRAMID_ENTRY = "matrix[m, t] = S + int(t * u) - m"

MUTATIONS = [
    # The research sketch's mutation: truncation applied after the row is subtracted.
    _mutation("int_outside_the_subtraction", (PYRAMID_ENTRY, "matrix[m, t] = S + int(t * u - m)")),
    _mutation("round_instead_of_int", (PYRAMID_ENTRY, "matrix[m, t] = S + round(t * u) - m")),
    _mutation("height_missing_final_row", (
        "height = S + int((horizon - 1) * u) + 1", "height = S + int((horizon - 1) * u)",
    )),
    _mutation("height_uses_horizon_not_last_index", (
        "height = S + int((horizon - 1) * u) + 1", "height = S + int(horizon * u) + 1",
    )),
    _mutation("no_upper_clip", ("return matrix.clamp(0, S)", "return matrix.clamp(min=0)")),
    _mutation("autoregressive_keeps_the_scale", (
        "return _pyramid(horizon, S, S)", "return _pyramid(horizon, S, uncertainty_scale)",
    )),
    _mutation("full_sequence_without_clean_row", (
        "torch.arange(S, -1, -1, dtype=torch.long)", "torch.arange(S, 0, -1, dtype=torch.long)",
    )),
    _mutation("trains_on_one_level_per_sequence", (
        "levels = torch.randint(0, timesteps, (num_frames, batch_size), generator=generator)",
        "levels = torch.randint(0, timesteps, (1, batch_size), generator=generator).expand(num_frames, batch_size).clone()",
    )),
    _mutation("missing_frames_set_clean", (
        "torch.full_like(levels, timesteps - 1)", "torch.zeros_like(levels)",
    )),
    _mutation("missing_frames_level_off_by_one", (
        "torch.full_like(levels, timesteps - 1)", "torch.full_like(levels, timesteps)",
    )),
    _mutation("context_at_pure_noise", (
        "context = torch.zeros(n_context, dtype=torch.long)",
        "context = torch.full((n_context,), int(matrix.max()), dtype=torch.long)",
    )),
    _mutation("updates_every_frame", (
        "return from_levels, to_levels, from_levels != to_levels",
        "return from_levels, to_levels, torch.ones_like(from_levels, dtype=torch.bool)",
    )),
    _mutation("stabilization_off_by_one", (
        "torch.full_like(real, stabilization_level - 1)", "torch.full_like(real, stabilization_level)",
    )),
    _mutation("no_stabilization", (
        "return torch.where(real < 0, torch.full_like(real, stabilization_level - 1), real)",
        "return real.clamp(min=0)",
    )),
    _mutation("real_steps_start_at_zero", (
        "torch.linspace(-1, timesteps - 1, sampling_timesteps + 1).long()",
        "torch.linspace(0, timesteps - 1, sampling_timesteps + 1).long()",
    )),
    _mutation("accepts_a_level_above_S", (
        "int(levels.max()) > sampling_timesteps", "int(levels.max()) > timesteps",
    )),
    _mutation("uses_the_global_rng", (
        "levels = torch.randint(0, timesteps, (num_frames, batch_size), generator=generator)",
        "levels = torch.randint(0, timesteps, (num_frames, batch_size))",
    )),
]


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task, known_ids=set(TASKS))
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert set(task["advisory_prerequisites"]) <= set(TASKS)
    assert len(task["interview_questions"]) == 7


def test_mutation_names_are_unique():
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {mutation.name for mutation in MUTATIONS}
