"""Mutation gate for the pi0_flow_matching_chunk exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "pi0_flow_matching_chunk"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """Apply exact-substring edits to the reference; each old string must occur once."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"edit target not unique: {old!r}"
        code = code.replace(old, new)
    return code


MUTATIONS = [
    # Research section 3.1 V4: noise-at-one interpolation with the noise-at-zero target.
    Mutation("mixed_conventions", edit((
        "            v_target = noise - actions\n",
        "            v_target = actions - noise\n",
    ))),
    # Research section 3.1 V4: dt with the wrong sign (step away from the data).
    Mutation("dt_wrong_sign", edit(("x = x + dt * velocity_fn(x, time)", "x = x - dt * velocity_fn(x, time)"))),
    # Research section 3.1 V4: prefix clamped only after the step, not before.
    Mutation("prefix_clamped_only_after", edit((
        "                x = torch.where(keep, hard_prefix, x)  # clamp before the velocity call\n",
        "",
    ))),
    Mutation("prefix_clamped_only_before", edit((
        "            if hard_prefix is not None:\n                x = torch.where(keep, hard_prefix, x)  # and after the update\n",
        "",
    ))),
    Mutation("time_one_step_late", edit((
        "t_k = (1.0 + k * dt) if noise_at_one else k * dt",
        "t_k = (1.0 + (k + 1) * dt) if noise_at_one else (k + 1) * dt",
    ))),
    Mutation("linspace_schedule_with_endpoint", edit((
        "t_k = (1.0 + k * dt) if noise_at_one else k * dt",
        "t_k = (1.0 - k / max(num_steps - 1, 1)) if noise_at_one else k / max(num_steps - 1, 1)",
    ))),
    Mutation("sampler_ignores_convention", edit((
        "        clean_time = 0.0 if noise_at_one else 1.0\n        dt = (-1.0 if noise_at_one else 1.0) / num_steps\n",
        "        noise_at_one = True\n        clean_time = 0.0\n        dt = -1.0 / num_steps\n",
    ))),
    Mutation("training_ignores_convention", edit((
        "        if noise_at_one:\n            x_t = t * noise",
        "        if True:\n            x_t = t * noise",
    ))),
    Mutation("prefix_gets_noise_time", edit((
        "            clean_time = 0.0 if noise_at_one else 1.0\n            per_position",
        "            clean_time = 1.0 if noise_at_one else 0.0\n            per_position",
    ))),
    Mutation("sampler_prefix_time_not_clean", edit((
        "                time = torch.where(hard_prefix_mask, clean, time[:, None].expand(batch, horizon))\n",
        "",
    ))),
    Mutation("target_zeroed_on_prefix", edit((
        "        return x_t, v_target, model_time",
        "        if prefix_mask is not None:\n"
        "            v_target = v_target.masked_fill(prefix_mask[:, :, None], 0.0)\n"
        "        return x_t, v_target, model_time",
    ))),
    Mutation("in_place_update", edit(("x = x + dt * velocity_fn(x, time)", "x += dt * velocity_fn(x, time)"))),
    Mutation("time_always_float32", edit((
        "time = torch.full((batch,), t_k, dtype=noise.dtype, device=noise.device)",
        "time = torch.full((batch,), t_k, dtype=torch.float32, device=noise.device)",
    ))),
    Mutation("time_shape_not_validated", edit((
        "        if time.shape != actions.shape[:1]:\n            raise ValueError(\"time must have shape (B,)\")\n",
        "",
    ))),
    Mutation("model_time_scalar_with_prefix", edit((
        "            model_time = torch.where(prefix_mask, torch.full_like(per_position, clean_time), per_position)\n"
        "            t = model_time[:, :, None]\n",
        "            model_time = time\n"
        "            t = torch.where(prefix_mask, torch.full_like(per_position, clean_time), per_position)[:, :, None]\n",
    ))),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {mutation.name for mutation in MUTATIONS}


def test_mutation_names_are_unique():
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=set(TASKS))
    assert not any(key.endswith("_zh") for key in task)
    assert set(task["advisory_prerequisites"]) <= set(TASKS)
    assert len(task["interview_questions"]) == 7
