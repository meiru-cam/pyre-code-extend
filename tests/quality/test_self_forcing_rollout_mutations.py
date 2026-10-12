"""Mutation gate for the Self Forcing training-rollout exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "self_forcing_rollout"

STEP_LOOP = (
    "        with torch.no_grad():\n"
    "            for j in range(exit_step):\n"
    "                x0_hat = model.denoise(x, steps[j], cache)\n"
    "                x = renoise(x0_hat, steps[j + 1])\n"
)
EXIT_CALL = "        if start >= grad_from_frame:\n"
CACHE_UPDATE = (
    "        with torch.no_grad():\n"
    "            context = renoise(x0_hat, context_noise)\n"
    "            cache.append(model.kv(context, context_noise, cache))\n"
)
DROP_ZERO = (
    "    if steps and steps[-1] == 0:\n"
    "        steps = steps[:-1]  # the clean end of the schedule is not a denoising step\n"
)
RENOISE_DRAW = "        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    _mutation(
        "gradient through every denoising step",
        (STEP_LOOP, STEP_LOOP.replace("        with torch.no_grad():\n", "        if True:\n")),
    ),
    _mutation(
        "cache computed with gradients",
        (CACHE_UPDATE, CACHE_UPDATE.replace("        with torch.no_grad():\n", "        if True:\n")),
    ),
    _mutation(
        "cache from the clean prediction without context re-noise",
        (CACHE_UPDATE, CACHE_UPDATE.replace("renoise(x0_hat, context_noise)", "x0_hat")),
    ),
    _mutation(
        "cache from the noisy exit input",
        (CACHE_UPDATE, CACHE_UPDATE.replace("renoise(x0_hat, context_noise)", "renoise(x, context_noise)")),
    ),
    _mutation(
        "re-noise to the current timestep",
        (STEP_LOOP, STEP_LOOP.replace("steps[j + 1]", "steps[j]")),
    ),
    _mutation(
        "exit step taken one index late",
        (STEP_LOOP, STEP_LOOP.replace("range(exit_step)", "range(min(exit_step + 1, len(steps) - 1))")),
        ("model.denoise(x, steps[exit_step], cache)  # the only", "model.denoise(x, steps[min(exit_step + 1, len(steps) - 1)], cache)  # the only"),
    ),
    _mutation(
        "trailing zero timestep kept",
        (DROP_ZERO, ""),
    ),
    _mutation(
        "first block's exit step used for every block",
        ("    for b, exit_step in enumerate(exit_steps):\n", "    for b, _ in enumerate(exit_steps):\n        exit_step = exit_steps[0]\n"),
    ),
    _mutation(
        "gradient start compared with the block's last frame",
        (EXIT_CALL, "        if start + frames_per_block - 1 >= grad_from_frame:\n"),
    ),
    _mutation(
        "context re-noise skipped when the timestep is zero",
        (CACHE_UPDATE, CACHE_UPDATE.replace("renoise(x0_hat, context_noise)", "renoise(x0_hat, context_noise) if context_noise else x0_hat")),
    ),
    _mutation(
        "cache entry built at timestep zero",
        (CACHE_UPDATE, CACHE_UPDATE.replace("model.kv(context, context_noise, cache)", "model.kv(context, 0, cache)")),
    ),
    _mutation(
        "output is the cached context instead of the prediction",
        ("        outputs.append(x0_hat)\n", ""),
        ("            cache.append(model.kv(context, context_noise, cache))\n",
         "            cache.append(model.kv(context, context_noise, cache))\n        outputs.append(context)\n"),
    ),
    _mutation(
        "noise drawn from the global generator",
        (RENOISE_DRAW, "        eps = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)\n"),
    ),
    _mutation(
        "cache passed with the current block's entry",
        ("        outputs.append(x0_hat)\n", "        outputs.append(x0_hat)\n        cache.append(model.kv(x0_hat.detach(), context_noise, cache))\n"),
    ),
]


def test_task_validates():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_fail_unshown_cases(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {mutation.name for mutation in MUTATIONS}
