"""Mutation gate for the world-model CEM refit exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "world_model_cem_step"

ORDER_LINE = 'order = torch.argsort(energies, descending=(mode == "tdmpc"), stable=True)'
WEIGHT_LINE = "w = torch.exp(temperature * (values - values.max()))"
SIGMA_LINE = "sigma = torch.sqrt((w * (elites - mu) ** 2).sum(dim=0))"
MEAN_BLEND = "new_mean = mu * (1.0 - momentum_mean) + mean * momentum_mean"
STD_BLEND = "new_std = sigma * (1.0 - momentum_std) + std * momentum_std"


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
        "vjepa2 keeps the highest energies (topk largest=True)",
        (ORDER_LINE, "order = torch.argsort(energies, descending=True, stable=True)"),
    ),
    _mutation(
        "tdmpc keeps the lowest values",
        (ORDER_LINE, "order = torch.argsort(energies, descending=False, stable=True)"),
    ),
    _mutation(
        "ties go to the higher sample index",
        (ORDER_LINE, 'order = S - 1 - torch.argsort(energies.flip(0), descending=(mode == "tdmpc"), stable=True)'),
    ),
    _mutation(
        "momentum applied the wrong way round",
        (MEAN_BLEND, "new_mean = mu * momentum_mean + mean * (1.0 - momentum_mean)"),
        (STD_BLEND, "new_std = sigma * momentum_std + std * (1.0 - momentum_std)"),
    ),
    _mutation(
        "std blended with the mean momentum",
        (STD_BLEND, "new_std = sigma * (1.0 - momentum_mean) + std * momentum_mean"),
    ),
    _mutation(
        "softmax without subtracting the max",
        (WEIGHT_LINE, "w = torch.exp(temperature * values)"),
    ),
    _mutation(
        "temperature ignored",
        (WEIGHT_LINE, "w = torch.exp(values - values.max())"),
    ),
    _mutation(
        "vjepa2 uses the biased std",
        ("sigma = elites.std(dim=0, correction=1)", "sigma = elites.std(dim=0, correction=0)"),
    ),
    _mutation(
        "tdmpc std divides by topk - 1",
        (SIGMA_LINE, "sigma = torch.sqrt((w * (elites - mu) ** 2).sum(dim=0) * topk / max(topk - 1, 1))"),
    ),
    _mutation(
        "tdmpc ignores the weights in the mean",
        ("mu = (w * elites).sum(dim=0)", "mu = elites.mean(dim=0)"),
    ),
    _mutation(
        "vjepa2 accepts topk 1 and returns a NaN std",
        ('    if mode == "vjepa2" and topk < 2:\n        raise ValueError("vjepa2 mode needs topk >= 2 for an unbiased std")\n', ""),
    ),
    _mutation(
        "accepts non-finite energies",
        ('    if not torch.isfinite(energies).all():\n        raise ValueError("energies must be finite")\n', ""),
    ),
    _mutation(
        "blends the mean in place",
        (MEAN_BLEND, "new_mean = mean.mul_(momentum_mean).add_(mu * (1.0 - momentum_mean))"),
    ),
    _mutation(
        "casts to float32",
        ("    # Stable sort:", "    samples, energies, mean, std = (t.float() for t in (samples, energies, mean, std))\n    # Stable sort:"),
    ),
]


def test_world_model_cem_step_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_world_model_cem_step_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
