"""Mutation gate for the multi-part Real-Time Chunking prefix guidance exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.parts_gate import assert_first_fails_in_part, mutant

TASK_ID = "rtc_prefix_guidance"

MUTATIONS = [
    # Part 1: prefix weights.
    ("linspace keeps both endpoints", 1, [
        ("ramp = torch.linspace(1, 0, end - start + 2)[1:-1]", "ramp = torch.linspace(1, 0, end - start)"),
    ]),
    ("linspace drops only the 1", 1, [
        ("ramp = torch.linspace(1, 0, end - start + 2)[1:-1]", "ramp = torch.linspace(1, 0, end - start + 1)[1:]"),
    ]),
    ("exp without the linear factor", 1, [
        ("ramp = ramp * torch.expm1(ramp) / (math.e - 1)", "ramp = torch.expm1(ramp) / (math.e - 1)"),
    ]),
    ("ramp slope keeps the unclamped end", 1, [
        ("    end = min(end, total)\n    start = min(start, end)\n",
         "    start = min(start, end)\n    full = end\n    end = min(end, total)\n"),
        ("ramp = torch.linspace(1, 0, end - start + 2)[1:-1]",
         "ramp = torch.linspace(1, 0, full - start + 2)[1:-1][: end - start] if full > start else torch.zeros(0)"),
    ]),
    ("start not clamped to end", 1, [
        ("    start = min(start, end)\n    steps", "    start = min(start, total)\n    steps"),
        ("torch.cat([torch.ones(start), ramp, torch.zeros(total - end)])",
         "torch.cat([torch.ones(start), ramp, torch.zeros(max(total - max(end, start), 0))])"),
    ]),
    ("ones schedule stops at start", 1, [
        ("return (steps < end).float()", "return (steps < start).float()"),
    ]),
    ("unknown schedule treated as linear", 1, [
        ("        raise ValueError(f\"unknown schedule {schedule!r}\")\n", "        schedule = \"linear\"\n"),
    ]),
    # Part 2: guided velocity.
    ("tau is time, not 1 - time", 2, [
        ("tau = torch.tensor(1.0 - float(time), dtype=torch.float64)", "tau = torch.tensor(float(time), dtype=torch.float64)"),
    ]),
    ("no clamp at max_guidance_weight", 2, [
        ("    gain = torch.clamp(gain, max=max_w)\n", ""),
    ]),
    ("velocity computed before x requires grad", 2, [
        ("        x = x_t.detach().clone().requires_grad_(True)\n        v = v_fn(x)\n",
         "        x = x_t.detach().clone()\n        v = v_fn(x)\n        x.requires_grad_(True)\n"),
    ]),
    ("grad not enabled under no_grad", 2, [
        ("    with torch.enable_grad():\n", "    if True:\n"),
    ]),
    ("horizon not capped by the previous chunk", 2, [
        ("horizon = min(execution_horizon, prev_steps)", "horizon = execution_horizon"),
    ]),
    ("nan at time 0 not zeroed", 2, [
        ("gain = torch.nan_to_num(c * inv_r2, nan=0.0, posinf=max_w)", "gain = c * inv_r2"),
    ]),
    ("guidance sign flipped", 2, [
        ("return (v - gain.to(x_t.dtype) * correction).detach()", "return (v + gain.to(x_t.dtype) * correction).detach()"),
    ]),
    ("prefix weights ignored", 2, [
        ("err = ((target - x1) * weights).detach()", "err = (target - x1).detach()"),
    ]),
    ("previous chunk padded at the front", 2, [
        ("(0, 0, 0, steps - prev_steps)", "(0, 0, steps - prev_steps, 0)"),
    ]),
    ("x1 uses the noise-at-zero convention", 2, [
        ("x1 = x - time * v", "x1 = x + (1 - time) * v"),
    ]),
    ("result left attached", 2, [
        ("return (v - gain.to(x_t.dtype) * correction).detach()", "return v - gain.to(x_t.dtype) * correction"),
    ]),
    ("velocity evaluated twice", 2, [
        ("        v = v_fn(x)\n", "        v_fn(x_t)\n        v = v_fn(x)\n"),
    ]),
    ("narrow action dim accepted", 2, [
        ("prev_chunk.shape[-1] != x_t.shape[-1] or ", ""),
    ]),
]


def _named(original: str) -> list[Mutation]:
    return [Mutation(name, mutant(original, edits)) for name, _, edits in MUTATIONS]


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected_by_unshown_cases(_repeat):
    original = get_task(TASK_ID)["solution"]
    rejected = assert_mutations_rejected(TASK_ID, _named(original), require_unshown=True)
    assert set(rejected) == {name for name, _, _ in MUTATIONS}


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert len(MUTATIONS) >= 6
