"""Mutation gate for the latent_exit_criteria (Huginn adaptive exit) exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "latent_exit_criteria"

KL_LINE = 'values = F.kl_div(log_p, prev, reduction="none", log_target=True).sum(dim=-1)'
LATENT_LINE = "change = (s[1:] - s[:-1]).norm(dim=-1) / s[1:].norm(dim=-1)"
STEPS_LINE = "steps = torch.arange(1, R + 1, device=fires.device).view(R, 1).expand(R, N)"


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
        "KL direction reversed (current to previous)",
        (KL_LINE, 'values = F.kl_div(prev, log_p, reduction="none", log_target=True).sum(dim=-1)'),
    ),
    _mutation(
        "KL with no uniform start (step 1 cannot fire)",
        ("prev = torch.cat([uniform, log_p[:-1]])", "prev = torch.cat([log_p[:1] + float('inf'), log_p[:-1]])"),
    ),
    _mutation(
        "KL uniform start uses probabilities instead of log-probabilities",
        ("uniform = torch.full_like(log_p[:1], math.log(1.0 / V))", "uniform = torch.full_like(log_p[:1], 1.0 / V)"),
    ),
    _mutation(
        "absolute instead of relative latent change",
        (LATENT_LINE, "change = (s[1:] - s[:-1]).norm(dim=-1)"),
    ),
    _mutation(
        "latent change divided by the previous state",
        (LATENT_LINE, "change = (s[1:] - s[:-1]).norm(dim=-1) / s[:-1].norm(dim=-1)"),
    ),
    _mutation(
        "latent-diff compares step 1 with a zero state",
        ('values = torch.cat([torch.full_like(s[:1, :, 0], float("inf")), change])',
         "values = torch.cat([torch.ones_like(s[:1, :, 0]), change])"),
    ),
    _mutation(
        "entropy sentinel 0.0 instead of 100.0",
        ("torch.full_like(entropy[:1], 100.0)", "torch.full_like(entropy[:1], 0.0)"),
    ),
    _mutation(
        "entropy-diff cannot fire at step 1",
        ("torch.full_like(entropy[:1], 100.0)", "torch.full_like(entropy[:1], float('inf'))"),
    ),
    _mutation(
        "returns the step after the condition holds",
        (STEPS_LINE, "steps = torch.arange(2, R + 2, device=fires.device).view(R, 1).expand(R, N)"),
    ),
    _mutation(
        "0-based step index",
        (STEPS_LINE, "steps = torch.arange(0, R, device=fires.device).view(R, 1).expand(R, N)"),
    ),
    _mutation(
        "never-exit tokens return -1",
        ("never = torch.full_like(steps, R)", "never = torch.full_like(steps, R + 1)"),
        ("return torch.where(fires, steps, never).min(dim=0).values",
         "out = torch.where(fires, steps, never).min(dim=0).values\n    return torch.where(out > R, -1, out)"),
    ),
    _mutation(
        "difference criteria fire on ties (<=)",
        ("fires = values < float(threshold)", "fires = values <= float(threshold)"),
    ),
    _mutation(
        "argmax counter must exceed the threshold (>)",
        ("fires = torch.stack(counts) >= int(threshold)", "fires = torch.stack(counts) > int(threshold)"),
    ),
    _mutation(
        "argmax window counts step 1 as stable",
        ("count = torch.zeros(N, dtype=torch.long, device=argmax.device)",
         "count = torch.ones(N, dtype=torch.long, device=argmax.device)"),
    ),
    _mutation(
        "argmax takes the last index among tied logits",
        ("argmax = logits_per_step.argmax(dim=-1)  # (R, N)",
         "argmax = logits_per_step.shape[-1] - 1 - logits_per_step.flip(-1).argmax(dim=-1)"),
    ),
    _mutation(
        "kl 'auto' uses the paper's 5e-4",
        ('AUTO_THRESHOLDS = {"kl": 1e-3,', 'AUTO_THRESHOLDS = {"kl": 5e-4,'),
    ),
    _mutation(
        "latent-diff 'auto' is 1e-3",
        ('"latent-diff": 0.03,', '"latent-diff": 1e-3,'),
    ),
    _mutation(
        "argmax-stability 'auto' is 4",
        ('"argmax-stability": 5}', '"argmax-stability": 4}'),
    ),
    _mutation(
        "accepts mismatched inputs",
        ("if R < 1 or latents_per_step.shape[:2] != (R, N):", "if R < 1:"),
    ),
]


def test_latent_exit_criteria_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_latent_exit_criteria_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
