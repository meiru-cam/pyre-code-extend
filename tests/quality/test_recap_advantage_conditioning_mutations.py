"""Mutation gate for the RECAP advantage-conditioning exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "recap_advantage_conditioning"

QUANTILE = "        return float(torch.quantile(advantages.double(), 1.0 - positive_fraction))\n"
COMPARE = "            positive[rows] = advantages[rows].double() > eps\n"
CODES = "        codes = (positive | is_correction.bool()).long()  # human corrections are always positive\n"
DROP = "        return torch.where(drop_u < drop_prob, torch.full_like(codes, -1), codes)\n"
TERMINAL = "        rewards[length] = 0.0 if success else -float(c_fail)  # terminal reward, Eq. 5\n"
BOOTSTRAP = "        return tail[idx] - tail[end] + padded[end] - values\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    # Section V-D read literally: the threshold is the 30th percentile, so about 70% is positive.
    _mutation(
        "threshold at the positive_fraction quantile (main-text reading)",
        (QUANTILE, "        return float(torch.quantile(advantages.double(), positive_fraction))\n"),
    ),
    _mutation(
        "threshold at the median regardless of the fraction (CFGRL-style fixed cut)",
        (QUANTILE, "        return float(torch.quantile(advantages.double(), 0.5))\n"),
    ),
    _mutation(
        "ties at the threshold count as positive",
        (COMPARE, "            positive[rows] = advantages[rows].double() >= eps\n"),
    ),
    _mutation(
        "one global threshold across tasks",
        ("            eps = self.threshold(advantages[rows], positive_fraction)  # one threshold per task\n",
         "            eps = self.threshold(advantages, positive_fraction)\n"),
    ),
    _mutation(
        "corrections are not forced positive",
        (CODES, "        codes = positive.long()\n"),
    ),
    _mutation(
        "corrections are excluded from the threshold population",
        ("            eps = self.threshold(advantages[rows], positive_fraction)  # one threshold per task\n",
         "            keep = rows & ~is_correction.bool()\n"
         "            eps = self.threshold(advantages[keep if keep.any() else rows], positive_fraction)\n"),
    ),
    _mutation(
        "dropped samples become negative instead of unconditioned",
        (DROP, "        return torch.where(drop_u < drop_prob, torch.zeros_like(codes), codes)\n"),
    ),
    _mutation(
        "dropout keeps corrections",
        (DROP, "        drop = (drop_u < drop_prob) & ~is_correction.bool()\n"
               "        return torch.where(drop, torch.full_like(codes, -1), codes)\n"),
    ),
    _mutation(
        "drops when u is above drop_prob",
        (DROP, "        return torch.where(drop_u > 1 - drop_prob, torch.full_like(codes, -1), codes)\n"),
    ),
    _mutation(
        "failure adds -c_fail on top of the step penalty",
        (TERMINAL, "        rewards[length] = 0.0 if success else -1.0 - float(c_fail)\n"),
    ),
    _mutation(
        "success terminal reward is -1",
        (TERMINAL, "        rewards[length] = -1.0 if success else -float(c_fail)\n"),
    ),
    _mutation(
        "bootstraps from the last value when t + N runs past the end",
        ("        padded = torch.cat([values, values.new_zeros(1)])  # no value after the last observation\n",
         "        padded = torch.cat([values, values[-1:]])\n"),
    ),
    _mutation(
        "bootstraps from values[t + N - 1] (off by one)",
        (BOOTSTRAP, "        return tail[idx] - tail[end] + padded[(end - 1).clamp(min=0)] - values\n"),
    ),
    _mutation(
        "sums N + 1 rewards before bootstrapping",
        (BOOTSTRAP, "        end2 = (idx + lookahead + 1).clamp(max=steps)\n"
                    "        return tail[idx] - tail[end2] + padded[end] - values\n"),
    ),
    _mutation(
        "drops the terminal observation's value when t + N lands on it",
        ("        padded = torch.cat([values, values.new_zeros(1)])  # no value after the last observation\n",
         "        padded = torch.cat([values[:-1], values.new_zeros(2)])\n"),
    ),
    _mutation(
        "floors returns into bins",
        ("        index = torch.round((returns.double() + 1.0) / width)\n",
         "        index = torch.floor((returns.double() + 1.0) / width)\n"),
    ),
    _mutation(
        "expected value uses bin indices",
        ("        return probs @ self.bin_centers(logits.dtype, logits.device)\n",
         "        return probs @ torch.arange(self.num_bins, dtype=logits.dtype)\n"),
    ),
    _mutation(
        "guidance mixes from the conditional output",
        ("        return v_uncond + beta * (v_cond - v_uncond)\n",
         "        return v_cond + beta * (v_cond - v_uncond)\n"),
    ),
    _mutation(
        "swaps the indicator texts",
        ('_TEXT = {1: "Advantage: positive", 0: "Advantage: negative", -1: ""}\n',
         '_TEXT = {1: "Advantage: negative", 0: "Advantage: positive", -1: ""}\n'),
    ),
    _mutation(
        "accepts positive_fraction of 0 or 1",
        ("        if not 0.0 < positive_fraction < 1.0:\n", "        if not 0.0 <= positive_fraction <= 1.0:\n"),
    ),
]


def test_recap_advantage_conditioning_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])
    assert {s["kind"] for s in task["sources"]} == {"paper"}, "the oracle for this exercise is the paper alone"


@pytest.mark.parametrize("_repeat", range(3))
def test_recap_advantage_conditioning_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
