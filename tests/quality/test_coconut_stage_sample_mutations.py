"""Mutation gate for the Coconut curriculum stage-sample exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "coconut_stage_sample"


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
        "one latent per step instead of c_thought",
        ("    n_latent = n_groups * c_thought\n", "    n_latent = n_groups\n"),
    ),
    _mutation(
        "above max skips only max_latent_stage steps",
        ("        n_skip = len(step_ids)\n", "        n_skip = max_latent_stage\n"),
    ),
    _mutation(
        "label mask forgets the two markers",
        ("    labels = [-100] * len(prefix) + tail\n",
         "    cut = len(question_ids) + n_latent\n    labels = [-100] * cut + input_ids[cut:]\n"),
    ),
    _mutation(
        "ignores pad_latent_to_max",
        ("        if pad_latent_to_max:\n", "        if True:\n"),
    ),
    _mutation(
        "pad_latent_to_max inverted",
        ("        if pad_latent_to_max:\n", "        if not pad_latent_to_max:\n"),
    ),
    _mutation(
        "no cap on groups above max",
        ("            n_groups = max_latent_stage\n", "            n_groups = stage\n"),
    ),
    _mutation(
        "caps groups by the step count at every stage",
        ("        n_skip, n_groups = stage, stage\n",
         "        n_skip, n_groups = stage, min(stage, len(step_ids))\n"),
    ),
    _mutation(
        "skips the last steps instead of the first",
        ("step_ids[n_skip:]", "step_ids[:len(step_ids) - n_skip]"),
    ),
    _mutation(
        "labels only the answer",
        ("    labels = [-100] * len(prefix) + tail\n",
         "    labels = [-100] * (len(input_ids) - len(answer_ids)) + list(answer_ids)\n"),
    ),
    _mutation(
        "pre-shifted labels",
        ("    labels = [-100] * len(prefix) + tail\n",
         "    labels = ([-100] * len(prefix) + tail)[1:] + [-100]\n"),
    ),
    _mutation(
        "markers kept under no_special_marker",
        ('    bot = [] if no_special_marker else [special_ids["bot"]]\n', '    bot = [special_ids["bot"]]\n'),
        ('    eot = [] if no_special_marker else [special_ids["eot"]]\n', '    eot = [special_ids["eot"]]\n'),
    ),
    _mutation(
        "stage 0 drops the marker pair",
        ("    bot = [] if no_special_marker else", "    bot = [] if no_special_marker or stage == 0 else"),
        ("    eot = [] if no_special_marker else", "    eot = [] if no_special_marker or stage == 0 else"),
    ),
    _mutation(
        "position ids start at 1",
        ("torch.arange(len(input_ids), dtype=torch.long)", "torch.arange(1, len(input_ids) + 1, dtype=torch.long)"),
    ),
    _mutation(
        "returns int32 tensors",
        ('"input_ids": torch.tensor(input_ids, dtype=torch.long)', '"input_ids": torch.tensor(input_ids, dtype=torch.int32)'),
    ),
    _mutation(
        "consumes the step list in place",
        ("    tail = [token for step in step_ids[n_skip:]",
         "    del step_ids[:n_skip]\n    tail = [token for step in step_ids"),
    ),
    _mutation(
        "accepts bools and floats",
        ("isinstance(value, bool) or not isinstance(value, int) or value < low", "value < low"),
    ),
]


def test_coconut_stage_sample_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_coconut_stage_sample_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
