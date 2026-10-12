"""Mutation gate for the pause-token training mask exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "pause_token_training_mask"

CAT_LINE = "    input_ids = torch.cat([prefix_ids.long(), pauses, target_ids.long().to(device)])\n"
PREFIX_MASK = "    labels[:answer_start] = -100\n"
PAUSE_MASK = "    labels[input_ids == int(pause_id)] = -100\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    # The paper's S_ignore copied into aligned labels without the shift: masks the
    # position predicting a pause, so the last pause keeps its label and the answer
    # token before an in-target pause loses its own.
    _mutation(
        "masks the position before each pause instead of the pause (off-by-one)",
        (PREFIX_MASK, "    labels[:prefix_ids.numel()] = -100\n"),
        (PAUSE_MASK, "    labels[:-1][input_ids[1:] == int(pause_id)] = -100\n"),
    ),
    _mutation(
        "computes loss on pause targets",
        (PREFIX_MASK, "    labels[:prefix_ids.numel()] = -100\n"),
        (PAUSE_MASK, ""),
    ),
    _mutation(
        "prepends the pauses instead of appending them",
        (CAT_LINE, "    input_ids = torch.cat([pauses, prefix_ids.long(), target_ids.long().to(device)])\n"),
    ),
    _mutation(
        "masks the first answer token too",
        (PREFIX_MASK, "    labels[:answer_start + 1] = -100\n"),
    ),
    _mutation(
        "trains on a pause inside the target",
        (PAUSE_MASK, ""),
    ),
    _mutation(
        "pre-shifts the labels",
        ("    return input_ids, labels, answer_start\n",
         "    labels = torch.cat([labels[1:], labels.new_full((1,), -100)])\n"
         "    return input_ids, labels, answer_start\n"),
    ),
    _mutation(
        "finds answer_start by searching for the last pause",
        ("    answer_start = prefix_ids.numel() + n_pause\n",
         "    hits = (input_ids == int(pause_id)).nonzero()\n"
         "    answer_start = int(hits[-1]) + 1 if len(hits) else prefix_ids.numel()\n"),
    ),
    _mutation(
        "answer_start points at the first pause",
        ("    answer_start = prefix_ids.numel() + n_pause\n",
         "    answer_start = prefix_ids.numel()\n"),
        (PREFIX_MASK, "    labels[:answer_start + n_pause] = -100\n"),
    ),
    _mutation(
        "labels alias input_ids",
        ("    labels = input_ids.clone()\n", "    labels = input_ids\n"),
    ),
    _mutation(
        "keeps the input dtype",
        (CAT_LINE,
         "    input_ids = torch.cat([prefix_ids, pauses.to(prefix_ids.dtype), target_ids.to(prefix_ids.dtype)])\n"),
    ),
    _mutation(
        "accepts a negative or bool n_pause",
        ("    if isinstance(n_pause, bool) or not isinstance(n_pause, int) or n_pause < 0:\n",
         "    if not isinstance(n_pause, int):\n"),
        ("    pauses = torch.full((n_pause,)", "    pauses = torch.full((max(int(n_pause), 0),)"),
    ),
    _mutation(
        "accepts an empty prefix",
        ('        raise ValueError("prefix_ids must not be empty")\n', "        pass\n"),
    ),
]


def test_pause_token_training_mask_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])
    assert [s["kind"] for s in task["sources"]] == ["paper"], "the oracle for this exercise is the paper alone"


@pytest.mark.parametrize("_repeat", range(3))
def test_pause_token_training_mask_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
