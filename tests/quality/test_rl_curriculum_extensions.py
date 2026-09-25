"""Behavioral gate for the added RL algorithm and framework exercises."""

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


NEW_TASKS = (
    "reinforce_discounted_returns",
    "reinforce_policy_loss",
    "dapo_clip_higher_loss",
    "verl_dataproto_filter_chunk",
    "rollout_weight_sync_staleness",
    "slime_custom_generate_hook",
)


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_new_rl_exercise_reference_passes_its_contract(task_id):
    task = get_task(task_id)
    assert task is not None, f"{task_id} is not registered"
    result = _execute_tests(task["solution"], task, capture_output=False)
    assert result.allPassed, [(r.name, r.error) for r in result.results if not r.passed]


def _changed(task_id, name, old, new):
    original = get_task(task_id)["solution"]
    assert old in original, f"mutation target drifted: {task_id}/{name}"
    return Mutation(name, original.replace(old, new, 1))


@pytest.mark.parametrize("_repeat", range(3))
@pytest.mark.parametrize("task_id,mutations", [
    ("reinforce_discounted_returns", [
        ("crosses episode boundary", "* (~dones[index])", "* 1"),
        ("ignores discount", "gamma * carry", "carry"),
    ]),
    ("reinforce_policy_loss", [
        ("trains the baseline", "baseline.detach()", "baseline"),
        ("dilutes loss with masked tokens", "/ mask.sum()", "/ mask.numel()"),
    ]),
    ("dapo_clip_higher_loss", [
        ("uses low cap for positive tokens", "1 + clip_high", "1 + clip_low"),
        ("trains masked tokens", "valid = mask", "valid = torch.ones_like(mask)"),
    ]),
    ("verl_dataproto_filter_chunk", [
        ("chunks before filtering", "data.select_idxs(indices).chunk(chunks)", "data.chunk(chunks)"),
    ]),
    ("rollout_weight_sync_staleness", [
        ("rejects the exact lag bound", "trainer_version - version > max_staleness", "trainer_version - version >= max_staleness"),
        ("updates weights during generation", "and in_flight > 0", "and False"),
    ]),
    ("slime_custom_generate_hook", [
        ("treats length as complete", '"length": sample.Status.TRUNCATED', '"length": sample.Status.COMPLETED'),
        ("counts prompt tokens as response", 'sample.response_length = len(ids)', 'sample.response_length = len(sample.tokens)'),
    ]),
])
def test_new_rl_exercises_reject_realistic_mistakes(task_id, mutations, _repeat):
    rejected = assert_mutations_rejected(
        task_id,
        [_changed(task_id, name, old, new) for name, old, new in mutations],
        require_unshown=True,
    )
    assert len(rejected) == len(mutations)
