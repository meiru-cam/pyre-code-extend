"""Mutation gate for trajectory finalization at the rollout length boundary."""

from __future__ import annotations

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


TASK_ID = "agent_rollout_truncation"


def _mutation(name: str, old: str, new: str) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    assert old in solution, f"mutation target drifted: {name}"
    return Mutation(name, solution.replace(old, new, 1))


def test_agent_rollout_truncation_reference_contract():
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_agent_rollout_truncation_rejects_documented_mistakes():
    mutations = [
        _mutation(
            "treats an exact-fit trajectory as truncated",
            "    truncated = original_length > max_response_length",
            "    truncated = original_length >= max_response_length",
        ),
        _mutation(
            "keeps the newest tokens instead of the trajectory prefix",
            "    kept_ids = response_ids[:kept].clone()",
            "    kept_ids = response_ids[-kept:].clone()",
        ),
        _mutation(
            "takes the mask from a different slice",
            "    kept_mask = response_mask[:kept].clone()",
            "    kept_mask = response_mask[-kept:].clone()",
        ),
        _mutation(
            "takes rollout log-probabilities from a different slice",
            "    kept_log_probs = rollout_log_probs[:kept].clone()",
            "    kept_log_probs = rollout_log_probs[-kept:].clone()",
        ),
        _mutation(
            "adds the overlong penalty instead of subtracting it",
            "    effective_reward = float(reward) - (penalty if truncated else 0.0)",
            "    effective_reward = float(reward) + (penalty if truncated else 0.0)",
        ),
        _mutation(
            "penalizes every trajectory",
            "    effective_reward = float(reward) - (penalty if truncated else 0.0)",
            "    effective_reward = float(reward) - penalty",
        ),
        _mutation(
            "puts terminal reward on the last policy token",
            "    token_rewards[-1] = effective_reward",
            "    token_rewards[kept_mask.nonzero()[-1]] = effective_reward",
        ),
        _mutation(
            "returns the caller's tensor views",
            "    kept_ids = response_ids[:kept].clone()",
            "    kept_ids = response_ids[:kept]",
        ),
        _mutation(
            "returns the caller's mask view",
            "    kept_mask = response_mask[:kept].clone()",
            "    kept_mask = response_mask[:kept]",
        ),
        _mutation(
            "returns the caller's log-probability view",
            "    kept_log_probs = rollout_log_probs[:kept].clone()",
            "    kept_log_probs = rollout_log_probs[:kept]",
        ),
        _mutation(
            "accepts complex token ids as integers",
            " or torch.is_complex(response_ids)",
            "",
        ),
        _mutation(
            "accepts a NaN penalty",
            "    if math.isnan(penalty) or penalty < 0:",
            "    if penalty < 0:",
        ),
    ]
    rejected = assert_mutations_rejected(TASK_ID, mutations)
    assert set(rejected) == {mutation.name for mutation in mutations}
