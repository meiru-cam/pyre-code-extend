"""Mutation gates for the interview-oriented agentic RL exercises."""

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


def _mutation(task_id, name, old, new):
    solution = get_task(task_id)["solution"]
    assert old in solution, f"mutation target drifted: {task_id}/{name}"
    return Mutation(name, solution.replace(old, new, 1))


def _contract(task_id):
    task = get_task(task_id)
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [(r.name, r.error) for r in response.results if not r.passed]


def test_agentic_rl_interview_references_pass():
    for task_id in ("rl_eval_loop", "async_agent_rollout", "agent_env_adapter", "rollout_train_boundary"):
        _contract(task_id)


@pytest.mark.parametrize("_repeat", range(3))
def test_rl_eval_loop_rejects_common_mistakes(_repeat):
    rejected = assert_mutations_rejected("rl_eval_loop", [
        _mutation("rl_eval_loop", "ignores truncation",
                   "if terminated or truncated:",
                   "if terminated:"),
        _mutation("rl_eval_loop", "counts reward as success",
                   'success = bool(info.get("success", False))',
                   "success = bool(reward > 0)"),
        _mutation("rl_eval_loop", "reuses one seed",
                   "seed=seed + episode",
                   "seed=seed"),
    ], require_unshown=True)
    assert len(rejected) == 3


@pytest.mark.parametrize("_repeat", range(3))
def test_async_agent_rollout_rejects_common_mistakes(_repeat):
    rejected = assert_mutations_rejected("async_agent_rollout", [
        _mutation("async_agent_rollout", "drops the concurrency semaphore",
                   "async with semaphore:",
                   "if True:"),
        _mutation("async_agent_rollout", "does not enforce timeout",
                   "value = await asyncio.wait_for(rollout_fn(item), timeout=timeout)",
                   "value = await rollout_fn(item)"),
        _mutation("async_agent_rollout", "returns completion order",
                   "return [entry for _, entry in sorted(results)]",
                   "return [entry for _, entry in sorted(results, reverse=True)]"),
    ], require_unshown=True)
    assert len(rejected) == 3


@pytest.mark.parametrize("_repeat", range(3))
def test_agent_env_adapter_rejects_common_mistakes(_repeat):
    rejected = assert_mutations_rejected("agent_env_adapter", [
        _mutation("agent_env_adapter", "marks every token trainable",
                   'response_mask.extend([True] * len(token_ids))',
                   "response_mask.extend([False] * len(token_ids))"),
        _mutation("agent_env_adapter", "steps after terminal",
                   "if terminated or truncated:\n            break",
                   "if False:\n            break"),
        _mutation("agent_env_adapter", "omits reward accumulation",
                   "rewards.append(float(reward))",
                   "rewards.append(0.0)"),
    ], require_unshown=True)
    assert len(rejected) == 3


@pytest.mark.parametrize("_repeat", range(3))
def test_rollout_train_boundary_rejects_common_mistakes(_repeat):
    rejected = assert_mutations_rejected("rollout_train_boundary", [
        _mutation("rollout_train_boundary", "trains on environment tokens",
                   '"advantages"].append(total_reward if trainable else 0.0)',
                   '"advantages"].append(total_reward)'),
        _mutation("rollout_train_boundary", "drops environment tokens from input",
                   'output["input_ids"].append(token_id)',
                   'if trainable: output["input_ids"].append(token_id)'),
        _mutation("rollout_train_boundary", "normalizes across episodes",
                   "total_reward = sum(float(reward) for reward in rewards)",
                   "total_reward = sum(float(reward) for reward in rewards) / len(rewards)"),
    ], require_unshown=True)
    assert len(rejected) == 3


@pytest.mark.parametrize("_repeat", range(3))
def test_fully_async_rollout_buffer_rejects_common_mistakes(_repeat):
    task_id = "fully_async_rollout_buffer"
    rejected = assert_mutations_rejected(task_id, [
        _mutation(task_id, "waits for every rollout before training",
                   "            await emit_ready()",
                   "            pass"),
        _mutation(task_id, "sends failed samples to training",
                   '        if result["status"] == "ok":',
                   "        if True:"),
        _mutation(task_id, "drops the final partial batch",
                   "    await emit_ready(force=True)",
                   "    await emit_ready()"),
    ], require_unshown=True)
    assert len(rejected) == 3
