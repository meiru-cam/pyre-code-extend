"""Mutation gates and metadata contracts for the agent runtime path."""

from __future__ import annotations

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


def _mutation(task_id: str, name: str, old: str, new: str) -> Mutation:
    solution = get_task(task_id)["solution"]
    assert old in solution, f"mutation target drifted: {task_id}/{name}"
    return Mutation(name, solution.replace(old, new))


def test_tool_registry_reference_contract():
    task = get_task("tool_registry")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_budgeted_agent_loop_reference_contract():
    task = get_task("budgeted_agent_loop")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_supervisor_orchestration_reference_contract():
    task = get_task("supervisor_orchestration")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_agent_runtime_metadata_contracts_and_source_coverage():
    for task_id in ("tool_registry", "budgeted_agent_loop", "supervisor_orchestration"):
        task = get_task(task_id)
        validate_task(task_id, task)
        assert task["version"] == 1
        assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
        assert any(test.get("visibility") == "unshown" for test in task["tests"])
        assert task["model_connections"]
        assert task["pro_con_analysis"]["pros"] and task["pro_con_analysis"]["cons"]
        assert len(task["design_note_rubric"]) == 8
        assert all(len(source["commit"]) == 40 for source in task["sources"])
    combined = " ".join(
        source["url"]
        for task_id in ("tool_registry", "budgeted_agent_loop", "supervisor_orchestration")
        for source in get_task(task_id)["sources"]
    )
    for application in ("openclaw/openclaw", "NousResearch/hermes-agent", "microsoft/agent-framework"):
        assert application in combined
    loop = get_task("budgeted_agent_loop")
    assert "RetryableModelError" in loop["hints"][1]["content"]
    assert "model.failed" in loop["description_en"] and "model.retry" in loop["description_en"]


def _registry_mutations():
    return [
        _mutation("tool_registry", "duplicate_overwrite", '''        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
''', ""),
        _mutation("tool_registry", "bool_is_integer", "type(arguments[argument]) is not declared", "not isinstance(arguments[argument], declared)"),
        _mutation("tool_registry", "drops_idempotency", "idempotency_key=idempotency_key", "idempotency_key=None"),
        _mutation("tool_registry", "leaks_handler", '''                "parameters": {
''', '''                "handler": tool,
                "parameters": {
'''),
        _mutation("tool_registry", "accepts_unsupported_schema", '''        if any(not isinstance(name, str) or declared not in self._TYPE_NAMES
               for name, declared in tool.argument_schema.items()):
            raise TypeError("argument schema contains an unsupported entry")
''', ""),
    ]


def _loop_mutations():
    return [
        _mutation("budgeted_agent_loop", "retry_all_errors", "except PermanentToolError as error:", "except Exception as error:"),
        _mutation("budgeted_agent_loop", "never_retries", "if attempt == max_attempts:", "if True:"),
        _mutation("budgeted_agent_loop", "does_not_advance_clock", "clock.sleep(delay)", "pass"),
        _mutation("budgeted_agent_loop", "iteration_off_by_one", 'if iterations >= limits["max_iterations"]:', 'if iterations > limits["max_iterations"]:'),
        _mutation("budgeted_agent_loop", "changes_retry_key", "registry.invoke(name, deepcopy(arguments), idempotency_key)", 'registry.invoke(name, deepcopy(arguments), f"{idempotency_key}:{attempt}")'),
        _mutation("budgeted_agent_loop", "ignores_cancellation", "if cancelled():", "if False:"),
        _mutation("budgeted_agent_loop", "does_not_count_attempts", "            tool_calls += 1\n", ""),
        _mutation("budgeted_agent_loop", "misclassifies_unexpected_tool_error",
                  '"type": "tool_execution_failure",',
                  '"type": "permanent_tool_failure",'),
        _mutation("budgeted_agent_loop", "never_retries_model_api",
                  "if model_attempt == max_attempts:", "if True:"),
        _mutation("budgeted_agent_loop", "swallows_model_harness_failure", '''            except HarnessFailure:
                raise
            except PermanentModelError as error:
''', '''            except PermanentModelError as error:
'''),
        _mutation("budgeted_agent_loop", "swallows_tool_harness_failure", '''            except HarnessFailure:
                raise
            except PermanentToolError as error:
''', '''            except PermanentToolError as error:
'''),
    ]


def _supervisor_mutations():
    return [
        _mutation("supervisor_orchestration", "unbounded_fanout", "min(max_concurrency, len(pending))", "len(pending)"),
        _mutation("supervisor_orchestration", "ignores_queue_pressure", "admitted = remaining[:max_queue]", "admitted = remaining"),
        _mutation("supervisor_orchestration", "reruns_checkpoints",
                  'if checkpoint_store.contains(run_id, task["task_id"]):',
                  "if False:"),
        _mutation("supervisor_orchestration", "unfair_retry_front", "pending.append(task)", "pending.appendleft(task)"),
        _mutation("supervisor_orchestration", "loses_partial_status", '''    if failures and results:
        status = "partial"
''', '''    if failures and results:
        status = "failed"
'''),
        _mutation("supervisor_orchestration", "wrong_correlation", '''                f"{run_id}:{task_id}:{attempt_index}", correlation_id, run_id,
''', '''                f"{run_id}:{task_id}:{attempt_index}", run_id, run_id,
'''),
        _mutation("supervisor_orchestration", "failure_as_success", '''                failures[task_id] = {"type": "dead_letter", "reason": "permanent",
                                     "worker": worker.name, "message": str(error)}
''', '''                results[task_id] = {"error": str(error)}
'''),
        _mutation("supervisor_orchestration", "confuses_missing_and_null_checkpoint",
                  'if checkpoint_store.contains(run_id, task["task_id"]):',
                  'if checkpoint_store.get(run_id, task["task_id"]) is not None:'),
        _mutation("supervisor_orchestration", "misclassifies_unexpected_worker_error",
                  '"reason": "unexpected",',
                  '"reason": "permanent",'),
        _mutation("supervisor_orchestration", "publishes_without_checkpoint",
                  "checkpoint_store.put(run_id, task_id, value)", "pass"),
        _mutation("supervisor_orchestration", "swallows_worker_harness_failure", '''            except HarnessFailure:
                raise
            except PermanentToolError as error:
''', '''            except PermanentToolError as error:
'''),
    ]


@pytest.mark.parametrize("_repeat", range(3))
def test_tool_registry_mutations(_repeat):
    mutations = _registry_mutations()
    assert set(assert_mutations_rejected("tool_registry", mutations)) == {item.name for item in mutations}


@pytest.mark.parametrize("_repeat", range(3))
def test_budgeted_loop_mutations(_repeat):
    mutations = _loop_mutations()
    assert set(assert_mutations_rejected("budgeted_agent_loop", mutations)) == {item.name for item in mutations}


@pytest.mark.parametrize("_repeat", range(3))
def test_supervisor_mutations(_repeat):
    mutations = _supervisor_mutations()
    assert set(assert_mutations_rejected("supervisor_orchestration", mutations)) == {item.name for item in mutations}
