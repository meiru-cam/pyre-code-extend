"""Contracts for the deterministic agent and scenario-data harness."""

from __future__ import annotations

import pytest

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents import (
    AgentResult,
    InMemoryBus,
    InMemoryCheckpointStore,
    FakeTool,
    MessageEnvelope,
    PermanentToolError,
    OrchestrationResult,
    RoundRobinScheduler,
    RetryableToolError,
    Scenario,
    ScriptedModel,
    ScriptedWorker,
    TraceRecorder,
    VirtualClock,
    assert_correlation_chain,
    assert_budget_usage,
    assert_event_order,
    assert_final_state,
    assert_max_concurrency,
    assert_no_duplicate_effects,
    assert_retry_delays,
)


def test_agent_result_isolates_messages_and_failure_data():
    messages = [{"role": "user", "content": "go"}]
    failure = {"type": "budget", "limit": "tokens"}
    result = AgentResult("failed", None, messages, 1, 0, 4, 0.1, 2.0, failure)
    messages[0]["content"] = "changed"
    failure["type"] = "changed"
    assert result.messages == [{"role": "user", "content": "go"}]
    assert result.failure == {"type": "budget", "limit": "tokens"}


def test_orchestration_doubles_are_deterministic_and_isolated():
    envelope = MessageEnvelope("m", "c", "r", "s", "w", None, 0, "k", {"x": 1})
    worker_a = ScriptedWorker("a", {"search"}, [{"answer": "a"}])
    worker_b = ScriptedWorker("b", {"search"}, [{"answer": "b"}])
    scheduler = RoundRobinScheduler()
    assert scheduler.choose("search", [worker_a, worker_b]).name == "a"
    assert scheduler.choose("search", [worker_a, worker_b]).name == "b"
    assert worker_a.run(envelope) == {"answer": "a"}
    assert worker_a.run(envelope) == {"answer": "a"}
    assert worker_a.calls == 2 and worker_a.effects == 1
    bus = InMemoryBus(); bus.send(envelope)
    envelope.payload["x"] = 9
    assert bus.messages[0].payload == {"x": 1}
    store = InMemoryCheckpointStore(); store.put("r", "task", {"answer": "saved"})
    loaded = store.get("r", "task"); loaded["answer"] = "changed"
    assert store.get("r", "task") == {"answer": "saved"}
    assert store.contains("r", "task") is True
    assert store.contains("r", "missing") is False
    store.put("r", "null-task", None)
    assert store.contains("r", "null-task") is True
    assert store.get("r", "null-task") is None


def test_orchestration_result_isolates_nested_state():
    results = {"a": {"value": 1}}
    failures = {"b": {"type": "dead_letter"}}
    value = OrchestrationResult("partial", results, failures, {"a": 1, "b": 2}, 2, ["a"])
    results["a"]["value"] = 9; failures["b"]["type"] = "changed"
    assert value.results == {"a": {"value": 1}}
    assert value.failures == {"b": {"type": "dead_letter"}}


def test_message_envelope_validates_and_copies_json_payload():
    payload = {"query": "weather", "tags": ["safe"]}
    envelope = MessageEnvelope("m1", "c1", "run", "supervisor", "worker", 4.0, 0, "key", payload)
    payload["query"] = "changed"
    assert envelope.payload == {"query": "weather", "tags": ["safe"]}
    with pytest.raises(HarnessFailure, match="message_id"):
        MessageEnvelope("", "c", "r", "s", "w", None, 0, None, {})
    with pytest.raises(HarnessFailure, match="JSON"):
        MessageEnvelope("m", "c", "r", "s", "w", None, 0, None, {"bad": object()})


def test_scripted_model_records_isolated_calls_and_exhaustion_is_author_error():
    model = ScriptedModel([{"type": "final", "content": "done"}])
    messages = [{"role": "user", "content": "go"}]
    assert model.call(messages)["content"] == "done"
    messages[0]["content"] = "mutated"
    assert model.calls[0][0]["content"] == "go"
    with pytest.raises(HarnessFailure, match="exhausted"):
        model.call([])


def test_scripted_model_raises_scripted_failure():
    model = ScriptedModel([RetryableToolError("busy", retry_after=2)])
    with pytest.raises(RetryableToolError, match="busy"):
        model.call([])


def test_fake_tool_replays_idempotent_result_without_duplicate_effect():
    tool = FakeTool("charge", "charge once", {"amount": int}, outcomes=[{"receipt": "r1"}])
    first = tool.invoke({"amount": 3}, idempotency_key="pay-1")
    second = tool.invoke({"amount": 3}, idempotency_key="pay-1")
    assert first == second == {"receipt": "r1"}
    assert tool.calls == 2 and tool.effects == 1
    assert tool.invocation_keys == ["pay-1", "pay-1"]
    assert_no_duplicate_effects(tool)


def test_fake_tool_failure_schedule_distinguishes_retryable_and_permanent():
    tool = FakeTool(
        "api", "scripted", {"q": str},
        outcomes=[RetryableToolError("rate", retry_after=1), PermanentToolError("denied")],
    )
    with pytest.raises(RetryableToolError):
        tool.invoke({"q": "x"}, "k")
    with pytest.raises(PermanentToolError):
        tool.invoke({"q": "x"}, "k")
    assert tool.effects == 0


def test_virtual_clock_never_sleeps_real_time():
    clock = VirtualClock(start=5.0)
    clock.sleep(1.25)
    assert clock.now() == 6.25 and clock.sleeps == [1.25]
    with pytest.raises(HarnessFailure, match="non-negative"):
        clock.sleep(-1)


def test_scenario_loader_validates_version_tools_messages_budgets_and_trace():
    raw = {
        "version": 1,
        "tools": ["search"],
        "messages": [{
            "message_id": "m", "correlation_id": "c", "parent_run_id": "r",
            "sender": "s", "recipient": "w", "deadline": None, "attempt": 0,
            "idempotency_key": None, "payload": {"q": "x"},
        }],
        "failure_schedule": {"search": ["retryable", "result"]},
        "budgets": {"max_iterations": 3},
        "expected_trace": [{"kind": "run.started"}],
    }
    scenario = Scenario.load(raw, known_tools={"search"})
    assert scenario.version == 1 and scenario.messages[0].message_id == "m"
    with pytest.raises(HarnessFailure, match="unknown tools"):
        Scenario.load({**raw, "tools": ["missing"]}, known_tools={"search"})
    with pytest.raises(HarnessFailure, match="version"):
        Scenario.load({**raw, "version": 2}, known_tools={"search"})


def test_trace_assertions_cover_order_delays_effects_and_correlations():
    trace = TraceRecorder()
    trace.emit("run.started", 0, "run", "c", {})
    trace.emit("tool.retry", 0, "run", "c", {"delay": 1.0})
    trace.emit("tool.retry", 1, "run", "c", {"delay": 2.0})
    trace.emit("run.completed", 3, "run", "c", {})
    assert_event_order(trace, ["run.started", "tool.retry", "run.completed"])
    assert_retry_delays(trace, [1.0, 2.0])
    assert_correlation_chain(trace, "run", "c")
    with pytest.raises(AssertionError, match="order"):
        assert_event_order(trace, ["run.completed", "run.started"])


def test_trace_assertions_cover_budgets_concurrency_and_final_state():
    trace = TraceRecorder()
    trace.emit("wave.started", 0, "run", "c", {"size": 2})
    trace.emit("wave.started", 1, "run", "c", {"size": 1})
    assert_budget_usage({"tokens": 5, "cost": 0.2}, {"tokens": 5, "cost": 1.0})
    assert_max_concurrency(trace, 2)
    value = OrchestrationResult("partial", {"a": 1}, {"b": {"type": "failed"}}, {"a": 1, "b": 1}, 2, [])
    assert_final_state(value, "partial", result_ids={"a"}, failure_ids={"b"})
    with pytest.raises(AssertionError, match="budget"):
        assert_budget_usage({"tokens": 6}, {"tokens": 5})
    with pytest.raises(AssertionError, match="concurrency"):
        assert_max_concurrency(trace, 1)
    with pytest.raises(AssertionError, match="status"):
        assert_final_state(value, "completed")
