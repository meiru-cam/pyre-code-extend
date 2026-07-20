"""Learner-facing assertions for agent traces and side effects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch_judge.harness.agents.fake_tools import FakeTool
from torch_judge.harness.agents.trace import TraceRecorder


def assert_event_order(trace: TraceRecorder, expected: Sequence[str]) -> None:
    cursor = 0
    for event in trace.events:
        if cursor < len(expected) and event.kind == expected[cursor]:
            cursor += 1
    assert cursor == len(expected), (
        f"event order differs: expected subsequence {list(expected)}, "
        f"got {[event.kind for event in trace.events]}"
    )


def assert_retry_delays(trace: TraceRecorder, expected: Sequence[float]) -> None:
    actual = [event.data.get("delay") for event in trace.events if event.kind == "tool.retry"]
    assert actual == list(expected), f"retry delays differ: expected {list(expected)}, got {actual}"


def assert_no_duplicate_effects(tool: FakeTool) -> None:
    keyed = [key for key in tool.effect_keys if key is not None]
    assert len(keyed) == len(set(keyed)), f"duplicate side effects found for idempotency keys {keyed}"
    assert tool.effects == len(tool.effect_keys), "side-effect counter does not match recorded effects"


def assert_correlation_chain(trace: TraceRecorder, run_id: str, correlation_id: str) -> None:
    mismatches = [
        event for event in trace.events
        if event.run_id != run_id or event.correlation_id != correlation_id
    ]
    assert not mismatches, "trace contains events outside the expected run/correlation chain"


def assert_budget_usage(usage: dict[str, float], limits: dict[str, float]) -> None:
    """Assert every named cumulative usage value stays within its limit."""
    for name, limit in limits.items():
        assert name in usage, f"budget usage is missing {name!r}"
        assert usage[name] <= limit, (
            f"budget {name} exceeded: limit {limit}, observed {usage[name]}"
        )


def assert_max_concurrency(trace: TraceRecorder, limit: int) -> None:
    """Assert recorded wave/in-flight observations do not exceed a limit."""
    observations = [
        event.data[key]
        for event in trace.events
        for key in ("in_flight", "size")
        if key in event.data
    ]
    observed = max(observations, default=0)
    assert observed <= limit, (
        f"concurrency exceeded: limit {limit}, observed {observed}"
    )


def assert_final_state(
    result: Any,
    status: str,
    *,
    result_ids: set[str] | None = None,
    failure_ids: set[str] | None = None,
) -> None:
    """Assert terminal status and, when requested, result/failure partitions."""
    actual_status = getattr(result, "status", None)
    assert actual_status == status, f"final status differs: expected {status}, got {actual_status}"
    if result_ids is not None:
        actual = set(getattr(result, "results", {}))
        assert actual == result_ids, f"final result ids differ: expected {result_ids}, got {actual}"
    if failure_ids is not None:
        actual = set(getattr(result, "failures", {}))
        assert actual == failure_ids, f"final failure ids differ: expected {failure_ids}, got {actual}"
