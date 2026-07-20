"""Versioned, validated offline scenario fixtures (the data harness)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Collection

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.protocol import MessageEnvelope, copy_json


@dataclass(frozen=True)
class Scenario:
    version: int
    tools: tuple[str, ...]
    messages: tuple[MessageEnvelope, ...]
    failure_schedule: dict[str, tuple[str, ...]]
    budgets: dict[str, float]
    expected_trace: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, raw: dict[str, Any], *, known_tools: Collection[str]) -> "Scenario":
        data = copy_json(raw, field="scenario JSON")
        if not isinstance(data, dict):
            raise HarnessFailure("scenario JSON must be an object")
        if data.get("version") != 1:
            raise HarnessFailure("unsupported scenario version; expected version 1")
        tools = data.get("tools")
        if not isinstance(tools, list) or not all(isinstance(tool, str) and tool for tool in tools):
            raise HarnessFailure("scenario tools must be a list of non-empty strings")
        unknown = sorted(set(tools) - set(known_tools))
        if unknown:
            raise HarnessFailure(f"scenario references unknown tools: {unknown}")
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list):
            raise HarnessFailure("scenario messages must be a list")
        try:
            messages = tuple(MessageEnvelope(**message) for message in raw_messages)
        except TypeError as error:
            raise HarnessFailure(f"malformed scenario message: {error}") from error
        schedule = data.get("failure_schedule")
        if not isinstance(schedule, dict) or set(schedule) - set(tools):
            raise HarnessFailure("failure schedule must contain only declared tools")
        allowed = {"retryable", "permanent", "result"}
        normalized_schedule: dict[str, tuple[str, ...]] = {}
        for name, outcomes in schedule.items():
            if not isinstance(outcomes, list) or any(outcome not in allowed for outcome in outcomes):
                raise HarnessFailure("failure schedule values must be retryable, permanent, or result")
            normalized_schedule[name] = tuple(outcomes)
        budgets = data.get("budgets")
        if not isinstance(budgets, dict):
            raise HarnessFailure("scenario budgets must be an object")
        normalized_budgets: dict[str, float] = {}
        for name, value in budgets.items():
            if (
                not isinstance(name, str)
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or value < 0
            ):
                raise HarnessFailure("scenario budgets must be non-negative finite numbers")
            normalized_budgets[name] = value
        expected = data.get("expected_trace")
        if not isinstance(expected, list) or any(
            not isinstance(event, dict) or not isinstance(event.get("kind"), str)
            for event in expected
        ):
            raise HarnessFailure("expected trace must be a list of events with kind")
        return cls(1, tuple(tools), messages, normalized_schedule, normalized_budgets, tuple(expected))
