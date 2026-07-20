"""Typed, immutable values shared by deterministic agent exercises."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from torch_judge.harness import HarnessFailure


def copy_json(value: Any, *, field: str) -> Any:
    """Validate and isolate a JSON-like value used by a harness fixture."""

    def validate(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if isfinite(item):
                return
            raise HarnessFailure(f"{field} must contain finite JSON numbers")
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise HarnessFailure(f"{field} must be a JSON-like value")

    validate(value)
    return deepcopy(value)


def require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessFailure(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class MessageEnvelope:
    message_id: str
    correlation_id: str
    parent_run_id: str
    sender: str
    recipient: str
    deadline: float | None
    attempt: int
    idempotency_key: str | None
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        for field in ("message_id", "correlation_id", "parent_run_id", "sender", "recipient"):
            require_string(getattr(self, field), field=field)
        if self.deadline is not None and (
            not isinstance(self.deadline, (int, float))
            or isinstance(self.deadline, bool)
            or not isfinite(float(self.deadline))
            or self.deadline < 0
        ):
            raise HarnessFailure("deadline must be a non-negative finite number or None")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 0:
            raise HarnessFailure("attempt must be a non-negative integer")
        if self.idempotency_key is not None:
            require_string(self.idempotency_key, field="idempotency_key")
        payload = copy_json(self.payload, field="payload JSON")
        if not isinstance(payload, dict):
            raise HarnessFailure("payload JSON must be an object")
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True)
class AgentResult:
    status: str
    content: str | None
    messages: list[dict[str, Any]]
    iterations: int
    tool_calls: int
    tokens: int
    cost: float
    elapsed: float
    failure: dict[str, Any] | None

    def __post_init__(self) -> None:
        require_string(self.status, field="agent result status")
        if self.content is not None and not isinstance(self.content, str):
            raise HarnessFailure("agent result content must be a string or None")
        messages = copy_json(self.messages, field="agent result messages JSON")
        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            raise HarnessFailure("agent result messages JSON must be a list of objects")
        for field in ("iterations", "tool_calls", "tokens"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise HarnessFailure(f"agent result {field} must be a non-negative integer")
        for field in ("cost", "elapsed"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or value < 0
            ):
                raise HarnessFailure(f"agent result {field} must be a non-negative finite number")
        failure = None if self.failure is None else copy_json(
            self.failure, field="agent result failure JSON"
        )
        if failure is not None and not isinstance(failure, dict):
            raise HarnessFailure("agent result failure JSON must be an object or None")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "elapsed", float(self.elapsed))
        object.__setattr__(self, "failure", failure)


@dataclass(frozen=True)
class OrchestrationResult:
    status: str
    results: dict[str, Any]
    failures: dict[str, dict[str, Any]]
    attempts: dict[str, int]
    max_in_flight: int
    resumed: list[str]

    def __post_init__(self) -> None:
        require_string(self.status, field="orchestration result status")
        results = copy_json(self.results, field="orchestration results JSON")
        failures = copy_json(self.failures, field="orchestration failures JSON")
        attempts = copy_json(self.attempts, field="orchestration attempts JSON")
        resumed = copy_json(self.resumed, field="orchestration resumed JSON")
        if not isinstance(results, dict):
            raise HarnessFailure("orchestration results JSON must be an object")
        if not isinstance(failures, dict) or not all(
            isinstance(value, dict) for value in failures.values()
        ):
            raise HarnessFailure("orchestration failures JSON must map task ids to objects")
        if not isinstance(attempts, dict) or not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in attempts.items()
        ):
            raise HarnessFailure("orchestration attempts JSON must map task ids to counts")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or self.max_in_flight < 0
        ):
            raise HarnessFailure("max_in_flight must be a non-negative integer")
        if not isinstance(resumed, list) or not all(isinstance(item, str) for item in resumed):
            raise HarnessFailure("orchestration resumed JSON must be a list of task ids")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "resumed", resumed)
