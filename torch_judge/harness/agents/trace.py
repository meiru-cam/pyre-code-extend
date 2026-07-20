"""Typed trace events for deterministic runtime assertions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.protocol import copy_json, require_string


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    timestamp: float
    run_id: str
    correlation_id: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        require_string(self.kind, field="trace kind")
        require_string(self.run_id, field="trace run_id")
        require_string(self.correlation_id, field="trace correlation_id")
        if not isinstance(self.timestamp, (int, float)) or not isfinite(float(self.timestamp)):
            raise HarnessFailure("trace timestamp must be a finite number")
        data = copy_json(self.data, field="trace data JSON")
        if not isinstance(data, dict):
            raise HarnessFailure("trace data JSON must be an object")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "data", data)


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(
        self,
        kind: str,
        timestamp: float,
        run_id: str,
        correlation_id: str,
        data: dict[str, Any],
    ) -> TraceEvent:
        event = TraceEvent(kind, timestamp, run_id, correlation_id, data)
        self.events.append(event)
        return event
