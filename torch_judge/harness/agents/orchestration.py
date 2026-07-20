"""Deterministic doubles for supervisor/worker orchestration exercises."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Iterable, Sequence

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.protocol import MessageEnvelope, copy_json, require_string


class ScriptedWorker:
    """A worker double with declared capabilities and idempotent effects."""

    def __init__(self, name: str, capabilities: set[str], outcomes: Iterable[Any]) -> None:
        self.name = require_string(name, field="worker name")
        if not isinstance(capabilities, set) or not all(
            isinstance(capability, str) and capability for capability in capabilities
        ):
            raise HarnessFailure("worker capabilities must be non-empty strings in a set")
        self.capabilities = frozenset(capabilities)
        self._outcomes = deque(outcomes)
        self._results: dict[str, Any] = {}
        self.envelopes: list[MessageEnvelope] = []
        self.calls = 0
        self.effects = 0

    def run(self, envelope: MessageEnvelope) -> Any:
        if not isinstance(envelope, MessageEnvelope):
            raise HarnessFailure("worker input must be a MessageEnvelope")
        self.calls += 1
        isolated = deepcopy(envelope)
        self.envelopes.append(isolated)
        key = envelope.idempotency_key
        if key is not None and key in self._results:
            return deepcopy(self._results[key])
        if not self._outcomes:
            raise HarnessFailure(f"scripted worker {self.name!r} exhausted")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        result = copy_json(outcome, field="worker result JSON")
        self.effects += 1
        if key is not None:
            self._results[key] = deepcopy(result)
        return deepcopy(result)


class RoundRobinScheduler:
    """Choose eligible workers fairly and deterministically per capability."""

    def __init__(self) -> None:
        self._cursors: dict[str, int] = defaultdict(int)

    def choose(self, capability: str, workers: Sequence[ScriptedWorker]) -> ScriptedWorker | None:
        eligible = [worker for worker in workers if capability in worker.capabilities]
        if not eligible:
            return None
        cursor = self._cursors[capability]
        chosen = eligible[cursor % len(eligible)]
        self._cursors[capability] = cursor + 1
        return chosen


class InMemoryBus:
    """Record isolated envelopes sent over the logical transport."""

    def __init__(self) -> None:
        self.messages: list[MessageEnvelope] = []

    def send(self, envelope: MessageEnvelope) -> None:
        if not isinstance(envelope, MessageEnvelope):
            raise HarnessFailure("bus messages must be MessageEnvelope values")
        self.messages.append(deepcopy(envelope))


class InMemoryCheckpointStore:
    """A copy-isolated `(run_id, task_id)` checkpoint map."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], Any] = {}
        self.puts = 0

    def get(self, run_id: str, task_id: str) -> Any | None:
        return deepcopy(self._values.get((run_id, task_id)))

    def contains(self, run_id: str, task_id: str) -> bool:
        require_string(run_id, field="checkpoint run_id")
        require_string(task_id, field="checkpoint task_id")
        return (run_id, task_id) in self._values

    def put(self, run_id: str, task_id: str, value: Any) -> None:
        require_string(run_id, field="checkpoint run_id")
        require_string(task_id, field="checkpoint task_id")
        self._values[(run_id, task_id)] = copy_json(value, field="checkpoint JSON")
        self.puts += 1
