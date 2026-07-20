"""Deterministic tool doubles with failure and side-effect accounting."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from math import isfinite
from typing import Any, Iterable

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.protocol import copy_json, require_string


class RetryableToolError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        if retry_after is not None and (
            not isinstance(retry_after, (int, float))
            or isinstance(retry_after, bool)
            or not isfinite(float(retry_after))
            or retry_after < 0
        ):
            raise HarnessFailure("retry_after must be a non-negative finite number or None")
        self.retry_after = retry_after


class PermanentToolError(RuntimeError):
    pass


class FakeTool:
    def __init__(
        self,
        name: str,
        description: str,
        argument_schema: dict[str, type],
        *,
        outcomes: Iterable[Any],
    ) -> None:
        self.name = require_string(name, field="tool name")
        self.description = require_string(description, field="tool description")
        if not isinstance(argument_schema, dict) or not all(
            isinstance(key, str) and isinstance(value, type)
            for key, value in argument_schema.items()
        ):
            raise HarnessFailure("argument_schema must map strings to Python types")
        self.argument_schema = dict(argument_schema)
        self._outcomes = deque(outcomes)
        self._results: dict[str, Any] = {}
        self.effect_keys: list[str | None] = []
        self.invocation_keys: list[str | None] = []
        self.calls = 0
        self.effects = 0

    def invoke(self, arguments: dict[str, Any], idempotency_key: str | None = None) -> Any:
        copy_json(arguments, field="tool arguments JSON")
        if idempotency_key is not None:
            require_string(idempotency_key, field="idempotency_key")
        self.calls += 1
        self.invocation_keys.append(idempotency_key)
        if idempotency_key is not None and idempotency_key in self._results:
            return deepcopy(self._results[idempotency_key])
        if not self._outcomes:
            raise HarnessFailure(f"fake tool {self.name!r} exhausted")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        result = copy_json(outcome, field="tool result JSON")
        self.effects += 1
        self.effect_keys.append(idempotency_key)
        if idempotency_key is not None:
            self._results[idempotency_key] = deepcopy(result)
        return deepcopy(result)
