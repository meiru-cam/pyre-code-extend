"""Deterministic model double used by offline agent exercises."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from math import isfinite
from typing import Any, Iterable

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.protocol import copy_json


class RetryableModelError(RuntimeError):
    """A provider failure that may succeed when the same model call is retried."""

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


class PermanentModelError(RuntimeError):
    """A provider failure that must not be retried by the agent loop."""


class ScriptedModel:
    def __init__(self, outcomes: Iterable[dict[str, Any] | Exception]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[list[dict[str, Any]]] = []

    def call(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        copied = copy_json(messages, field="model messages JSON")
        if not isinstance(copied, list) or not all(isinstance(item, dict) for item in copied):
            raise HarnessFailure("model messages JSON must be a list of objects")
        self.calls.append(copied)
        if not self._outcomes:
            raise HarnessFailure("scripted model exhausted before the run completed")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        response = copy_json(outcome, field="scripted response JSON")
        if not isinstance(response, dict):
            raise HarnessFailure("scripted response JSON must be an object")
        return deepcopy(response)
