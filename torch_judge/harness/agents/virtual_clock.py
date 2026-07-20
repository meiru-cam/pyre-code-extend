"""A monotonic clock that advances without wall-clock sleeping."""

from __future__ import annotations

from math import isfinite

from torch_judge.harness import HarnessFailure


class VirtualClock:
    def __init__(self, *, start: float = 0.0) -> None:
        if not isinstance(start, (int, float)) or isinstance(start, bool) or not isfinite(float(start)):
            raise HarnessFailure("clock start must be a finite number")
        self._time = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._time

    def sleep(self, delay: float) -> None:
        if (
            not isinstance(delay, (int, float))
            or isinstance(delay, bool)
            or not isfinite(float(delay))
            or delay < 0
        ):
            raise HarnessFailure("sleep delay must be a non-negative finite number")
        delay = float(delay)
        self.sleeps.append(delay)
        self._time += delay
