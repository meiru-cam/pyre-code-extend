A `TimeMap` records, for a set of string keys, the history of values a key has held over time. A *timestamp* is a real number (an `int` or a `float`); a larger timestamp means a later point in time. Implement the following four parts.

### Part 1 — Basic implementation

`TimeMap()` creates an empty store.

`set(key, value, timestamp)` records that, as of `timestamp`, `key` holds `value`. A key can have any number of values recorded at different timestamps, and calls are not required to arrive in increasing order of `timestamp` for a given key: a call with an earlier timestamp may happen after one with a later timestamp has already been recorded.

`get(key, timestamp)` returns the value with the largest recorded timestamp that is less than or equal to `timestamp`, among everything recorded so far for `key`. If `key` has never been passed to `set`, or every timestamp recorded for `key` is greater than the one being queried, `get` returns `None`.

If two calls to `set` use the same `key` and exactly the same `timestamp`, the later of the two calls (in the order `set` was invoked) is the one `get` uses for that timestamp, as if the earlier call had never happened.

```py
from typing import Any


class TimeMap:
    def __init__(self) -> None: ...
    def set(self, key: str, value: Any, timestamp: float) -> None: ...
    def get(self, key: str, timestamp: float) -> Any | None: ...
```

Example:

```text
tm = TimeMap()
tm.set("probe-1", 68.0, 100.0)
tm.set("probe-1", 71.5, 140.0)
tm.get("probe-1", 130.0)          # -> 68.0   (the latest value recorded at or before 130.0)
tm.get("probe-1", 90.0)           # -> None   (nothing recorded yet at or before 90.0)
tm.get("probe-2", 140.0)          # -> None   (unknown key)
tm.set("probe-1", 65.0, 120.0)    # arrives after the 140.0 call, but timestamped earlier
tm.get("probe-1", 130.0)          # -> 65.0   (now the closest value at or before 130.0)
tm.set("probe-1", 70.0, 140.0)    # same timestamp as an earlier call
tm.get("probe-1", 140.0)          # -> 70.0   (the later call wins)
```

### Part 2 — Testability

Testing `TimeMap` should not depend on the real clock: a test that calls `time.time()` gets a different timestamp on every run, and separating two calls with `time.sleep` makes the test slow.

`TimeMap`'s constructor accepts an optional *clock*: any object with a method `now() -> float`. `set` and `get` both accept `timestamp` as optional; when it is left out (or passed as `None`), the call uses `clock.now()` in its place. An explicit `timestamp` always overrides the clock, whether or not a clock was injected. When no clock is given, `TimeMap` falls back to one that calls the real `time.time()`.

Tests use the following manual clock, which never touches the real time:

```python
class ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt   # dt may be negative: this simulates the clock moving backward
```

```py
class TimeMap:
    def __init__(self, clock: Any = None) -> None: ...
    def set(self, key: str, value: Any, timestamp: float | None = None) -> None: ...
    def get(self, key: str, timestamp: float | None = None) -> Any | None: ...
```

Example:

```text
clock = ManualClock(start=1000.0)
tm = TimeMap(clock=clock)
tm.set("probe-1", 68.0)            # timestamp defaults to clock.now() == 1000.0
clock.advance(40.0)
tm.set("probe-1", 71.5)            # timestamp defaults to clock.now() == 1040.0
tm.get("probe-1")                  # timestamp defaults to clock.now() == 1040.0 -> 71.5
tm.get("probe-1", 1000.0)          # an explicit timestamp overrides the clock -> 68.0
```

### Part 3 — Keeping timestamps strictly increasing

From this part on, the timestamp recorded for a given key must be strictly increasing. For a key `key`, let `last(key)` be the largest timestamp recorded for it so far (undefined before its first `set`). When `set(key, value, timestamp)` is called — whether `timestamp` was passed explicitly or came from `clock.now()` — the timestamp that ends up recorded is:

- `timestamp` itself, if `last(key)` is undefined or `timestamp` is strictly greater than `last(key)`;
- `last(key) + 1` otherwise — this covers both the clock moving backward and an explicit `timestamp` that is not larger than the last one recorded for that key.

This constraint, and that adjustment, apply independently to each key; a key with few writes is not compared against a key with many. This replaces the two Part 1 rules for a call whose timestamp is not larger than the previous one recorded for that key: such a call is no longer inserted before an existing entry, and no longer overwrites an entry at the same timestamp — it is always appended, at whichever timestamp this rule computes. From this part on, `set` returns the timestamp it actually recorded, which may differ from the `timestamp` argument.

```py
class TimeMap:
    def __init__(self, clock: Any = None) -> None: ...
    def set(self, key: str, value: Any, timestamp: float | None = None) -> float: ...
    def get(self, key: str, timestamp: float | None = None) -> Any | None: ...
```

Example:

```text
clock = ManualClock(start=100.0)
tm = TimeMap(clock=clock)
tm.set("probe-1", 68.0)            # -> 100.0   (clock.now())
tm.set("probe-1", 70.0, 100.0)     # explicit 100.0 is not greater than last (100.0) -> recorded at 101.0
tm.get("probe-1", 100.0)           # -> 68.0    (100.0 still holds the first value)
tm.get("probe-1", 101.0)           # -> 70.0
clock.advance(-50.0)               # the clock moves backward: now() == 50.0
tm.set("probe-1", 65.0)            # 50.0 is not greater than last (101.0) -> recorded at 102.0
tm.get("probe-1", 102.0)           # -> 65.0
```

### Part 4 — Concurrent access

Multiple threads may call `set` and `get` on the same store at the same time, for the same key or for different keys, in any interleaving and under any scheduling. Implement `ThreadSafeTimeMap`, offering the same `set`/`get` semantics as Part 3's `TimeMap` (including the strictly-increasing rule), plus the following three guarantees, which must hold for any number of threads and any scheduling:

- Every call to `set` or `get` behaves as if it took effect at a single instant, and a call that omits `timestamp` reads `clock.now()` at that instant: the store's internal state is never left inconsistent (for a given key, the recorded timestamps stay strictly increasing, and there are exactly as many recorded timestamps as recorded values), and no call raises an exception caused only by another thread's concurrent call.
- Every value returned by `get(key, ...)` is either `None` or exactly the `value` argument of some `set` call on `key` — never a value paired with a timestamp that a different `set` call recorded.
- Part 3's strictly-increasing rule still holds when several threads call `set` for the same key at the same time: whichever order their calls actually take effect in, the sequence of timestamps recorded for that key stays strictly increasing.

Implement and compare at least two locking strategies: one lock shared by the whole store, and one lock per key, created the first time that key is touched. If you also consider a read/write lock that lets concurrent `get` calls run together, say whether it would actually help here.

```py
class ThreadSafeTimeMap:
    def __init__(self, clock: Any = None) -> None: ...
    def set(self, key: str, value: Any, timestamp: float | None = None) -> float: ...
    def get(self, key: str, timestamp: float | None = None) -> Any | None: ...
```

Example:

```text
Thread A: set("probe-1", 71.5)     # no explicit timestamp; clock.now() == 1040.0 for both threads
Thread B: set("probe-1", 69.0)     # A and B run at the same time, in either order

# Whichever call actually takes effect first, "probe-1" ends up with two strictly increasing
# timestamps, e.g. 1040.0 and 1041.0 -- never two entries at 1040.0.
get("probe-1", 1041.0)  # -> 71.5 or 69.0, whichever of A/B took effect second, never anything else
```
