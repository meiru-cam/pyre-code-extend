"""A versioned key-value store whose requirements grow one interview part at a time."""

from ._interview import interview

_MANUAL_CLOCK = r"""
class ManualClock:
    def __init__(self, start=0.0):
        self._now = start
    def now(self):
        return self._now
    def advance(self, dt):
        self._now += dt
"""

TASK = {
    "title": "Time-Travel Key-Value Store",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TimeMap",
    "description_en": r"""Build `TimeMap`, a key-value store that keeps every value a key has held and answers "what was this key's value at time t?".

The requirement arrives in parts, the way an interviewer adds them. Each part keeps every earlier behavior, so one `TimeMap` class passes all parts at the end. Pass every test of the current part to reveal the next one.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first part is a warm-up (a sorted list per key plus binary search). The later parts add one requirement at a time and test whether your design absorbs each one with a small change.

**Where it is used:** config and feature-flag histories, metrics stores and MVCC databases all answer "value as of time t" with the same per-key sorted history.

Adapted from the time-based key-value store question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, with part 1 narrowed so that the parts build on each other.""",
    "parts": [
        {
            "title": "Basic store",
            "description_en": r"""**Signature:** `TimeMap()`, `set(key, value, timestamp) -> None`, `get(key, timestamp) -> value | None`

- `set` records that `key` holds `value` from `timestamp` on. A timestamp is an `int` or a `float`.
- `get` returns the value recorded at the largest timestamp less than or equal to `timestamp`.
- `get` returns `None` for a key that was never set, or when every recorded timestamp for the key is later than `timestamp`.
- Keys are independent: a `set` on one key never changes what `get` returns for another.
- For one key, `set` calls arrive with strictly increasing timestamps.
- `get` must take time logarithmic in the number of values recorded for that key.

**Example:**
- `tm.set("probe", 68.0, 100)`, then `tm.set("probe", 71.5, 140)`
- `tm.get("probe", 130)` returns `68.0`; `tm.get("probe", 140)` returns `71.5`
- `tm.get("probe", 90)` returns `None`; `tm.get("other", 140)` returns `None`""",
        },
        {
            "title": "Injectable clock",
            "description_en": r"""Tests must not depend on the real clock. Keep every Part 1 behavior and add:

**Signature:** `TimeMap(clock=None)`, `set(key, value, timestamp=None)`, `get(key, timestamp=None)`

- A clock is any object with a method `now() -> float`.
- When `timestamp` is omitted or `None`, `set` and `get` use `clock.now()` at the moment of the call.
- An explicit `timestamp` always wins over the clock.
- With no clock given, `TimeMap` uses the real time, `time.time()`.

**Example** (a test clock whose `now()` returns a number the test controls):
- `clock.now()` is `1000`: `tm.set("probe", 68.0)` records at `1000`
- the clock moves to `1040`: `tm.set("probe", 71.5)`, then `tm.get("probe")` returns `71.5`
- `tm.get("probe", 1000)` returns `68.0`""",
        },
        {
            "title": "Strictly increasing timestamps",
            "description_en": r"""The Part 1 assumption goes away: a `set` may now carry a timestamp that is not larger than the last one recorded for its key, either passed explicitly or because the clock moved backward. Keep Parts 1–2 and add:

**Signature:** `set(key, value, timestamp=None) -> float`, returning the timestamp actually recorded.

- Let `last` be the largest timestamp recorded for `key` so far.
- If the key has no values yet, or the requested timestamp is greater than `last`, record at the requested timestamp.
- Otherwise record at `last + 1`. Nothing is inserted before or overwritten.
- Each key has its own `last`.

**Example:**
- `tm.set("probe", 68.0, 100)` returns `100`
- `tm.set("probe", 70.0, 100)` returns `101`; `tm.get("probe", 100)` is still `68.0`
- the clock reads `50`: `tm.set("probe", 65.0)` returns `102`""",
        },
        {
            "title": "Concurrent callers",
            "description_en": r"""Several threads now call `set` and `get` on the same `TimeMap` at once, on the same key or different keys. Keep Parts 1–3 and guarantee, for any number of threads and any scheduling:

- Each `set` and `get` behaves as if it happened at a single instant. A call without `timestamp` reads the clock inside that instant.
- For each key, the recorded timestamps stay strictly increasing, even when threads set the same key at the same time. No two `set` calls on one key return the same timestamp.
- `get(key, ...)` returns `None` or a value that some `set` on that key stored. Never another key's value, and never a half-written entry.
- No call raises because of another thread's call.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What per-key structure lets you find the last timestamp at or before t without scanning? If timestamps arrive in order, where does each new one go? Is there one code path every write goes through, so that a new requirement changes only that path?"},
        {"level": 2, "kind": "analysis", "content": "Keep two parallel lists per key, timestamps and values, and append to both. For get, bisect_right on the timestamps gives the insertion point; the answer sits one to the left, or there is none. Route every write through a single block that settles the timestamp, records it and returns it, and every read through a single lookup: each later part then changes one place, not several."},
    ],
    "model_connections": [
        "Multi-version concurrency control keeps a sorted version list per row and reads the newest version at or before a snapshot timestamp, which is part 1 at database scale.",
        "Hybrid logical clocks bump a timestamp to last + 1 when the physical clock has not advanced, the same fix part 3 applies per key.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Parallel sorted lists give O(log n) reads with bisect and O(1) appends when writes arrive in order.",
            "An injected clock makes time-dependent behavior testable without sleeping.",
            "Enforcing monotonic timestamps inside the store keeps every caller's history consistent.",
        ],
        "cons": [
            "Memory grows with every version; a real store needs compaction or a retention window.",
            "Rewriting a backward timestamp to last + 1 hides clock skew instead of reporting it.",
            "One store-wide lock serializes unrelated keys; per-key locks add bookkeeping and their own creation race.",
        ],
    },
    "tests": [
        {"name": "Part 1: latest value at or before t", "part": 1, "behavior": "state.invariant", "code": r"""
tm = {fn}()
tm.set("probe", 68.0, 100)
tm.set("probe", 71.5, 140)
assert tm.get("probe", 130) == 68.0
assert tm.get("probe", 140) == 71.5
assert tm.get("probe", 10**9) == 71.5
"""},
        {"name": "Part 1: missing key or too early", "part": 1, "behavior": "edge.empty_or_boundary", "code": r"""
tm = {fn}()
assert tm.get("probe", 100) is None
tm.set("probe", 68.0, 100)
assert tm.get("probe", 99.999) is None
assert tm.get("other", 100) is None
"""},
        {"name": "Part 1: keys are independent", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A set on one key changed what get returns for another key; keep one history per key.",
         "code": r"""
tm = {fn}()
tm.set("a", "a1", 1)
tm.set("b", "b1", 2)
tm.set("a", "a2", 3)
assert tm.get("a", 2) == "a1"
assert tm.get("b", 1) is None
assert tm.get("b", 3) == "b1"
assert tm.get("a", 3) == "a2"
"""},
        {"name": "Part 1: exact and in-between timestamps", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "get must include a value recorded exactly at the queried timestamp and pick the latest one not after it.",
         "code": r"""
tm = {fn}()
for t in range(0, 100, 10):
    tm.set("k", f"v{t}", t)
for q in range(0, 100):
    assert tm.get("k", q) == f"v{q - q % 10}", q
assert tm.get("k", 9.5) == "v0"
assert tm.get("k", 10.0) == "v10"
assert tm.get("k", -1) is None
"""},
        {"name": "Part 1: get is logarithmic", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "get compared too many timestamps; binary-search the key's sorted history instead of scanning it.",
         "code": r"""
class Stamp(float):
    compares = 0
    def _count(self):
        Stamp.compares += 1
    def __lt__(self, other): self._count(); return float(self) < float(other)
    def __le__(self, other): self._count(); return float(self) <= float(other)
    def __gt__(self, other): self._count(); return float(self) > float(other)
    def __ge__(self, other): self._count(); return float(self) >= float(other)
    def __eq__(self, other): self._count(); return float(self) == float(other)
    __hash__ = float.__hash__

tm = {fn}()
n = 4096
for i in range(n):
    tm.set("k", i, Stamp(i))
Stamp.compares = 0
for q in (0, 1000, 2047, 3000, 4095, 5000):
    assert tm.get("k", Stamp(q)) == min(q, n - 1)
assert Stamp.compares <= 6 * 40, f"{Stamp.compares} comparisons for 6 lookups in 4096 entries"
"""},
        {"name": "Part 2: timestamps default to the clock", "part": 2, "behavior": "contract.signature", "code": _MANUAL_CLOCK + r"""
clock = ManualClock(1000.0)
tm = {fn}(clock=clock)
tm.set("probe", 68.0)
clock.advance(40.0)
tm.set("probe", 71.5)
assert tm.get("probe") == 71.5
assert tm.get("probe", 1000.0) == 68.0
"""},
        {"name": "Part 2: explicit timestamps beat the clock", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "An explicit timestamp must be used as given even when a clock is injected.",
         "code": _MANUAL_CLOCK + r"""
clock = ManualClock(500.0)
tm = {fn}(clock=clock)
tm.set("k", "early", 100.0)
tm.set("k", "now")
assert tm.get("k", 100.0) == "early"
assert tm.get("k", 499.0) == "early"
assert tm.get("k") == "now"
tm.set("j", "x", timestamp=None)
assert tm.get("j", 500.0) == "x"
"""},
        {"name": "Part 2: the clock is read at call time", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Read clock.now() on every call that omits the timestamp, not once when the store is built.",
         "code": _MANUAL_CLOCK + r"""
clock = ManualClock(0.0)
tm = {fn}(clock=clock)
clock.advance(10.0)
tm.set("k", "a")
assert tm.get("k", 9.0) is None
clock.advance(5.0)
assert tm.get("k") == "a"
clock.advance(-10.0)
assert tm.get("k") is None
"""},
        {"name": "Part 2: default clock is real time", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Without a clock argument, fall back to a clock whose now() returns time.time().",
         "code": r"""
import time
tm = {fn}()
before = time.time()
tm.set("k", "v")
assert tm.get("k", before - 60) is None
assert tm.get("k", time.time() + 60) == "v"
assert tm.get("k") == "v"
"""},
        {"name": "Part 3: backward timestamps become last + 1", "part": 3, "behavior": "events.ordering", "code": _MANUAL_CLOCK + r"""
clock = ManualClock(100.0)
tm = {fn}(clock=clock)
assert tm.set("probe", 68.0) == 100.0
assert tm.set("probe", 70.0, 100.0) == 101.0
assert tm.get("probe", 100.0) == 68.0
assert tm.get("probe", 101.0) == 70.0
clock.advance(-50.0)
assert tm.set("probe", 65.0) == 102.0
assert tm.get("probe", 102.0) == 65.0
"""},
        {"name": "Part 3: each key keeps its own last timestamp", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Compare a set only with the last timestamp recorded for the same key.",
         "code": r"""
tm = {fn}()
assert tm.set("busy", 1, 50) == 50
assert tm.set("busy", 2, 50) == 51
assert tm.set("quiet", 3, 10) == 10
assert tm.set("quiet", 4, 11) == 11
assert tm.get("quiet", 10) == 3
"""},
        {"name": "Part 3: returns the requested timestamp when it is new", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "set must return the timestamp it recorded, which is the requested one when it is later than the key's last.",
         "code": r"""
tm = {fn}()
assert tm.set("k", "a", 1.5) == 1.5
assert tm.set("k", "b", 7.25) == 7.25
assert tm.set("k", "c", 3) == 8.25
assert tm.set("k", "d", 8.25) == 9.25
assert tm.set("k", "e", 100) == 100
# Later by less than 1 is still later: record it as requested, not at last + 1.
assert tm.set("k", "f", 100.5) == 100.5
assert [tm.get("k", t) for t in (1.5, 7.25, 8.25, 9.25, 100, 100.5)] == ["a", "b", "c", "d", "e", "f"]
"""},
        {"name": "Part 3: nothing is overwritten", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A repeated or backward timestamp must append a new entry at last + 1, not replace an existing one.",
         "code": _MANUAL_CLOCK + r"""
clock = ManualClock(10.0)
tm = {fn}(clock=clock)
recorded = [tm.set("k", i) for i in range(5)]
assert recorded == [10.0, 11.0, 12.0, 13.0, 14.0]
assert [tm.get("k", t) for t in recorded] == [0, 1, 2, 3, 4]
"""},
        {"name": "Part 4: concurrent sets on one key stay strictly increasing", "part": 4, "behavior": "concurrency.thread_safety", "code": r"""
import random, threading, time

def pause():
    # A random sleep hands the GIL to another thread and shuffles who resumes first.
    time.sleep(random.random() * 0.0004)

class SlowStamp(float):
    def __lt__(self, other): pause(); return float(self) < float(other)
    def __le__(self, other): pause(); return float(self) <= float(other)
    def __gt__(self, other): pause(); return float(self) > float(other)
    def __ge__(self, other): pause(); return float(self) >= float(other)

class SlowKey(str):
    # Every dict lookup on this key yields, including between two writes that belong together.
    def __hash__(self):
        pause()
        return str.__hash__(self)
    __eq__ = str.__eq__

class SlowClock:
    def now(self):
        pause()
        return SlowStamp(1000.0)

HOT = SlowKey("hot")
tm = {fn}(clock=SlowClock())
threads, per_thread = 8, 15
recorded = [[] for _ in range(threads)]
errors = []

def worker(i):
    try:
        for j in range(per_thread):
            recorded[i].append((tm.set(HOT, (i, j)), (i, j)))
    except Exception as error:
        errors.append(error)

pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
for t in pool: t.start()
for t in pool: t.join()
assert not errors, errors
stamps = [stamp for per in recorded for stamp, _ in per]
assert len(set(stamps)) == len(stamps), "two sets on one key recorded the same timestamp"
assert sorted(stamps) == [1000.0 + k for k in range(threads * per_thread)]
for per in recorded:
    for stamp, value in per:
        assert tm.get(HOT, stamp) == value, "a value was paired with another set's timestamp"
"""},
        {"name": "Part 4: readers only see stored values", "part": 4, "visibility": "unshown", "behavior": "concurrency.thread_safety",
         "failure_message": "Under concurrent calls, get returned a value no set stored for that key or raised; guard reads and writes with the same lock.",
         "code": r"""
import sys, threading
old = sys.getswitchinterval()
sys.setswitchinterval(1e-6)
try:
    tm = {fn}()
    errors = []
    def writer(key):
        try:
            for i in range(1500):
                tm.set(key, (key, i), 0)
        except Exception as error:
            errors.append(error)
    def reader(key):
        try:
            for i in range(1500):
                value = tm.get(key, i)
                assert value is None or value[0] == key, value
        except Exception as error:
            errors.append(error)
    pool = [threading.Thread(target=f, args=(k,)) for k in ("a", "b") for f in (writer, reader)]
    for t in pool: t.start()
    for t in pool: t.join()
finally:
    sys.setswitchinterval(old)
assert not errors, errors[:3]
for key in ("a", "b"):
    assert [tm.get(key, t) for t in range(1500)] == [(key, t) for t in range(1500)]
"""},
    ],
    "solution": r'''import bisect
import threading
import time


class _RealClock:
    def now(self):
        return time.time()


class TimeMap:
    def __init__(self, clock=None):
        self._clock = clock if clock is not None else _RealClock()
        self._stamps = {}
        self._values = {}
        self._lock = threading.Lock()

    def set(self, key, value, timestamp=None):
        with self._lock:
            if timestamp is None:
                timestamp = self._clock.now()
            stamps = self._stamps.setdefault(key, [])
            if stamps and timestamp <= stamps[-1]:
                timestamp = stamps[-1] + 1
            stamps.append(timestamp)
            self._values.setdefault(key, []).append(value)
            return timestamp

    def get(self, key, timestamp=None):
        with self._lock:
            if timestamp is None:
                timestamp = self._clock.now()
            stamps = self._stamps.get(key)
            if not stamps:
                return None
            index = bisect.bisect_right(stamps, timestamp)
            return self._values[key][index - 1] if index else None
''',
    "interview_questions": interview(
        # Concept and deep-dive questions are answered before coding, so they stay inside
        # part 1. Questions about later parts sit in tradeoffs, which a multi-part
        # exercise shows only once every part is unlocked.
        concept=[
            "What data structure do you keep per key, and why does it make get logarithmic?",
            "What should get return when the key exists but every recorded timestamp is later than the query?",
        ],
        deep_dive=[
            "Which binary-search variant finds the latest timestamp at or before t, and how does it treat an exact match?",
        ],
        tradeoffs=[
            "Why inject a clock instead of calling time.time() inside set and get?",
            "Walk through what set does when the requested timestamp is not larger than the key's last one, and why appending at last + 1 keeps get correct.",
            "Which steps of set must happen atomically for part 4, and what goes wrong if the clock is read outside the lock?",
            "Compare one lock for the whole store with one lock per key: what does each cost, and when does the per-key version win?",
            "Would a read-write lock that lets concurrent gets run together help here? Why or why not?",
        ],
    ),
}
