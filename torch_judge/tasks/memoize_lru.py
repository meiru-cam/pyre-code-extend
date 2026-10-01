"""An LRU memoizer: fix its bugs, make it survive a crash, then make it safe for threads."""

from ._interview import interview

_SETUP = r"""
import os, tempfile

def counting(fn):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

workdir = tempfile.mkdtemp()
log = os.path.join(workdir, "cache.log")
"""

TASK = {
    "title": "LRU Memoizer: Bugfix, Durability, Threads",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "Memoize",
    "description_en": r"""`Memoize(func, capacity)` wraps `func` and caches what it returns, in the style of `functools.lru_cache`, holding at most `capacity` entries and evicting the least recently used one first.

The requirement arrives in parts, the way an interviewer adds them. Each part keeps every earlier behavior, so one `Memoize` class passes all parts at the end. Pass every test of the current part to reveal the next one.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** reading someone else's code and finding the bug is a separate skill from writing code, and an LRU cache is small enough to read in a minute. The later parts check that the fix holds up as the requirements grow.

**Where it is used:** `functools.lru_cache`, HTTP and DNS caches, and model-serving caches all key results by their arguments and evict by recency.

Adapted from the LRU cache bug-fix question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, with every part on one class.""",
    "parts": [
        {
            "title": "Fix three bugs",
            "description_en": r"""The starter code has three bugs: two in `generate_key`, one in `__call__`. Fix them so that all of these hold.

**Signature:** `Memoize(func, capacity)`, called as `cache(*args, **kwargs)`, with counters `hits` and `misses`

- A call whose arguments match an earlier cached call returns the cached value without calling `func`, and counts as a hit. Any other call calls `func`, caches the result and counts as a miss.
- Two calls match exactly when they pass equal positional arguments in the same positions and equal keyword arguments, compared by name and value in any order.
- `f(1, 2)` and `f((1, 2))` never match. `f(1, b=2)` and `f(1, ("b", 2))` never match. A positional argument never matches a keyword argument.
- At most `capacity` entries are kept. Adding one more evicts the least recently used entry.
- A hit counts as a use, exactly like a miss.
- An unhashable argument, such as a `list`, raises `TypeError`.

**Example** (`capacity=2`):
- `cache(1, b=2, c=3)` miss, then `cache(1, c=3, b=2)` hit
- `cache((1, 2))` miss, then `cache(1, 2)` miss; the first entry is evicted
- `cache((1, 2))` hit, then `cache(1, 3)` miss; `f(1, 2)` is evicted, not `f((1, 2))`""",
        },
        {
            "title": "Survive a crash",
            "description_en": r"""A crash, meaning the process killed at any instant, must lose no call that finished before it. Keep Part 1 and add:

**Signature:** `Memoize(func, capacity, path=None)`, `compact() -> None`

- With `path=None`, nothing is written anywhere and `Memoize` behaves exactly as in Part 1.
- With a `path`, every call appends one record to the file at exactly `path` before returning: a miss records the new entry, a hit records the use. Flush and `os.fsync` the file before returning.
- Recording a call costs time and space proportional to that call only, never to `capacity` or to earlier calls. The file is append-only.
- Creating `Memoize(func, capacity, path)` on an existing file recovers, inside `__init__`, the same entries, values and least-to-most-recently-used order the crashed instance had after its last finished call.
- A record cut short or corrupted by the crash is detected from a length and a checksum. It and everything after it are discarded without raising, and later calls append after the last good record.
- `hits` and `misses` are not persisted. They start at 0.
- Arguments and return values are JSON values: `None`, `bool`, `int`, `float`, `str`, `list`, `dict` with `str` keys, and `tuple` in arguments. Argument values round-trip exactly, tuples included; a returned tuple may come back as a list.
- `compact()` rewrites `path` so its size depends only on the entries currently cached. It writes a temporary file next to `path` and replaces `path` atomically.""",
        },
        {
            "title": "Concurrent callers",
            "description_en": r"""Several threads now call the same `Memoize` at once. Keep Parts 1–2 and guarantee, for any number of threads and any scheduling:

- Every call behaves as if it happened at a single instant, and the cache never holds more than `capacity` entries.
- For one key, `func` is never running more than once at a time. A thread that asks for a key whose value is being computed waits for that result instead of calling `func` again.
- A call for a key never waits on a running `func` for a different key: `func` runs without holding any lock the cache uses.
- If `func` raises, every thread waiting on that call gets the same exception, and nothing is cached or recorded.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which two calls in the example produce the same flattened tuple of arguments? Does the key keep a boundary between positional and keyword arguments? Which OrderedDict call marks an entry as recently used, and does the hit path make it?"},
        {"level": 2, "kind": "analysis", "content": "A key that is just args + kwargs.items() loses where positional ends and keyword begins, and unwrapping a one-element key makes f((1, 2)) look like f(1, 2). Build the key as a pair of the args tuple and the sorted kwargs items, never unwrap it, and call move_to_end on a hit before returning."},
    ],
    "model_connections": [
        "functools.lru_cache builds its key from args plus a sentinel-separated kwargs tuple, the same boundary the Part 1 fix restores.",
        "Write-ahead logs in databases frame each record with a length and a checksum and truncate at the first bad record on recovery, which is what Part 2 asks for.",
        "Request coalescing in CDNs and Go's singleflight package let one caller compute a missing value while the others wait for it, the Part 3 guarantee.",
    ],
    "pro_con_analysis": {
        "pros": [
            "An append-only log makes each call's durability cost independent of the cache size.",
            "Length plus checksum framing lets recovery stop cleanly at a torn write instead of trusting partial bytes.",
            "Single-flight avoids running an expensive or side-effecting func twice for the same key.",
        ],
        "cons": [
            "fsync on every call bounds throughput by disk latency; batching trades durability for speed.",
            "The log grows until compact() runs, and choosing when to compact is left to the caller.",
            "Waiters share the leader's exception, so one failure fails every concurrent caller of that key.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
calls = []
def f(*args, **kwargs):
    calls.append((args, kwargs))
    return (args, kwargs)

cache = {fn}(f, capacity=2)
cache(1, b=2, c=3)
cache(1, c=3, b=2)
assert len(calls) == 1, "reordered kwargs must hit the same entry"
cache((1, 2))
cache(1, 2)
assert len(calls) == 3, "f((1, 2)) and f(1, 2) are different calls"
cache((1, 2))
assert len(calls) == 3, "f((1, 2)) is still cached"
cache(1, 3)
cache((1, 2))
assert len(calls) == 4, "the hit on f((1, 2)) made it recent, so f(1, 2) was evicted instead"
cache(1, 2)
assert len(calls) == 5
assert (cache.hits, cache.misses) == (3, 5)
"""},
        {"name": "Part 1: keyword order does not matter", "part": 1, "behavior": "state.invariant", "code": r"""
n = [0]
def f(**kw):
    n[0] += 1
    return sorted(kw)
cache = {fn}(f, capacity=4)
assert cache(a=1, b=2, c=3) == ["a", "b", "c"]
assert cache(c=3, a=1, b=2) == ["a", "b", "c"]
assert cache(b=2, c=3, a=1) == ["a", "b", "c"]
assert n[0] == 1
"""},
        {"name": "Part 1: positional and keyword arguments never mix", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Two different calls shared one cache entry; keep positional and keyword arguments apart in the key and never unwrap a one-element key.",
         "code": r"""
seen = []
def f(*args, **kwargs):
    seen.append((args, kwargs))
    return len(seen)
cache = {fn}(f, capacity=10)
distinct = [
    lambda: cache(1, 2),
    lambda: cache((1, 2)),
    lambda: cache(1, b=2),
    lambda: cache(1, ("b", 2)),
    lambda: cache(b=2),
    lambda: cache(("b", 2)),
    lambda: cache(1),
    lambda: cache((1,)),
]
results = [call() for call in distinct]
assert len(seen) == len(distinct), f"only {len(seen)} of {len(distinct)} distinct calls reached func"
assert [call() for call in distinct] == results
assert len(seen) == len(distinct)
"""},
        {"name": "Part 1: a hit refreshes recency", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A hit must mark its entry as most recently used, so eviction picks an entry untouched for longer.",
         "code": r"""
calls = []
cache = {fn}(lambda x: calls.append(x) or x, capacity=3)
for x in (1, 2, 3):
    cache(x)
cache(1)
cache(2)
cache(4)
before = len(calls)
cache(1); cache(2); cache(4)
assert len(calls) == before, "1, 2 and 4 should all still be cached"
cache(3)
assert len(calls) == before + 1, "3 was the least recently used entry and should have been evicted"
"""},
        {"name": "Part 1: capacity, counters and unhashable arguments", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Keep at most capacity entries, count hits and misses, and let an unhashable argument raise TypeError.",
         "code": r"""
cache = {fn}(lambda x: x * 10, capacity=1)
assert cache(1) == 10 and cache(1) == 10 and cache(2) == 20 and cache(1) == 10
assert (cache.hits, cache.misses) == (1, 3)
try:
    cache([1, 2])
except TypeError:
    pass
else:
    raise AssertionError("a list argument must raise TypeError")
"""},
        {"name": "Part 2: recovery after a crash keeps entries and order", "part": 2, "behavior": "checkpoint.recovery", "code": _SETUP + r"""
square = counting(lambda n: n * n)
w = {fn}(square, capacity=2, path=log)
assert w(3) == 9 and w(4) == 16 and w(3) == 9
del w

square2 = counting(lambda n: n * n)
r = {fn}(square2, capacity=2, path=log)
assert (r.hits, r.misses) == (0, 0)
# The hit on 3 made it more recent than 4, so a new entry must evict 4, not 3.
r(5)
assert square2.calls == 1
assert r(3) == 9 and square2.calls == 1, "3 should still be cached: recovery lost the use of 3"
assert r(4) == 16 and square2.calls == 2, "4 was least recently used and should have been evicted"
"""},
        {"name": "Part 2: a torn last record is discarded", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "A record cut short by a crash must be detected and dropped, recovering the state from before that call, and later calls must append after the last good record.",
         "code": _SETUP + r"""
import shutil
w = {fn}(lambda n: n + 100, capacity=3, path=log)
for n in (1, 2, 3, 1):
    w(n)
before = os.path.getsize(log)
w(4)
after = os.path.getsize(log)
assert after > before + 1
for cut in sorted({before + 1, (before + after) // 2, after - 1}):
    torn = os.path.join(workdir, f"torn{cut}.log")
    shutil.copyfile(log, torn)
    with open(torn, "r+b") as fh:
        fh.truncate(cut)
    f = counting(lambda n: n + 100)
    r = {fn}(f, capacity=3, path=torn)
    assert r(2) == 102 and r(3) == 103 and r(1) == 101 and f.calls == 0, f"cut at {cut}: entries before the torn call were lost"
    assert r(4) == 104 and f.calls == 1, f"cut at {cut}: the torn call must count as never made"
    del r
    f2 = counting(lambda n: n + 100)
    again = {fn}(f2, capacity=3, path=torn)
    assert again(4) == 104 and f2.calls == 0, f"cut at {cut}: a call after recovery was not persisted"
"""},
        {"name": "Part 2: a corrupted record is caught by its checksum", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "Recovery must verify each record's checksum and discard a damaged record and everything after it, without raising.",
         "code": _SETUP + r"""
w = {fn}(lambda s: s.upper(), capacity=4, path=log)
w("a"); w("b")
before = os.path.getsize(log)
w("cccccccccccccccccccc")
after = os.path.getsize(log)
with open(log, "r+b") as fh:
    fh.seek((before + after) // 2 + 2)
    byte = fh.read(1)
    fh.seek(-1, 1)
    fh.write(bytes([byte[0] ^ 0x5A]))
f = counting(lambda s: s.upper())
r = {fn}(f, capacity=4, path=log)
assert r("cccccccccccccccccccc") == "CCCCCCCCCCCCCCCCCCCC" and f.calls == 1, "the damaged call must count as never made"
# Capacity 4 now holds a, b, c and x1. A damaged record recovered as some other entry
# would have taken a slot, so x1 would have evicted "a".
r("x1")
assert r("a") == "A" and r("b") == "B" and f.calls == 2, "recovery kept an entry from the damaged record"
"""},
        {"name": "Part 2: argument types round-trip exactly", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "Recovered keys must equal the original keys, so a tuple argument must come back as a tuple and keyword arguments by name.",
         "code": _SETUP + r"""
w = {fn}(lambda *a, **k: [list(a), sorted(k)], capacity=8, path=log)
calls = [((1, (2, 3)), {}), (((1, 2),), {}), ((1, 2), {}), ((), {"x": (1, "y"), "z": None}), (("s", 1.5, True), {})]
expected = [w(*a, **k) for a, k in calls]
del w
f = counting(lambda *a, **k: [list(a), sorted(k)])
r = {fn}(f, capacity=8, path=log)
for (a, k), value in zip(calls, expected):
    assert r(*a, **k) == value
assert f.calls == 0, f"{f.calls} recovered calls missed the cache"
"""},
        {"name": "Part 2: no path writes nothing", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "With path=None the cache must not create or write any file.",
         "code": _SETUP + r"""
here = os.getcwd()
os.chdir(workdir)
try:
    c = {fn}(lambda x: x, capacity=2)
    c(1); c(1); c(2)
    c2 = {fn}(lambda x: x, capacity=2, path=None)
    c2(1)
    assert os.listdir(workdir) == [], os.listdir(workdir)
finally:
    os.chdir(here)
"""},
        {"name": "Part 2: each call appends a bounded record", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Recording one call must cost space proportional to that call, not to the cache; append one record instead of rewriting the file.",
         "code": _SETUP + r"""
w = {fn}(lambda n: n, capacity=1000, path=log)
sizes = [0]
for n in range(10000, 10300):
    w(n)
    sizes.append(os.path.getsize(log))
first = sizes[2] - sizes[1]
last = sizes[-1] - sizes[-2]
assert last <= 2 * first + 16, f"record grew from {first} to {last} bytes as the cache filled"
"""},
        {"name": "Part 2: compact shrinks the log and keeps the state", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "compact() must leave a file whose size depends only on the cached entries, from which recovery gives the same entries and order.",
         "code": _SETUP + r"""
fresh = os.path.join(workdir, "fresh.log")
f0 = {fn}(lambda n: -n, capacity=3, path=fresh)
for n in (1, 2, 3):
    f0(n)
fresh_size = os.path.getsize(fresh)

w = {fn}(lambda n: -n, capacity=3, path=log)
for n in (1, 2, 3):
    w(n)
for _ in range(300):
    w(2); w(1)
grown = os.path.getsize(log)
w.compact()
compacted = os.path.getsize(log)
assert compacted <= 3 * fresh_size, f"compacted log is {compacted} bytes; three entries take {fresh_size}"
assert compacted < grown
w(4)
del w
f = counting(lambda n: -n)
r = {fn}(f, capacity=3, path=log)
assert r(1) == -1 and r(2) == -2 and r(4) == -4 and f.calls == 0
r(5)
r(3)
assert f.calls == 2, "3 was evicted before the crash and must stay evicted"
"""},
        {"name": "Part 3: one computation per key", "part": 3, "behavior": "concurrency.thread_safety", "code": r"""
import threading
release = threading.Event()
started = threading.Event()
runs = []
def slow(x):
    runs.append(x)
    started.set()
    assert release.wait(5)
    return x * 2

cache = {fn}(slow, capacity=8)
results = []
def caller():
    results.append(cache(21))
pool = [threading.Thread(target=caller, daemon=True) for _ in range(6)]
pool[0].start()
assert started.wait(5)
for t in pool[1:]:
    t.start()
release.set()
for t in pool:
    t.join(5)
assert not any(t.is_alive() for t in pool), "a caller is still waiting"
assert runs == [21], f"func ran {len(runs)} times for one key"
assert results == [42] * 6
"""},
        {"name": "Part 3: other keys do not wait on a slow func", "part": 3, "visibility": "unshown", "behavior": "concurrency.thread_safety",
         "failure_message": "A call for one key blocked behind func running for another key; run func outside every lock the cache holds.",
         "code": r"""
import threading
release = threading.Event()
started = threading.Event()
def f(x):
    if x == "slow":
        started.set()
        release.wait(5)
    return x
cache = {fn}(f, capacity=8)
cache("cached")
blocker = threading.Thread(target=lambda: cache("slow"), daemon=True)
blocker.start()
assert started.wait(5)
done = []
other = threading.Thread(target=lambda: done.append((cache("fast"), cache("cached"))), daemon=True)
other.start()
other.join(2)
finished = not other.is_alive()
release.set()
blocker.join(5)
assert finished, "calls for other keys waited on the slow func"
assert done == [("fast", "cached")]
"""},
        {"name": "Part 3: a failure reaches every waiter and caches nothing", "part": 3, "visibility": "unshown", "behavior": "concurrency.thread_safety",
         "failure_message": "When func raises, every thread waiting on that call must get the same exception, and nothing may be cached.",
         "code": r"""
import threading
release = threading.Event()
started = threading.Event()
runs = []
def flaky(x):
    runs.append(x)
    if len(runs) == 1:
        started.set()
        release.wait(5)
        raise ValueError("boom")
    return x

cache = {fn}(flaky, capacity=4)
outcomes = []
def caller():
    try:
        outcomes.append(cache(7))
    except ValueError as error:
        outcomes.append(str(error))
pool = [threading.Thread(target=caller, daemon=True) for _ in range(4)]
pool[0].start()
assert started.wait(5)
for t in pool[1:]:
    t.start()
release.set()
for t in pool:
    t.join(5)
assert not any(t.is_alive() for t in pool), "a waiter never returned after func raised"
assert outcomes == ["boom"] * 4, outcomes
assert cache(7) == 7 and runs == [7, 7], "the failed call must not be cached"
"""},
        {"name": "Part 3: concurrent mixed keys stay consistent", "part": 3, "visibility": "unshown", "behavior": "concurrency.thread_safety",
         "failure_message": "Under concurrent calls the cache returned a wrong value, raised, or exceeded capacity; guard every update of the shared state with a lock.",
         "code": r"""
import random, threading, time

class SlowKey(int):
    # Hashing yields the thread, so unguarded cache updates interleave.
    def __hash__(self):
        time.sleep(random.random() * 0.0003)
        return int.__hash__(self)

cache = {fn}(lambda k: int(k) * 3, capacity=5)
errors = []
def worker(seed):
    rng = random.Random(seed)
    try:
        for _ in range(60):
            k = rng.randrange(12)
            assert cache(SlowKey(k)) == k * 3
    except Exception as error:
        errors.append(error)
pool = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(8)]
for t in pool: t.start()
for t in pool: t.join(20)
assert not errors, errors[:3]
assert len(cache.cache) <= 5, f"cache holds {len(cache.cache)} entries for capacity 5"
"""},
    ],
    "solution": r'''import json
import os
import struct
import threading
import zlib
from collections import OrderedDict

_HEADER = struct.Struct(">II")  # payload length, crc32 of the payload


def _encode(value):
    # JSON has no tuple, so tag containers to bring keys back exactly.
    if isinstance(value, tuple):
        return {"t": [_encode(v) for v in value]}
    if isinstance(value, list):
        return {"l": [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {"d": {k: _encode(v) for k, v in value.items()}}
    return value


def _decode(value):
    if isinstance(value, dict):
        (tag, inner), = value.items()
        if tag == "t":
            return tuple(_decode(v) for v in inner)
        if tag == "l":
            return [_decode(v) for v in inner]
        return {k: _decode(v) for k, v in inner.items()}
    return value


class _Flight:
    def __init__(self):
        self.done = threading.Event()
        self.result = None
        self.error = None


class Memoize:
    def __init__(self, func, capacity, path=None):
        self.func = func
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.path = path
        self._lock = threading.Lock()
        self._flights = {}
        self._log = None
        if path is not None:
            self._recover()
            self._log = open(path, "ab")

    def generate_key(self, *args, **kwargs):
        return (args, tuple(sorted(kwargs.items())))

    def __call__(self, *args, **kwargs):
        key = self.generate_key(*args, **kwargs)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                self._append({"op": "touch", "key": _encode(key)})
                return self.cache[key]
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                flight = self._flights[key] = _Flight()
                self.misses += 1
        if not leader:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            return flight.result
        try:
            result = self.func(*args, **kwargs)
        except BaseException as error:
            with self._lock:
                del self._flights[key]
            flight.error = error
            flight.done.set()
            raise
        with self._lock:
            self._put(key, result)
            self._append({"op": "put", "key": _encode(key), "value": _encode(result)})
            del self._flights[key]
        flight.result = result
        flight.done.set()
        return result

    def compact(self):
        if self.path is None:
            return
        with self._lock:
            tmp = self.path + ".compact"
            with open(tmp, "wb") as out:
                for key, value in self.cache.items():
                    out.write(self._frame({"op": "put", "key": _encode(key), "value": _encode(value)}))
                out.flush()
                os.fsync(out.fileno())
            self._log.close()
            os.replace(tmp, self.path)
            self._log = open(self.path, "ab")

    def _put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    @staticmethod
    def _frame(record):
        payload = json.dumps(record, separators=(",", ":")).encode()
        return _HEADER.pack(len(payload), zlib.crc32(payload)) + payload

    def _append(self, record):
        if self._log is None:
            return
        self._log.write(self._frame(record))
        self._log.flush()
        os.fsync(self._log.fileno())

    def _recover(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            data = fh.read()
        offset = 0
        while offset + _HEADER.size <= len(data):
            length, crc = _HEADER.unpack_from(data, offset)
            payload = data[offset + _HEADER.size: offset + _HEADER.size + length]
            if len(payload) < length or zlib.crc32(payload) != crc:
                break
            try:
                record = json.loads(payload)
                key = _decode(record["key"])
            except (ValueError, KeyError, TypeError):
                break
            if record["op"] == "put":
                self._put(key, _decode(record["value"]))
            elif key in self.cache:
                self.cache.move_to_end(key)
            offset += _HEADER.size + length
        if offset < len(data):
            with open(self.path, "r+b") as fh:
                fh.truncate(offset)
''',
    "interview_questions": interview(
        concept=[
            "What does an LRU cache evict, and why must a hit count as a use?",
            "What two properties must a memoization key have, and which calls should share one?",
        ],
        deep_dive=[
            "Walk through the key the starter builds for f(1, 2), f((1, 2)) and f(1, b=2). Which pairs collide, and how does your key keep them apart?",
        ],
        tradeoffs=[
            "Why an append-only log instead of rewriting the cache to disk on every call, and what does compaction buy back?",
            "How does recovery tell a torn final record from a good one, and why must later writes start after the last good record?",
            "Compare one lock around the whole call with a lock around the bookkeeping plus single-flight. What does each cost?",
        ],
    ),
}
