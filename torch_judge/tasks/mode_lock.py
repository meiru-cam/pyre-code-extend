"""A lock shared by any number of holders in one mode, then served fairly in arrival order, then with timeouts."""

from ._interview import interview

# A rig that drives named threads through an exact sequence of arrivals and releases.
_RIG = r"""
import threading, time, random

def wait_until(pred, what="the lock to reach the expected state", limit=10.0):
    end = time.monotonic() + limit
    while not pred():
        if time.monotonic() > end:
            raise AssertionError(f"timed out waiting for {what} (a deadlock or a missed wake-up?)")
        time.sleep(0.0005)

class Rig:
    def __init__(self, lock):
        self.lock, self.workers = lock, {}
    def arrive(self, name, mode, timeout=None):
        w = {"mode": mode, "got": threading.Event(), "go": threading.Event(), "done": threading.Event(), "result": [], "error": []}
        self.workers[name] = w
        threading.Thread(target=self._run, args=(w, timeout), daemon=True).start()
        wait_until(lambda: w["got"].is_set() or w["result"] or self._parked(w), f"{name} to be granted or to block")
    def _parked(self, w):
        others = sum(1 for v in self.workers.values() if v is not w and not v["got"].is_set() and not v["result"])
        return self.lock.waiting_count() > others
    def _run(self, w, timeout):
        try:
            ok = self.lock.acquire(w["mode"]) if timeout is None else self.lock.acquire(w["mode"], timeout)
        except Exception as e:
            w["error"].append(e)
            w["result"].append(None)
            return
        if timeout is not None and not ok:
            w["result"].append(False)
            return
        w["got"].set()
        w["go"].wait()
        try:
            self.lock.release(w["mode"])
        except Exception as e:
            w["error"].append(e)
        w["done"].set()
    def release(self, name):
        w = self.workers[name]
        w["go"].set()
        wait_until(lambda: w["done"].is_set(), f"{name} to release")
        assert not w["error"], f"{name}: {w['error'][0]!r}"
    def holders(self):
        return {n for n, w in self.workers.items() if w["got"].is_set() and not w["done"].is_set()}
    def expect(self, names):
        wait_until(lambda: self.holders() == set(names), f"holders {sorted(names)}")
        time.sleep(0.01)
        assert self.holders() == set(names), (sorted(self.holders()), sorted(names))

def raises(name, call):
    try:
        call()
    except Exception as e:
        assert type(e).__name__ == name, f"expected {name}, got {type(e).__name__}"
        return
    raise AssertionError(f"expected {name}")

def stress(lock, threads=12, rounds=40, modes=("p", "q", "r")):
    seen, guard, errors = {}, threading.Lock(), []
    def worker(k):
        rng = random.Random(k)
        for _ in range(rounds):
            mode = rng.choice(modes)
            lock.acquire(mode)
            with guard:
                seen[mode] = seen.get(mode, 0) + 1
                if len(seen) > 1:
                    errors.append(dict(seen))
            time.sleep(rng.random() * 0.0005)
            with guard:
                seen[mode] -= 1
                if not seen[mode]:
                    del seen[mode]
            lock.release(mode)
    ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
        assert not t.is_alive(), "a thread never finished: deadlock or missed wake-up"
    assert not errors, f"two modes held at once: {errors[0]}"
    assert lock.waiting_count() == 0
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "scheduler.concurrency", "code": _RIG + r"""
lock = {fn}()
rig = Rig(lock)
rig.arrive("a", "ingest"); rig.expect({"a"})
rig.arrive("b", "query")
rig.arrive("c", "query")
assert lock.waiting_count() == 2
rig.arrive("d", "ingest"); rig.expect({"a", "d"})
rig.release("a"); rig.expect({"d"})
raises("ModeMismatchError", lambda: lock.release("query"))
rig.release("d"); rig.expect({"b", "c"})
assert lock.waiting_count() == 0
rig.release("b"); rig.release("c")
"""},
    {"name": "Part 1: waking, counting and bad releases", "part": 1, "visibility": "unshown", "behavior": "concurrency.thread_safety",
     "failure_message": "When the last holder releases, every waiter of one mode must be granted together; waiting_count counts blocked calls across modes; release on an idle lock or in a mode not held raises ModeMismatchError and changes nothing; the same mode stays open to newcomers while others wait.",
     "code": _RIG + r"""
lock = {fn}()
rig = Rig(lock)
rig.arrive("h", "x")
for i in range(4):
    rig.arrive(f"y{i}", "y")
rig.arrive("z", "z")
assert lock.waiting_count() == 5
raises("ModeMismatchError", lambda: lock.release("y"))
rig.arrive("h2", "x"); rig.expect({"h", "h2"})
rig.release("h"); rig.release("h2")
wait_until(lambda: rig.holders() in ({"y0", "y1", "y2", "y3"}, {"z"}), "one whole mode to be granted")
group = rig.holders()
assert lock.waiting_count() == 5 - len(group)
for name in sorted(group):
    rig.release(name)
rest = {"z"} if group != {"z"} else {"y0", "y1", "y2", "y3"}
rig.expect(rest)
for name in sorted(rest):
    rig.release(name)
assert lock.waiting_count() == 0
raises("ModeMismatchError", lambda: lock.release("x"))
lock.acquire("solo"); lock.release("solo")
"""},
    {"name": "Part 1: many threads", "part": 1, "visibility": "unshown", "behavior": "concurrency.thread_safety",
     "failure_message": "With 12 threads acquiring and releasing random modes, two modes were held at once, a thread never finished, or waiting_count did not return to 0.",
     "code": _RIG + r"""
stress({fn}())
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "events.ordering", "code": _RIG + r"""
lock = {fn}(fair=True)
rig = Rig(lock)
rig.arrive("a", "query"); rig.expect({"a"})
rig.arrive("b", "ingest")
rig.arrive("c", "ingest")
rig.arrive("d", "query")
assert lock.waiting_count() == 3
rig.release("a"); rig.expect({"b", "c"})
rig.arrive("e", "ingest")
raises("ModeMismatchError", lambda: lock.release("query"))
rig.release("b"); rig.release("c"); rig.expect({"d"})
rig.release("d"); rig.expect({"e"})
rig.release("e")
assert lock.waiting_count() == 0
"""},
    {"name": "Part 2: only the last batch can be joined", "part": 2, "visibility": "unshown", "behavior": "events.ordering",
     "failure_message": "A fair lock lets a call join only the last batch in the queue, running or not; a batch is granted as a whole when it reaches the front and leaves once every member released; release raises ModeMismatchError unless its mode is the front batch's and that batch has a holder; fair=False keeps Part 1.",
     "code": _RIG + r"""
lock = {fn}(fair=True)
rig = Rig(lock)
rig.arrive("a", "x")
rig.arrive("b", "y")
rig.arrive("c", "y")
rig.arrive("d", "z")
rig.arrive("e", "y")
rig.release("a"); rig.expect({"b", "c"})
rig.arrive("f", "y")
rig.expect({"b", "c"})
raises("ModeMismatchError", lambda: lock.release("z"))
rig.release("b"); rig.release("c"); rig.expect({"d"})
rig.release("d"); rig.expect({"e", "f"})
rig.arrive("g", "y"); rig.expect({"e", "f", "g"})
for name in "efg":
    rig.release(name)
raises("ModeMismatchError", lambda: lock.release("y"))
rig.arrive("h", "x"); rig.expect({"h"})
for i in range(5):
    rig.arrive(f"w{i}", "w")
rig.release("h"); rig.expect({f"w{i}" for i in range(5)})
for i in range(5):
    rig.release(f"w{i}")
plain = {fn}()
pr = Rig(plain)
pr.arrive("a", "x"); pr.arrive("b", "y"); pr.arrive("c", "x"); pr.expect({"a", "c"})
pr.release("a"); pr.release("c"); pr.expect({"b"}); pr.release("b")
"""},
    {"name": "Part 2: many threads, fairly", "part": 2, "visibility": "unshown", "behavior": "concurrency.thread_safety",
     "failure_message": "With 12 threads on a fair lock, two modes were held at once, a thread never finished (wake every member of a batch, not one), or waiting_count did not return to 0.",
     "code": _RIG + r"""
stress({fn}(fair=True))
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "budget.enforcement", "code": _RIG + r"""
lock = {fn}(fair=True)
rig = Rig(lock)
rig.arrive("a", "query"); rig.expect({"a"})
assert lock.acquire("ingest", timeout=0) is False
assert lock.waiting_count() == 0
assert lock.acquire("query", timeout=0) is True
lock.release("query")
rig.arrive("b", "ingest")
rig.arrive("c", "query")
rig.arrive("d", "ingest", timeout=0.05)
wait_until(lambda: rig.workers["d"]["result"] == [False], "d to give up")
rig.release("a"); rig.expect({"b"})
rig.release("b"); rig.expect({"c"})
rig.release("c")
assert lock.acquire("any") is True
lock.release("any")
"""},
    {"name": "Part 3: giving up leaves no trace", "part": 3, "visibility": "unshown", "behavior": "retry.classification",
     "failure_message": "A call that times out returns False and leaves no trace; an emptied batch is removed, and if its neighbours share a mode they become one batch; a batch with other members stays; a timeout long enough to be granted returns True; an unfair lock times out the same way.",
     "code": _RIG + r"""
lock = {fn}(fair=True)
rig = Rig(lock)
rig.arrive("a", "x")
rig.arrive("b", "y", timeout=0.05)
rig.arrive("c", "x")
rig.expect({"a"})
wait_until(lambda: rig.workers["b"]["result"] == [False], "b to give up")
rig.expect({"a", "c"})
rig.arrive("d", "y")
rig.arrive("e", "y", timeout=0.05)
wait_until(lambda: rig.workers["e"]["result"] == [False], "e to give up")
assert lock.waiting_count() == 1
rig.arrive("f", "x")
rig.release("a"); rig.release("c"); rig.expect({"d"})
rig.release("d"); rig.expect({"f"})
rig.arrive("g", "z", timeout=5.0)
rig.release("f"); rig.expect({"g"})
assert rig.workers["g"]["result"] == []
rig.release("g")
plain = {fn}()
pr = Rig(plain)
pr.arrive("a", "x")
assert plain.acquire("y", 0) is False and plain.acquire("x", 0) is True
plain.release("x")
pr.arrive("b", "y", timeout=0.05)
wait_until(lambda: pr.workers["b"]["result"] == [False], "b to give up")
assert plain.waiting_count() == 0
pr.release("a")
assert plain.acquire("y") is True
plain.release("y")
"""},
]

TASK = {
    "title": "Mode Lock",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ModeLock",
    "description_en": r"""Build `ModeLock`, a lock that any number of threads may hold together as long as they all use the same mode, then serve it fairly in arrival order, then let a waiting call give up.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ModeLock` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A mode is a string compared only by equality. At any instant every holder of the lock uses the same mode.
- `acquire(mode)` blocks until the caller holds the lock in `mode`. `release(mode)` gives back one hold in `mode`.
- `release(mode)` raises `ModeMismatchError` and changes nothing when nobody holds the lock in `mode`. Define the error class yourself; it is recognised by name.
- `waiting_count()` returns how many `acquire` calls are blocked right now, across every mode.
- No busy waiting: blocked calls sleep on a condition and wake when something changes.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it tests condition variables beyond the textbook mutex: rechecking after a wake-up, waking a whole group, and keeping counts consistent. Each later part adds one requirement: fairness, so one busy mode cannot starve the others, then timeouts that leave the queue as if the call never came.

**Where it is used:** search indexes that serve many queries or many ingest writers but never both at once, shared/exclusive table locks in databases, and phase barriers in batch pipelines.

Adapted from the ModalLock question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as one class renamed `ModeLock` instead of `ModalLock` and `FairModalLock`: fairness is the constructor flag `fair`. Part 3 extends one of the source's follow-ups, timeouts; `acquire` returning a bool, `timeout=0` and the merging of same-mode neighbours are new.""",
    "parts": [
        {
            "title": "Shared holds by mode",
            "description_en": r"""**Signatures:** `ModeLock()`, `acquire(mode)`, `release(mode)`, `waiting_count() -> int`, and `class ModeMismatchError(Exception)`.

- `acquire(mode)` returns at once when the lock is idle or already held in `mode`, even if other calls are waiting.
- When the last holder releases, the lock is idle, and the waiters of one mode are all granted together.

**Example:** threads arrive one after another and each step finishes before the next:
- `a` holds `"ingest"`; `b` and `c` ask for `"query"` and block, so `waiting_count()` is `2`
- `d` asks for `"ingest"` and holds next to `a`, though others are waiting
- `a` releases; while `d` still holds, `release("query")` raises `ModeMismatchError`
- `d` releases, and `b` and `c` hold together""",
        },
        {
            "title": "Fair order",
            "description_en": r"""Keep Part 1. `ModeLock(fair=True)` serves modes in arrival order; `ModeLock()` and `fair=False` keep Part 1.

- Calls form batches: a call joins the last batch in the queue if it has the same mode, and otherwise starts a new batch at the end.
- The batch at the front holds the lock, and every member of it holds together. A call that joins the front batch holds at once.
- The front batch leaves once all its members have released, and the next batch is granted.
- `release(mode)` raises `ModeMismatchError` unless the front batch has `mode` and someone in it still holds.

**Example:** `a` holds `"query"`; `b` asks for `"ingest"` and waits, and `c` asks for `"ingest"` and joins `b`'s batch:
- `d` asks for `"query"` and waits: the last batch is `"ingest"`, so `d` cannot join `a`
- `a` releases, so `b` and `c` hold together; `e` asks for `"ingest"` and queues behind `d`
- while `b` and `c` hold, `release("query")` raises `ModeMismatchError`; the next grants are `{d}`, then `{e}`""",
        },
        {
            "title": "Giving up",
            "description_en": r"""Keep Parts 1–2. `acquire(mode, timeout=None) -> bool` now returns `True` once the lock is held.

- With a `timeout` in seconds, it returns `False` if the lock is not held by then. `timeout=0` only checks, never waits.
- A call that gives up holds nothing and no longer counts as waiting.
- In a fair lock, a batch that loses its last member to a timeout leaves the queue. If the batches on either side of it have the same mode, they become one batch, so waiters in the later one may hold at once.

**Example:** with `fair=True`, `a` holds `"query"`:
- `acquire("ingest", timeout=0)` is `False`, and `acquire("query", timeout=0)` is `True` since nobody waits
- `b` asks for `"ingest"`, `c` for `"query"`, and `d` for `"ingest"` with `timeout=0.05`; `d` returns `False`
- releases then grant `{b}` and `{c}`, as if `d` had never come""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does a blocked acquire need to see before it may hold the lock, and what must a waiter check again when it wakes, given that other threads may have run first? When the last holder releases, how many waiters should wake?"},
        {"level": 2, "kind": "analysis", "content": "Use one threading.Condition. In acquire, add yourself to a blocked counter, loop on wait() until nobody holds the lock or the holders use your mode, then leave the counter, store the mode and count one more holder. In release, refuse a mode nobody holds before touching anything; the last release clears the mode and wakes every waiter."},
    ],
    "model_connections": [
        "An embedding index must pause lookups while ingest jobs rebuild it, yet many lookups, or many ingest shards, may run together.",
        "Serving systems batch requests of the same kind, such as one adapter or one model, and fairness decides how long other kinds wait.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Any number of same-mode holders run together, which an ordinary mutex cannot do.",
            "Fair batches bound the wait of every mode by the batches already ahead of it.",
            "Timeouts let a caller give up without leaving a ghost batch in the queue.",
        ],
        "cons": [
            "The unfair lock starves a mode while the current mode keeps gaining holders.",
            "Fairness costs throughput: a newcomer of the running mode waits behind other modes.",
            "notify_all wakes every waiter, most of whom go back to sleep, which costs CPU with many waiters.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import threading
from collections import deque


class ModeMismatchError(Exception):
    pass


class _Batch:
    def __init__(self, mode):
        self.mode = mode
        self.waiting = 0   # members still blocked in acquire
        self.active = 0    # members holding the lock
        self.into = None   # the batch this one was merged into, if any

    def root(self):
        batch = self
        while batch.into is not None:
            batch = batch.into
        return batch


class ModeLock:
    def __init__(self, fair=False):
        self._fair = fair
        self._cond = threading.Condition()
        self._mode = None      # unfair: the mode held, or None when idle
        self._holders = 0
        self._waiting = 0
        self._queue = deque()  # fair: batches in arrival order; the front one holds the lock

    def acquire(self, mode, timeout=None):
        with self._cond:
            if self._fair:
                return self._acquire_fair(mode, timeout)
            self._waiting += 1
            try:
                # wait_for rechecks after every wake-up: another mode may have taken the lock first
                granted = self._cond.wait_for(lambda: self._mode in (None, mode), timeout)
            finally:
                self._waiting -= 1
            if not granted:
                return False
            self._mode = mode
            self._holders += 1
            return True

    def _acquire_fair(self, mode, timeout):
        if self._queue and self._queue[-1].mode == mode:
            batch = self._queue[-1]  # only the last batch may be joined, so nobody cuts in line
        else:
            batch = _Batch(mode)
            self._queue.append(batch)
        batch.waiting += 1
        granted = self._cond.wait_for(lambda: self._queue[0] is batch.root(), timeout)
        batch = batch.root()
        batch.waiting -= 1
        if granted:
            batch.active += 1
            return True
        if batch.waiting == 0 and batch.active == 0:
            self._drop(batch)
        return False

    def _drop(self, batch):
        """Remove an empty batch that gave up; neighbours of the same mode then become one batch."""
        i = self._queue.index(batch)
        del self._queue[i]
        if 0 < i < len(self._queue) and self._queue[i - 1].mode == self._queue[i].mode:
            left, right = self._queue[i - 1], self._queue[i]
            left.waiting += right.waiting
            left.active += right.active
            right.into = left
            del self._queue[i]
        self._cond.notify_all()  # the merged waiters may now be at the front

    def release(self, mode):
        with self._cond:
            if self._fair:
                batch = self._queue[0] if self._queue else None
                if batch is None or batch.mode != mode or batch.active == 0:
                    raise ModeMismatchError(mode)
                batch.active -= 1
                if batch.active == 0 and batch.waiting == 0:  # woken members may not have left wait yet
                    self._queue.popleft()
                    self._cond.notify_all()
                return
            if self._holders == 0 or self._mode != mode:
                raise ModeMismatchError(mode)
            self._holders -= 1
            if self._holders == 0:
                self._mode = None
                self._cond.notify_all()  # every waiter of the next mode may join together

    def waiting_count(self):
        with self._cond:
            if self._fair:
                return sum(batch.waiting for batch in self._queue)
            return self._waiting
''',
    "interview_questions": interview(
        concept=[
            "Why must a woken waiter check the mode again in a loop instead of assuming it may now hold?",
            "Why does the last release call notify_all instead of notify?",
        ],
        deep_dive=[
            "How do you keep waiting_count exact, including calls that are granted without ever blocking?",
        ],
        tradeoffs=[
            "How can the unfair lock starve a mode, and why does joining only the last batch prevent it?",
            "Why must a fair batch stay at the front until both its holders and its woken-but-not-yet-running members are gone?",
            "When a timed-out batch leaves the queue, why merge its neighbours, and what would go wrong without it?",
            "How would you build a shared/exclusive lock on top of this, and what must change for the exclusive mode?",
        ],
    ),
}
