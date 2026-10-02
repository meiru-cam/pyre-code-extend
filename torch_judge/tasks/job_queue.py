"""A work queue with reservation tokens, then lease timeouts on a manual clock, then a retry budget with a dead-letter list."""

from ._interview import interview

# A naive model that rescans every task on each call, a clock, and a replay that maps the learner's tokens to the model's.
_MODEL = r"""
import random, time

class Clock:
    def __init__(self, start=0):
        self.t = start
    def now(self):
        return self.t
    def advance(self, dt):
        self.t += dt

class Model:
    def __init__(self, clock=None, lease=None, max_attempts=None):
        self.clock, self.lease, self.max = clock, lease, max_attempts
        self.tasks = {}   # id -> dict(state, token, deadline, attempts, payload)
        self.ready = []
        self.dead = []
        self.count = 0

    def reclaim(self):
        if self.clock is None:
            return
        now = self.clock.now()
        gone = sorted((t["deadline"], tid) for tid, t in self.tasks.items() if t["state"] == "reserved" and t["deadline"] < now)
        for _, tid in gone:
            self.release(tid)

    def release(self, tid):
        t = self.tasks[tid]
        t["token"] = None
        if self.max is not None and t["attempts"] >= self.max:
            t["state"] = "dead"
            self.dead.append(tid)
        else:
            t["state"] = "ready"
            self.ready.append(tid)

    def submit(self, tid, payload):
        self.reclaim()
        if tid in self.tasks:
            raise ValueError(tid)
        self.tasks[tid] = {"state": "ready", "token": None, "deadline": None, "attempts": 0, "payload": payload}
        self.ready.append(tid)

    def reserve(self, worker):
        self.reclaim()
        if not self.ready:
            return None
        tid = self.ready.pop(0)
        t = self.tasks[tid]
        self.count += 1
        t["state"], t["token"], t["attempts"] = "reserved", self.count, t["attempts"] + 1
        t["deadline"] = None if self.clock is None else self.clock.now() + self.lease
        return (tid, self.count, t["payload"])

    def held(self, tid, token):
        self.reclaim()
        t = self.tasks.get(tid)
        if t is None:
            raise KeyError("UnknownTaskError")
        if t["state"] != "reserved" or t["token"] != token:
            raise KeyError("InvalidReservationError")

    def complete(self, tid, token):
        self.held(tid, token)
        self.tasks[tid]["state"], self.tasks[tid]["token"] = "completed", None

    def fail(self, tid, token):
        self.held(tid, token)
        self.release(tid)

    def dead_letters(self):
        self.reclaim()
        return list(self.dead)

    def requeue_dead(self, tid):
        self.reclaim()
        t = self.tasks.get(tid)
        if t is None:
            raise KeyError("UnknownTaskError")
        if t["state"] != "dead":
            raise ValueError(tid)
        self.dead.remove(tid)
        t["state"], t["attempts"] = "ready", 0
        self.ready.append(tid)

def outcome(call):
    try:
        return ("ok", call())
    except KeyError as e:
        return ("raise", e.args[0])
    except Exception as e:
        return ("raise", type(e).__name__)

IDS = ["a", "b", "c", "d", "e"]

def random_ops(rng, steps, part):
    ops = []
    for _ in range(steps):
        r = rng.random()
        if r < 0.2:
            ops.append(("submit", rng.choice(IDS)))
        elif r < 0.45:
            ops.append(("reserve",))
        elif r < 0.7:
            ops.append((rng.choice(["complete", "fail"]), rng.choice(IDS + ["zz"]), rng.random()))
        elif r < 0.85 and part >= 2:
            ops.append(("advance", rng.choice([0, 1, 2, 3, 5])))
        elif part >= 3 and r < 0.92:
            ops.append(("dead_letters",))
        elif part >= 3:
            ops.append(("requeue_dead", rng.choice(IDS + ["zz"])))
        else:
            ops.append(("reserve",))
    return ops

def replay(make, ops, part, label, lease=3, max_attempts=2):
    clock = Clock() if part >= 2 else None
    args = () if part == 1 else (clock, lease) if part == 2 else (clock, lease, max_attempts)
    q, m = make(*args), Model(*args)
    seen = {}  # task id -> list of (learner token, model token), newest last
    for i, op in enumerate(ops):
        name = op[0]
        if name == "advance":
            clock.advance(op[1])
            continue
        if name == "submit":
            got, want = outcome(lambda: q.submit(op[1], op[1].upper())), outcome(lambda: m.submit(op[1], op[1].upper()))
        elif name == "reserve":
            got, want = outcome(lambda: q.reserve("w")), outcome(lambda: m.reserve("w"))
            if got[0] == "ok" and want[0] == "ok" and got[1] is not None and want[1] is not None:
                seen.setdefault(want[1][0], []).append((got[1][1], want[1][1]))
                got, want = ("ok", (got[1][0], got[1][2])), ("ok", (want[1][0], want[1][2]))
        elif name in ("complete", "fail"):
            pairs = seen.get(op[1], [])
            if pairs and op[2] < 0.85:
                mine, theirs = pairs[min(len(pairs) - 1, int(op[2] * 2 * len(pairs)) % len(pairs))] if op[2] < 0.3 else pairs[-1]
            else:
                mine, theirs = object(), -1
            got = outcome(lambda: getattr(q, name)(op[1], mine))
            want = outcome(lambda: getattr(m, name)(op[1], theirs))
        else:
            got = outcome(lambda: getattr(q, name)(*op[1:]))
            want = outcome(lambda: getattr(m, name)(*op[1:]))
        assert got == want, (label, i, op, got, want, ops[:i])
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
def kind(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__
    return None
q = {fn}()
q.submit("j1", "resize")
q.submit("j2", "encode")
q.submit("j3", "upload")
a = q.reserve("w1")
b = q.reserve("w1")
assert (a[0], a[2]) == ("j1", "resize") and b[0] == "j2"
assert q.fail("j1", a[1]) is None
assert q.reserve("w2")[0] == "j3"
c = q.reserve("w2")
assert c[0] == "j1" and c[1] != a[1]
assert kind(lambda: q.complete("j1", b[1])) == "InvalidReservationError"
assert kind(lambda: q.complete("j9", c[1])) == "UnknownTaskError"
assert kind(lambda: q.submit("j2", "again")) == "ValueError"
assert q.complete("j2", b[1]) is None
assert q.fail("j1", c[1]) is None
assert kind(lambda: q.complete("j1", c[1])) == "InvalidReservationError"
assert q.reserve("w3")[0] == "j1"
"""},
    {"name": "Part 1: tokens and states", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "Every reservation gets a token no other reservation shares; another task's token, an old token, a token after complete and any token for a ready or completed task raise InvalidReservationError; an unknown id raises UnknownTaskError first; payloads come back unchanged; ready tasks keep first-in, first-out order.",
     "code": _MODEL + r"""
q = {fn}()
payload = {"rows": [1, 2]}
q.submit("x", payload)
q.submit("y", None)
rx = q.reserve("w")
assert rx[2] is payload
ry = q.reserve("w")
assert rx[1] != ry[1]
assert outcome(lambda: q.complete("x", ry[1])) == ("raise", "InvalidReservationError")
assert outcome(lambda: q.fail("y", rx[1])) == ("raise", "InvalidReservationError")
q.complete("x", rx[1])
assert outcome(lambda: q.complete("x", rx[1])) == ("raise", "InvalidReservationError")
assert outcome(lambda: q.fail("x", rx[1])) == ("raise", "InvalidReservationError")
q.fail("y", ry[1])
assert outcome(lambda: q.fail("y", ry[1])) == ("raise", "InvalidReservationError"), "y is ready again"
assert outcome(lambda: q.fail("ghost", ry[1])) == ("raise", "UnknownTaskError")
tokens = set()
for i in range(50):
    q.submit(f"t{i}", i)
order = []
while True:
    r = q.reserve("w")
    if r is None:
        break
    order.append(r[0])
    assert r[1] not in tokens
    tokens.add(r[1])
    if r[0] == "y" or int(r[0][1:]) % 3:
        q.complete(r[0], r[1])
    else:
        q.fail(r[0], r[1])
        r2 = q.reserve("w")
        tokens.add(r2[1])
        q.complete(r2[0], r2[1])
        order.append(r2[0])
assert order[:2] == ["y", "t0"] and order[2] == "t1", order[:4]
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random sequences of submit, reserve, complete and fail with fresh, stale and foreign tokens, a result or error differed from a model that keeps one first-in, first-out ready list.",
     "code": _MODEL + r"""
for seed in range(300):
    replay({fn}, random_ops(random.Random(seed), 60, 1), 1, seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "retry.classification", "code": r"""
class Clock:
    def __init__(self):
        self.t = 0
    def now(self):
        return self.t
    def advance(self, dt):
        self.t += dt
def kind(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__
    return None
clock = Clock()
q = {fn}(clock, 10)
q.submit("j1", "resize")
a = q.reserve("w1")
clock.advance(10)
assert q.complete("j1", a[1]) is None, "a lease is valid through its deadline"
q.submit("j2", "encode")
b = q.reserve("w1")
clock.advance(11)
q.submit("j3", "upload")
c = q.reserve("w2")
assert c[0] == "j2", "j2 was reclaimed when submit was called, ahead of j3"
assert kind(lambda: q.complete("j2", b[1])) == "InvalidReservationError"
assert q.reserve("w2")[0] == "j3"
"""},
    {"name": "Part 2: reclaiming order and timing", "part": 2, "visibility": "unshown", "behavior": "retry.classification",
     "failure_message": "Every call, submit included, first reclaims leases with deadline < now, ordered by deadline and then task id; a reclaimed reservation's token stops working; a lease expiring during complete or fail makes that call raise InvalidReservationError; nothing happens between calls.",
     "code": _MODEL + r"""
clock = Clock()
q = {fn}(clock, 5)
for tid in ["m", "k", "z"]:
    q.submit(tid, tid)
rm = q.reserve("w")
rk = q.reserve("w")
clock.advance(1)
rz = q.reserve("w")
clock.advance(5)
assert outcome(lambda: q.fail("m", rm[1])) == ("raise", "InvalidReservationError")
assert [q.reserve("w")[0] for _ in range(2)] == ["k", "m"], "same deadline: by task id"
assert outcome(lambda: q.complete("z", rz[1])) == ("ok", None)
clock.advance(100)
q2 = {fn}(clock, 1)
q2.submit("p", 1)
rp = q2.reserve("w")
clock.advance(1)
assert outcome(lambda: q2.complete("p", rp[1])) == ("ok", None)
q2.submit("s", 2)
rs = q2.reserve("w")
clock.advance(2)
assert outcome(lambda: q2.complete("s", rs[1])) == ("raise", "InvalidReservationError")
assert q2.reserve("w")[0] == "s"
"""},
    {"name": "Part 2: random calls with a clock", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random sequences with clock advances, a result or error differed from a model that reclaims every expired lease at the start of each call.",
     "code": _MODEL + r"""
for seed in range(300):
    replay({fn}, random_ops(random.Random(100 + seed), 70, 2), 2, seed)
"""},
    {"name": "Part 2: many open leases", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "With 30,000 open leases, 60,000 further calls took too long: keep leases in a heap by deadline so each call only looks at the ones that have expired.",
     "code": _MODEL + r"""
clock = Clock()
q = {fn}(clock, 10**6)
for i in range(30000):
    q.submit(f"t{i}", i)
for i in range(30000):
    q.reserve("w")
start = time.perf_counter()
for i in range(60000):
    assert q.reserve("w") is None
elapsed = time.perf_counter() - start
assert elapsed < 4.0, f"{elapsed:.2f}s for 60,000 calls with 30,000 open leases"
clock.advance(10**6 + 1)
assert q.reserve("w")[0] == "t0"
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "retry.backoff", "code": r"""
class Clock:
    def __init__(self):
        self.t = 0
    def now(self):
        return self.t
    def advance(self, dt):
        self.t += dt
def kind(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__
    return None
clock = Clock()
q = {fn}(clock, 6, 3)
q.submit("j1", "resize")
q.submit("j2", "encode")
a = q.reserve("w1")
q.fail("j1", a[1])
b = q.reserve("w1")
assert b[0] == "j2"
q.complete("j2", b[1])
c = q.reserve("w1")
q.fail("j1", c[1])
assert q.dead_letters() == []
q.reserve("w2")
clock.advance(7)
assert q.dead_letters() == ["j1"], "the expired lease used up the third attempt"
q.submit("j3", "upload")
assert q.requeue_dead("j1") is None
assert [q.reserve("w3")[0], q.reserve("w3")[0]] == ["j3", "j1"]
assert kind(lambda: q.requeue_dead("j1")) == "ValueError"
assert kind(lambda: q.requeue_dead("j7")) == "UnknownTaskError"
"""},
    {"name": "Part 3: attempts, order of death and requeue", "part": 3, "visibility": "unshown", "behavior": "retry.backoff",
     "failure_message": "Attempts count reservations; a task dies when a failed or expired reservation brings its count to max_attempts; dead_letters lists tasks in the order they died and returns a new list; requeue_dead resets the count, puts the task at the back, and raises ValueError for a task that is not dead; a completed task never dies.",
     "code": _MODEL + r"""
clock = Clock()
q = {fn}(clock, 100, 1)
for tid in ["p", "q", "r"]:
    q.submit(tid, tid)
rp, rq, rr = q.reserve("w"), q.reserve("w"), q.reserve("w")
q.fail("q", rq[1])
q.complete("r", rr[1])
q.fail("p", rp[1])
dead = q.dead_letters()
assert dead == ["q", "p"]
dead.append("junk")
assert q.dead_letters() == ["q", "p"]
assert outcome(lambda: q.requeue_dead("r")) == ("raise", "ValueError")
q.submit("s", "s")
q.requeue_dead("p")
assert q.reserve("w")[0] == "s" and q.reserve("w")[0] == "p"
q3 = {fn}(clock, 2, 3)
q3.submit("x", 0)
for attempt in range(3):
    r = q3.reserve("w")
    assert r is not None, attempt
    clock.advance(3)
assert q3.dead_letters() == ["x"]
"""},
    {"name": "Part 3: random calls with retries", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random sequences with a retry budget, a result or error differed from a model that sends a task to the dead list once a failed or expired reservation reaches max_attempts.",
     "code": _MODEL + r"""
for seed in range(300):
    replay({fn}, random_ops(random.Random(200 + seed), 80, 3), 3, seed)
"""},
]

TASK = {
    "title": "Fault-Tolerant Job Queue",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "JobQueue",
    "description_en": r"""Build `JobQueue`, which hands jobs to workers under reservation tokens, takes jobs back from workers whose lease runs out, and moves jobs that keep failing to a dead-letter list.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `JobQueue` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Each task has a caller-chosen `task_id`, never submitted twice, and a `payload` the queue stores and returns untouched.
- A task is `ready`, `reserved` by a worker, or `completed`, which is final. Part 3 adds `dead`.
- Ready tasks leave in the order they became ready, whether they were just submitted or came back.
- Errors are classes you define, recognised by name: `UnknownTaskError` and `InvalidReservationError`. `ValueError` is the built-in.
- Calls never overlap, so no locking is needed.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is the core of every job system, small enough to write in one sitting. Each later part adds one requirement: tokens stop a slow worker from finishing a job someone else now holds, leases recover from crashed workers, and the retry budget stops one bad job from cycling forever.

**Where it is used:** Amazon SQS visibility timeouts, Celery and RQ acknowledgements, Kubernetes job retries, and dead-letter queues in every message broker.

Adapted from the fault-tolerant work queue question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class renamed from `WorkQueue` to `JobQueue`. A reservation is a tuple instead of a named tuple, the source's size limits are dropped, Part 2 adds a speed check with 30,000 open leases, and the clock, lease and retry budget arrive as optional constructor arguments so that earlier parts keep working. The test clock is any object with `now()`.""",
    "parts": [
        {
            "title": "Reserve, complete and fail",
            "description_en": r"""**Signatures:**
- `JobQueue().submit(task_id, payload)` adds a ready task. It raises `ValueError` if the id was used before.
- `reserve(worker_id) -> tuple | None` takes the ready task that has waited longest, marks it reserved and returns `(task_id, token, payload)`. The token is a new value that no other reservation ever gets. It returns `None` when nothing is ready.
- `complete(task_id, token)` marks the task completed. `fail(task_id, token)` puts it back at the end of the ready tasks. Both return `None`.
- `complete` and `fail` raise `UnknownTaskError` for an id never submitted. Otherwise, unless the task is reserved under exactly this token, they raise `InvalidReservationError` and change nothing.

**Example:** submit `j1`, `j2` and `j3`, then `a = reserve("w1")` and `b = reserve("w1")`, which give `j1` and `j2`:
- `fail("j1", a[1])` sends `j1` behind `j3`, so the next two reserves give `j3`, then `j1` again as `c`, with a new token
- `complete("j1", b[1])` raises `InvalidReservationError`: `b` is a token for `j2`. `complete("j9", c[1])` raises `UnknownTaskError`, and `submit("j2", "again")` raises `ValueError`
- `complete("j2", b[1])` works; `fail("j1", c[1])` sends `j1` back once more, so `complete("j1", c[1])` now raises `InvalidReservationError`""",
        },
        {
            "title": "Leases on a clock",
            "description_en": r"""Keep Part 1. The constructor now also takes `JobQueue(clock, lease_duration)`; with no clock, nothing expires.

- `clock.now()` returns the current integer time. Read it only when a method is called.
- A reservation made at time `t` has the deadline `t + lease_duration`. It is still valid at the deadline and expires once `now()` is greater.
- Every call, `submit` included, first takes back every expired reservation, as if `fail` had been called on it, in order of deadline and then ascending task id. Only then does the call do its own work.
- A token whose reservation was taken back works no more than any other old token. 30,000 open leases must not slow other calls down.

**Example:** with `lease_duration = 10`, submit `j1` and `a = reserve("w1")` at time `0`:
- at time `10`, `complete("j1", a[1])` still works
- then submit `j2` and `b = reserve("w1")` at `10`, and move the clock to `21`: `submit("j3", ...)` first takes `j2` back, so the next `reserve` gives `j2`, ahead of `j3`
- `complete("j2", b[1])` now raises `InvalidReservationError`""",
        },
        {
            "title": "Retry budget and dead letters",
            "description_en": r"""Keep Parts 1–2. The constructor now also takes `JobQueue(clock, lease_duration, max_attempts)`; with none, retries are unlimited.

- Each task counts its reservations. When a reservation ends by `fail` or by expiry and the count has reached `max_attempts`, the task becomes `dead` instead of ready.
- `dead_letters() -> list[str]` returns a new list of the dead tasks' ids, in the order they died.
- `requeue_dead(task_id)` puts a dead task at the back of the ready tasks with its count back at `0`. It raises `UnknownTaskError` for an unknown id and `ValueError` for a task that is not dead.
- Both new methods also start by taking back expired reservations.

**Example:** with `lease_duration = 6` and `max_attempts = 3`, submit `j1` and `j2` at time `0`:
- reserve `j1` and fail it, reserve and complete `j2`, then reserve `j1` and fail it again
- reserve `j1` a third time and let the lease run out: at time `7`, `dead_letters()` is `["j1"]`
- submit `j3`, then `requeue_dead("j1")`: the next reserves give `j3`, then `j1`
- `requeue_dead("j1")` now raises `ValueError`, since `j1` is reserved, and `requeue_dead("j7")` raises `UnknownTaskError`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "A worker that failed a task still holds its old token. If the token were just the task id, how could complete tell that worker apart from the one holding the task now? Which structure gives you the longest-waiting ready task in O(1)?"},
        {"level": 2, "kind": "analysis", "content": "A counter that only goes up gives every reservation a token nobody else ever had. Store, per task id, whether it is ready, reserved or completed, plus the token of the reservation now in force. Ready ids wait in a deque: reserve takes from the left and fail appends on the right. Check the id first, then the state and token, and change nothing until both checks pass."},
    ],
    "model_connections": [
        "Distributed training and evaluation clusters hand shards to workers with leases, so a crashed worker's shard goes to someone else instead of being lost.",
        "Inference and data pipelines retry failed requests a bounded number of times and park poison inputs in a dead-letter list for inspection.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A fresh token per reservation makes a late reply from an old worker harmless.",
            "Reclaiming at the start of each call needs no background thread and stays deterministic in tests.",
            "A heap of deadlines with stale entries skipped makes reclaiming cheap however many leases are open.",
        ],
        "cons": [
            "Reclaiming only happens when someone calls the queue, so an idle queue holds expired leases.",
            "Stale heap entries stay until their deadline passes, so memory can grow with many quick completions.",
            "With unlimited retries a bad task can cycle forever, which is what Part 3 fixes.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import heapq
import itertools
from collections import deque


class UnknownTaskError(Exception):
    pass


class InvalidReservationError(Exception):
    pass


class _Task:
    def __init__(self, payload):
        self.payload = payload
        self.state = "ready"  # ready, reserved, completed or dead
        self.token = None     # the current reservation's token while reserved
        self.attempts = 0


class JobQueue:
    def __init__(self, clock=None, lease_duration=None, max_attempts=None):
        self._clock = clock
        self._lease = lease_duration
        self._max = max_attempts
        self._tasks = {}
        self._ready = deque()  # task ids in the order they became ready
        self._leases = []      # heap of (deadline, task_id, token); stale entries are skipped
        self._dead = {}        # task id -> None, in the order tasks died
        self._tokens = itertools.count(1)

    def _reclaim(self):
        if self._clock is None:
            return
        now = self._clock.now()
        while self._leases and self._leases[0][0] < now:  # a lease is valid through its deadline
            _, task_id, token = heapq.heappop(self._leases)
            task = self._tasks[task_id]
            if task.state == "reserved" and task.token == token:
                self._release(task_id, task)

    def _release(self, task_id, task):
        """End a reservation without success: back of the queue, or dead once out of attempts."""
        task.token = None
        if self._max is not None and task.attempts >= self._max:
            task.state = "dead"
            self._dead[task_id] = None
        else:
            task.state = "ready"
            self._ready.append(task_id)

    def _held(self, task_id, token):
        self._reclaim()  # before the check: this very lease may have just expired
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        if task.state != "reserved" or task.token != token:
            raise InvalidReservationError(task_id)
        return task

    def submit(self, task_id, payload):
        self._reclaim()  # reclaimed tasks queue ahead of this one
        if task_id in self._tasks:
            raise ValueError(f"task {task_id!r} was already submitted")
        self._tasks[task_id] = _Task(payload)
        self._ready.append(task_id)

    def reserve(self, worker_id):
        self._reclaim()
        if not self._ready:
            return None
        task_id = self._ready.popleft()
        task = self._tasks[task_id]
        task.state, task.token = "reserved", next(self._tokens)  # a fresh token for every reservation
        task.attempts += 1
        if self._clock is not None:
            heapq.heappush(self._leases, (self._clock.now() + self._lease, task_id, task.token))
        return (task_id, task.token, task.payload)

    def complete(self, task_id, token):
        task = self._held(task_id, token)
        task.state, task.token = "completed", None

    def fail(self, task_id, token):
        self._release(task_id, self._held(task_id, token))

    def dead_letters(self):
        self._reclaim()
        return list(self._dead)

    def requeue_dead(self, task_id):
        self._reclaim()
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        if task.state != "dead":
            raise ValueError(f"task {task_id!r} is not dead")
        del self._dead[task_id]
        task.state, task.attempts = "ready", 0
        self._ready.append(task_id)
''',
    "interview_questions": interview(
        concept=[
            "Why does each reservation need its own token instead of identifying it by task id and worker id?",
            "Why must complete and fail reject a token for a task that is ready or already completed?",
        ],
        deep_dive=[
            "Which data structures make submit, reserve, complete and fail all O(1), and why must a failed task go to the back?",
        ],
        tradeoffs=[
            "Why reclaim expired leases at the start of every call instead of with a background thread, and what does that cost?",
            "How do you keep reclaiming cheap with many open leases when completed reservations stay in the heap?",
            "Should max_attempts count reservations or retries after the first, and why does that choice matter to users?",
            "What would you change to run this queue across several processes or machines?",
        ],
    ),
}
