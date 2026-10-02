"""Fix four merged pull requests on a thread-shared job queue: an atomic batch submit, a race in submit, unsafe persistence, and a mutable default in a policy refactor."""

from ._interview import interview

# A model queue, a lock that counts acquisitions, and an id whose hashing yields to other threads.
_MODEL = r"""
import json, os, pickle, random, sys, tempfile, threading, time

def is_value_error(e):
    return "ValueError" in [c.__name__ for c in type(e).__mro__]

def raises_value_error(fn):
    try:
        fn()
    except Exception as e:
        assert is_value_error(e), f"expected ValueError, got {type(e).__name__}: {e}"
        return
    raise AssertionError("expected ValueError, nothing was raised")

class Model:
    def __init__(self):
        self.jobs, self.order, self.seq = {}, [], 0
    def add(self, i, pri, max_attempts=3):
        self.jobs[i] = {"priority": pri, "status": "pending", "attempts": 0, "max_attempts": max_attempts, "seq": self.seq}
        self.seq += 1
    def pop(self):
        ready = [(-j["priority"], j["seq"], i) for i, j in self.jobs.items() if j["status"] == "pending"]
        if not ready:
            return None
        i = min(ready)[2]
        self.jobs[i]["status"] = "running"
        return i
    def fail(self, i):
        j = self.jobs[i]
        j["attempts"] += 1
        if j["attempts"] < j["max_attempts"]:
            j["status"], j["seq"] = "pending", self.seq
            self.seq += 1
        else:
            j["status"] = "failed"
    def counts(self):
        out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for j in self.jobs.values():
            out[j["status"]] += 1
        return out

class CountingLock:
    def __init__(self, lock):
        self.lock, self.count = lock, 0
    def acquire(self, *a, **k):
        self.count += 1
        return self.lock.acquire(*a, **k)
    def release(self):
        return self.lock.release()
    def locked(self):
        return self.lock.locked()
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, *exc):
        self.release()

class SlowId(str):
    # hashing sleeps, so a thread that looks this id up hands the GIL to the others
    def __hash__(self):
        time.sleep(0.005)
        return str.__hash__(self)

def run_threads(targets, seconds=4):
    errors, threads = [], []
    def wrap(fn):
        def body():
            try:
                fn()
            except BaseException as e:
                errors.append(e)
        return body
    threads = [threading.Thread(target=wrap(t), daemon=True) for t in targets]
    for t in threads:
        t.start()
    deadline = time.monotonic() + seconds
    for t in threads:
        t.join(max(0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), f"threads did not finish within {seconds} seconds"
    return errors

def tmp_path(name):
    return os.path.join(tempfile.mkdtemp(), name)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "effects.idempotency", "code": _MODEL + r"""
q = {fn}()
q.submit("seed", {})
raises_value_error(lambda: q.submit_many([("j1", {}, 1), ("j2", {}, 4), ("seed", {}, 9)]))
assert q.counts()["pending"] == 1, ("a rejected batch must add nothing", q.counts())
jobs = q.submit_many([("j1", {}, 1), ("j2", {}, 4)])
assert [j.id for j in jobs] == ["j1", "j2"], [j.id for j in jobs]
assert q.pop_ready().id == "j2"
"""},
    {"name": "Part 1: batches, duplicates and the queue's own rules", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Against a model queue, submit_many added part of a batch that held a duplicate (against the queue or inside the batch), failed on a generator, took the lock more than once per batch, or the queue's pop order, retries or counts changed.",
     "code": _MODEL + r"""
for seed in range(60):
    rng = random.Random(seed)
    q, m = {fn}(), Model()
    for step in range(40):
        op = rng.random()
        if op < 0.35:
            batch = [(f"j{rng.randrange(30)}", {"n": step}, rng.randint(0, 3)) for _ in range(rng.randint(0, 4))]
            ids = [b[0] for b in batch]
            ok = len(set(ids)) == len(ids) and not set(ids) & set(m.jobs)
            lock = q._lock = CountingLock(q._lock)
            try:
                got = q.submit_many(iter(batch) if rng.random() < 0.5 else batch)
            except Exception as e:
                assert not ok and is_value_error(e), (seed, step, batch, type(e).__name__)
            else:
                assert ok, (seed, step, "a batch with a duplicate was accepted", batch)
                assert [j.id for j in got] == ids, (seed, step, [j.id for j in got], ids)
                for i, _, pri in batch:
                    m.add(i, pri)
            assert lock.count == 1, (seed, step, f"submit_many took the lock {lock.count} times")
            q._lock = lock.lock
        elif op < 0.5:
            i = f"j{rng.randrange(30)}"
            if i in m.jobs:
                raises_value_error(lambda: q.submit(i, {}))
            else:
                pri = rng.randint(0, 3)
                q.submit(i, {}, pri, max_attempts=2)
                m.add(i, pri, 2)
        elif op < 0.8:
            got, want = q.pop_ready(), m.pop()
            assert (got.id if got else None) == want, (seed, step, got and got.id, want)
            if want is not None and rng.random() < 0.5:
                q.fail(want)
                m.fail(want)
            elif want is not None:
                q.complete(want)
                m.jobs[want]["status"] = "done"
        assert q.counts() == m.counts(), (seed, step, q.counts(), m.counts())
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "concurrency.thread_safety", "code": _MODEL + r"""
q = {fn}()
key = SlowId("job-7")
errors = run_threads([lambda: q.submit(key, {"from": 1}), lambda: q.submit(key, {"from": 2})])
assert len(errors) == 1 and is_value_error(errors[0]), [type(e).__name__ for e in errors]
assert q.counts()["pending"] == 1, q.counts()
"""},
    {"name": "Part 2: many threads, single and batch submits", "part": 2, "visibility": "unshown", "behavior": "concurrency.thread_safety",
     "failure_message": "With 8 threads submitting overlapping ids through submit and submit_many while id lookups yield, an id was accepted twice, a duplicate did not raise ValueError, or popping everything handed out a job twice.",
     "code": _MODEL + r"""
for round_ in range(3):
    q = {fn}()
    ids = [SlowId(f"r{round_}-{k}") for k in range(6)]
    ok, lock = [], threading.Lock()
    def single(k):
        def go():
            for i in ids[k % 3::2]:
                try:
                    q.submit(i, {})
                    with lock:
                        ok.append(i)
                except Exception as e:
                    assert is_value_error(e), type(e).__name__
        return go
    def batch(k):
        def go():
            for i in ids[k % 2::3]:
                try:
                    q.submit_many([(i, {}, k)])
                    with lock:
                        ok.append(i)
                except Exception as e:
                    assert is_value_error(e), type(e).__name__
        return go
    errors = run_threads([single(k) for k in range(4)] + [batch(k) for k in range(4)])
    assert not errors, errors
    assert sorted(ok) == sorted(set(ok)), ("an id was accepted twice", sorted(ok))
    popped = []
    while (j := q.pop_ready()) is not None:
        popped.append(j.id)
    assert sorted(popped) == sorted(ok), (sorted(popped), sorted(ok))
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "security.injection", "code": _MODEL + r"""
q = {fn}()
q.submit("a", {"rows": [1, 2]}, 2)
q.submit("b", "text", 5)
q.pop_ready()
path = tmp_path("queue.json")
q.save(path)
back = {fn}.load(path)
assert back.get("b").status == "running" and back.get("a").payload == {"rows": [1, 2]}, back.get("a").payload
assert back.pop_ready().id == "a"
class Trap:
    def __reduce__(self):
        return (exec, ("import sys; sys._pq_trap_ran = True",))
sys._pq_trap_ran = False
evil = tmp_path("evil.bin")
with open(evil, "wb") as f:
    pickle.dump(Trap(), f)
raises_value_error(lambda: {fn}.load(evil))
assert not getattr(sys, "_pq_trap_ran", False), "load ran code from the file"
"""},
    {"name": "Part 3: payloads, bad files and round trips", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "save with a payload JSON cannot hold did not raise TypeError or damaged the earlier file, a file that is not a JSON list of job rows (unknown field, missing field, unknown status) did not raise ValueError, or a round trip lost a field or changed which jobs are poppable.",
     "code": _MODEL + r"""
q = {fn}()
q.submit("keep", [1, "two"], 1)
path = tmp_path("q.json")
q.save(path)
before = open(path, "rb").read()
q.submit("bad", {3, 4})
try:
    q.save(path)
except Exception as e:
    assert type(e).__name__ == "TypeError" or "TypeError" in [c.__name__ for c in type(e).__mro__], type(e).__name__
else:
    raise AssertionError("saving a set payload must raise TypeError")
assert open(path, "rb").read() == before, "a failed save changed the file"
row = {"id": "x", "payload": None, "priority": 0, "status": "pending", "attempts": 0, "max_attempts": 3}
for bad in ([dict(row, owner="me")], [{k: v for k, v in row.items() if k != "attempts"}], [dict(row, status="paused")], {"jobs": [row]}, 7, "plain text"):
    p = tmp_path("bad.json")
    with open(p, "w") as f:
        f.write(bad if isinstance(bad, str) else json.dumps(bad))
    raises_value_error(lambda: {fn}.load(p))
for seed in range(30):
    rng = random.Random(100 + seed)
    q = {fn}()
    for k in range(rng.randint(1, 8)):
        q.submit(f"s{k}", {"k": k, "tags": ["a"] * k}, rng.randint(0, 3), max_attempts=rng.randint(1, 3))
    for _ in range(rng.randint(0, 6)):
        j = q.pop_ready()
        if j is None:
            break
        rng.choice([q.complete, q.fail, lambda i: None])(j.id)
    path = tmp_path("r.json")
    q.save(path)
    back = {fn}.load(path)
    for k in range(8):
        i = f"s{k}"
        try:
            a = q.get(i)
        except KeyError:
            continue
        b = back.get(i)
        assert (a.payload, a.priority, a.status, a.attempts, a.max_attempts) == (b.payload, b.priority, b.status, b.attempts, b.max_attempts), (seed, i)
    assert back.counts() == q.counts(), (seed, back.counts(), q.counts())
    pending = sorted(i for i in (f"s{k}" for k in range(8)) if i in back._jobs and back.get(i).status == "pending")
    popped, pris = [], []
    while (j := back.pop_ready()) is not None:
        popped.append(j.id)
        pris.append(j.priority)
    assert sorted(popped) == pending and pris == sorted(pris, reverse=True), (seed, popped, pending)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": _MODEL + r"""
q = {fn}()
q.submit("t1", {}).metadata["tenant"] = "acme"
assert q.submit("t2", {}).metadata == {}, "every job needs its own metadata dict"
raises_value_error(lambda: {fn}(policy="round_robin"))
"""},
    {"name": "Part 4: metadata, persistence and policies", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Metadata was shared between jobs, not copied from submit's argument, lost in a save and load, or not defaulted to {} for rows saved without it; an unknown policy name did not raise ValueError; or a policy instance was not used, or not called while the queue's lock is held.",
     "code": _MODEL + r"""
given = {"tenant": "beta"}
q = {fn}()
j = q.submit("m1", {}, metadata=given)
given["tenant"] = "changed"
assert j.metadata == {"tenant": "beta"}, j.metadata
q.submit("m2", {}).metadata["n"] = 2
assert q.get("m1").metadata == {"tenant": "beta"}, q.get("m1").metadata
path = tmp_path("m.json")
q.save(path)
back = {fn}.load(path)
assert back.get("m2").metadata == {"n": 2} and back.get("m1").metadata == {"tenant": "beta"}, (back.get("m1").metadata, back.get("m2").metadata)
old = tmp_path("old.json")
with open(old, "w") as f:
    json.dump([{"id": o, "payload": 1, "priority": 0, "status": "pending", "attempts": 0, "max_attempts": 3} for o in ("o1", "o2")], f)
loaded = {fn}.load(old)
loaded.get("o1").metadata["seen"] = True
assert loaded.get("o2").metadata == {}, ("rows loaded without metadata must not share one dict", loaded.get("o2").metadata)
for name in ("fifo", "", "PRIORITY"):
    if name:
        raises_value_error(lambda: {fn}(policy=name))
base = {fn}.__init__.__globals__["SchedulingPolicy"]
class Lowest(base):
    def __init__(self):
        self.held = []
    def select(self, heap, jobs):
        self.held.append(queue._lock.locked())
        ready = [j for j in jobs.values() if j.status == "pending"]
        return min(ready, key=lambda j: j.priority) if ready else None
policy = Lowest()
queue = {fn}(policy=policy)
queue.submit("hi", {}, 9)
queue.submit("lo", {}, 1)
assert queue.pop_ready().id == "lo" and queue.get("lo").status == "running"
assert policy.held == [True], policy.held
"""},
]

TASK = {
    "title": "Job Queue Pull Request Review",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "JobQueue",
    "description_en": r"""The starter code is a thread-shared priority job queue with four pull requests merged exactly as their authors wrote them. Each part reviews one pull request: keep what it was for, and fix what it broke.

The requirement arrives in parts. Each part keeps every earlier behavior, so one fixed `JobQueue` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `JobQueue` hands out pending jobs by priority, higher first, and equal priorities in the order they became pending. `pop_ready` marks the job `"running"`; `fail` makes it pending again while attempts remain, else `"failed"`; `complete` marks it `"done"`.
- Job statuses are `"pending"`, `"running"`, `"done"` and `"failed"`, and `counts()` reports how many jobs are in each.
- A duplicate job id raises `ValueError`.
- Keep `self._lock`, the one `threading.Lock` that guards the queue, and use it as a context manager. Keep every public method and the `Job` class.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** review rounds hand you plausible pull requests and an AI assistant, and judge whether you can say exactly what each one breaks and prove it with a test. Each later part adds one requirement: a race, unsafe deserialization, then a mutable default hidden in a refactor.

**Where it is used:** job queues like this one feed training, evaluation and data pipelines. Partial batches, duplicate jobs and a persistence format that can run code are the incidents a careful review prevents.

Adapted from the agentic pull request review question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The four pull requests come merged into the starter instead of shown as diffs, and the written review, the choice of one pull request and the rewritten description are left out; each part grades the fixed behavior instead. The richer payloads that the pickle change wanted are declined rather than supported. `submit` gains a `metadata` argument, and a policy name that is not registered raises `ValueError`.""",
    "parts": [
        {
            "title": "Batch submit",
            "description_en": r"""**Signature:** `JobQueue(policy=None).submit_many(items) -> list[Job]`

- `items` is any iterable of `(job_id, payload, priority)` triples. Either every job is added, in order, and the created jobs are returned in order, or none is.
- An id already in the queue, or repeated inside the batch, raises `ValueError` and adds nothing.
- `submit_many` takes `self._lock` exactly once per call.

**Example:** after `submit("seed", {})`:
- `submit_many([("j1", {}, 1), ("j2", {}, 4), ("seed", {}, 9)])` raises `ValueError`, and only `seed` is pending
- `submit_many([("j1", {}, 1), ("j2", {}, 4)])` returns jobs `j1` and `j2`, and `pop_ready()` then returns `j2`""",
        },
        {
            "title": "The duplicate check",
            "description_en": r"""Keep Part 1. The second pull request moved `submit`'s duplicate check out of the lock to save time.

- Threads may call `submit` and `submit_many` at the same moment, with the same ids. Each id is accepted exactly once, and every other attempt raises `ValueError`.

**Example:** two threads submit `"job-7"` at the same moment, and hashing the id is slow enough that both are inside `submit` together:
- exactly one call raises `ValueError`
- one job is pending""",
        },
        {
            "title": "Persistence",
            "description_en": r"""Keep Parts 1–2. The third pull request switched `save` and `load` to pickle, so loading a file can run any code inside it.

- `save(path)` writes JSON. If a payload cannot be written as JSON, it raises `TypeError` and leaves any existing file at `path` unchanged.
- `JobQueue.load(path)` reads it back as data only; nothing in the file may run. Loading restores every job's fields, and exactly the pending jobs are poppable again.
- A file that is not a JSON list of job rows raises `ValueError`. A row must have exactly the fields `id`, `payload`, `priority`, `status`, `attempts` and `max_attempts`, with a known status.

**Example:** jobs `a` (payload `{"rows": [1, 2]}`, priority 2) and `b` (priority 5), with `b` popped, are saved and loaded:
- `b` is `"running"`, `a` keeps its payload, and `pop_ready()` returns `a`
- loading a pickle file built to run a function raises `ValueError`, and the function never runs""",
        },
        {
            "title": "Policies and metadata",
            "description_en": r"""Keep Parts 1–3. The fourth pull request added scheduling policies and a `metadata` dict on every job.

- Every job has its own `metadata` dict, empty by default. `submit(..., metadata=None)` stores a copy of the dict it is given.
- `metadata` is saved and loaded with the job; it is the one optional field in a row, and a row without it loads with `{}`.
- `JobQueue(policy=...)` accepts `None`, a registered name (only `"priority"`), or a `SchedulingPolicy` instance. Any other name raises `ValueError`.
- `pop_ready` calls the policy's `select(heap, jobs)` while holding `self._lock`, and marks the job it returns `"running"`.

**Example:**
- setting `metadata["tenant"]` on one job leaves the next job's `metadata` equal to `{}`
- `JobQueue(policy="round_robin")` raises `ValueError`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "The batch is three jobs and the third id already exists. What does the merged submit_many leave in the queue, and is that what a caller replaying a backlog expects? How many times does it take the lock, and what happens if you take it again inside?"},
        {"level": 2, "kind": "analysis", "content": "submit_many calls submit per item, so a duplicate halfway through leaves the earlier items added and the lock is taken once per job. Turn items into a list, take the lock once, check every id against the queue and a set of ids already seen in the batch, and only then add them all with a helper that assumes the lock is held."},
    ],
    "model_connections": [
        "Training and evaluation pipelines run on shared job queues where a duplicate or half-submitted batch silently wastes accelerator hours.",
        "Model checkpoints moved from pickle to formats such as safetensors because loading a pickle can execute arbitrary code.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Checking a whole batch before adding any of it keeps the queue consistent when one item is bad.",
            "Holding one lock across check and insert removes the race at the cost of a few instructions inside the lock.",
            "JSON persistence loads only data and fails loudly on payloads it cannot represent.",
        ],
        "cons": [
            "A single lock serialises every caller; sharding the queue would scale better.",
            "JSON cannot hold richer payloads, so callers must convert them or store a reference instead.",
            "A policy interface that receives the raw heap couples every policy to the queue's internals.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
from abc import ABC, abstractmethod

import heapq
import itertools
import json
import os
import threading


class SchedulingPolicy(ABC):
    """Decides which pending job pop_ready() hands out next."""

    @abstractmethod
    def select(self, heap, jobs):
        """heap: the queue's (-priority, seq, job_id) list. jobs: id -> Job. Returns a ready Job or None."""


class PriorityPolicy(SchedulingPolicy):
    def select(self, heap, jobs):
        while heap:
            _, _, job_id = heapq.heappop(heap)
            job = jobs[job_id]
            if job.status == "pending":
                return job
        return None


_POLICIES = {"priority": PriorityPolicy}


def _resolve_policy(policy):
    if isinstance(policy, SchedulingPolicy):
        return policy
    name = policy or "priority"
    if name not in _POLICIES:
        raise ValueError(f"unknown policy {name!r}")
    return _POLICIES[name]()


_FIELDS = {"id", "payload", "priority", "status", "attempts", "max_attempts"}
_STATUSES = {"pending", "running", "done", "failed"}


class Job:
    def __init__(self, id, payload, priority=0, status="pending", attempts=0, max_attempts=3, metadata=None):
        self.id = id
        self.payload = payload
        self.priority = priority
        self.status = status  # "pending" | "running" | "done" | "failed"
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.metadata = {} if metadata is None else metadata  # never one dict shared by every job

    def to_dict(self):
        return dict(id=self.id, payload=self.payload, priority=self.priority,
                    status=self.status, attempts=self.attempts, max_attempts=self.max_attempts,
                    metadata=self.metadata)


class JobQueue:
    """A priority job queue shared by worker threads. Higher priority first; equal priorities in submission order."""

    def __init__(self, policy=None):
        self._lock = threading.Lock()
        self._jobs = {}  # id -> Job, every job ever submitted
        self._heap = []  # (-priority, seq, id), pending jobs only
        self._counter = itertools.count()
        self._policy = _resolve_policy(policy)

    def submit(self, job_id, payload, priority=0, max_attempts=3, metadata=None):
        with self._lock:  # check and insert together, or two callers can both see the id as free
            if job_id in self._jobs:
                raise ValueError(f"duplicate job id {job_id!r}")
            return self._add(job_id, payload, priority, max_attempts, metadata)

    def _add(self, job_id, payload, priority, max_attempts=3, metadata=None):
        job = Job(job_id, payload, priority, "pending", 0, max_attempts, dict(metadata or {}))
        self._jobs[job_id] = job
        heapq.heappush(self._heap, (-priority, next(self._counter), job_id))
        return job

    def submit_many(self, items):
        """items: an iterable of (job_id, payload, priority) triples. All are submitted, or none."""
        items = list(items)
        with self._lock:  # once per batch
            seen = set()
            for job_id, _, _ in items:
                if job_id in self._jobs or job_id in seen:
                    raise ValueError(f"duplicate job id {job_id!r}")
                seen.add(job_id)
            return [self._add(job_id, payload, priority) for job_id, payload, priority in items]

    def pop_ready(self):
        """The highest-priority pending job, marked "running" so no other caller gets it; None if none."""
        with self._lock:
            job = self._policy.select(self._heap, self._jobs)
            if job is not None:
                job.status = "running"
            return job

    def complete(self, job_id):
        with self._lock:
            self._jobs[job_id].status = "done"

    def fail(self, job_id):
        """Record a failed attempt: pending again while attempts remain, else "failed"."""
        with self._lock:
            job = self._jobs[job_id]
            job.attempts += 1
            if job.attempts < job.max_attempts:
                job.status = "pending"
                heapq.heappush(self._heap, (-job.priority, next(self._counter), job_id))
            else:
                job.status = "failed"

    def get(self, job_id):
        return self._jobs[job_id]

    def counts(self):
        with self._lock:
            out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for job in self._jobs.values():
                out[job.status] += 1
            return out

    def save(self, path):
        with self._lock:
            rows = [job.to_dict() for job in self._jobs.values()]
        text = json.dumps(rows)  # a payload JSON cannot hold raises here, before the file is touched
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        queue = cls()
        with open(path, "rb") as f:
            rows = json.loads(f.read().decode("utf-8"))  # data only: nothing in the file can run code
        if not isinstance(rows, list):
            raise ValueError("a saved queue is a list of jobs")
        for row in rows:
            if not isinstance(row, dict) or not _FIELDS <= set(row) <= _FIELDS | {"metadata"}:
                raise ValueError(f"bad job row {row!r}")
            if row["status"] not in _STATUSES:
                raise ValueError(f"bad status {row['status']!r}")
            job = Job(**row)
            queue._jobs[job.id] = job
            if job.status == "pending":
                heapq.heappush(queue._heap, (-job.priority, next(queue._counter), job.id))
        return queue
''',
    "interview_questions": interview(
        concept=[
            "What does the merged submit_many leave behind when the third of five ids is a duplicate, and how do you make a batch all or nothing?",
            "Why must submit_many take the lock once rather than calling submit for each item?",
        ],
        deep_dive=[
            "Which test would you add to the pull request to prove the partial-batch bug, and which to prove the fix?",
        ],
        tradeoffs=[
            "Why is a duplicate check outside the lock a race, and how do you write a test that hits it reliably?",
            "Why is pickle a security problem for a file the queue loads, and what do you tell the author who wanted richer payloads?",
            "Why is a mutable default argument such as metadata={} a bug, and how would a code review catch it?",
            "Which of the four pull requests would you approve after changes, and which would you send back for a redesign?",
        ],
    ),
}
