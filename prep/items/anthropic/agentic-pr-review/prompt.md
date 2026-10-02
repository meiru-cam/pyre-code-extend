You are working in a browser-based editor with an integrated terminal, open on a small Python repository. A separate code-hosting page lists four open pull requests against this repository; each is shown below as a unified *diff* (a text description of the lines a change adds and removes, in `diff` format) together with the description its author wrote. An AI coding assistant is available in the terminal and may be used for any part of the task, but every review comment and every line of code you submit is your responsibility: being unable to explain why a comment or a change is correct is treated the same as not having made it.

The repository has one module, `jobqueue.py`:

```python
import heapq
import itertools
import json
import threading


class Job:
    def __init__(self, id, payload, priority=0, status="pending", attempts=0, max_attempts=3):
        self.id = id
        self.payload = payload
        self.priority = priority
        self.status = status                # "pending" | "running" | "done" | "failed"
        self.attempts = attempts
        self.max_attempts = max_attempts

    def to_dict(self):
        return dict(id=self.id, payload=self.payload, priority=self.priority,
                    status=self.status, attempts=self.attempts, max_attempts=self.max_attempts)


class JobQueue:
    """A priority job queue, safe to share between several worker threads in one process. Ready jobs
    are handed out by priority (higher first); jobs of equal priority are handed out in submission order."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}                     # id -> Job, every job ever submitted
        self._heap = []                     # (-priority, seq, id), pending jobs only
        self._counter = itertools.count()

    def submit(self, job_id, payload, priority=0, max_attempts=3):
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"duplicate job id {job_id!r}")
            job = Job(job_id, payload, priority, "pending", 0, max_attempts)
            self._jobs[job_id] = job
            heapq.heappush(self._heap, (-priority, next(self._counter), job_id))
        return job

    def pop_ready(self):
        """Return the highest-priority pending job, marking it "running" so no other caller can
        receive the same job. Returns None if no job is pending."""
        with self._lock:
            while self._heap:
                _, _, job_id = heapq.heappop(self._heap)
                job = self._jobs[job_id]
                if job.status == "pending":
                    job.status = "running"
                    return job
            return None

    def complete(self, job_id):
        with self._lock:
            self._jobs[job_id].status = "done"

    def fail(self, job_id):
        """Record a failed attempt at a running job. Reschedules it as pending if attempts remain
        (it becomes poppable again immediately), else marks it permanently "failed"."""
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
        """Number of jobs currently in each status, for monitoring."""
        with self._lock:
            out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for job in self._jobs.values():
                out[job.status] += 1
            return out

    def save(self, path):
        with self._lock:
            rows = [job.to_dict() for job in self._jobs.values()]
        with open(path, "w") as f:
            json.dump(rows, f)

    @classmethod
    def load(cls, path):
        queue = cls()
        with open(path) as f:
            rows = json.load(f)
        for row in rows:
            job = Job(**row)
            queue._jobs[job.id] = job
            if job.status == "pending":
                heapq.heappush(queue._heap, (-job.priority, next(queue._counter), job.id))
        return queue
```

The repository also has `test_jobqueue.py`, with `unittest` tests for `submit` (rejects a duplicate id), `pop_ready` (priority order, then submission order among equal priorities), `fail` (retries until `max_attempts` is reached, then marks the job `"failed"`), and `save`/`load` (a round trip through a file reproduces every job's fields, and only jobs left `"pending"` are poppable again afterwards). The repository's `README` names its test command: `python -m unittest -v`.

**PR #1 — Add `submit_many` for batch job submission** (opened by `grace-oyelaran`)

> We seed the queue with a backlog of a few hundred jobs at startup, replayed from the previous day's > snapshot, and today that means every caller writes its own loop over `submit`. This adds `submit_many`, > which takes the same `(job_id, payload, priority)` triples callers already build and submits them in one > call, saving that loop and taking the lock once per batch instead of once per job. No change to `submit` > or to any other method.

```diff
diff --git a/jobqueue.py b/jobqueue.py
--- a/jobqueue.py
+++ b/jobqueue.py
@@ class JobQueue:
             heapq.heappush(self._heap, (-priority, next(self._counter), job_id))
         return job

+    def submit_many(self, items):
+        """items: an iterable of (job_id, payload, priority) triples. Submits each in order and
+        returns the list of created Job objects."""
+        return [self.submit(job_id, payload, priority) for job_id, payload, priority in items]
+
     def pop_ready(self):
         """Return the highest-priority pending job, marking it "running" so no other caller can
diff --git a/test_jobqueue.py b/test_jobqueue.py
--- a/test_jobqueue.py
+++ b/test_jobqueue.py
@@ class TestJobQueue(unittest.TestCase):
     def test_submit_rejects_duplicate_id(self):
         q = JobQueue()
         q.submit("a", {})
         with self.assertRaises(ValueError):
             q.submit("a", {})
+
+    def test_submit_many_creates_all_jobs_in_order(self):
+        q = JobQueue()
+        jobs = q.submit_many([("a", {}, 0), ("b", {}, 5), ("c", {}, 0)])
+        self.assertEqual([j.id for j in jobs], ["a", "b", "c"])
+        self.assertEqual(q.pop_ready().id, "b")
```

**PR #2 — Reduce lock contention in `submit()`** (opened by `marcus-kaelin`)

> Under load, `submit` shows up in profiling as the most contended lock: every call takes `self._lock` for > the whole method, even though the duplicate-id check never touches shared mutable state as long as the id > turns out to be free. This moves the check ahead of the lock, so the common case — a fresh id — does > less work while holding the lock. Behavior is unchanged: a duplicate id still raises `ValueError`.

```diff
diff --git a/jobqueue.py b/jobqueue.py
--- a/jobqueue.py
+++ b/jobqueue.py
@@ class JobQueue:
     def submit(self, job_id, payload, priority=0, max_attempts=3):
-        with self._lock:
-            if job_id in self._jobs:
-                raise ValueError(f"duplicate job id {job_id!r}")
+        if job_id in self._jobs:
+            raise ValueError(f"duplicate job id {job_id!r}")
+        with self._lock:
             job = Job(job_id, payload, priority, "pending", 0, max_attempts)
             self._jobs[job_id] = job
             heapq.heappush(self._heap, (-priority, next(self._counter), job_id))
         return job
```

**PR #3 — Persist queue state with pickle instead of JSON** (opened by `owen-t`)

> A couple of internal callers want to stash richer objects in a job's `payload` — namedtuples, > dataclasses, even a small numpy array for one use case — and JSON can't represent those, so `save` > silently can't be used for them today. Pickle handles anything picklable with no schema work on our side, > and it is a drop-in replacement: same file, same call sites, same round trip. Swapped both `save` and > `load` over; nothing else in the file changes.

```diff
diff --git a/jobqueue.py b/jobqueue.py
--- a/jobqueue.py
+++ b/jobqueue.py
@@
 import heapq
 import itertools
-import json
+import pickle
 import threading
@@ class JobQueue:
     def save(self, path):
         with self._lock:
             rows = [job.to_dict() for job in self._jobs.values()]
-        with open(path, "w") as f:
-            json.dump(rows, f)
+        with open(path, "wb") as f:
+            pickle.dump(rows, f)

     @classmethod
     def load(cls, path):
         queue = cls()
-        with open(path) as f:
-            rows = json.load(f)
+        with open(path, "rb") as f:
+            rows = pickle.load(f)
         for row in rows:
             job = Job(**row)
             queue._jobs[job.id] = job
```

**PR #4 — Pluggable scheduling policies** (opened by `felix-wu`)

> Priority-then-FIFO will not stay the only scheduling rule we need — the on-call rotation for the > batch-import team has already asked for round-robin across tenants, so one noisy tenant cannot starve the > others. Rather than hard-code a second rule into `pop_ready` later, this introduces a `SchedulingPolicy` > interface and a small registry, and moves the existing behavior into a `PriorityPolicy` so nothing changes > by default. Also added a general-purpose `metadata` field on `Job`, since a per-tenant policy will need > somewhere to keep its own bookkeeping per job.

```diff
diff --git a/jobqueue.py b/jobqueue.py
--- a/jobqueue.py
+++ b/jobqueue.py
@@
+from abc import ABC, abstractmethod
+
 import heapq
 import itertools
 import json
 import threading


+class SchedulingPolicy(ABC):
+    """Decides which pending job pop_ready() should hand out next."""
+
+    @abstractmethod
+    def select(self, heap, jobs):
+        """heap: the queue's internal (-priority, seq, job_id) list. jobs: id -> Job.
+        Returns a ready Job, or None if none is ready."""
+
+
+class PriorityPolicy(SchedulingPolicy):
+    def select(self, heap, jobs):
+        while heap:
+            _, _, job_id = heapq.heappop(heap)
+            job = jobs[job_id]
+            if job.status == "pending":
+                return job
+        return None
+
+
+_POLICIES = {"priority": PriorityPolicy}
+
+
+def _resolve_policy(policy):
+    if isinstance(policy, SchedulingPolicy):
+        return policy
+    return _POLICIES[policy or "priority"]()
+
+
 class Job:
-    def __init__(self, id, payload, priority=0, status="pending", attempts=0, max_attempts=3):
+    def __init__(self, id, payload, priority=0, status="pending", attempts=0, max_attempts=3, metadata={}):
         self.id = id
         self.payload = payload
         self.priority = priority
         self.status = status
         self.attempts = attempts
         self.max_attempts = max_attempts
+        self.metadata = metadata
@@ class JobQueue:
-    def __init__(self):
+    def __init__(self, policy=None):
         self._lock = threading.Lock()
         self._jobs = {}
         self._heap = []
         self._counter = itertools.count()
+        self._policy = _resolve_policy(policy)
@@ class JobQueue:
     def pop_ready(self):
         """Return the highest-priority pending job, marking it "running" so no other caller can
         receive the same job. Returns None if no job is pending."""
         with self._lock:
-            while self._heap:
-                _, _, job_id = heapq.heappop(self._heap)
-                job = self._jobs[job_id]
-                if job.status == "pending":
-                    job.status = "running"
-                    return job
-            return None
+            job = self._policy.select(self._heap, self._jobs)
+            if job is not None:
+                job.status = "running"
+            return job
```

### Part 1 — Review the four pull requests

For each pull request, in the order given, write:

- The product or engineering problem it is trying to solve, in one or two sentences.
- A summary of its implementation approach and which parts of `jobqueue.py` (or its tests) it touches.
- Review comments covering correctness, edge cases, maintainability, tests, performance, and security or compatibility, each pointing at a specific line or a specific behavior; state which tests are missing for the new or changed behavior.
- Whether the pull request is mergeable as it stands; if not, exactly what blocks it.

### Part 2 — Improve one pull request

Choose exactly one of the four pull requests to improve; it need not be the one with the most obvious problems. State which one and why, including why each of the other three was not chosen. Then:

- Modify `jobqueue.py` (and `test_jobqueue.py`) directly so that the chosen pull request's remaining problems are fixed.
- Add, update, or fix tests for every piece of new or changed behavior. Do not delete a test unless the behavior it checked no longer exists.
- Run the repository's test command, `python -m unittest -v`, and report its result.
- Rewrite the pull request's description: a summary of the change, the design trade-offs made, the test results, and any known limitations that remain.
