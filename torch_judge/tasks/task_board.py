"""Per-user tasks with ids, edits and completion, then sorted lists, expiry, and lists at past timestamps."""

from ._interview import interview

# A naive model that keeps a log of every call and rebuilds the state for each question, plus a random call generator.
_MODEL = r"""
import random, time

class Model:
    def __init__(self):
        self.log = []  # (timestamp, op, args) for every state change, in call order
        self.count = 0

    def _state(self, at):
        tasks = {}
        for ts, op, args in self.log:
            if ts > at:
                break
            if op == "add":
                tid, user, title, priority, ttl, seq = args
                tasks[tid] = {"user": user, "title": title, "priority": priority, "created": ts, "ttl": ttl, "done": False, "seq": seq}
            elif op == "edit":
                tid, title, priority = args
                tasks[tid]["title"], tasks[tid]["priority"] = title, priority
            else:
                tasks[args[0]]["done"] = True
        return {tid: t for tid, t in tasks.items()
                if not t["done"] and (t["ttl"] is None or at < t["created"] + t["ttl"])}

    def _live(self, at, user, tid):
        t = self._state(at).get(tid)
        return t if t is not None and t["user"] == user else None

    def add(self, ts, user, title, priority, ttl=None):
        self.count += 1
        tid = f"t{self.count}"
        self.log.append((ts, "add", (tid, user, title, priority, ttl, self.count)))
        return tid

    def edit(self, ts, user, tid, title, priority):
        if self._live(ts, user, tid) is None:
            return False
        self.log.append((ts, "edit", (tid, title, priority)))
        return True

    def finish(self, ts, user, tid):
        if self._live(ts, user, tid) is None:
            return False
        self.log.append((ts, "finish", (tid,)))
        return True

    def get(self, ts, user, tid):
        t = self._live(ts, user, tid)
        return None if t is None else (tid, t["title"], t["priority"], t["created"])

    def list_tasks(self, at, user, min_priority=None):
        rows = [(tid, t) for tid, t in self._state(at).items()
                if t["user"] == user and (min_priority is None or t["priority"] >= min_priority)]
        rows.sort(key=lambda r: (-r[1]["priority"], r[1]["seq"]))
        return [(tid, t["title"], t["priority"], t["created"]) for tid, t in rows]

USERS = ["ana", "bo", "cy"]
TITLES = ["sync", "Sync", "deploy", "notes"]

def random_calls(rng, steps, part):
    ts, calls, made = 0, [], 0
    for _ in range(steps):
        ts += rng.choice([0, 0, 1, 1, 2, 3])
        user = rng.choice(USERS)
        tid = f"t{rng.randint(1, made + 1)}"
        kind = rng.random()
        if kind < 0.3 or made == 0:
            ttl = rng.choice([None, None, 1, 2, 4, 7]) if part >= 3 else None
            args = (ts, user, rng.choice(TITLES), rng.randint(0, 4)) + ((ttl,) if ttl is not None else ())
            calls.append(("add", args))
            made += 1
        elif kind < 0.5:
            calls.append(("edit", (ts, user, tid, rng.choice(TITLES), rng.randint(0, 4))))
        elif kind < 0.65:
            calls.append(("finish", (ts, user, tid)))
        elif kind < 0.8 or part == 1:
            calls.append(("get", (ts, user, tid)))
        else:
            at = rng.randint(0, ts) if part >= 4 else ts
            calls.append(("list_tasks", (at, user) + ((rng.randint(0, 4),) if rng.random() < 0.4 else ())))
    return calls

def replay(board, calls, label):
    model = Model()
    for i, (name, args) in enumerate(calls):
        want = getattr(model, name)(*args)
        got = getattr(board, name)(*args)
        assert got == want, (label, i, name, args, got, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
b = {fn}()
assert b.add(5, "mia", "draft slides", 2) == "t1"
assert b.add(5, "raj", "draft slides", 4) == "t2"
assert b.edit(6, "mia", "t1", "final slides", 6) is True
assert b.get(6, "mia", "t1") == ("t1", "final slides", 6, 5)
assert b.finish(7, "raj", "t2") is True and b.get(8, "raj", "t2") is None
assert b.edit(8, "raj", "t1", "x", 1) is False and b.finish(8, "raj", "t1") is False
"""},
    {"name": "Part 1: owners, missing ids and failed calls", "part": 1, "visibility": "unshown", "behavior": "security.permission",
     "failure_message": "Ids come from one counter shared by every user; a task owned by someone else, a finished task and an unknown id all give None or False; a failed edit or finish changes nothing; an edit keeps created_at.",
     "code": r"""
b = {fn}()
ids = [b.add(1, u, "same", 1) for u in ["a", "b", "a", "c"]]
assert ids == ["t1", "t2", "t3", "t4"]
assert b.edit(2, "b", "t1", "stolen", 9) is False and b.get(2, "a", "t1") == ("t1", "same", 1, 1)
assert b.finish(2, "a", "t9") is False and b.get(2, "a", "t9") is None and b.edit(2, "a", "t9", "x", 1) is False
assert b.edit(3, "a", "t3", "same", 1) is True, "an edit to the same values still succeeds"
assert b.edit(4, "a", "t3", "renamed", 0) is True and b.get(4, "a", "t3") == ("t3", "renamed", 0, 1)
assert b.finish(5, "a", "t3") is True and b.finish(5, "a", "t3") is False
assert b.add(6, "a", "next", 2) == "t5", "a finished task's id is never reused"
assert b.get(6, "A", "t5") is None, "user ids are case-sensitive"
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of add, edit, finish and get, a return value differed from a model that replays every call.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(seed), 60, 1), seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "metrics.ties", "code": r"""
b = {fn}()
assert [b.add(1, "raj", "logs", 2), b.add(1, "raj", "alerts", 7), b.add(2, "raj", "docs", 2)] == ["t1", "t2", "t3"]
assert b.finish(3, "raj", "t2") is True
assert b.add(3, "raj", "oncall", 7) == "t4"
assert b.list_tasks(3, "raj") == [("t4", "oncall", 7, 3), ("t1", "logs", 2, 1), ("t3", "docs", 2, 2)]
assert b.list_tasks(3, "raj", 3) == [("t4", "oncall", 7, 3)]
assert b.list_tasks(3, "mia") == []
"""},
    {"name": "Part 2: ties, filters and fresh lists", "part": 2, "visibility": "unshown", "behavior": "metrics.ties",
     "failure_message": "Lists sort by priority, highest first, then by creation order (the id counter), never by title or timestamp; min_priority keeps priorities >= it, including negative ones; an edit moves a task but keeps its place among equals; each call returns a new list.",
     "code": r"""
b = {fn}()
b.add(4, "u", "zeta", 1)
b.add(4, "u", "alpha", 1)
b.add(4, "u", "mid", -3)
assert [t[0] for t in b.list_tasks(4, "u")] == ["t1", "t2", "t3"]
assert b.list_tasks(4, "u", -3) == b.list_tasks(4, "u") and b.list_tasks(4, "u", 2) == []
assert [t[0] for t in b.list_tasks(4, "u", 0)] == ["t1", "t2"], "0 is a real minimum, not 'no filter'"
b.edit(5, "u", "t1", "zeta", 0)
b.edit(5, "u", "t3", "mid", 1)
assert [t[0] for t in b.list_tasks(5, "u")] == ["t2", "t3", "t1"]
got = b.list_tasks(5, "u")
got.clear()
assert len(b.list_tasks(5, "u")) == 3
"""},
    {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence that also lists tasks, a return value differed from a model that replays every call.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(100 + seed), 60, 2), seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
b = {fn}()
assert b.add(10, "mia", "renew cert", 3, ttl=4) == "t1"
assert b.add(11, "mia", "backup", 1) == "t2"
assert b.list_tasks(13, "mia") == [("t1", "renew cert", 3, 10), ("t2", "backup", 1, 11)]
assert b.edit(13, "mia", "t1", "renew certs", 5) is True
assert b.list_tasks(14, "mia") == [("t2", "backup", 1, 11)]
assert b.finish(14, "mia", "t1") is False and b.get(14, "mia", "t1") is None
"""},
    {"name": "Part 3: expiry edges", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "A task with ttl is active while timestamp < created_at + ttl and inactive from that instant on, in get, list_tasks, edit and finish; edits never move the deadline; ttl=None never expires; ttl=1 lives for one time unit.",
     "code": r"""
b = {fn}()
b.add(0, "u", "short", 1, ttl=1)
b.add(0, "u", "long", 1, 100)
b.add(0, "u", "none", 1, None)
assert [t[0] for t in b.list_tasks(0, "u")] == ["t1", "t2", "t3"]
assert b.get(0, "u", "t1") == ("t1", "short", 1, 0)
assert b.edit(1, "u", "t1", "late", 1) is False, "t1 expired at 0 + 1"
assert b.edit(99, "u", "t2", "x", 2) is True and b.edit(99, "u", "t2", "y", 3) is True
assert b.get(99, "u", "t2") == ("t2", "y", 3, 0)
assert b.finish(100, "u", "t2") is False and b.list_tasks(100, "u") == [("t3", "none", 1, 0)]
assert b.get(10**9, "u", "t3") == ("t3", "none", 1, 0)
"""},
    {"name": "Part 3: random calls", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence with ttls, a return value differed from a model that replays every call and checks expiry at the time asked.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(200 + seed), 70, 3), seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "checkpoint.recovery", "code": r"""
b = {fn}()
assert b.add(2, "raj", "warmup", 5, ttl=6) == "t1"
assert b.add(2, "mia", "audit", 1) == "t2"
assert b.add(4, "raj", "eval", 3) == "t3"
assert b.finish(6, "raj", "t1") is True
assert b.edit(7, "raj", "t3", "eval v2", 9) is True
assert b.list_tasks(1, "raj") == []
assert b.list_tasks(5, "raj") == [("t1", "warmup", 5, 2), ("t3", "eval", 3, 4)]
assert b.list_tasks(6, "raj") == [("t3", "eval", 3, 4)]
assert b.list_tasks(7, "raj") == [("t3", "eval v2", 9, 4)]
assert b.list_tasks(7, "mia") == [("t2", "audit", 1, 2)]
"""},
    {"name": "Part 4: same-time calls and early questions", "part": 4, "visibility": "unshown", "behavior": "events.ordering",
     "failure_message": "A past list must include every call with timestamp <= at_timestamp, including several edits at one timestamp, where the last one wins; a finish at t hides the task at t; before a user's first task the list is []; asking about the past changes nothing.",
     "code": r"""
b = {fn}()
b.add(2, "u", "a", 1)
b.edit(2, "u", "t1", "b", 2)
b.edit(2, "u", "t1", "c", 3)
b.add(4, "u", "d", 3)
b.edit(6, "u", "t2", "e", 5)
b.finish(6, "u", "t2")
assert b.list_tasks(1, "u") == [] and b.list_tasks(0, "nobody") == []
assert b.list_tasks(2, "u") == [("t1", "c", 3, 2)]
assert b.list_tasks(5, "u", 3) == [("t1", "c", 3, 2), ("t2", "d", 3, 4)]
assert b.list_tasks(6, "u") == [("t1", "c", 3, 2)]
assert b.list_tasks(3, "u") == [("t1", "c", 3, 2)]
assert b.get(7, "u", "t1") == ("t1", "c", 3, 2) and b.add(7, "u", "f", 0) == "t3"
assert b.list_tasks(7, "u") == [("t1", "c", 3, 2), ("t3", "f", 0, 7)]
"""},
    {"name": "Part 4: random past questions", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence that also lists tasks at past timestamps, a return value differed from replaying only the calls up to that timestamp.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(300 + seed), 70, 4), seed)
"""},
    {"name": "Part 4: past questions on a long history", "part": 4, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "2,000 past lists over 50 tasks and 100,000 edits took too long: keep each task's versions in time order and binary-search them instead of replaying the history for each question.",
     "code": r"""
import random, time
rng = random.Random(4)
b = {fn}()
for i in range(50):
    b.add(0, "u", f"task{i}", 0)
for ts in range(1, 100001):
    b.edit(ts, "u", f"t{rng.randint(1, 50)}", "v", rng.randint(0, 9))
start = time.perf_counter()
sizes = [len(b.list_tasks(rng.randint(0, 100000), "u", 5)) for _ in range(2000)]
elapsed = time.perf_counter() - start
assert len(b.list_tasks(0, "u")) == 50 and b.list_tasks(0, "u", 1) == []
assert all(0 <= s <= 50 for s in sizes)
assert elapsed < 2.0, f"{elapsed:.2f}s for 2,000 past lists"
"""},
]

TASK = {
    "title": "Task Board with Time Travel",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TaskBoard",
    "description_en": r"""Build `TaskBoard`, which keeps each user's tasks with ids, edits and completion, then adds sorted lists, expiry, and lists as they were at an earlier time.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `TaskBoard` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Each task stores its owner's user id, a title (titles may repeat), an integer priority (bigger means more urgent) and `created_at`, the time of its `add`.
- Ids are `t` followed by a number that goes up by one with every `add`, whoever calls it: `t1`, `t2`, and so on. An id is never reused.
- Methods return a task as the tuple `(task_id, title, priority, created_at)`.
- Every method takes an integer timestamp first. Timestamps of `add`, `edit` and `finish` calls never go down, though neighbours may be equal.
- A task is active from its `add` until it is finished. A task that is not active, belongs to another user, or does not exist gives the same failure value.
- User ids are case-sensitive. A call that fails changes nothing. No method raises.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment where each level builds on the class you already wrote. Each later part adds one requirement, and the last one punishes a design that only ever kept the current state.

**Where it is used:** issue trackers and to-do services, job queues with deadlines, and audit views that show what a list looked like at a given moment.

Adapted from the task manager online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class renamed from `TaskManager` to `TaskBoard`, with shorter method names: `add_task` becomes `add`, `update_task` becomes `edit`, `complete_task` becomes `finish`, `get_task` becomes `get` and `get_task_list` becomes `list_tasks`. Tasks are tuples instead of a dataclass, and ids are `t1`, `t2`, and so on. Part 4 adds a speed check for past lists.""",
    "parts": [
        {
            "title": "Add, edit, finish and get",
            "description_en": r"""**Signatures:**
- `TaskBoard().add(timestamp, user, title, priority) -> str` creates an active task owned by `user` and returns its id.
- `edit(timestamp, user, task_id, title, priority) -> bool` replaces the title and priority and keeps `created_at`. It returns `False` if the task is not an active task of `user`.
- `finish(timestamp, user, task_id) -> bool` makes the task inactive for good. It returns `False` if the task is not an active task of `user`.
- `get(timestamp, user, task_id) -> tuple | None` returns the task if it is an active task of `user`, and `None` otherwise.

**Example:**
- `add(5, "mia", "draft slides", 2)` is `"t1"` and `add(5, "raj", "draft slides", 4)` is `"t2"`: same title, same timestamp, two tasks
- `edit(6, "mia", "t1", "final slides", 6)` is `True`, and `get(6, "mia", "t1")` is `("t1", "final slides", 6, 5)`
- `finish(7, "raj", "t2")` is `True`; after it, `get(8, "raj", "t2")` is `None`
- `edit(8, "raj", "t1", "x", 1)` and `finish(8, "raj", "t1")` are `False`: `t1` belongs to `mia`""",
        },
        {
            "title": "Sorted lists",
            "description_en": r"""Keep Part 1 and add a list.

**Signature:** `list_tasks(at_timestamp, user, min_priority=None) -> list[tuple]`

- Return `user`'s active tasks with `priority >= min_priority`, or all of them when `min_priority` is `None`.
- Sort by priority, highest first, then by creation order: the earlier `add` call first, whatever the timestamps or titles.
- In this part `at_timestamp` is always the largest timestamp passed to any call so far.
- Each call returns a new list.

**Example:** `add(1, "raj", "logs", 2)`, `add(1, "raj", "alerts", 7)`, `add(2, "raj", "docs", 2)`, then `finish(3, "raj", "t2")` and `add(3, "raj", "oncall", 7)`:
- `list_tasks(3, "raj")` is `[("t4", "oncall", 7, 3), ("t1", "logs", 2, 1), ("t3", "docs", 2, 2)]`
- `list_tasks(3, "raj", 3)` is `[("t4", "oncall", 7, 3)]`, and `list_tasks(3, "mia")` is `[]`""",
        },
        {
            "title": "Expiry",
            "description_en": r"""Keep Parts 1–2 and let tasks expire.

**Signature:** `add(timestamp, user, title, priority, ttl=None) -> str`

- With a positive integer `ttl`, the task counts as finished at `created_at + ttl`: from that timestamp on, `get`, `list_tasks`, `edit` and `finish` treat it as finished.
- `ttl=None` never expires. `edit` never changes the deadline.
- Expiry is checked when a method is called, against that call's timestamp; nothing removes tasks in the background.

**Example:** `add(10, "mia", "renew cert", 3, ttl=4)` is `"t1"`, active while the timestamp is below `14`, and `add(11, "mia", "backup", 1)` is `"t2"`:
- `list_tasks(13, "mia")` holds both, and `edit(13, "mia", "t1", "renew certs", 5)` is `True`, with the deadline still `14`
- `list_tasks(14, "mia")` is `[("t2", "backup", 1, 11)]`, and `finish(14, "mia", "t1")` is `False`""",
        },
        {
            "title": "Lists at past timestamps",
            "description_en": r"""Keep Parts 1–3. `list_tasks` may now ask about any timestamp, including one earlier than calls already made.

- Answer as if the board had received only the calls with `timestamp <= at_timestamp`, then ask Part 3's `list_tasks` at that time. Every call at `at_timestamp` itself counts.
- With no such calls for `user`, that is `[]`. Asking about the past changes nothing.
- A history of 100,000 edits must still answer thousands of past lists in about a second.

**Example:** `add(2, "raj", "warmup", 5, ttl=6)`, `add(2, "mia", "audit", 1)`, `add(4, "raj", "eval", 3)`, `finish(6, "raj", "t1")`, then `edit(7, "raj", "t3", "eval v2", 9)`:
- `list_tasks(1, "raj")` is `[]`, and `list_tasks(5, "raj")` is `[("t1", "warmup", 5, 2), ("t3", "eval", 3, 4)]`
- `list_tasks(6, "raj")` is `[("t3", "eval", 3, 4)]`: the finish at `6` counts
- `list_tasks(7, "raj")` is `[("t3", "eval v2", 9, 4)]`, while `list_tasks(7, "mia")` is `[("t2", "audit", 1, 2)]`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which single lookup does every method start with, and what three conditions decide whether a task counts as found? If the id counter lived per user, what would two users' first tasks be called?"},
        {"level": 2, "kind": "analysis", "content": "Keep a dict from task id to a small record: owner, title, priority, created_at and an active flag, plus one counter for ids. Write one helper that returns the record only if it exists, belongs to the user and is active, and use it in get, edit and finish. Each call is then an O(1) dict lookup."},
    ],
    "model_connections": [
        "Experiment and job trackers keep runs per user with priorities and deadlines, and their dashboards sort by priority and age.",
        "Audit and evaluation tooling often asks what a queue or leaderboard looked like at a past moment, which needs versioned records rather than only the latest state.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One helper for 'active task of this user' keeps get, edit and finish consistent.",
            "Checking expiry at read time needs no background job and no clock of its own.",
            "Per-task version lists answer a past list with one binary search per task.",
        ],
        "cons": [
            "Listing sorts the user's tasks on every call, O(k log k) for k tasks.",
            "Version lists grow with every edit and are never trimmed.",
            "A past list still touches every task the user ever had, even finished ones.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import bisect


class _Task:
    def __init__(self, user, created, ttl, title, priority):
        self.user = user
        self.created = created
        self.ttl = ttl
        self.times = [created]  # when each version was written, never decreasing
        self.versions = [(title, priority)]
        self.finished = None  # timestamp of the finish call, if any

    def active(self, at):
        if at < self.created:
            return False
        if self.finished is not None and self.finished <= at:
            return False
        return self.ttl is None or at < self.created + self.ttl

    def version(self, at):
        return self.versions[bisect.bisect_right(self.times, at) - 1]  # the last write at or before at


class TaskBoard:
    def __init__(self):
        self._tasks = {}
        self._by_user = {}  # user -> task ids in creation order
        self._count = 0

    def _live(self, timestamp, user, task_id):
        task = self._tasks.get(task_id)
        if task is None or task.user != user or not task.active(timestamp):
            return None
        return task

    def _row(self, task_id, task, at):
        title, priority = task.version(at)
        return (task_id, title, priority, task.created)

    def add(self, timestamp, user, title, priority, ttl=None):
        self._count += 1
        task_id = f"t{self._count}"
        self._tasks[task_id] = _Task(user, timestamp, ttl, title, priority)
        self._by_user.setdefault(user, []).append(task_id)
        return task_id

    def edit(self, timestamp, user, task_id, title, priority):
        task = self._live(timestamp, user, task_id)
        if task is None:
            return False
        task.times.append(timestamp)
        task.versions.append((title, priority))
        return True

    def finish(self, timestamp, user, task_id):
        task = self._live(timestamp, user, task_id)
        if task is None:
            return False
        task.finished = timestamp
        return True

    def get(self, timestamp, user, task_id):
        task = self._live(timestamp, user, task_id)
        return None if task is None else self._row(task_id, task, timestamp)

    def list_tasks(self, at_timestamp, user, min_priority=None):
        rows = []
        for task_id in self._by_user.get(user, []):  # creation order, so a stable sort keeps ties in it
            task = self._tasks[task_id]
            if task.active(at_timestamp):
                row = self._row(task_id, task, at_timestamp)
                if min_priority is None or row[2] >= min_priority:
                    rows.append(row)
        rows.sort(key=lambda row: -row[2])
        return rows
''',
    "interview_questions": interview(
        concept=[
            "Why does one id counter shared by every user keep ids unique when titles and users repeat?",
            "Why should get, edit and finish treat a finished task, another user's task and an unknown id the same way?",
        ],
        deep_dive=[
            "What must edit check before it writes anything, and how do you keep that check in one place?",
        ],
        tradeoffs=[
            "How would you keep list_tasks fast for a user with many tasks whose priorities change often?",
            "Why check expiry when a method is called instead of deleting expired tasks in the background?",
            "How do you answer a list for a past timestamp without replaying every call since the start?",
            "Compare a version list per task with a full snapshot after every call: what does each cost in memory and per question?",
        ],
    ),
}
