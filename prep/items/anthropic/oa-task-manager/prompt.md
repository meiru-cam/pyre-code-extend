Implement the four levels below in order: a level's tests must pass before the next level's tests run, and every level extends the same class rather than replacing it. A *task* belongs to exactly one *user* (a string user id) and has a `task_id`, a `title`, and an integer `priority` (a larger priority means a more urgent task); the same title may repeat, within one user's tasks or across different users, but every `task_id` is unique across the whole instance. Every method that changes state (`add_task`, `update_task`, `complete_task`) takes an integer *timestamp* as its first argument, and, across the whole sequence of such calls made to one instance, timestamps never decrease; two different calls, for the same task or for different tasks, may share a timestamp. Every read returns a task as a `Task` object, a plain immutable record:

```py
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    priority: int
    created_at: int
```

### Level 1 — Core operations

A task is *active* from the moment it is created until, at the latest, the moment it is completed; `get_task`, `update_task`, and `complete_task` all treat a task that is not active exactly as if its `task_id` did not exist.

```py
class TaskManager:
    def add_task(self, timestamp: int, user: str, title: str, priority: int) -> str:
        """Creates a new task owned by user, with the given title and priority, created at
        timestamp. Returns the new task's id. Ids are assigned from one counter shared by every
        user, in call order: the first call to add_task on this instance returns "task-1", the
        second "task-2", and so on -- ids are always unique, whatever title or user they carry."""

    def update_task(self, timestamp: int, user: str, task_id: str, title: str, priority: int) -> bool:
        """Replaces the title and priority of task_id, leaving created_at unchanged. Returns True
        on success. Returns False, and changes nothing, if task_id does not exist, is not owned by
        user, or is not active."""

    def complete_task(self, timestamp: int, user: str, task_id: str) -> bool:
        """Marks task_id completed: from this call on, it is no longer active. Returns True on
        success. Returns False, and changes nothing, if task_id does not exist, is not owned by
        user, or is already not active (including a task that is already completed)."""

    def get_task(self, timestamp: int, user: str, task_id: str) -> Task | None:
        """Returns task_id's current title, priority, and creation timestamp if it exists, is
        owned by user, and is active. None otherwise."""
```

Example:

```text
tm = TaskManager()
tm.add_task(10, "alice", "write report", 3)     # -> "task-1"
tm.add_task(12, "alice", "review PR", 5)        # -> "task-2"
tm.add_task(14, "bob", "write report", 1)       # -> "task-3"   -- title reused; owned by a different user
tm.get_task(15, "alice", "task-1")   # -> Task("task-1", "write report", 3, 10)
tm.get_task(15, "alice", "task-3")   # -> None                  -- task-3 belongs to bob, not alice
tm.update_task(16, "alice", "task-1", "write Q3 report", 4)   # -> True
tm.get_task(17, "alice", "task-1")   # -> Task("task-1", "write Q3 report", 4, 10)   -- created_at unchanged
tm.complete_task(18, "alice", "task-1")   # -> True
tm.complete_task(19, "alice", "task-1")   # -> False             -- already completed
tm.get_task(20, "alice", "task-1")   # -> None                   -- a completed task is not active
tm.update_task(21, "alice", "task-1", "x", 1)   # -> False       -- cannot update a task that is not active
```

### Level 2 — Listing and filters

```py
class TaskManager:
    def get_task_list(self, at_timestamp: int, user: str, min_priority: int | None = None) -> list[Task]:
        """Returns every active task owned by user whose priority is >= min_priority (every active
        task owned by user, if min_priority is None), evaluated at at_timestamp. Through Level 3,
        at_timestamp is guaranteed to equal the greatest timestamp passed to any call made so far
        on this instance -- every list here asks about the present, never the past. Tasks are
        sorted by priority descending; ties are broken by creation order ascending -- the order
        add_task was called, i.e. by task_id's numeric suffix, not by timestamp, since two tasks
        may be created with the same timestamp. [] if no task of user's matches."""
```

Example:

```text
tm = TaskManager()
tm.add_task(0, "alice", "write report", 3)      # -> "task-1"
tm.add_task(1, "alice", "review PR", 5)         # -> "task-2"
tm.add_task(2, "alice", "file expenses", 5)     # -> "task-3"     -- same priority as task-2, added later
tm.add_task(3, "alice", "renew badge", 1)       # -> "task-4"
tm.get_task_list(3, "alice")
# -> [Task("task-2", "review PR", 5, 1), Task("task-3", "file expenses", 5, 2),
#     Task("task-1", "write report", 3, 0), Task("task-4", "renew badge", 1, 3)]
#    priority 5 before 3 before 1; within priority 5, task-2 (created first) before task-3
tm.get_task_list(3, "alice", min_priority=3)
# -> [Task("task-2", ...), Task("task-3", ...), Task("task-1", ...)]   -- renew badge (priority 1) filtered out
tm.get_task_list(3, "carol")   # -> []                                 -- carol owns no tasks
```

### Level 3 — Expiry

From this level on, `add_task` accepts an optional `ttl`: a positive integer number of time units. A task created with a `ttl` is active only while `at_timestamp < created_at + ttl`; at `created_at + ttl` itself, and at every later timestamp, it is no longer active — exactly as if `complete_task` had been called on it at that instant, for every purpose: it disappears from `get_task` and `get_task_list`, and `update_task`/`complete_task` on it return `False`. A task created without a `ttl` (`ttl=None`, the default) never expires. `update_task` never changes a task's `ttl` or `created_at` — only its title and priority — so a task's expiry deadline, once `add_task` sets it, is fixed for the task's whole life. Every read applies this rule lazily, by comparing `created_at + ttl` against `at_timestamp`/`timestamp` at call time; no background process ever removes an expired task.

```py
class TaskManager:
    def add_task(self, timestamp: int, user: str, title: str, priority: int, ttl: int | None = None) -> str:
        """Same as before, but a positive ttl makes the task active only while
        at_timestamp < timestamp + ttl. ttl=None (the default) means the task never expires."""
```

Example:

```text
tm = TaskManager()
tm.add_task(0, "alice", "flush cache", 2, ttl=5)   # -> "task-1"   active while at_timestamp < 5
tm.add_task(0, "alice", "renew badge", 1)           # -> "task-2"   no ttl -> never expires
tm.get_task_list(4, "alice")   # -> [task-1, task-2]     4 < 5: task-1 still active
tm.get_task_list(5, "alice")   # -> [task-2]             5 == 0 + 5: task-1 has just expired
tm.get_task(5, "alice", "task-1")             # -> None
tm.update_task(5, "alice", "task-1", "x", 9)  # -> False               -- expired, behaves unknown
tm.complete_task(6, "alice", "task-2")        # -> True
tm.get_task_list(6, "alice")   # -> []
```

### Level 4 — Time travel

The guarantee that `at_timestamp` is always the latest timestamp seen so far is lifted: `get_task_list` may now be called with an `at_timestamp` earlier than one or more `add_task`, `update_task`, or `complete_task` calls already made — a genuine query into the past. It must return exactly the list a live `get_task_list` call would have returned at that instant: the tasks active at `at_timestamp`, reconstructed from only the calls with `timestamp <= at_timestamp` (every later call — whether it added a task, changed one, or completed one — must be invisible), with Level 3's expiry rule applied at `at_timestamp` itself, sorted and filtered exactly as in Level 2. `at_timestamp` may name any moment, including one before `user`'s first task was ever added, in which case the result is `[]`.

Example:

```text
tm = TaskManager()
tm.add_task(0, "alice", "write report", 3)                    # -> "task-1"
tm.add_task(2, "alice", "review PR", 5, ttl=10)                # -> "task-2"   active while at_timestamp < 12
tm.update_task(4, "alice", "task-1", "write Q3 report", 4)     # -> True
tm.complete_task(6, "alice", "task-2")                          # -> True
tm.get_task_list(20, "alice")
# -> [Task("task-1", "write Q3 report", 4, 0)]           -- "now": task-2 completed, task-1 already renamed
tm.get_task_list(1, "alice")
# -> [Task("task-1", "write report", 3, 0)]              -- before task-2 existed (created at 2 > 1) and
#                                                             before task-1's rename (at 4 > 1)
tm.get_task_list(3, "alice")
# -> [Task("task-2", "review PR", 5, 2), Task("task-1", "write report", 3, 0)]
#    task-2 exists and outranks by priority; task-1's rename (at 4) has not happened yet at 3
tm.get_task_list(6, "alice")
# -> [Task("task-1", "write Q3 report", 4, 0)]           -- task-2's completion at 6 already applies (inclusive)
```
