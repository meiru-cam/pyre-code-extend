A *work queue* holds tasks that workers pull, process, and report back on. Each task is submitted with a caller-chosen `task_id`, unique among every task ever submitted, and an opaque `payload` that the queue never inspects. A worker is identified by a `worker_id` string. In one run, the calls to the methods below number at most $2 \times 10^5$ in total.

Every public method call is atomic: no two calls are ever interleaved with each other. A task moves through a small state machine. This part only needs three of its states:

- **`ready`**: submitted, not currently held by any worker, eligible for `reserve`
- **`reserved`**: held by a worker, being processed
- **`completed`**: processed successfully; terminal

### Part 1 — Reservation, completion and failure

`submit(task_id, payload)` adds a new task in the `ready` state. Raises `ValueError` if `task_id` was already submitted.

`reserve(worker_id)` hands the calling worker, out of the tasks currently `ready`, the one that entered `ready` earliest — first in, first out, regardless of whether a task got to `ready` by being freshly submitted or by being returned there. It moves that task to `reserved` and returns a `Reservation` holding `task_id`, `payload`, and a fresh *token*: a value, unique to this one act of reserving this one task, that `complete` and `fail` must present back. Returns `None` if no task is `ready`.

`complete(task_id, token)` marks the task `completed`. This only succeeds if the task is currently `reserved` and `token` matches the token issued by that reservation; otherwise it raises `InvalidReservationError`. If `task_id` was never submitted, it raises `UnknownTaskError` instead.

`fail(task_id, token)` checks `token` under the exact same rule as `complete`. Once it matches, the task goes back to `ready`, at the back of the queue. This part places no limit on how many times a task may fail and be retried.

Reserving a task again — after it fails, or after any other event returns it to `ready` — issues a brand new token. A token from an earlier reservation of the task never works again for `complete` or `fail`, even in the hands of a worker that still believes it owns the task.

```py
from typing import Any, NamedTuple


class Reservation(NamedTuple):
    task_id: str
    token: object
    payload: Any


class WorkQueue:
    def __init__(self) -> None: ...
    def submit(self, task_id: str, payload: Any) -> None: ...
    def reserve(self, worker_id: str) -> "Reservation | None": ...
    def complete(self, task_id: str, token: object) -> None: ...
    def fail(self, task_id: str, token: object) -> None: ...
```

Example:

```text
q = WorkQueue()
q.submit('t1', 'build')
q.submit('t2', 'test')

r1 = q.reserve('w1')          # r1.task_id == 't1'
r2 = q.reserve('w2')          # r2.task_id == 't2'
q.complete('t1', r1.token)    # 't1' -> completed
q.fail('t2', r2.token)        # 't2' -> ready, back of the queue

r3 = q.reserve('w3')          # r3.task_id == 't2' again, with a new token
q.complete('t2', r2.token)    # raises InvalidReservationError: r2.token is stale
q.complete('t2', r3.token)    # succeeds
```

### Part 2 — Lease timeouts

A worker can crash or hang after reserving a task without ever calling `complete` or `fail`. To bound how long a task can stay stuck, every reservation now carries a *lease*: a deadline after which it is considered abandoned.

Time comes from a manual clock, given below; the queue never reads the system clock.

```python
class ManualClock:
    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt
```

`WorkQueue`'s constructor now also takes `clock` (a `ManualClock`) and a positive integer `lease_duration`. When `reserve` issues a reservation at time `t = clock.now()`, its *lease deadline* is `t + lease_duration`; the reservation stays valid through and including that instant, and has *expired* once `clock.now()` is strictly greater than the deadline. `lease_duration` and every value the clock returns are at most $10^9$.

There is no background thread scanning for expirations. Instead, every call to a `WorkQueue` method, `submit` included, begins with a *reclaiming pass*: each reservation whose lease has expired by then is ended with the same effect as if its worker had called `fail` on it at that moment, so its task re-enters `ready` at the back of the queue. One pass handles the expired reservations one at a time, in ascending order of lease deadline, breaking ties by ascending `task_id`. Only then does the call do its own work.

Once a reservation is reclaimed this way, its token stops working for `complete`/`fail` under the exact same rule as Part 1 — a worker that still believes it holds the lease gets `InvalidReservationError`, indistinguishable from having raced an explicit `fail`.

```py
class WorkQueue:
    def __init__(self, clock: "ManualClock", lease_duration: int) -> None: ...
```

Example, with `lease_duration = 5`:

```text
t=0  q.submit('t1', 'x')
t=0  rA = q.reserve('A')          # deadline = 0 + 5 = 5, valid through t=5
t=7  clock.advance(7)             # no call yet -- expiry is not detected until the next call
t=7  rB = q.reserve('B')          # reclaims t1 first (7 > 5): back to ready, then re-reserved by B
t=7  q.complete('t1', rA.token)   # raises InvalidReservationError
t=7  q.complete('t1', rB.token)   # succeeds
```

### Part 3 — Retry budget and dead-letter queue

The constructor takes one more argument, a positive integer `max_attempts`. Each task now keeps an *attempt count*: the number of times it has been reserved, starting from 0 when it is submitted. There is also a fourth state, `dead`, which only `requeue_dead` can leave.

Whenever a `reserved` task leaves that state without completing — through `fail` or through lease expiry — look at its attempt count, which already includes the reservation that just ended. If the count has reached `max_attempts`, the task becomes `dead` instead of `ready`; otherwise it becomes `ready`, exactly as before. `complete` is unaffected: a completed task still ends in the terminal `completed` state.

`dead_letters()` returns the `task_id` of every task currently `dead`, in the order they became `dead`.

`requeue_dead(task_id)` moves a `dead` task to the back of the `ready` queue and resets its attempt count to 0, as if it had just been submitted, so it gets `max_attempts` more reservations. Raises `UnknownTaskError` if `task_id` was never submitted, or `ValueError` if it is not currently `dead`. Like every other method, `dead_letters` and `requeue_dead` begin with a reclaiming pass.

```py
class WorkQueue:
    def __init__(self, clock: "ManualClock", lease_duration: int, max_attempts: int) -> None: ...
    def dead_letters(self) -> list[str]: ...
    def requeue_dead(self, task_id: str) -> None: ...
```

Example, with `lease_duration = 3` and `max_attempts = 2`:

```text
t=0  q.submit('t1', 'x')
t=0  rA = q.reserve('A')          # 1st reservation
t=0  q.fail('t1', rA.token)       # 1 < 2 attempts so far -> ready
t=0  rB = q.reserve('B')          # 2nd reservation
t=0  q.fail('t1', rB.token)       # 2 >= 2 attempts -> dead
     q.dead_letters()             # -> ['t1']
     q.reserve('C')               # -> None, nothing ready
     q.requeue_dead('t1')         # attempt count reset to 0
     rC = q.reserve('C')          # 1st reservation again
```
