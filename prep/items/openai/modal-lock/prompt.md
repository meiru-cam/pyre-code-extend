A *mode* is a string that names how a shared resource is being used at a given moment -- for example, whether a GPU is currently running `"train"` jobs or `"infer"` jobs. `ModalLock` protects such a resource: at any instant, every thread holding the lock is using the same mode, and any number of threads may hold it together as long as they all ask for that one mode. A thread that asks for a different mode than the one currently held must block until every current holder has released.

The examples below are given in terms of a small tool, `Rig`, that drives a lock through an exact, chosen sequence of arrivals and releases without depending on sleeps or timing: every step blocks until its effect on the lock is actually visible, using `waiting_count()` -- the number of `acquire()` calls currently blocked, across every mode -- as the signal that a call has genuinely parked rather than been granted immediately.

```python
import threading
import time


def wait_until(predicate, timeout=5.0):
    """Block until predicate() is true. The timeout only guards against a real deadlock; it plays
    no role when the lock behaves correctly, so it does not make the check timing-dependent."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the lock to reach the expected state (deadlock?)")
        time.sleep(0.0005)


class _Waiter:
    def __init__(self, mode):
        self.mode = mode
        self.acquired = threading.Event()
        self.go_release = threading.Event()
        self.released = threading.Event()


class Rig:
    """name -> a dedicated thread that calls lock.acquire(mode) as soon as it is told to arrive,
    and lock.release(mode) as soon as it is told to release."""

    def __init__(self, lock):
        self.lock = lock
        self._w: dict[str, _Waiter] = {}

    def arrive(self, name, mode):
        """Start thread `name`, which calls acquire(mode) immediately. Returns only once that call
        has either been granted, or is genuinely parked inside acquire() -- so the caller's next
        step is guaranteed to happen strictly after this arrival, in program order."""
        w = _Waiter(mode)
        self._w[name] = w
        threading.Thread(target=self._run, args=(w,), daemon=True).start()
        wait_until(lambda: w.acquired.is_set() or self._parked(w))

    def _parked(self, w):
        # NOTE: count the other not-yet-granted calls BEFORE reading waiting_count(). That count can
        # only shrink meanwhile (a call woken by an earlier release may still be on its way out of
        # acquire()), so a waiting_count() above it can only come from w itself having parked.
        others = sum(1 for v in self._w.values() if v is not w and not v.acquired.is_set())
        return self.lock.waiting_count() > others

    def release(self, name):
        """Have `name` call release(mode) now. Returns once that call has returned."""
        w = self._w[name]
        w.go_release.set()
        wait_until(lambda: w.released.is_set())

    def holders(self):
        """The names whose acquire() has returned and whose release() has not."""
        return frozenset(n for n, w in self._w.items() if w.acquired.is_set() and not w.released.is_set())

    def _run(self, w):
        self.lock.acquire(w.mode)
        w.acquired.set()
        w.go_release.wait()
        self.lock.release(w.mode)
        w.released.set()
```

### Part 1 — ModalLock

`acquire(mode)` blocks until the lock's current mode is either absent (the lock is idle) or equal to `mode`, then makes `mode` the current mode and adds the caller to the set of holders. `release(mode)` removes the caller from that set; once the last holder of the current mode releases, the lock becomes idle again. `release(mode)` raises `ModeMismatchError` when the lock currently has no holders at all, or when its current mode is not `mode` -- both mean this call does not correspond to any reservation it actually holds.

```py
class ModalLock:
    def __init__(self) -> None: ...
    def acquire(self, mode: str) -> None: ...
    def release(self, mode: str) -> None: ...
    def waiting_count(self) -> int: ...
```

Example, with a fresh `ModalLock` and threads `a, b, c, d`: `a` requests `"train"` (granted at once); `b` requests `"train"` (joins `a`); `c` requests `"infer"` (blocks, `"train"` is held); `a` releases (`b` still holds `"train"`, so `c` keeps blocking); `b` releases (`"train"` drains to zero, `c` is granted `"infer"`); `d` requests `"infer"` (joins `c`). The holder set after each of these six steps is, in order: `{a}`, `{a, b}`, `{a, b}`, `{b}`, `{c}`, `{c, d}`.

### Part 2 — FairModalLock

`ModalLock` can starve a mode forever: as long as the current mode keeps gaining new same-mode holders before its last one releases, the lock is never idle, so a waiter for a different mode never gets a turn. `FairModalLock` grants modes in the order their requests arrived, by the following rule.

Every call to `acquire(mode)` joins a *batch*: a run of consecutively-arrived, same-mode requests that are served together. Batches sit in a single FIFO queue, oldest first, and the batch that currently holds the lock stays at the front of that queue. A call that finds the queue's *last* batch asking for `mode` joins that batch, even if it is the batch currently holding the lock; otherwise it starts a new batch at the back of the queue. A batch is served -- every one of its members becomes a holder -- as soon as it reaches the *front* of the queue, and a call that joins the front batch becomes a holder at once. The front batch leaves the queue once every member that ever joined it has released, and the next batch (if any) is served. So `acquire(mode)` returns at once on an idle lock, and also while the lock is held in `mode` with nothing queued behind the holders; but once a request for another mode is waiting, a new `acquire(mode)` queues behind it even though `mode` is the mode currently held. `release(mode)` raises `ModeMismatchError` under the same conditions as `ModalLock.release`: the lock has no holders at all, or its holders are not using `mode`.

```py
class FairModalLock:
    def __init__(self) -> None: ...
    def acquire(self, mode: str) -> None: ...
    def release(self, mode: str) -> None: ...
    def waiting_count(self) -> int: ...
```

The first three scenarios below use only two modes; each column after "Arrivals" happens after every earlier column, and the last column lists the batches in the order they run, each as the set of names that hold the lock together.

- Scenario: Basic fairness · Arrivals (name:mode): a:train, b:infer, c:train · Releases: a, b, c · Batches: `{a}`, `{b}`, `{c}`
- Scenario: Staggered arrivals · Arrivals (name:mode): a:train, b:infer, c:train, d:infer, e:train · Releases: a, b, c, d, e · Batches: `{a}`, `{b}`, `{c}`, `{d}`, `{e}`
- Scenario: Same-mode burst · Arrivals (name:mode): a:train, then b:infer, c:infer, d:infer · Releases: a, then b, c, d · Batches: `{a}`, `{b, c, d}`

In the basic-fairness row, `c` requests the same mode as `a` but must not join `a`'s already-running batch, because `b` is already queued behind it; `c` waits its own turn behind `b`. In the same-mode-burst row, all three `"infer"` arrivals queue up while `"train"` is held, then all three become holders together the instant `a` releases.

A third mode makes the "no cutting in line" rule visible with modes on both sides of the queue at once. With a fresh `FairModalLock`: `a` requests `"train"` (granted); `b` requests `"infer"` (queues); `c` requests `"export"` (queues); `d` requests `"infer"` (queues -- the queue's last batch is `c`'s `"export"` batch, so `d` cannot join `b`); `a` releases (`b`'s batch runs, holders `{b}`); `e` requests `"infer"` (the queue's last batch is now `d`'s, so `e` joins `d`, not `b`); `b` releases (`c`'s batch runs, holders `{c}`); `c` releases (`d`'s batch runs together with `e`, holders `{d, e}`); `d` and `e` release. The batches run in the order `{a}`, `{b}`, `{c}`, `{d, e}`.
