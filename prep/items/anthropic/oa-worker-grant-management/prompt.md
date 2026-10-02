A *worker* is identified by a `worker_id`, a non-empty string, unique among every worker ever registered — once registered, a `worker_id` stays registered for the lifetime of the instance. Implement the four levels below, in order: a level's checks must pass before the next level's checks run, and every level extends the same class rather than replacing it. Methods are called in the order their underlying events actually happen; `clock_in` and `clock_out` additionally carry that instant's `timestamp`, a non-negative integer, and across every `clock_in`/`clock_out` call made to one instance, regardless of which worker, timestamps never decrease. Every interval in this problem — an office session, the window given to `calculate_pay`, and a double-pay period — is half-open: `[a, b)` includes `a` and excludes `b`. An `hourly_rate` is a positive integer: a stretch of duration `d`, in the same units as a timestamp, at a constant rate `r` earns `d * r`. A `position` is a non-empty string, set when a worker is registered and replaced, together with the rate, by a promotion.

### Level 1 — Office sessions

```py
class Workforce:
    def add_worker(self, worker_id: str, position: str, hourly_rate: int) -> bool:
        """Registers a new worker with this position and hourly_rate. Returns True, or False
        without effect, if worker_id is already registered."""

    def clock_in(self, worker_id: str, timestamp: int) -> bool:
        """Opens a new office session for worker_id, starting at timestamp. Returns True, or
        False without effect, if worker_id is not registered, or if worker_id already has an
        open session (a clock_in with no clock_out since)."""

    def clock_out(self, worker_id: str, timestamp: int) -> bool:
        """Closes worker_id's open session as [start, timestamp), start being the timestamp of
        the clock_in that opened it. Returns True, or False without effect, if worker_id is not
        registered, or has no open session."""
```

Example:

```text
add_worker("alice", "Engineer", 15)   # -> True
add_worker("alice", "Manager", 30)    # -> False   -- "alice" is already registered
clock_in("alice", 100)                # -> True    -- opens [100, ...)
clock_in("alice", 105)                # -> False   -- already has an open session
clock_out("alice", 140)               # -> True    -- closes the session as [100, 140)
clock_out("alice", 150)               # -> False   -- no open session
clock_in("bob", 120)                  # -> False   -- "bob" was never registered
```

### Level 2 — Total time and the leaderboard

A worker's *closed* sessions are the ones `clock_out` has completed; a session still open (no matching `clock_out` yet) is not among them.

```py
class Workforce:
    def get_total_work_time(self, worker_id: str) -> int | None:
        """Returns the sum of the duration of every closed session of worker_id. Returns None
        if worker_id is not registered. An open session contributes nothing until it is closed."""

    def top_k_workers(self, k: int) -> list[str]:
        """Returns up to k worker_ids, ranked by get_total_work_time descending (workers tied on
        that total are ordered by worker_id ascending), or every registered worker_id in that
        order if fewer than k are registered. k is a positive integer."""
```

Example:

```text
add_worker("alice", "Engineer", 15); add_worker("bob", "Engineer", 20); add_worker("cy", "Designer", 15)
clock_in("alice", 0); clock_out("alice", 40)      # closed session, 40 ticks
clock_in("bob", 0);   clock_out("bob", 25)        # closed session, 25 ticks
clock_in("cy", 0);    clock_out("cy", 40)         # closed session, 40 ticks -- ties alice
get_total_work_time("alice")   # -> 40
get_total_work_time("dee")     # -> None   -- not registered
top_k_workers(2)                # -> ["alice", "cy"]          tied at 40: "alice" before "cy"
top_k_workers(10)               # -> ["alice", "cy", "bob"]   only 3 workers are registered
clock_in("bob", 100)             # bob's new session is open, not yet closed
top_k_workers(1)                 # -> ["alice"]                bob's open session still counts as 0
```

### Level 3 — Promotions and pay

`set_promotion` schedules a change of `position` and `hourly_rate` for a worker; it takes effect at that worker's next `clock_in`, never immediately and never retroactively — an already-open session keeps the rate it started with. At most one promotion can be pending per worker: calling `set_promotion` again before the pending one is consumed replaces it outright (the earlier call's `new_position` and `new_hourly_rate` are both discarded, not merged field by field).

```py
class Workforce:
    def set_promotion(self, worker_id: str, new_position: str, new_hourly_rate: int) -> bool:
        """Schedules new_position and new_hourly_rate to replace worker_id's current ones,
        effective at worker_id's next clock_in. Returns True, or False without effect, if
        worker_id is not registered."""

    def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:
        """Returns worker_id's total pay earned during [start, end): the sum, over every session
        that overlaps [start, end), of the overlap's duration times the rate that was in effect
        when that session's own clock_in happened. A session still open when calculate_pay is
        called is treated as ending exactly at end, as if it were still ongoing through the whole
        query, rather than at any later, real clock_out. Returns None if worker_id is not
        registered. 0 <= start <= end."""
```

Example:

```text
add_worker("alice", "Engineer", 10)
clock_in("alice", 0); clock_out("alice", 50)               # session A: [0, 50) at $10 -> 500
set_promotion("alice", "Senior Engineer", 18)               # pending -- session A is unaffected
clock_in("alice", 80)                                       # consumes the pending promotion: rate now 18
clock_out("alice", 100)                                     # session B: [80, 100) at $18 -> 360
calculate_pay("alice", 0, 100)    # -> 860     -- 500 (A) + 360 (B)
calculate_pay("alice", 10, 130)   # -> 760     -- A clipped to [10,50)=400; B fully in: 360
calculate_pay("bob", 0, 100)      # -> None    -- not registered
clock_in("alice", 150)                                       # session C is opened, still open
calculate_pay("alice", 120, 200)  # -> 900     -- C clamped to [150, 200): 50 ticks at 18
calculate_pay("alice", 0, 130)    # -> 860     -- C clamped to [150, 130) is empty: clock_in (150) >= end (130)
```

### Level 4 — Double-pay periods

`set_double_pay(start, end)` marks `[start, end)` as a *double-pay period*: any pay earned during it, by any worker, is doubled. Double-pay periods may be set so that they overlap each other; the overlap between two of them is still only ever doubled once, never quadrupled. `calculate_pay` is extended accordingly: within the overlap of a session with `[start, end)` computed as in Level 3, the sub-interval that additionally falls inside at least one double-pay period is paid at twice the session's own rate; the rest of that overlap is paid at the session's own rate, unchanged.

```py
class Workforce:
    def set_double_pay(self, start: int, end: int) -> None:
        """Marks every timestamp in [start, end) as double pay for every worker. 0 <= start <=
        end. Double-pay periods accumulate; a later call never removes or shrinks an earlier one,
        even where they overlap."""
```

`calculate_pay` keeps the signature of Level 3; only the value it computes changes.

Example:

```text
add_worker("alice", "Engineer", 10)
clock_in("alice", 0); clock_out("alice", 50)          # session A: [0, 50) at $10
set_promotion("alice", "Senior Engineer", 20)
clock_in("alice", 60); clock_out("alice", 120)        # session B: [60, 120) at $20
set_double_pay(40, 70)                                 # double pay on [40, 70)
set_double_pay(65, 100)                                # double pay on [65, 100), overlapping the first
                                                        # -- union is [40, 100); the shared part [65,70)
                                                        # is still only ever 2x, never 4x
calculate_pay("alice", 0, 120)
# session A [0, 50): [0, 40) at 1x -> 40 * 10 = 400; [40, 50) at 2x -> 10 * 10 * 2 = 200 -> 600
# session B [60, 120): [60, 100) at 2x -> 40 * 20 * 2 = 1600; [100, 120) at 1x -> 20 * 20 = 400 -> 2000
# total: 600 + 2000 = 2600
```
