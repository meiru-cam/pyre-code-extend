Worth confirming up front, because reasonable designs differ on them: how an open session — one with no `clock_out` yet — should be treated by `calculate_pay` (here, clamped to the query's own `end`, since the query carries no separate "as of" time of its own); and whether a repeated `clock_in` while already clocked in, or a `set_promotion` that arrives before the pending one is consumed, should raise or fail quietly (here, both fail quietly or overwrite, matching how `clock_out` and `add_worker` already signal failure).

### Level 1

A worker is a small object carrying its `position` and `hourly_rate`, the timestamp of an open session if one exists, and a list of closed sessions. A closed session is stored as `(start, end, rate)`, the rate being the one in effect for that whole session — recording it here, at `clock_out` time, costs nothing and is exactly what `calculate_pay` needs from Level 3 on, since a later promotion must never change a session that has already closed. `clock_in` also consumes a pending promotion, if any, before opening the session, so that the very session it opens already uses the new rate.

```python
class _Worker:
    def __init__(self, position: str, hourly_rate: int):
        self.position = position
        self.hourly_rate = hourly_rate
        self.open_clock_in: int | None = None
        self.closed_sessions: list[tuple[int, int, int]] = []   # (start, end, rate), rate fixed at close
        self.pending_promotion: tuple[str, int] | None = None


class Workforce:
    def __init__(self):
        self._workers: dict[str, _Worker] = {}

    def add_worker(self, worker_id: str, position: str, hourly_rate: int) -> bool:
        if worker_id in self._workers:
            return False
        self._workers[worker_id] = _Worker(position, hourly_rate)
        return True

    def clock_in(self, worker_id: str, timestamp: int) -> bool:
        worker = self._workers.get(worker_id)
        if worker is None or worker.open_clock_in is not None:
            return False
        if worker.pending_promotion is not None:      # NOTE: consumed here -- not when set_promotion was called
            worker.position, worker.hourly_rate = worker.pending_promotion
            worker.pending_promotion = None
        worker.open_clock_in = timestamp
        return True

    def clock_out(self, worker_id: str, timestamp: int) -> bool:
        worker = self._workers.get(worker_id)
        if worker is None or worker.open_clock_in is None:
            return False
        # NOTE: worker.hourly_rate cannot have changed since this session opened: clock_in is the only
        # place a promotion is consumed, and a second clock_in is refused while a session is open.
        worker.closed_sessions.append((worker.open_clock_in, timestamp, worker.hourly_rate))
        worker.open_clock_in = None
        return True
```

Every Level 1 method is $O(1)$.

### Level 2

`get_total_work_time` sums the duration of the closed sessions only, matching the statement's "closed" definition; `top_k_workers` sorts every registered worker by a key that negates that sum, so the largest total sorts first, with `worker_id` itself, not negated, breaking ties ascending.

```python
def _closed_time(worker: _Worker) -> int:
    return sum(end - start for start, end, _ in worker.closed_sessions)


def get_total_work_time(self, worker_id: str) -> int | None:
    worker = self._workers.get(worker_id)
    return None if worker is None else _closed_time(worker)


def top_k_workers(self, k: int) -> list[str]:
    ranked = sorted(self._workers.items(), key=lambda item: (-_closed_time(item[1]), item[0]))
    return [worker_id for worker_id, _ in ranked[:k]]


Workforce.get_total_work_time = get_total_work_time
Workforce.top_k_workers = top_k_workers
```

$O(n)$ for `get_total_work_time` in the number of sessions; $O(m \log m)$ for `top_k_workers` in the number of registered workers $m$ (dominated by the sort; slicing to $k$ afterwards adds nothing a separate heap would meaningfully save at this scale).

### Level 3

`set_promotion` only ever replaces `pending_promotion` outright, so an unconsumed promotion followed by another one leaves just the second. `calculate_pay` needs the overlap, in ticks, between two half-open intervals; the same `_overlap` helper serves every session, including the open one, by first building a uniform list of `(start, end, rate)` triples where the open session (if any) is given `end` itself as its end — the query's own `end`, never a later real `clock_out`. Passing that clamped interval through the very same overlap arithmetic as every closed session, rather than special-casing it, also makes it disappear correctly on its own: when the open session's `start` is at or past the query's `end`, the clamped interval is empty and `_overlap` already returns 0 for it.

```python
def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the overlap between half-open [a_start, a_end) and [b_start, b_end)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _sessions_as_of(worker: _Worker, query_end: int) -> list[tuple[int, int, int]]:
    """Every closed session, plus the open one (if any) clamped to end at query_end."""
    sessions = list(worker.closed_sessions)
    if worker.open_clock_in is not None:      # NOTE: open at query time -- treated as ongoing through query_end
        sessions.append((worker.open_clock_in, query_end, worker.hourly_rate))
    return sessions


def set_promotion(self, worker_id: str, new_position: str, new_hourly_rate: int) -> bool:
    worker = self._workers.get(worker_id)
    if worker is None:
        return False
    worker.pending_promotion = (new_position, new_hourly_rate)   # NOTE: replaces, does not merge, any older one
    return True


def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:   # NOTE: redefined by Level 4 below
    worker = self._workers.get(worker_id)
    if worker is None:
        return None
    return sum(_overlap(s, e, start, end) * rate for s, e, rate in _sessions_as_of(worker, end))


Workforce.set_promotion = set_promotion
Workforce.calculate_pay = calculate_pay
```

$O(1)$ for `set_promotion`; $O(n)$ for `calculate_pay` in the number of sessions the worker has, closed or open.

### Level 4

Double-pay periods are kept merged into a sorted, disjoint list, so that summing the overlap of one segment against every stored period equals its overlap against their union — no period is ever counted twice for a tick both cover. `_merge_intervals` is the standard sweep: sort by start, and extend the last interval instead of starting a new one whenever the next one starts at or before its end (`<=`, not `<`, so touching intervals such as `[10, 20)` and `[20, 30)` merge into one — this changes nothing about the pay any interval computes, only how compactly the periods are stored). `calculate_pay` is redefined rather than patched: for every session's overlap with `[start, end)`, the part of that overlap that additionally falls inside the merged periods is paid at `2 * rate`; the remainder, at `rate`.

```python
def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def set_double_pay(self, start: int, end: int) -> None:
    if not hasattr(self, "_bonus_periods"):     # NOTE: lazy init -- predates __init__, as in Level 1
        self._bonus_periods: list[tuple[int, int]] = []
    self._bonus_periods = _merge_intervals(self._bonus_periods + [(start, end)])


def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:
    worker = self._workers.get(worker_id)
    if worker is None:
        return None
    bonus_periods = getattr(self, "_bonus_periods", [])
    total = 0
    for s, e, rate in _sessions_as_of(worker, end):
        seg_start, seg_end = max(s, start), min(e, end)
        if seg_start >= seg_end:
            continue
        # NOTE: bonus_periods is already merged, so these overlaps are disjoint and simply add up --
        # a chain of three overlapping double-pay windows still doubles their union once, not 3x or 4x.
        doubled = sum(_overlap(seg_start, seg_end, bs, be) for bs, be in bonus_periods)
        total += (seg_end - seg_start - doubled) * rate + doubled * rate * 2
    return total


Workforce.set_double_pay = set_double_pay
Workforce.calculate_pay = calculate_pay
```

`set_double_pay` is $O(p \log p)$ in the number of periods stored so far, from the sort inside `_merge_intervals`; `calculate_pay` is $O(n + p)$ per session, for $n$ sessions and $p$ merged periods.

### Follow-ups

- A variant where a double-pay period only counts for a session it covers completely, rather than prorating a partial overlap: `calculate_pay` would check containment session by session instead of summing `_overlap` against `_bonus_periods`, and the per-tick doubling above would collapse to an all-or-nothing multiplier per session.
- A promotion effective from a specific future timestamp, rather than "the next `clock_in`": that would need a session's rate to change partway through it, splitting `calculate_pay`'s per-session overlap into a before and an after segment, the way a double-pay period already splits one.
- An aggregate across every worker, such as the total of every doubled tick paid out in a range: it reuses `_bonus_periods` and `_overlap` unchanged, iterating every worker's sessions instead of one.
- Concurrent `clock_in`/`clock_out` for the same worker: both read and write `open_clock_in` without a lock, so two threads racing to clock the same worker in could both see it unset and both open a session. A per-worker lock held for the duration of `clock_in` and `clock_out` serializes exactly the calls that touch one worker's state, without blocking calls on other workers.
- A worker with a very long session history queried often: `calculate_pay` rescans every session on every call. Sorting sessions by `start` once and walking only the ones that could overlap `[start, end)`, found with `bisect`, bounds the work by the sessions actually touched instead of the worker's whole history.

```python
# ---- Level 1 example ----
w = Workforce()
assert w.add_worker("alice", "Engineer", 15) is True
assert w.add_worker("alice", "Manager", 30) is False
assert w.clock_in("alice", 100) is True
assert w.clock_in("alice", 105) is False
assert w.clock_out("alice", 140) is True
assert w.clock_out("alice", 150) is False
assert w.clock_in("bob", 120) is False

# ---- Level 2 example ----
w = Workforce()
for wid, pos, rate in [("alice", "Engineer", 15), ("bob", "Engineer", 20), ("cy", "Designer", 15)]:
    w.add_worker(wid, pos, rate)
w.clock_in("alice", 0); w.clock_out("alice", 40)
w.clock_in("bob", 0); w.clock_out("bob", 25)
w.clock_in("cy", 0); w.clock_out("cy", 40)
assert w.get_total_work_time("alice") == 40
assert w.get_total_work_time("dee") is None
assert w.top_k_workers(2) == ["alice", "cy"]
assert w.top_k_workers(10) == ["alice", "cy", "bob"]
w.clock_in("bob", 100)
assert w.top_k_workers(1) == ["alice"]

# ---- Level 3 example ----
w = Workforce()
w.add_worker("alice", "Engineer", 10)
w.clock_in("alice", 0); w.clock_out("alice", 50)
w.set_promotion("alice", "Senior Engineer", 18)
w.clock_in("alice", 80)
w.clock_out("alice", 100)
assert w.calculate_pay("alice", 0, 100) == 860
assert w.calculate_pay("alice", 10, 130) == 760
assert w.calculate_pay("bob", 0, 100) is None
w.clock_in("alice", 150)
assert w.calculate_pay("alice", 120, 200) == 900
assert w.calculate_pay("alice", 0, 130) == 860

# ---- Level 4 example ----
w = Workforce()
w.add_worker("alice", "Engineer", 10)
w.clock_in("alice", 0); w.clock_out("alice", 50)
w.set_promotion("alice", "Senior Engineer", 20)
w.clock_in("alice", 60); w.clock_out("alice", 120)
w.set_double_pay(40, 70)
w.set_double_pay(65, 100)
assert w.calculate_pay("alice", 0, 120) == 2600
print("all four levels' worked examples replayed exactly")

# ---- edge cases the examples do not reach ----

# add_worker returning False leaves the original position/rate untouched
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.add_worker("a", "Manager", 999)
w.clock_in("a", 0); w.clock_out("a", 10)
assert w.calculate_pay("a", 0, 10) == 100   # still $10/tick, not 999

# a promotion set while a session is open only reaches the session after the NEXT clock_in
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.clock_in("a", 0)                        # session open
w.set_promotion("a", "Senior", 50)        # pending -- the open session must not see it
w.clock_out("a", 10)
assert w.calculate_pay("a", 0, 10) == 100           # closed at the old rate
w.clock_in("a", 10)                                  # consumes the pending promotion now
w.clock_out("a", 20)
assert w.calculate_pay("a", 10, 20) == 500           # new session at the new rate

# a second set_promotion before the first is consumed: only the later one survives
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.set_promotion("a", "Mid", 20)
w.set_promotion("a", "Senior", 30)        # replaces the pending "Mid"/20 outright
w.clock_in("a", 0); w.clock_out("a", 10)
assert w.calculate_pay("a", 0, 10) == 300

# a zero-duration session (clock_in and clock_out at the same timestamp) contributes nothing
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.clock_in("a", 5); w.clock_out("a", 5)
assert w.get_total_work_time("a") == 0
assert w.calculate_pay("a", 0, 100) == 0

# an open session that starts at or after the query's end contributes nothing
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.clock_in("a", 50)
assert w.calculate_pay("a", 0, 50) == 0
assert w.calculate_pay("a", 0, 49) == 0

# three pairwise-overlapping double-pay periods still double each tick exactly once
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.clock_in("a", 0); w.clock_out("a", 30)
w.set_double_pay(0, 12)
w.set_double_pay(8, 20)
w.set_double_pay(15, 30)                  # union of all three is exactly [0, 30)
assert w.calculate_pay("a", 0, 30) == 30 * 10 * 2

# two separate (non-overlapping) double-pay periods inside one session stay independently additive
w = Workforce()
w.add_worker("a", "Engineer", 10)
w.clock_in("a", 0); w.clock_out("a", 30)
w.set_double_pay(0, 5)      # 5 ticks doubled
w.set_double_pay(20, 25)    # 5 ticks doubled
# [0,5) 2x=100, [5,20) 1x=150, [20,25) 2x=100, [25,30) 1x=50
assert w.calculate_pay("a", 0, 30) == 100 + 150 + 100 + 50

# set_double_pay before any worker, or any session, exists is harmless
w2 = Workforce()
w2.set_double_pay(0, 100)
assert w2.calculate_pay("nobody", 0, 100) is None
w2.add_worker("z", "Engineer", 5)
assert w2.calculate_pay("z", 0, 100) == 0

# top_k_workers with k larger than the number registered, everyone tied at 0
w3 = Workforce()
for wid in ("m", "a", "z"):
    w3.add_worker(wid, "Engineer", 10)
assert w3.top_k_workers(5) == ["a", "m", "z"]   # nobody has clocked in: all tied at 0, ascending id
print("edge cases OK")
```

```python
# ---- independent reference model: tick-by-tick simulation from the raw call log ----
# Shares no code with the Workforce implementation above -- calculate_pay is answered by walking every
# integer tick of the query window one at a time, rather than with interval-overlap arithmetic.
import random


def _slow_replay(log):
    workers: dict[str, dict] = {}
    bonus: list[tuple[int, int]] = []
    for entry in log:
        kind = entry[0]
        if kind == "add_worker":
            _, wid, pos, rate = entry
            if wid not in workers:
                workers[wid] = {"position": pos, "rate": rate, "pending": None, "open": None, "closed": []}
        elif kind == "clock_in":
            _, wid, ts = entry
            rec = workers.get(wid)
            if rec is not None and rec["open"] is None:
                if rec["pending"] is not None:
                    rec["position"], rec["rate"] = rec["pending"]
                    rec["pending"] = None
                rec["open"] = ts
        elif kind == "clock_out":
            _, wid, ts = entry
            rec = workers.get(wid)
            if rec is not None and rec["open"] is not None:
                rec["closed"].append((rec["open"], ts, rec["rate"]))
                rec["open"] = None
        elif kind == "set_promotion":
            _, wid, pos, rate = entry
            rec = workers.get(wid)
            if rec is not None:
                rec["pending"] = (pos, rate)
        else:                                    # "set_double_pay"
            _, s, e = entry
            bonus.append((s, e))
    return workers, bonus


def _slow_total_time(workers, wid):
    return None if wid not in workers else sum(e - s for s, e, _ in workers[wid]["closed"])


def _slow_top_k(workers, k):
    return sorted(workers, key=lambda wid: (-_slow_total_time(workers, wid), wid))[:k]


def _slow_calculate_pay(workers, bonus, wid, start, end):
    if wid not in workers:
        return None
    rec = workers[wid]
    sessions = list(rec["closed"])
    if rec["open"] is not None:
        sessions.append((rec["open"], end, rec["rate"]))
    total = 0
    for tick in range(start, end):
        for s, e, rate in sessions:
            if s <= tick < e:                   # NOTE: one worker's sessions never overlap each other
                total += rate * 2 if any(bs <= tick < be for bs, be in bonus) else rate
                break
    return total


def _apply(workforce, entry):
    getattr(workforce, entry[0])(*entry[1:])


def _run_trial(seed):
    rng = random.Random(seed)
    ids = ["a", "b", "c"]
    workforce, log, ts = Workforce(), [], 0
    for _ in range(rng.randint(15, 25)):
        kind = rng.choices(
            ["add_worker", "clock_in", "clock_out", "set_promotion", "set_double_pay"],
            weights=[2, 5, 5, 3, 2])[0]
        wid = rng.choice(ids)
        if kind == "add_worker":
            entry = ("add_worker", wid, rng.choice(["Eng", "Designer", "Manager"]), rng.randint(2, 12))
        elif kind == "clock_in":
            ts += rng.choice([0, 0, 1, 2])
            entry = ("clock_in", wid, ts)
        elif kind == "clock_out":
            ts += rng.choice([0, 1, 2, 3])
            entry = ("clock_out", wid, ts)
        elif kind == "set_promotion":
            entry = ("set_promotion", wid, rng.choice(["Eng II", "Staff", "Lead"]), rng.randint(2, 12))
        else:
            s = rng.randint(0, min(ts, 40) + 5)
            entry = ("set_double_pay", s, s + rng.randint(0, 8))

        _apply(workforce, entry)
        log.append(entry)
        workers, bonus = _slow_replay(log)

        for wid2 in ids + ["ghost"]:
            assert workforce.get_total_work_time(wid2) == _slow_total_time(workers, wid2), \
                (seed, "get_total_work_time", wid2, log)
        for k in (1, 2, len(ids) + 2):
            assert workforce.top_k_workers(k) == _slow_top_k(workers, k), (seed, "top_k_workers", k, log)
        hi = min(ts + 8, 60)
        for wid2 in ids + ["ghost"]:
            a, b = sorted(rng.choices(range(hi + 1), k=2))
            got, want = workforce.calculate_pay(wid2, a, b), _slow_calculate_pay(workers, bonus, wid2, a, b)
            assert got == want, (seed, "calculate_pay", wid2, a, b, got, want, log)


for seed in range(200):
    _run_trial(seed)
print("cross-validated 200 random operation sequences against an independent tick-by-tick reference model")
print("all checks passed")
```
