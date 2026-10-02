"""Workers clock in and out, then totals and a leaderboard, promotions with pay over a window, and double-pay periods."""

from ._interview import interview

# A tick-by-tick model of every rule, independent of the interval arithmetic in the reference, and a random call generator.
_MODEL = r"""
import random

class Model:
    def __init__(self):
        self.w = {}
        self.double = set()  # every double-pay tick

    def add_worker(self, wid, position, rate):
        if wid in self.w:
            return False
        self.w[wid] = {"pos": position, "rate": rate, "next": None, "open": None, "done": []}
        return True

    def clock_in(self, wid, ts):
        w = self.w.get(wid)
        if w is None or w["open"] is not None:
            return False
        if w["next"] is not None:
            w["pos"], w["rate"] = w["next"]
            w["next"] = None
        w["open"] = (ts, w["rate"])
        return True

    def clock_out(self, wid, ts):
        w = self.w.get(wid)
        if w is None or w["open"] is None:
            return False
        w["done"].append((w["open"][0], ts, w["open"][1]))
        w["open"] = None
        return True

    def get_total_work_time(self, wid):
        w = self.w.get(wid)
        return None if w is None else sum(e - s for s, e, _ in w["done"])

    def top_k_workers(self, k):
        return sorted(self.w, key=lambda wid: (-self.get_total_work_time(wid), wid))[:k]

    def set_promotion(self, wid, position, rate):
        if wid not in self.w:
            return False
        self.w[wid]["next"] = (position, rate)
        return True

    def get_position(self, wid):
        w = self.w.get(wid)
        return None if w is None else w["pos"]

    def calculate_pay(self, wid, start, end):
        w = self.w.get(wid)
        if w is None:
            return None
        sessions = list(w["done"]) + ([(w["open"][0], end, w["open"][1])] if w["open"] else [])
        total = 0
        for s, e, rate in sessions:
            for t in range(max(s, start), min(e, end)):
                total += rate * (2 if t in self.double else 1)
        return total

    def set_double_pay(self, start, end):
        self.double.update(range(start, end))

IDS = ["kim", "ada", "lu", "Ada"]

def random_calls(rng, steps, part):
    ts, calls = 0, []
    for _ in range(steps):
        wid = rng.choice(IDS)
        r = rng.random()
        if r < 0.12:
            calls.append(("add_worker", (wid, rng.choice(["dev", "ops"]), rng.randint(1, 9))))
        elif r < 0.45:
            ts += rng.choice([0, 0, 1, 2, 5])
            calls.append((rng.choice(["clock_in", "clock_out"]), (wid, ts)))
        elif part == 1:
            calls.append(("clock_in", (wid, ts)))
        elif r < 0.6:
            calls.append(("get_total_work_time", (wid,)))
        elif r < 0.7 or part == 2:
            calls.append(("top_k_workers", (rng.randint(1, 5),)))
        elif r < 0.78:
            calls.append(("set_promotion", (wid, rng.choice(["lead", "dev"]), rng.randint(1, 9))))
        elif r < 0.83:
            calls.append(("get_position", (wid,)))
        elif r < 0.92 or part == 3:
            a = rng.randint(0, ts + 5)
            calls.append(("calculate_pay", (wid, a, a + rng.randint(0, 30))))
        else:
            a = rng.randint(0, ts + 10)
            calls.append(("set_double_pay", (a, a + rng.randint(0, 12))))
    return calls

def replay(sheet, calls, label):
    model = Model()
    for i, (name, args) in enumerate(calls):
        want = getattr(model, name)(*args)
        got = getattr(sheet, name)(*args)
        assert got == want, (label, i, name, args, got, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
t = {fn}()
assert t.add_worker("w-07", "Annotator", 12) is True
assert t.add_worker("w-07", "Lead", 40) is False
assert t.clock_out("w-07", 3) is False
assert t.clock_in("w-07", 3) is True and t.clock_in("w-07", 4) is False
assert t.clock_out("w-07", 9) is True
assert t.clock_in("w-99", 9) is False
assert t.clock_in("w-07", 9) is True, "a new session may start when the last one ended"
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of add_worker, clock_in and clock_out, a return value differed from a model: a worker id is registered once, clock_in needs no open session, and clock_out needs one.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 50, 1), seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "metrics.ties", "code": r"""
t = {fn}()
for wid in ["kim", "ada", "lu"]:
    t.add_worker(wid, "Labeler", 10)
t.clock_in("ada", 0)
t.clock_in("kim", 5)
t.clock_in("lu", 10)
t.clock_out("kim", 20)
t.clock_in("kim", 25)
t.clock_out("ada", 30)
t.clock_out("kim", 40)
assert t.get_total_work_time("kim") == 30 and t.get_total_work_time("ada") == 30
assert t.get_total_work_time("lu") == 0, "an open session counts once it is closed"
assert t.top_k_workers(3) == ["ada", "kim", "lu"] and t.top_k_workers(1) == ["ada"]
assert t.get_total_work_time("zed") is None
"""},
    {"name": "Part 2: ties, case and short lists", "part": 2, "visibility": "unshown", "behavior": "metrics.ties",
     "failure_message": "top_k_workers ranks by closed time, highest first, then by worker id in Python's string order (so 'B' comes before 'a'); it returns every worker when k is larger, and counts zero-length and open sessions as 0.",
     "code": r"""
t = {fn}()
for wid in ["a", "B", "c"]:
    t.add_worker(wid, "x", 1)
assert t.top_k_workers(5) == ["B", "a", "c"]
t.clock_in("c", 4)
t.clock_out("c", 4)
t.clock_in("a", 4)
assert t.top_k_workers(5) == ["B", "a", "c"] and t.get_total_work_time("c") == 0
t.clock_out("a", 6)
assert t.top_k_workers(1) == ["a"] and t.top_k_workers(2) == ["a", "B"]
"""},
    {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence that also asks for totals and the leaderboard, a return value differed from a model that adds up closed sessions.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(100 + seed), 60, 2), seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
t = {fn}()
t.add_worker("ada", "Labeler", 10)
t.clock_in("ada", 0)
assert t.set_promotion("ada", "Reviewer", 15) is True
assert t.set_promotion("ada", "Senior", 25) is True
t.clock_out("ada", 20)
assert t.get_position("ada") == "Labeler"
t.clock_in("ada", 30)
assert t.get_position("ada") == "Senior"
t.clock_out("ada", 40)
assert t.calculate_pay("ada", 15, 35) == 5 * 10 + 5 * 25
t.clock_in("ada", 50)
assert t.calculate_pay("ada", 45, 60) == 10 * 25
assert t.calculate_pay("ada", 0, 50) == 20 * 10 + 10 * 25
assert t.calculate_pay("bo", 0, 50) is None and t.set_promotion("bo", "x", 1) is False
"""},
    {"name": "Part 3: promotions and windows", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "A promotion replaces position and rate together at the next clock_in, never during an open session or for past ones, and a second one before then replaces the first; pay sums the overlap of each session with [start, end) at that session's own rate; an open session counts as running until end; an empty window pays 0.",
     "code": r"""
t = {fn}()
t.add_worker("u", "dev", 3)
t.set_promotion("u", "lead", 7)
t.clock_in("u", 10)
t.set_promotion("u", "dev", 5)
assert t.calculate_pay("u", 0, 12) == 2 * 7, "the first promotion applied at clock_in"
t.clock_out("u", 14)
t.clock_in("u", 14)
assert t.get_position("u") == "dev"
assert t.calculate_pay("u", 12, 16) == 2 * 7 + 2 * 5
assert t.calculate_pay("u", 16, 16) == 0 and t.calculate_pay("u", 0, 10) == 0
assert t.calculate_pay("u", 0, 10**6) == 4 * 7 + (10**6 - 14) * 5
t.clock_out("u", 20)
assert t.calculate_pay("u", 0, 10**6) == 4 * 7 + 6 * 5, "a closed session stops at its clock_out"
assert t.get_position("nobody") is None
"""},
    {"name": "Part 3: random calls", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence with promotions and pay questions, a return value differed from a model that pays every time unit of every session at its rate.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(200 + seed), 70, 3), seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": r"""
t = {fn}()
t.add_worker("ada", "Labeler", 10)
t.clock_in("ada", 0)
t.clock_out("ada", 30)
assert t.set_double_pay(10, 20) is None
t.set_double_pay(15, 25)
t.set_double_pay(50, 50)
assert t.calculate_pay("ada", 0, 30) == 10 * 10 + 15 * 20 + 5 * 10
assert t.calculate_pay("ada", 12, 18) == 6 * 20
"""},
    {"name": "Part 4: periods for every worker and every time", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "A double-pay period covers [start, end) for every worker and for sessions before or after the call that set it; overlapping or touching periods double a time unit once, never more; empty periods change nothing.",
     "code": r"""
t = {fn}()
t.add_worker("a", "x", 1)
t.add_worker("b", "x", 3)
t.set_double_pay(5, 10)
t.clock_in("a", 0)
t.clock_in("b", 2)
t.clock_out("a", 20)
t.set_double_pay(8, 12)
t.set_double_pay(12, 14)
t.set_double_pay(6, 9)
t.set_double_pay(30, 20)
assert t.calculate_pay("a", 0, 20) == 20 + 9
assert t.calculate_pay("b", 0, 7) == 5 * 3 + 2 * 3, "b's open session runs until the window's end"
assert t.calculate_pay("b", 13, 15) == 2 * 3 + 3
"""},
    {"name": "Part 4: random calls", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence with double-pay periods, a return value differed from a model that doubles each time unit covered by any period.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(300 + seed), 80, 4), seed)
"""},
]

TASK = {
    "title": "Timesheet with Promotions and Double Pay",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "Timesheet",
    "description_en": r"""Build `Timesheet`, which records when workers clock in and out, then ranks them by time worked, pays them with promotions, and doubles pay inside chosen periods.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Timesheet` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A worker id is a non-empty, case-sensitive string. Once registered, it stays registered.
- Timestamps are non-negative integers. Across all `clock_in` and `clock_out` calls, for every worker together, they never decrease.
- Every interval is half-open: `[a, b)` holds `a` but not `b`. A session is `[clock_in time, clock_out time)`.
- A rate is a positive integer. Time `d` at rate `r` earns `d * r`.
- A call that fails returns the failure value given for it and changes nothing. No method raises.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment where each level reuses the session records of the last. Each later part adds one requirement, and the pay parts reward storing the rate on each session instead of reading the worker's current one.

**Where it is used:** time tracking and payroll, contractor billing with rate changes, and surge or overtime pricing over time windows.

Adapted from the worker management online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class with a new name. `get_position` is added in Part 3 so that promotions can be checked, and `set_double_pay` returns `None`.""",
    "parts": [
        {
            "title": "Clocking in and out",
            "description_en": r"""**Signatures:**
- `Timesheet().add_worker(worker_id, position, hourly_rate) -> bool` registers a worker. It returns `False` if the id is taken.
- `clock_in(worker_id, timestamp) -> bool` opens a session. It returns `False` if the worker is not registered or already has an open session.
- `clock_out(worker_id, timestamp) -> bool` closes the open session. It returns `False` if the worker is not registered or has no open session.

**Example:**
- `add_worker("w-07", "Annotator", 12)` is `True`, and `add_worker("w-07", "Lead", 40)` is `False`
- `clock_out("w-07", 3)` is `False`: nothing is open yet
- `clock_in("w-07", 3)` is `True`, then `clock_in("w-07", 4)` is `False`, and `clock_out("w-07", 9)` is `True`
- `clock_in("w-99", 9)` is `False`, but `clock_in("w-07", 9)` is `True`: a session may start when the last one ended""",
        },
        {
            "title": "Totals and a leaderboard",
            "description_en": r"""Keep Part 1 and add two methods.

**Signatures:**
- `get_total_work_time(worker_id) -> int | None` is the summed length of the worker's closed sessions, or `None` if the worker is not registered. An open session adds nothing until it is closed.
- `top_k_workers(k) -> list[str]` returns up to `k` worker ids, most time first, ties by worker id in Python's string order. `k` is positive; with fewer workers, return all of them.

**Example:** `kim`, `ada` and `lu` are registered; `ada` works `[0, 30)`, `kim` works `[5, 20)` and `[25, 40)`, and `lu` clocks in at `10` and stays in:
- `get_total_work_time("kim")` is `30`, the same as `ada`, and `get_total_work_time("lu")` is `0`
- `top_k_workers(3)` is `["ada", "kim", "lu"]`, and `top_k_workers(1)` is `["ada"]`
- `get_total_work_time("zed")` is `None`""",
        },
        {
            "title": "Promotions and pay",
            "description_en": r"""Keep Parts 1–2 and add three methods.

**Signatures:**
- `set_promotion(worker_id, new_position, new_hourly_rate) -> bool` schedules a new position and rate together. They take effect at the worker's next `clock_in`, never for a session already open or closed. A second call before that replaces the first. It returns `False` if the worker is not registered.
- `get_position(worker_id) -> str | None` is the current position, or `None` if the worker is not registered.
- `calculate_pay(worker_id, start, end) -> int | None` sums, over each session, the length of its overlap with `[start, end)` times the rate in effect when that session began. An open session counts as running until `end`. It returns `None` if the worker is not registered. `0 <= start <= end`.

**Example:** `add_worker("ada", "Labeler", 10)` and `clock_in("ada", 0)`; then `set_promotion("ada", "Reviewer", 15)` and `set_promotion("ada", "Senior", 25)`, and `clock_out("ada", 20)`:
- `get_position("ada")` is still `"Labeler"`; after `clock_in("ada", 30)` it is `"Senior"`, and the rate is `25`
- after `clock_out("ada", 40)`, `calculate_pay("ada", 15, 35)` is `5 * 10 + 5 * 25 = 175`
- after `clock_in("ada", 50)`, `calculate_pay("ada", 45, 60)` is `250`, and `calculate_pay("ada", 0, 50)` is `450`: the open session starts at `50`, outside `[0, 50)`""",
        },
        {
            "title": "Double-pay periods",
            "description_en": r"""Keep Parts 1–3 and add double pay.

**Signature:** `set_double_pay(start, end) -> None`

- Every time in `[start, end)` becomes double pay, for every worker, for sessions before or after this call. `0 <= start`; if `end <= start` nothing changes.
- Periods only add up; none is ever removed. A time covered by several periods is still paid twice its rate, not more.
- `calculate_pay` pays the part of each overlap that lies inside any period at twice the session's rate, and the rest at its rate.

**Example:** `ada` at rate `10` works `[0, 30)`; then `set_double_pay(10, 20)`, `set_double_pay(15, 25)` and `set_double_pay(50, 50)`:
- the double-pay time is `[10, 25)`, so `calculate_pay("ada", 0, 30)` is `10 * 10 + 15 * 20 + 5 * 10 = 450`
- `calculate_pay("ada", 12, 18)` is `6 * 20 = 120`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does each worker need to remember between a clock_in and the matching clock_out? How does that one field let both methods decide in O(1) whether the call is allowed?"},
        {"level": 2, "kind": "analysis", "content": "Keep a dict from worker id to a record with the position, the rate and the start of the open session, or None. add_worker fails if the id is in the dict. clock_in fails for an unknown id or a non-None start, and otherwise stores the timestamp. clock_out fails for an unknown id or a None start, and otherwise appends (start, timestamp) to the worker's sessions and clears the start."},
    ],
    "model_connections": [
        "Compute billing for training and labeling jobs charges per GPU-hour at the rate when the job started, with surge windows priced differently.",
        "Annotation vendors pay labelers by session time, with rate changes after a review and bonus periods for urgent batches.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Storing the rate on each session makes promotions never rewrite past pay.",
            "Merging double-pay periods into a sorted union makes 'doubled once' automatic.",
            "Every method is a dict lookup plus work on one worker's sessions.",
        ],
        "cons": [
            "top_k_workers sorts every worker on each call, O(n log n).",
            "calculate_pay walks every session of the worker and every period, O(s * p) without binary search.",
            "Sessions are never trimmed, so memory grows with every clock_out.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
class _Worker:
    def __init__(self, position, rate):
        self.position = position
        self.rate = rate
        self.pending = None  # (position, rate) waiting for the next clock_in
        self.open = None     # (start, rate) of the session in progress
        self.sessions = []   # closed sessions as (start, end, rate)
        self.total = 0


def _overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


class Timesheet:
    def __init__(self):
        self._workers = {}
        self._periods = []  # double-pay periods as given
        self._union = []    # their sorted, merged union, rebuilt when a period is added

    def add_worker(self, worker_id, position, hourly_rate):
        if worker_id in self._workers:
            return False
        self._workers[worker_id] = _Worker(position, hourly_rate)
        return True

    def clock_in(self, worker_id, timestamp):
        worker = self._workers.get(worker_id)
        if worker is None or worker.open is not None:
            return False
        if worker.pending is not None:  # a promotion starts with the next session, never inside one
            worker.position, worker.rate = worker.pending
            worker.pending = None
        worker.open = (timestamp, worker.rate)
        return True

    def clock_out(self, worker_id, timestamp):
        worker = self._workers.get(worker_id)
        if worker is None or worker.open is None:
            return False
        start, rate = worker.open
        worker.sessions.append((start, timestamp, rate))
        worker.total += timestamp - start
        worker.open = None
        return True

    def get_total_work_time(self, worker_id):
        worker = self._workers.get(worker_id)
        return None if worker is None else worker.total

    def top_k_workers(self, k):
        ranked = sorted(self._workers, key=lambda wid: (-self._workers[wid].total, wid))
        return ranked[:k]

    def set_promotion(self, worker_id, new_position, new_hourly_rate):
        worker = self._workers.get(worker_id)
        if worker is None:
            return False
        worker.pending = (new_position, new_hourly_rate)  # replaces any earlier pending promotion
        return True

    def get_position(self, worker_id):
        worker = self._workers.get(worker_id)
        return None if worker is None else worker.position

    def calculate_pay(self, worker_id, start, end):
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        sessions = list(worker.sessions)
        if worker.open is not None:
            sessions.append((worker.open[0], end, worker.open[1]))  # still running through the window
        pay = 0
        for a, b, rate in sessions:
            lo, hi = max(a, start), min(b, end)
            if lo >= hi:
                continue
            doubled = sum(_overlap(lo, hi, p, q) for p, q in self._union)
            pay += (hi - lo) * rate + doubled * rate
        return pay

    def set_double_pay(self, start, end):
        if end <= start:
            return None
        self._periods.append((start, end))
        merged = []
        for p, q in sorted(self._periods):
            if merged and p <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], q)  # overlapping or touching periods join
            else:
                merged.append([p, q])
        self._union = [tuple(m) for m in merged]
        return None
''',
    "interview_questions": interview(
        concept=[
            "Why should clock_in fail while a session is open instead of closing the old session first?",
            "With half-open sessions, why can a worker clock out and clock in again at the same timestamp?",
        ],
        deep_dive=[
            "What does each worker record need so that add_worker, clock_in and clock_out are all O(1)?",
        ],
        tradeoffs=[
            "How would you keep top_k_workers fast with many workers and frequent clock-outs?",
            "Why store the rate on each session instead of reading the worker's rate when pay is asked for?",
            "Why does an open session count as running until the end of the pay window, and what would go wrong otherwise?",
            "How do you make sure overlapping double-pay periods double a time only once, and how would you answer pay questions fast with many periods?",
        ],
    ),
}
