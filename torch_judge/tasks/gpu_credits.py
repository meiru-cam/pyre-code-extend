"""A ledger of expiring GPU credit grants, first in order, then out of order, then fast."""

from ._interview import interview

# A slow model written straight from the statement: every query replays every call so far,
# sorted by timestamp, from an empty ledger.
_HELPERS = r"""
import random, time

class Replay:
    def __init__(self):
        self.calls = []
    def add_credit(self, credit_id, amount, timestamp, expiration):
        self.calls.append((timestamp, amount, timestamp + expiration))
    def subtract(self, amount, timestamp):
        self.calls.append((timestamp, -amount, None))
    def get_balance(self, timestamp):
        grants, debt = [], 0  # grants: [expires at, remaining]
        for ts, amount, end in sorted(self.calls):
            if ts > timestamp:
                break
            grants = [g for g in grants if g[0] >= ts]
            if amount > 0:
                paid = min(debt, amount)
                debt -= paid
                grants.append([end, amount - paid])
            else:
                need = -amount
                for grant in sorted(grants, key=lambda g: g[0]):
                    take = min(grant[1], need)
                    grant[1] -= take
                    need -= take
                debt += need
        value = sum(g[1] for g in grants if g[0] >= timestamp) - debt
        return None if value < 0 else value

def random_calls(rng, n, span=60):
    stamps = rng.sample(range(span), n)
    calls = []
    for i, ts in enumerate(stamps):
        if rng.random() < 0.55:
            calls.append(("add_credit", (f"g{i}", rng.randint(1, 9), ts, rng.randint(0, 25))))
        else:
            calls.append(("subtract", (rng.randint(1, 12), ts)))
    return calls

def call_ts(call):
    name, args = call
    return args[2] if name == "add_credit" else args[1]

def best_of_three(fn):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
"""

TASK = {
    "title": "GPU Credit Ledger",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "GPUCreditLedger",
    "description_en": r"""Build `GPUCreditLedger`, which issues GPU credits in expiring grants, spends them, and reports the balance at any time.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `GPUCreditLedger` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `add_credit(credit_id, amount, timestamp, expiration)` issues a grant of `amount` credits. It can be used at every time from `timestamp` to `timestamp + expiration`, both included.
- `subtract(amount, timestamp)` spends `amount`. It takes credit from the usable grants in order of their last usable time, soonest first, moving on when one runs out. Spent credit is gone.
- If the usable grants hold less than `amount`, the rest becomes debt. A grant issued later pays off the debt first, and only what is left becomes usable credit. Debt paid this way is never returned.
- `get_balance(timestamp)` considers only calls with a timestamp at or before it, applied in timestamp order. It returns the credit left in grants usable at that time minus the debt, or `None` if that is negative. With nothing usable and no debt it returns `0`.
- Amounts, timestamps and expirations are non-negative integers, amounts at least `1`. No two `add_credit` or `subtract` calls share a timestamp.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the rules are fiddly but each is small, and each later part adds one requirement.

**Where it is used:** cloud credit and prepaid billing systems, where promotional grants expire and usage arrives late.

Adapted from the GPU credits question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Calls in time order",
            "description_en": r"""**Signature:** `add_credit(credit_id, amount, timestamp, expiration) -> None`, `subtract(amount, timestamp) -> None`, `get_balance(timestamp) -> int | None`

- In this part every call, queries included, arrives in non-decreasing order of `timestamp`.

**Example:**
- `add_credit("a", 8, 0, 30)`: usable from 0 to 30; `get_balance(0)` is `8`
- `add_credit("b", 4, 5, 10)`: usable from 5 to 15
- `subtract(6, 10)` empties `b` first, then takes 2 from `a`; `get_balance(10)` is `6`
- `get_balance(31)` is `0`: `a` has expired
- `subtract(3, 40)` leaves a debt of 3; `get_balance(40)` is `None`
- `add_credit("c", 5, 50, 10)` pays the debt and keeps 2; `get_balance(50)` is `2`""",
        },
        {
            "title": "Calls in any order",
            "description_en": r"""Keep Part 1. Now `add_credit`, `subtract` and `get_balance` may arrive in any order of `timestamp`.

- `get_balance(timestamp)` still answers as if every call made so far with a timestamp at or before it had been applied in timestamp order.
- A call that arrives late changes later answers, including ones about times already queried.

**Example:**
- `subtract(5, 20)` arrives first; `get_balance(10)` is `0`
- `add_credit("a", 7, 15, 20)` arrives next; `get_balance(15)` is `7` and `get_balance(20)` is `2`""",
        },
        {
            "title": "Fast queries",
            "description_en": r"""Keep Parts 1–2 and make the common case fast. `U` is the number of `add_credit` and `subtract` calls, and `Q` the number of `get_balance` calls.

- When queries arrive in non-decreasing order of `timestamp`, each after every call at or before its time, all `Q` queries together cost `O(U log U + Q)`, not a replay per query.
- Any other order may take a slower path but must return the same answers as Part 2.

**Example:** 3,000 calls, each followed by a query at its own timestamp, take about ten times as long as 300 such calls, not a hundred times.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which grants are usable at a given time, and which one should a subtract drain first? What happens to the part of a subtract nothing could cover? When a new grant arrives while debt is outstanding, where does its amount go first?"},
        {"level": 2, "kind": "analysis", "content": "Keep a min-heap of [last usable time, remaining] for the grants issued so far, and a debt counter. Before each call, pop grants whose last usable time is before the call's timestamp. add_credit pays debt first and pushes the rest; subtract drains from the heap top and adds any shortfall to debt."},
    ],
    "model_connections": [
        "Cloud providers grant promotional and committed-use credits that expire, and their billing pipelines must apply late usage records to the right grant.",
        "Event-sourced ledgers rebuild balances by replaying a time-ordered log, and keep checkpoints so they do not replay from the start each time.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A min-heap keyed by expiry drains the soonest-expiring grant first and drops expired grants in O(log n).",
            "Replaying calls in timestamp order makes late arrivals correct with no special cases.",
            "Advancing one cursor through sorted calls makes in-order queries cost amortised O(log U) each.",
        ],
        "cons": [
            "Replaying everything per query costs O(U log U) per call, which is too slow for long histories.",
            "The incremental cursor is invalidated by a call behind it, so out-of-order traffic falls back to a rebuild.",
            "Keeping every call for rebuilds grows memory with the history; real systems checkpoint and compact.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
ledger = {fn}()
ledger.add_credit("a", 8, 0, 30)
assert ledger.get_balance(0) == 8
ledger.add_credit("b", 4, 5, 10)
ledger.subtract(6, 10)
assert ledger.get_balance(10) == 6
assert ledger.get_balance(15) == 6
assert ledger.get_balance(30) == 6
assert ledger.get_balance(31) == 0
ledger.subtract(3, 40)
assert ledger.get_balance(40) is None
ledger.add_credit("c", 5, 50, 10)
assert ledger.get_balance(50) == 2
assert ledger.get_balance(61) == 0
"""},
        {"name": "Part 1: boundaries and long debt", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A grant is usable at both ends of its window, an empty ledger reads 0, and debt stays until later grants pay it off in full.",
         "code": r"""
ledger = {fn}()
assert ledger.get_balance(0) == 0
ledger.add_credit("z", 3, 2, 0)
assert ledger.get_balance(2) == 3 and ledger.get_balance(3) == 0
ledger.subtract(10, 4)
assert ledger.get_balance(4) is None
ledger.add_credit("p", 4, 5, 100)
assert ledger.get_balance(5) is None, "6 of debt are still outstanding"
ledger.add_credit("q", 6, 6, 100)
assert ledger.get_balance(6) == 0
ledger.add_credit("r", 10**12, 7, 1)
ledger.subtract(1, 8)
assert ledger.get_balance(8) == 10**12 - 1
assert ledger.get_balance(200) == 0, "repaid debt does not come back when the grants expire"
"""},
        {"name": "Part 1: the soonest-expiring grant is drained first", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "subtract must drain usable grants in order of their last usable time, skipping expired ones and spilling into the next.",
         "code": r"""
ledger = {fn}()
ledger.add_credit("long", 5, 0, 100)
ledger.add_credit("short", 5, 1, 10)
ledger.add_credit("gone", 5, 2, 1)
ledger.subtract(7, 5)
assert ledger.get_balance(5) == 3
assert ledger.get_balance(12) == 3, "short was emptied first, then 2 came from long"
ledger.subtract(1, 101)
assert ledger.get_balance(101) is None
"""},
        {"name": "Part 1: random in-order calls match a replay", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random calls arriving in time order, a balance differed from replaying the calls.",
         "code": _HELPERS + r"""
for seed in range(200):
    rng = random.Random(seed)
    calls = sorted(random_calls(rng, rng.randint(1, 15)), key=call_ts)
    ledger, model = {fn}(), Replay()
    for name, args in calls:
        getattr(ledger, name)(*args)
        getattr(model, name)(*args)
        t = call_ts((name, args))
        for q in (t, t + rng.randint(0, 3)):
            assert ledger.get_balance(q) == model.get_balance(q), (seed, q)
        if rng.random() < 0.2:
            break
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
ledger = {fn}()
ledger.subtract(5, 20)
assert ledger.get_balance(10) == 0
ledger.add_credit("a", 7, 15, 20)
assert ledger.get_balance(15) == 7
assert ledger.get_balance(20) == 2
assert ledger.get_balance(12) == 0
"""},
        {"name": "Part 2: a late grant changes earlier answers", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A call that arrives late must be applied at its own timestamp, so it can change the answer for a time already queried.",
         "code": r"""
ledger = {fn}()
ledger.add_credit("a", 4, 0, 50)
ledger.subtract(10, 30)
assert ledger.get_balance(30) is None
ledger.add_credit("late", 10, 20, 50)
assert ledger.get_balance(30) == 4, "a (last usable at 50) is drained first, then 6 from late"
assert ledger.get_balance(51) == 4 and ledger.get_balance(71) == 0
ledger.subtract(1, 10)
assert ledger.get_balance(10) == 3
assert ledger.get_balance(30) == 3, "a holds only 3 at time 30, so late gives 7"
assert ledger.get_balance(51) == 3
"""},
        {"name": "Part 2: random shuffled calls match a replay", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random calls arriving in any order, a balance differed from replaying the calls in timestamp order.",
         "code": _HELPERS + r"""
for seed in range(200):
    rng = random.Random(seed)
    calls = random_calls(rng, rng.randint(1, 15))
    ledger, model = {fn}(), Replay()
    for name, args in calls:
        getattr(ledger, name)(*args)
        getattr(model, name)(*args)
        for _ in range(3):
            q = rng.randrange(-2, 90)
            assert ledger.get_balance(q) == model.get_balance(q), (seed, q)
"""},
        {"name": "Part 3: in-order queries do not replay", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "3,000 calls each followed by a query took far more than ten times as long as 300; keep a cursor and apply each call once instead of replaying per query.",
         "code": _HELPERS + r"""
def workload(n):
    rng = random.Random(n)
    calls = []
    for ts in range(0, 10 * n, 10):
        if rng.random() < 0.5:
            calls.append(("add_credit", (f"g{ts}", rng.randint(1, 50), ts, rng.randint(0, 300))))
        else:
            calls.append(("subtract", (rng.randint(1, 60), ts)))
    def run():
        ledger = {fn}()
        for call in calls:
            getattr(ledger, call[0])(*call[1])
            ledger.get_balance(call_ts(call))
            ledger.get_balance(call_ts(call) + 5)
    return run
ratio = best_of_three(workload(3000)) / best_of_three(workload(300))
assert ratio < 40, f"10x the calls took {ratio:.0f}x as long"
"""},
        {"name": "Part 3: every arrival order still matches a replay", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Mixing in-order queries with late calls and earlier queries gave a balance different from replaying the calls.",
         "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(500 + seed)
    calls = sorted(random_calls(rng, rng.randint(2, 20)), key=call_ts)
    late = [calls.pop(rng.randrange(len(calls))) for _ in range(rng.randint(0, 2))]
    ledger, model = {fn}(), Replay()
    for i, (name, args) in enumerate(calls):
        getattr(ledger, name)(*args)
        getattr(model, name)(*args)
        q = call_ts((name, args)) + rng.randint(0, 4)
        assert ledger.get_balance(q) == model.get_balance(q), (seed, q)
        if late and rng.random() < 0.3:
            extra = late.pop()
            getattr(ledger, extra[0])(*extra[1])
            getattr(model, extra[0])(*extra[1])
        if rng.random() < 0.2:
            back = rng.randrange(0, 70)
            assert ledger.get_balance(back) == model.get_balance(back), (seed, back)
"""},
    ],
    "solution": r'''import heapq


class GPUCreditLedger:
    """Applies calls in timestamp order up to a cursor; in-order traffic only moves it forward."""

    def __init__(self):
        self._calls = []  # (timestamp, amount, last usable time); negative amount = subtract
        self._reset()

    def _reset(self):
        self._pending = list(self._calls)
        heapq.heapify(self._pending)
        self._grants = []  # heap of [last usable time, remaining]
        self._credit = 0   # remaining credit across the heap
        self._debt = 0
        self._clock = -1   # every call at or before this time is applied

    def add_credit(self, credit_id, amount, timestamp, expiration):
        self._record((timestamp, amount, timestamp + expiration))

    def subtract(self, amount, timestamp):
        self._record((timestamp, -amount, None))

    def _record(self, call):
        self._calls.append(call)
        if call[0] <= self._clock:
            self._reset()
        else:
            heapq.heappush(self._pending, call)

    def _expire(self, before):
        while self._grants and self._grants[0][0] < before:
            self._credit -= heapq.heappop(self._grants)[1]

    def _apply(self, timestamp, amount, end):
        self._expire(timestamp)
        if amount > 0:
            paid = min(self._debt, amount)
            self._debt -= paid
            if amount > paid:
                heapq.heappush(self._grants, [end, amount - paid])
                self._credit += amount - paid
            return
        need = -amount
        while need and self._grants:
            grant = self._grants[0]
            take = min(grant[1], need)
            grant[1] -= take
            self._credit -= take
            need -= take
            if grant[1] == 0:
                heapq.heappop(self._grants)
        self._debt += need

    def get_balance(self, timestamp):
        if timestamp < self._clock:
            self._reset()
        while self._pending and self._pending[0][0] <= timestamp:
            self._apply(*heapq.heappop(self._pending))
        self._expire(timestamp)
        self._clock = timestamp
        value = self._credit - self._debt
        return None if value < 0 else value
''',
    "interview_questions": interview(
        concept=[
            "Which grants does a subtract at time t drain, and in what order?",
            "Where does a new grant's amount go when there is outstanding debt, and why does that debt not return later?",
        ],
        deep_dive=[
            "What structure gives you the soonest-expiring usable grant quickly, and how do expired grants leave it?",
        ],
        tradeoffs=[
            "How do you answer correctly when a call arrives after a query about a later time?",
            "What does replaying every call per query cost, and how does a cursor over sorted calls avoid it?",
            "When does the incremental cursor have to be thrown away, and what does the rebuild cost?",
        ],
    ),
}
