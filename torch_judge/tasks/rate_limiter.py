"""Fix a buggy rate limiter: requests count until exactly a period has passed, every rule decides before any records, every rule slides, and calls are thread-safe."""

from ._interview import interview

# A hand-moved clock and a model that keeps every admitted time.
_MODEL = r"""
import random, threading, time

BIG = (10**9, 10**9)

class Clock:
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self):
        return self.t

class Model:
    def __init__(self, rules):
        self.rules, self.times = rules, {}
    def allow(self, user, now):
        times = self.times.setdefault(user, [])
        ok = all(sum(1 for t in times if now - t <= period) < limit for limit, period in self.rules)
        if ok:
            times.append(now)
        return ok

def compare(fn, rules, rng, steps, label, users=("u", "v")):
    clock = Clock()
    lim, m = fn(rules, clock), Model(rules)
    for step in range(steps):
        clock.t += rng.choice([0, 0, 1, 2, 5, 10, 0.5])
        user = rng.choice(users)
        got, want = lim.allow(user), m.allow(user, clock.t)
        assert got == want, (label, step, rules, user, clock.t, got, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "budget.enforcement", "code": _MODEL + r"""
clock = Clock()
lim = {fn}([(2, 60), BIG], clock)
assert [lim.allow("ana") for _ in range(2)] == [True, True]
clock.t = 59
assert lim.allow("ana") is False
assert lim.allow("ben") is True
clock.t = 60
assert lim.allow("ana") is False, "a request made at 0 still counts at exactly 60"
clock.t = 60.5
assert lim.allow("ana") is True
"""},
    {"name": "Part 1: random times on one rule", "part": 1, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "With one rule that can reject, a result differed from a model where a request admitted at t counts while now - t <= period, for each user separately.",
     "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(seed)
    rules = [(rng.randint(1, 4), rng.choice([1, 2, 5, 10])), BIG]
    compare({fn}, rules, rng, 60, seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "effects.idempotency", "code": _MODEL + r"""
clock = Clock()
lim = {fn}([(2, 1000), (1, 10), BIG], clock)
assert lim.allow("ana") is True
clock.t = 5
assert lim.allow("ana") is False, "the second rule allows one request per 10 seconds"
clock.t = 20
assert lim.allow("ana") is True, "the rejected request at 5 must not count under the first rule"
clock.t = 30
assert lim.allow("ana") is False, "two admitted requests in 1000 seconds"
"""},
    {"name": "Part 2: random times on several rules", "part": 2, "visibility": "unshown", "behavior": "effects.idempotency",
     "failure_message": "With two rules that can reject, a result differed from a model that admits a request only if it fits under every rule and then records it under every rule; a rejected request must count nowhere.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(500 + seed)
    rules = [(rng.randint(1, 4), rng.choice([2, 5, 20, 50])) for _ in range(2)] + [BIG]
    compare({fn}, rules, rng, 60, seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "budget.enforcement", "code": _MODEL + r"""
clock = Clock(90)
lim = {fn}([(5, 10), (2, 100)], clock)
assert lim.allow("ana") is True
clock.t = 99
assert lim.allow("ana") is True
clock.t = 101
assert lim.allow("ana") is False, "the 100-second rule slides: 90 and 99 both still count"
clock.t = 190
assert lim.allow("ana") is False
clock.t = 190.5
assert lim.allow("ana") is True
"""},
    {"name": "Part 3: random times on every rule", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "With 1 to 3 rules that can all reject, including the last one, a result differed from a model where every rule is a sliding window over admitted requests.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(900 + seed)
    rules = [(rng.randint(1, 4), rng.choice([2, 5, 20, 50])) for _ in range(rng.randint(1, 3))]
    compare({fn}, rules, rng, 60, seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "concurrency.thread_safety", "code": _MODEL + r"""
for rules in ([(1, 60), BIG], [BIG, (1, 60)]):
    gate = threading.Barrier(2, timeout=0.5)
    def hold():
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
    lim = {fn}(rules, Clock(), hold)
    results = []
    threads = [threading.Thread(target=lambda: results.append(lim.allow("ana"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(results) == [False, True], (rules, results)
"""},
    {"name": "Part 4: many threads", "part": 4, "visibility": "unshown", "behavior": "concurrency.thread_safety",
     "failure_message": "16 threads calling allow at the same instant, with a checkpoint that yields, admitted more requests than a rule allows: decide and record under one lock.",
     "code": _MODEL + r"""
for rules in ([(5, 1000), BIG], [BIG, (5, 1000)], [(7, 1000), (5, 500)]):
    lim = {fn}(rules, Clock(), lambda: time.sleep(0.001))
    results = []
    def worker():
        for _ in range(3):
            results.append(lim.allow("ana"))
    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert results.count(True) == 5, (rules, results.count(True))
"""},
]

TASK = {
    "title": "Rate Limiter Bug Hunt",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "RateLimiter",
    "description_en": r"""Fix `RateLimiter`, which the starter code gives you with several bugs. It should admit a user's request only if the request fits under every rule, and it must stay correct when threads call it at once.

The requirement arrives in parts. Each part keeps every earlier behavior, so one fixed `RateLimiter` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `RateLimiter(rules, clock=time.time, checkpoint=lambda: None)`: `rules` is a list of `(max_requests, period_seconds)` pairs, and `clock()` returns the current time in seconds. Time never goes backward.
- `allow(user_id) -> bool` reads the clock once and returns `True` if the request is admitted. Users never affect each other.
- A request admitted at time `t` counts against a rule while `now - t <= period` for that rule, and not after. A request fits under a rule when fewer than `max_requests` of the user's admitted requests count against it.
- `allow` calls `checkpoint()` exactly once, after deciding and before recording anything.
- Keep the constructor and `allow`; the rest of the class is yours to change.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** debugging rounds test reading code against a written contract and proving each bug with a test before fixing it. Each later part adds one requirement, and each one exposes another bug in the starter: a boundary, a partial update, a shortcut that changes the rule, and a race.

**Where it is used:** API gateways and model-serving endpoints limit requests per key per minute, hour and day, and the same check-then-record pattern guards quotas, credits and inventory.

Adapted from the rate limiter bug hunt in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded. The starter is new code with its own bugs, the limiter takes a list of rules instead of three named tiers, `should_allow_request` is renamed `allow`, and the decorator is left out.""",
    "parts": [
        {
            "title": "The edge of the window",
            "description_en": r"""**Signature:** `RateLimiter(rules, clock, checkpoint).allow(user_id) -> bool`

- Make one rule exact: a request admitted at `t` still counts at `now = t + period` and stops counting just after.
- The tests here use one rule that can reject, followed by a rule too large to reject anything.

**Example:** `rules = [(2, 60), (10**9, 10**9)]`, with the clock moved by hand:
- at `0`, two calls for `"ana"` are admitted; at `59` a third is rejected, while `"ben"` is admitted
- at `60`, `"ana"` is still rejected: both requests from `0` count until the end of second `60`
- at `60.5`, `"ana"` is admitted""",
        },
        {
            "title": "Every rule decides before any records",
            "description_en": r"""Keep Part 1. Several rules can reject now.

- A request is admitted only if it fits under every rule. An admitted request then counts under every rule.
- A rejected request counts under no rule, even if some rules had room for it.

**Example:** `rules = [(2, 1000), (1, 10), (10**9, 10**9)]`:
- at `0`, `"ana"` is admitted; at `5` she is rejected by the second rule
- at `20` she is admitted: the rejected request at `5` used none of the first rule's two places
- at `30` she is rejected, with two admitted requests in the last `1000` seconds""",
        },
        {
            "title": "Every rule slides",
            "description_en": r"""Keep Parts 1–2. Every rule, the last one included, follows the same counting rule.

- No rule may count by fixed calendar windows, and no rule may count rejected requests.

**Example:** `rules = [(5, 10), (2, 100)]`:
- `"ana"` is admitted at `90` and at `99`
- at `101` she is rejected: both earlier requests are within `100` seconds, though a new hundred has begun
- at `190` she is still rejected, and at `190.5` she is admitted, since only the request from `99` still counts""",
        },
        {
            "title": "Threads",
            "description_en": r"""Keep Parts 1–3. Calls may now run on many threads at once.

- Concurrent calls must give results that some one-at-a-time order of the same calls would give. In particular, two calls must never both take a rule's last place.
- `checkpoint` may block or sleep, as the tests use it to hold a thread between deciding and recording.

**Example:** with `rules = [(1, 60), (10**9, 10**9)]`, two threads call `allow("ana")` at the same instant while `checkpoint` waits for both threads to arrive, giving up after `0.5` seconds:
- exactly one call returns `True`
- the same holds with the two rules swapped""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which line decides that an old request no longer counts? Compare its condition with the rule: at exactly t + period, should the request at t still count? Write the test for that instant before you change anything."},
        {"level": 2, "kind": "analysis", "content": "The eviction loop drops entries while now - t >= period, so a request stops counting at exactly t + period, one instant early. The contract keeps it while now - t <= period, so the loop must drop an entry only when now - t > period. A test at 0, 0, then exactly 60 with a limit of 2 per 60 seconds shows it."},
    ],
    "model_connections": [
        "Model-serving APIs cap requests and tokens per key over several windows, and an off-by-one at the window edge lets bursts through.",
        "Quota checks for GPU hours or credits use the same decide-then-record step, which must be atomic under concurrent jobs.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A log of admitted times per rule gives an exact sliding window.",
            "Deciding under every rule before recording keeps rejected requests from using any quota.",
            "One lock around decide and record makes concurrent calls behave like some serial order.",
        ],
        "cons": [
            "A log per rule costs memory proportional to the limit, which a fixed-window counter avoids at the price of bursts at window edges.",
            "One lock for all users serialises unrelated users; a lock per user scales better.",
            "Within one process only; across servers the log needs a shared store with atomic updates.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, rules, clock=time.time, checkpoint=lambda: None):
        self.rules = list(rules)  # (max_requests, period_seconds) pairs
        self.clock = clock
        self.checkpoint = checkpoint
        self.logs = defaultdict(lambda: [deque() for _ in self.rules])  # user_id -> admitted times, one log per rule
        self.lock = threading.Lock()

    def allow(self, user_id):
        with self.lock:  # deciding and recording form one step, so two calls cannot both take the last slot
            now = self.clock()
            logs = self.logs[user_id]
            fits = True
            for (limit, period), log in zip(self.rules, logs):
                while log and now - log[0] > period:  # a request still counts exactly period seconds later
                    log.popleft()
                fits = fits and len(log) < limit
            self.checkpoint()
            if fits:
                for log in logs:  # recorded under every rule, or under none
                    log.append(now)
            return fits
''',
    "interview_questions": interview(
        concept=[
            "At exactly t + period, should a request admitted at t still count, and which comparison in the eviction loop says so?",
            "How do you write a test that fails on the off-by-one without depending on the wall clock?",
        ],
        deep_dive=[
            "Why does a deque make eviction cheap, and what does eviction cost per call?",
        ],
        tradeoffs=[
            "Why must every rule decide before any rule records, and what does a rejected request cost otherwise?",
            "What does a fixed calendar window save over a sliding one, and which bursts does it let through?",
            "Why does a lock around deciding and recording fix the race, and how would you avoid one lock for every user?",
            "How would you enforce the same limits across several servers?",
        ],
    ),
}
