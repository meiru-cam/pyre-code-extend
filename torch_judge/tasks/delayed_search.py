"""Binary search through an answer that arrives one call late: one guess per call, then two, then many games sharing one channel."""

from ._interview import interview

# Judges that enforce each part's protocol and count calls; ceil_log avoids float rounding.
_JUDGES = r"""
import random

def ceil_log(n, base):
    k, v = 0, 1
    while v < n:
        k, v = k + 1, v * base
    return k

def compare(x, secret):
    return -1 if x < secret else (1 if x > secret else 0)

class Narrow:
    def __init__(self, n):
        self.lo, self.hi = 1, n
    def learn(self, x, r):
        if r < 0: self.lo = max(self.lo, x + 1)
        elif r > 0: self.hi = min(self.hi, x - 1)
        else: self.lo = self.hi = x

class BatchJudge:
    def __init__(self, n, secret, most):
        self.n, self.secret, self.most = n, secret, most
        self.used, self.last, self.calls = set(), None, 0
        self.known = Narrow(n)
    def send(self, xs):
        xs = list(xs)
        if self.known.lo == self.known.hi:
            raise ValueError("the secret was already determined, so no further call is allowed")
        if not 1 <= len(xs) <= self.most:
            raise ValueError(f"a call must carry 1 to {self.most} guesses, got {len(xs)}")
        for x in xs:
            if not isinstance(x, int) or not 1 <= x <= self.n:
                raise ValueError(f"guess {x!r} is outside [1, {self.n}]")
            if x in self.used or xs.count(x) > 1:
                raise ValueError(f"guess {x} was already used")
        self.calls += 1
        self.used.update(xs)
        answer = None
        if self.last is not None:
            answer = [compare(x, self.secret) for x in self.last]
            for x, r in zip(self.last, answer):
                self.known.learn(x, r)
        self.last = xs
        return answer
    def check(self, x):
        answer = self.send([x])
        return None if answer is None else answer[0]

class RoundJudge:
    def __init__(self, bounds, secrets):
        self.bounds, self.secrets = bounds, secrets
        self.used = {g: set() for g in bounds}
        self.known = {g: Narrow(n) for g, n in bounds.items()}
        self.last, self.calls = None, 0
    def check_round(self, guesses):
        for g, x in guesses.items():
            if g not in self.bounds:
                raise ValueError(f"unknown game {g!r}")
            if self.known[g].lo == self.known[g].hi:
                raise ValueError(f"game {g!r} is already determined")
            if not isinstance(x, int) or not 1 <= x <= self.bounds[g]:
                raise ValueError(f"guess {x!r} is outside [1, {self.bounds[g]}] for game {g!r}")
            if x in self.used[g]:
                raise ValueError(f"guess {x} was already used in game {g!r}")
        if self.calls and all(k.lo == k.hi for k in self.known.values()):
            raise ValueError("every secret was already determined")
        self.calls += 1
        for g, x in guesses.items():
            self.used[g].add(x)
        answer = None
        if self.last is not None:
            answer = {g: compare(x, self.secrets[g]) for g, x in self.last.items()}
            for g, r in answer.items():
                self.known[g].learn(self.last[g], r)
        self.last = dict(guesses)
        return answer

def run_one(fn, n, secret):
    j = BatchJudge(n, secret, 1)
    got = fn().find_secret(n, j.check)
    assert got == secret, (n, secret, got)
    assert j.calls <= 2 * ceil_log(n, 2) + 1, f"n={n}, secret={secret}: {j.calls} calls"

def run_two(fn, n, secret):
    j = BatchJudge(n, secret, 2)
    got = fn().find_secret_batched(n, j.send)
    assert got == secret, (n, secret, got)
    assert j.calls <= 2 * ceil_log(n, 3) + 1, f"n={n}, secret={secret}: {j.calls} calls"

def run_games(fn, bounds, secrets):
    j = RoundJudge(bounds, secrets)
    got = fn().solve_games(dict(bounds), j.check_round)
    assert got == secrets, (bounds, secrets, got)
    top = max(bounds.values(), default=1)
    assert j.calls <= 2 * ceil_log(top, 2) + 1, f"{len(bounds)} games up to {top}: {j.calls} rounds"
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": _JUDGES + r"""
j = BatchJudge(10, 3, 1)
assert [j.check(x) for x in (5, 10, 3, 9)] == [None, 1, 1, 0]
try:
    j.check(4)
    raise AssertionError("the judge should refuse a call once the secret is determined")
except ValueError:
    pass
run_one({fn}, 10, 3)
"""},
    {"name": "Part 1: every secret for small n", "part": 1, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "For some n up to 70 and some secret, the answer was wrong, the protocol was broken (a repeated or out-of-range guess, or a call after the secret was determined), or it took more than 2 * ceil(log2(n)) + 1 calls.",
     "code": _JUDGES + r"""
for n in range(1, 71):
    for secret in range(1, n + 1):
        run_one({fn}, n, secret)
"""},
    {"name": "Part 1: large n", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "For n up to 10**9 the answer or call count was wrong, or it was too slow: never loop over the whole range, and pick spare guesses from values already ruled out.",
     "code": _JUDGES + r"""
rng = random.Random(31)
for n in [2**20, 2**20 + 1, 10**6, 999_999_937, 10**9]:
    for secret in [1, 2, n // 2, n - 1, n] + [rng.randint(1, n) for _ in range(6)]:
        run_one({fn}, n, secret)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "contract.signature", "code": _JUDGES + r"""
j = BatchJudge(26, 5, 2)
assert j.send([9, 18]) is None and j.send([26]) == [1, 1]
assert j.send([3, 6]) == [1] and j.send([25]) == [-1, 1]
assert (j.known.lo, j.known.hi) == (4, 5)
run_two({fn}, 26, 5)
"""},
    {"name": "Part 2: every secret for small n", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "For some n up to 60 and some secret, the answer was wrong, a call carried 0 or more than 2 guesses, a guess repeated or left [1, n], or it took more than 2 * ceil(log3(n)) + 1 calls.",
     "code": _JUDGES + r"""
for n in range(1, 61):
    for secret in range(1, n + 1):
        run_two({fn}, n, secret)
"""},
    {"name": "Part 2: large n needs thirds", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "For n up to 10**9 the call count went over 2 * ceil(log3(n)) + 1: two guesses per real round must cut the candidates to about a third, not a half.",
     "code": _JUDGES + r"""
rng = random.Random(32)
for n in [3**12, 3**12 + 1, 10**6, 10**9]:
    for secret in [1, n // 3, n // 3 + 1, n] + [rng.randint(1, n) for _ in range(6)]:
        run_two({fn}, n, secret)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "contract.signature", "code": _JUDGES + r"""
games, secrets = {"a": 1, "b": 9, "c": 4}, {"a": 1, "b": 7, "c": 1}
j = RoundJudge(games, secrets)
assert j.check_round({"b": 5, "c": 2}) is None
assert j.check_round({"b": 8}) == {"b": -1, "c": 1}
assert (j.known["c"].lo, j.known["c"].hi) == (1, 1)
run_games({fn}, games, secrets)
"""},
    {"name": "Part 3: many games share the rounds", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "With many games the rounds went over 2 * ceil(log2(largest n)) + 1, a solved game was guessed again, or a result was wrong: every open game must guess in the same round so they all finish together.",
     "code": _JUDGES + r"""
rng = random.Random(33)
for trial in range(300):
    count = rng.randint(1, 6) if trial < 200 else rng.randint(20, 60)
    top = 12 if trial < 200 else 10 ** rng.randint(1, 9)
    bounds = {g: rng.randint(1, top) for g in range(count)}
    if trial % 5 == 0:
        bounds["one"] = 1
    secrets = {g: rng.randint(1, n) for g, n in bounds.items()}
    run_games({fn}, bounds, secrets)
"""},
]

TASK = {
    "title": "Search with Delayed Answers",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "DelayedSearch",
    "description_en": r"""Build `DelayedSearch`, which finds hidden numbers through a channel that answers each call one call late: first with one guess per call, then two, then for many games at once.

The requirement arrives in parts. Each part adds one method to the same `DelayedSearch` class, and the earlier methods must keep working. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A secret is an integer in `[1, n]`. A guess is compared with it: `-1` if the guess is smaller, `1` if it is larger, `0` if it is the secret.
- Each call's answer is about the call before it. The first call answers `None`, and the answer for your newest guess arrives only with your next call.
- Every guess lies in `[1, n]` and differs from every earlier guess for the same secret.
- Stop calling as soon as only one value is possible; for `n = 1`, return `1` without calling.
- The channel raises `ValueError` when a rule is broken. Each method returns the secret; each part sets its own call budget.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it turns binary search into a pipelining puzzle: every guess must be chosen before the previous one's answer is known. Each later part adds one requirement: two guesses per call cut the range to a third, and many games share one channel, so the round count, not the guess count, is what matters.

**Where it is used:** pipelined requests over a slow link, where each reply arrives after the next request has gone out; batched hyperparameter and threshold searches that send several probes per round; and several searches sharing one queue of evaluation jobs.

Adapted from the delayed-answer binary search question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as methods of one class instead of three functions. Part 2 caps a call at two guesses, which the source only implies; Part 3 also forbids an empty round once every game is settled; every call budget has one call to spare, as in the source's stated bounds.""",
    "parts": [
        {
            "title": "One guess per call",
            "description_en": r"""**Signature:** `DelayedSearch().find_secret(n, check) -> int`

- `check(x)` sends one guess and returns `None` on the first call, then the comparison of the previous call's guess.
- Use at most `2 * ceil(log2(n)) + 1` calls, for `n` up to `10**9`.
- A call whose only purpose is to bring back the previous answer still needs a new, unused guess.

**Example:** with `n = 10` and the secret `3`:
- `check(5)` answers `None`, and `check(10)` answers `1` for `5`: the secret is in `1..4`
- `check(3)` answers `1` for `10`, which adds nothing
- `check(9)` answers `0` for `3`: the secret is `3`, so stop; another call would raise `ValueError`""",
        },
        {
            "title": "Two guesses per call",
            "description_en": r"""Keep Part 1 and add a method.

**Signature:** `find_secret_batched(n, check_batch) -> int`

- `check_batch(xs)` sends a list of one or two guesses, all new. It returns `None` on the first call, then the list of answers for the previous call's guesses, in the same order.
- Use at most `2 * ceil(log3(n)) + 1` calls, for `n` up to `10**9`.

**Example:** with `n = 26` and the secret `5`:
- `check_batch([9, 18])` answers `None`, and `check_batch([26])` answers `[1, 1]`: the secret is in `1..8`
- `check_batch([3, 6])` answers `[1]`, for `26`
- `check_batch([25])` answers `[-1, 1]`: the secret is in `4..5`, so the search goes on""",
        },
        {
            "title": "Many games, few rounds",
            "description_en": r"""Keep Parts 1–2 and add a method.

**Signature:** `solve_games(bounds, check_round) -> dict`

- `bounds` maps each game id to that game's `n`. Every game has its own secret and its own set of used guesses.
- `check_round(guesses)` sends a dict from some game ids to one guess each; an empty dict is allowed. It returns `None` on the first call, then a dict with an answer for each guess of the previous call, whatever the current call holds.
- A game whose secret is determined must not be guessed again, and once every game is settled, make no further call. Return a dict from every game id to its secret.
- Use at most `2 * ceil(log2(m)) + 1` calls in total, where `m` is the largest `n`, however many games there are.

**Example:** with games `a` (`n = 1`), `b` (`n = 9`, secret `7`) and `c` (`n = 4`, secret `1`):
- `a` is settled from the start, so it is never guessed
- `check_round({"b": 5, "c": 2})` answers `None`
- `check_round({"b": 8})` answers `{"b": -1, "c": 1}`: `b` is in `6..9`, and `c` is settled at `1`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "When you choose a guess, which earlier guesses' answers do you already have? When there is nothing useful to ask, which values can a call send that break no rule?"},
        {"level": 2, "kind": "analysis", "content": "Keep [lo, hi] and a set of used values. One option: let every second call carry the midpoint, and let the calls between carry any unused value, best one you already know the answer for, since its only job is to bring the midpoint's answer back. Narrow on every answer that arrives. If a midpoint is already used, take the nearest free value. Stop once lo == hi."},
    ],
    "model_connections": [
        "Pipelined decoding and serving loops issue the next request before the previous result returns, so control logic must decide on stale information.",
        "Batched threshold and learning-rate searches send several probes per round and narrow on the results that come back.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Alternating real and spare calls keeps plain binary search's guarantee at twice the calls.",
            "Two probes per round cut the range to a third, which needs fewer calls for large n.",
            "Sharing rounds across games makes the total depend on the largest game, not the number of games.",
        ],
        "cons": [
            "Half of all calls in Part 1 carry no new probe, which is the price of the delay.",
            "Nudging a probe off a used midpoint cuts a little less than half, so the bound needs a spare call.",
            "A dropped or duplicated reply would desynchronise the pairing of answers and guesses.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).


class _Window:
    """The candidates [lo, hi] for one secret, plus every value already sent."""

    def __init__(self, n):
        self.lo, self.hi = 1, n
        self.used = set()
        self.top, self.bottom = n, 1  # next unused values to try above hi and below lo

    def learn(self, x, answer):
        if answer < 0:
            self.lo = max(self.lo, x + 1)
        elif answer > 0:
            self.hi = min(self.hi, x - 1)
        else:
            self.lo = self.hi = x

    def solved(self):
        return self.lo == self.hi

    def near(self, target, avoid=()):
        """The unused value inside [lo, hi] closest to target, or None."""
        for step in range(self.hi - self.lo + 1):
            for x in (target - step, target + step):
                if self.lo <= x <= self.hi and x not in self.used and x not in avoid:
                    return x
        return None

    def spare(self):
        """An unused value that teaches nothing new if possible: outside [lo, hi] first."""
        while self.top > self.hi:
            x, self.top = self.top, self.top - 1
            if x not in self.used:
                return x
        while self.bottom < self.lo:
            x, self.bottom = self.bottom, self.bottom + 1
            if x not in self.used:
                return x
        return self.near(self.lo)  # only early on, before anything is ruled out

    def probes(self, count):
        """count points that split [lo, hi] into count + 1 even pieces; one point once two candidates remain."""
        if self.hi - self.lo + 1 <= 2:
            count = 1
        picked = []
        for i in range(1, count + 1):
            x = self.near(self.lo + i * (self.hi - self.lo) // (count + 1), picked)
            if x is not None:
                picked.append(x)
        return picked


class DelayedSearch:
    def _search(self, n, send, width):
        """Alternate a round of real probes with a round that only fetches their answers."""
        w = _Window(n)
        if w.solved():
            return w.lo
        sent, real = None, True
        while True:
            batch = w.probes(width) if real else [w.spare()]
            w.used.update(batch)
            answers = send(batch)
            if sent is not None:
                for x, answer in zip(sent, answers):
                    w.learn(x, answer)
            if w.solved():
                return w.lo
            sent, real = batch, not real

    def find_secret(self, n, check):
        return self._search(n, lambda batch: None if (r := check(batch[0])) is None else [r], 1)

    def find_secret_batched(self, n, check_batch):
        return self._search(n, check_batch, 2)

    def solve_games(self, bounds, check_round):
        windows = {g: _Window(n) for g, n in bounds.items()}
        sent = {}
        while not all(w.solved() for w in windows.values()):
            # every open game guesses its midpoint, then an empty round fetches the answers
            guesses = {} if sent else {g: (w.lo + w.hi) // 2 for g, w in windows.items() if not w.solved()}
            answers = check_round(guesses) or {}
            for g, answer in answers.items():
                windows[g].learn(sent[g], answer)
            sent = guesses
        return {g: w.lo for g, w in windows.items()}
''',
    "interview_questions": interview(
        concept=[
            "When you choose the next guess, which answers do you have, and how does that change plain binary search?",
            "What can a spare call send so that it brings back the previous answer without breaking any rule?",
        ],
        deep_dive=[
            "Why does alternating a midpoint probe with a spare call stay within 2 * ceil(log2(n)) + 1 calls?",
        ],
        tradeoffs=[
            "Why do two guesses per call reach a third of the range, and what limits going to more guesses per call?",
            "When many games share one channel, why does the round count depend only on the largest game?",
            "Could some calls in the single-game case carry a new probe instead of a spare, and what would that gain?",
            "If calls could overlap in flight, how would the problem change?",
        ],
    ),
}
