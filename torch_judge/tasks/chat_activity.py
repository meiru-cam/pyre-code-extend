"""Counts and active chats over a sliding window of events, in order, then out of order, in bounded memory."""

from ._interview import interview

# A slow model written straight from the statement: it keeps every accepted event and scans them all.
_MODEL = r"""
import random, time, tracemalloc

class Model:
    def __init__(self, window):
        self.window, self.clock, self.events, self.seq = window, None, [], 0
    def record_event(self, user_id, chat_id, timestamp, event_type="ping"):
        if self.clock is not None and timestamp < self.clock - self.window:
            return
        self.seq += 1
        self.events.append((user_id, chat_id, timestamp, event_type, self.seq))
        self.clock = timestamp if self.clock is None else max(self.clock, timestamp)
    def recent_count(self, user_id, chat_id):
        if self.clock is None:
            return 0
        return sum(1 for u, c, ts, _, _ in self.events
                   if (u, c) == (user_id, chat_id) and self.clock - self.window <= ts <= self.clock)
    def active_sessions(self, user_id, now=None):
        if now is not None:
            self.clock = now if self.clock is None else max(self.clock, now)
        if self.clock is None:
            return 0
        active = 0
        for chat in {c for u, c, _, _, _ in self.events if u == user_id}:
            mine = [(ts, seq, kind) for u, c, ts, kind, seq in self.events if (u, c) == (user_id, chat)]
            pings = [(ts, seq) for ts, seq, kind in mine if kind == "ping"]
            closes = [(ts, seq) for ts, seq, kind in mine if kind == "close"]
            if pings and max(pings)[0] >= self.clock - self.window and (not closes or max(closes) < max(pings)):
                active += 1
        return active

def in_order_calls(rng, n, users="ab", chats="xyz", kinds=("ping",), gap=3):
    ts, calls = 0, []
    for _ in range(n):
        ts += rng.choice([0, 0, 1, 2, gap])
        calls.append((rng.choice(users), rng.choice(chats), ts, rng.choice(kinds)))
    return calls

def traced_peak_after(make_calls, tracker, checkpoints):
    # Current traced memory at each checkpoint (a count of calls made so far).
    tracemalloc.start()
    try:
        out = []
        for i, call in enumerate(make_calls):
            call(tracker)
            if i + 1 in checkpoints:
                out.append(tracemalloc.get_traced_memory()[0])
        return out
    finally:
        tracemalloc.stop()
"""

TASK = {
    "title": "Chat Activity Tracker",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ActivityTracker",
    "description_en": r"""Build `ActivityTracker(window)`, which watches a stream of chat events and answers questions about the last `window` seconds.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ActivityTracker` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `window` is a positive integer. Every event has a `user_id` (`str`), a `chat_id` (`str`) and a `timestamp`, a non-negative integer number of seconds.
- A chat is the pair `(user_id, chat_id)`: two users may both have a chat called `"x"`, and those are different chats.
- The tracker must not grow with the number of chats ever seen. At any moment it may hold state only for chats that have an event inside the current window, plus a constant amount.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the counting is easy; the work is deciding when state can be thrown away, and each later part adds one requirement.

**Where it is used:** rate limiters, presence indicators and live dashboards all count recent events per key without keeping history.

Adapted from the sliding-window event aggregation question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Recent event count",
            "description_en": r"""**Signature:** `ActivityTracker(window)`, `record_event(user_id, chat_id, timestamp) -> None`, `recent_count(user_id, chat_id) -> int`

- In this part, `record_event` calls arrive in non-decreasing `timestamp` order. Equal timestamps are allowed.
- Let `T` be the largest `timestamp` recorded so far, across all chats. The window is `T - window` to `T`, both included.
- `recent_count` returns how many of the chat's recorded events fall inside the window; `0` if none do or nothing was recorded.
- Each call costs `O(log n)` amortized for `n` events in the window: `recent_count` must not walk through the chat's events.

**Example**, `window = 10`:
- `record_event("ana", "x", 2)` twice; `recent_count("ana", "x")` is `2`
- `record_event("ana", "y", 7)`: `x` still has `2`, `y` has `1`
- `record_event("ana", "x", 15)`: the window is now 5 to 15, so `x` has `1` and `y` still has `1`
- `record_event("bo", "x", 18)`: the window is 8 to 18; `("ana", "y")` drops to `0`, `("ana", "x")` stays `1`, and `("bo", "x")` has `1`""",
        },
        {
            "title": "Active chats",
            "description_en": r"""Keep Part 1. Events now have a type.

**Signature:** `record_event(user_id, chat_id, timestamp, event_type="ping") -> None`, `active_sessions(user_id) -> int`

- `event_type` is `"ping"` (the user is using the chat) or `"close"` (the user ended it). Anything else raises `ValueError`. `recent_count` counts events of both types.
- A chat is active when its latest ping is inside the window and no close came after that ping. "After" compares timestamps, and between a ping and a close with the same timestamp, the one recorded later wins.
- `active_sessions(user_id)` returns how many of that user's chats are active.
- Calls still arrive in non-decreasing `timestamp` order.

**Example**, `window = 8`:
- `("ana", "a")` ping at `0`: `active_sessions("ana")` is `1`
- `("ana", "b")` ping at `1`: `2`
- `("ana", "a")` close at `4`: `1`
- `("ana", "b")` ping at `6`: still `1`
- `("bo", "c")` ping at `10`: the window is 2 to 10 and `b`'s ping at `6` is inside, so `ana` has `1`
- `("bo", "c")` ping at `15`: the window is 7 to 15, so `ana` has `0`""",
        },
        {
            "title": "Out-of-order arrival",
            "description_en": r"""Keep Parts 1–2. Events may now arrive in any `timestamp` order.

**Signature:** `active_sessions(user_id, now=None) -> int`

- The clock `C` is the largest value seen so far among every recorded `timestamp` and every `now` passed in. It never goes back: `now` is at least every timestamp recorded before it and at least every earlier `now`. With `now=None`, `C` is unchanged.
- The window is `C - window` to `C`. `recent_count` and `active_sessions` answer for that window.
- An event whose `timestamp` is below `C - window` when it arrives is ignored entirely: it changes nothing, now or later.
- Activity depends on timestamps, not arrival order: a chat is active when its ping with the highest timestamp is inside the window and no close has a higher timestamp. At equal timestamps the event recorded later wins, as before.
- The memory rule now uses `C`: state may be kept only for chats with an event inside `C - window` to `C`.

**Example**, `window = 5`:
- `("ana", "a")` ping at `20`; `active_sessions("ana", 20)` is `1`
- a close at `17` arrives late; it is before the ping, so `active_sessions("ana", 21)` is still `1`
- a close at `23`; `active_sessions("ana", 23)` is `0`
- a ping at `22` arrives late; the close at `23` is later, so `active_sessions("ana", 24)` is `0`
- a ping at `26`; `active_sessions("ana", 26)` is `1`
- `("ana", "b")` ping at `9`: below `26 - 5`, so it is ignored, and `recent_count("ana", "b")` is `0`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "When T moves forward, which events leave the window, and in what order do they leave? If you knew the oldest event still inside, could you find all leaving events without looking at the rest? What should a chat's entry look like once its last event has left?"},
        {"level": 2, "kind": "analysis", "content": "Keep one queue of (timestamp, chat) in arrival order, which is timestamp order here, and a dict chat -> number of its events in the window. On each record, push, update T, then pop from the front while the front's timestamp is below T - window, decrementing that chat's count and deleting the entry when it hits 0. recent_count is a dict lookup."},
    ],
    "model_connections": [
        "Serving systems track active sessions per user to enforce concurrency limits and to route follow-up requests to a warm KV cache.",
        "Rate limiters for model APIs count requests per key over a sliding window and must forget idle keys to keep memory bounded.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A queue in timestamp order lets each event be evicted exactly once, so the cost per call is amortized constant.",
            "Deleting a key's entry when its count reaches zero bounds memory by the chats in the window.",
            "Keeping each chat's highest ping and close timestamps makes the answer independent of arrival order.",
        ],
        "cons": [
            "Out-of-order arrival needs a heap instead of a queue, which costs O(log n) per event.",
            "Events later than the window are dropped; a real system needs a watermark contract with its producers.",
            "One very busy user or chat can still dominate a single machine's state and traffic.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
t = {fn}(10)
t.record_event("ana", "x", 2)
t.record_event("ana", "x", 2)
assert t.recent_count("ana", "x") == 2
t.record_event("ana", "y", 7)
assert t.recent_count("ana", "x") == 2 and t.recent_count("ana", "y") == 1
t.record_event("ana", "x", 15)
assert t.recent_count("ana", "x") == 1 and t.recent_count("ana", "y") == 1
t.record_event("bo", "x", 18)
assert t.recent_count("ana", "y") == 0
assert t.recent_count("ana", "x") == 1
assert t.recent_count("bo", "x") == 1
assert t.recent_count("nobody", "x") == 0
"""},
        {"name": "Part 1: random in-order streams", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random in-order events, a count differed from counting the chat's events between T - window and T, both included; chats with the same chat_id under different users are separate.",
         "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(seed)
    window = rng.choice([1, 2, 5, 9])
    tracker, model = {fn}(window), Model(window)
    assert tracker.recent_count("a", "x") == 0
    for user, chat, ts, _ in in_order_calls(rng, rng.randint(1, 40)):
        tracker.record_event(user, chat, ts)
        model.record_event(user, chat, ts)
        for u in "ab":
            for c in "xyz":
                assert tracker.recent_count(u, c) == model.recent_count(u, c), (seed, u, c, ts)
"""},
        {"name": "Part 1: memory follows the window", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Memory kept growing with the number of chats ever seen; delete a chat's state once its last event leaves the window.",
         "code": _MODEL + r"""
calls = [lambda t, i=i: t.record_event(f"user{i}", f"chat{i % 7}", i // 4) for i in range(20000)]
tracker = {fn}(5)
early, late = traced_peak_after(calls, tracker, {2000, 20000})
assert tracker.recent_count("user19999", "chat0") == 1 and tracker.recent_count("user0", "chat0") == 0, "wrong counts after the run"
assert late - early < 1_000_000, f"memory grew by {(late - early) / 1e6:.1f} MB over 18,000 more events"
"""},
        {"name": "Part 1: counting does not walk the events", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "With many events of one chat inside the window, recent_count took time proportional to them; keep a count per chat instead of scanning.",
         "code": _MODEL + r"""
def run(n):
    tracker = {fn}(10 ** 9)
    start = time.perf_counter()
    for i in range(n):
        tracker.record_event("hot", "chat", i)
        assert tracker.recent_count("hot", "chat") == i + 1
    return time.perf_counter() - start

def best(n):
    return min(run(n) for _ in range(3))

ratio = best(30000) / best(3000)
assert ratio < 40, f"10x the events took {ratio:.0f}x as long"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
t = {fn}(8)
t.record_event("ana", "a", 0, "ping")
assert t.active_sessions("ana") == 1
t.record_event("ana", "b", 1, "ping")
assert t.active_sessions("ana") == 2
t.record_event("ana", "a", 4, "close")
assert t.active_sessions("ana") == 1
t.record_event("ana", "b", 6, "ping")
assert t.active_sessions("ana") == 1
t.record_event("bo", "c", 10, "ping")
assert t.active_sessions("ana") == 1
t.record_event("bo", "c", 15, "ping")
assert t.active_sessions("ana") == 0
assert t.active_sessions("bo") == 1 and t.active_sessions("nobody") == 0
"""},
        {"name": "Part 2: random in-order pings and closes", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random in-order pings and closes, an active count or event count differed from the rules; at equal timestamps the event recorded later decides.",
         "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(seed)
    window = rng.choice([1, 3, 6])
    tracker, model = {fn}(window), Model(window)
    for user, chat, ts, kind in in_order_calls(rng, rng.randint(1, 40), kinds=("ping", "ping", "close")):
        tracker.record_event(user, chat, ts, kind)
        model.record_event(user, chat, ts, kind)
        for u in "abc":
            assert tracker.active_sessions(u) == model.active_sessions(u), (seed, u, ts)
        assert tracker.recent_count(user, chat) == model.recent_count(user, chat), (seed, user, chat)
t = {fn}(100)
t.record_event("u", "c", 5, "ping")
t.record_event("u", "c", 5, "close")
assert t.active_sessions("u") == 0, "a close recorded after a ping at the same timestamp wins"
t.record_event("u", "c", 5, "ping")
assert t.active_sessions("u") == 1
try:
    t.record_event("u", "c", 6, "open")
    raise AssertionError("an unknown event_type should raise ValueError")
except ValueError:
    pass
"""},
        {"name": "Part 2: memory follows the window for users too", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Memory kept growing with the number of users or chats ever seen; per-user state must go once the user has no chat with an event in the window.",
         "code": _MODEL + r"""
calls = []
for i in range(10000):
    calls.append(lambda t, i=i: t.record_event(f"user{i}", "c", i // 4, "ping"))
    calls.append(lambda t, i=i: t.record_event(f"user{i}", "c", i // 4, "close"))
early, late = traced_peak_after(calls, {fn}(5), {2000, 20000})
assert late - early < 1_000_000, f"memory grew by {(late - early) / 1e6:.1f} MB"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "events.ordering", "code": r"""
t = {fn}(5)
t.record_event("ana", "a", 20, "ping")
assert t.active_sessions("ana", 20) == 1
t.record_event("ana", "a", 17, "close")
assert t.active_sessions("ana", 21) == 1
t.record_event("ana", "a", 23, "close")
assert t.active_sessions("ana", 23) == 0
t.record_event("ana", "a", 22, "ping")
assert t.active_sessions("ana", 24) == 0
t.record_event("ana", "a", 26, "ping")
assert t.active_sessions("ana", 26) == 1
t.record_event("ana", "b", 9, "ping")
assert t.active_sessions("ana", 26) == 1
assert t.recent_count("ana", "b") == 0
"""},
        {"name": "Part 3: random out-of-order streams", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "With events arriving out of order and a clock moved by now, an answer differed from the rules: the clock is the largest timestamp or now seen, events below clock - window are ignored, and the highest timestamps decide activity.",
         "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(1000 + seed)
    window = rng.choice([2, 4, 7])
    tracker, model = {fn}(window), Model(window)
    clock = 0
    for step in range(rng.randint(1, 50)):
        if rng.random() < 0.25:
            clock += rng.randint(0, 3)
            now = rng.choice([None, clock])
            for u in "abc":
                assert tracker.active_sessions(u, now) == model.active_sessions(u, now), (seed, step, u)
                now = None
        else:
            ts = max(0, clock + rng.randint(-window - 3, 2))
            clock = max(clock, ts)
            user, chat, kind = rng.choice("ab"), rng.choice("xy"), rng.choice(["ping", "ping", "close"])
            tracker.record_event(user, chat, ts, kind)
            model.record_event(user, chat, ts, kind)
        for u in "ab":
            for c in "xy":
                assert tracker.recent_count(u, c) == model.recent_count(u, c), (seed, step, u, c)
"""},
        {"name": "Part 3: memory with late events and no queries", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "With out-of-order events and no queries, memory kept growing with the number of chats ever seen; recording must also move the clock forward and drop what left the window.",
         "code": _MODEL + r"""
rng = random.Random(9)
calls = []
for i in range(20000):
    ts = i // 4 + 50 - rng.randint(0, 8)
    kind = "close" if i % 3 == 0 else "ping"
    calls.append(lambda t, i=i, ts=ts, kind=kind: t.record_event(f"user{i % 50000}", f"chat{i}", ts, kind))
early, late = traced_peak_after(calls, {fn}(5), {2000, 20000})
assert late - early < 1_000_000, f"memory grew by {(late - early) / 1e6:.1f} MB"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import heapq


class ActivityTracker:
    def __init__(self, window):
        self.window = window
        self._clock = None   # C: the largest timestamp or now seen so far
        self._seq = 0        # arrival order, to break timestamp ties
        self._events = []    # heap of (timestamp, chat) for every event in the window
        self._counts = {}    # chat -> its events in the window; absent means 0
        self._chats = {}     # chat -> (latest ping, latest close, counted); each is (timestamp, seq) or None
        self._active = {}    # user -> active chats; absent means 0
        self._expiry = []    # heap of (forget at, seq, chat, ping, close); entries may be stale

    def record_event(self, user_id, chat_id, timestamp, event_type="ping"):
        if event_type not in ("ping", "close"):
            raise ValueError(f"unknown event_type {event_type!r}")
        if self._clock is not None and timestamp < self._clock - self.window:
            return  # too late to change any answer from now on
        self._seq += 1
        chat = (user_id, chat_id)
        heapq.heappush(self._events, (timestamp, chat))
        self._counts[chat] = self._counts.get(chat, 0) + 1

        ping, close, counted = self._chats.get(chat, (None, None, False))
        stamp = (timestamp, self._seq)
        if event_type == "ping":
            ping = stamp if ping is None else max(ping, stamp)
        else:
            close = stamp if close is None else max(close, stamp)
        active = ping is not None and (close is None or ping > close)
        if active != counted:
            self._bump(user_id, 1 if active else -1)
        self._chats[chat] = (ping, close, active)
        latest = max(s[0] for s in (ping, close) if s is not None)
        heapq.heappush(self._expiry, (latest, self._seq, chat, ping, close))
        self._advance(timestamp)  # recording moves the clock too, or a query-free stream leaks

    def _bump(self, user_id, delta):
        count = self._active.get(user_id, 0) + delta
        if count:
            self._active[user_id] = count
        else:
            del self._active[user_id]

    def _advance(self, t):
        self._clock = t if self._clock is None else max(self._clock, t)
        edge = self._clock - self.window
        while self._events and self._events[0][0] < edge:
            chat = heapq.heappop(self._events)[1]
            self._counts[chat] -= 1
            if not self._counts[chat]:
                del self._counts[chat]
        while self._expiry and self._expiry[0][0] < edge:
            _, _, chat, ping, close = heapq.heappop(self._expiry)
            current = self._chats.get(chat)
            if current is None or current[:2] != (ping, close):
                continue  # superseded by a later event
            if current[2]:
                self._bump(chat[0], -1)  # its latest ping just left the window
            del self._chats[chat]

    def recent_count(self, user_id, chat_id):
        return self._counts.get((user_id, chat_id), 0)

    def active_sessions(self, user_id, now=None):
        if now is not None:
            self._advance(now)
        return self._active.get(user_id, 0)
''',
    "interview_questions": interview(
        concept=[
            "Why does the window follow the largest timestamp seen rather than the caller's chat or the wall clock?",
            "When can a chat's entry be deleted, and what happens to memory if it never is?",
        ],
        deep_dive=[
            "Why does each event cost amortized O(1) to evict when events arrive in timestamp order?",
        ],
        tradeoffs=[
            "Why does arrival order stop mattering once you keep each chat's highest ping and close timestamps?",
            "Why can an event older than the clock minus the window be dropped without changing any later answer?",
            "How would you partition millions of events per second across machines, and what about one very busy user?",
            "After a crash, how much history must a machine replay to rebuild its state?",
        ],
    ),
}
