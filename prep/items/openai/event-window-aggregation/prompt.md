A chat product turns every user action into an *event*. Every event carries at least a `user_id`, a `chat_id`, and a `timestamp` — a non-negative integer number of seconds. `chat_id` is unique only *within* a `user_id`: two different users may independently use the same `chat_id`, so a chat's real identity is the pair `(user_id, chat_id)`, never `chat_id` alone.

### Part 1 — Recent event count

For this part, events across every user and chat arrive in non-decreasing order of `timestamp` (Part 3 drops this assumption). Implement a tracker, constructed once with a fixed positive integer `window`, that supports:

```py
class RecentEventCounter:
    def __init__(self, window: int): ...
    def record_event(self, user_id: str, chat_id: str, timestamp: int) -> None: ...
    def recent_count(self, user_id: str, chat_id: str) -> int: ...
```

`record_event` logs one event for that chat. `recent_count` returns how many events logged so far for that chat have a `timestamp` inside the closed interval `[T - window, T]`, where `T` is the largest `timestamp` passed to `record_event` so far, over every user and chat — not the caller's own chat, and not a wall clock.

At every point in the call sequence, the number of distinct `(user_id, chat_id)` pairs for which the tracker still holds any state must not exceed the number of pairs that currently have at least one event inside that window: it must never grow with how many distinct chats have ever been seen, only with how many are still inside the window. Aim for amortized O(1) time per call.

Example, with `window = 5`:

```text
record_event('alice', 'a1', 1)
record_event('alice', 'a1', 1)      # duplicate timestamps are allowed
recent_count('alice', 'a1')  -> 2   # T = 1, window [-4, 1]
record_event('alice', 'a2', 3)
recent_count('alice', 'a1')  -> 2   # T is now 3; both events at 1 are still inside [-2, 3]
recent_count('alice', 'a2')  -> 1
record_event('alice', 'a1', 9)
recent_count('alice', 'a1')  -> 1   # T = 9, window [4, 9]; only the new event remains
recent_count('alice', 'a2')  -> 0   # a2's only event, at 3, is also outside [4, 9]
```

### Part 2 — Active chat count

Events now carry a fourth field, `event_type`, one of `"ping"` (the user is actively interacting with the chat) or `"close"` (the user explicitly ended the chat — possibly much later, possibly never). A chat is *active* if the window contains a `"ping"` for it and, among the events recorded so far for that chat, no `"close"` has been recorded after that `"ping"` — "after" meaning later in the call sequence; when a `"ping"` and a `"close"` for the same chat share a `timestamp`, whichever one is recorded later in the call sequence wins (timestamps are still non-decreasing across calls, as in Part 1).

```py
class SessionActivityTracker:
    def __init__(self, window: int): ...
    def record(self, user_id: str, chat_id: str, event_type: str, timestamp: int) -> None: ...
    def active_sessions(self, user_id: str) -> int: ...
```

`active_sessions(user_id)` returns the number of that user's chats that are currently active. Part 1's memory rule still applies, now to whatever state decides activeness; in addition, no state the tracker keys by `user_id` alone may survive for a user whose `active_sessions` is currently `0`.

Example, with `window = 6`:

```text
record('alice', 'a1', 'ping', 0)
active_sessions('alice')  -> 1
record('alice', 'a2', 'ping', 2)
active_sessions('alice')  -> 2
record('alice', 'a1', 'close', 3)
active_sessions('alice')  -> 1
record('alice', 'a2', 'ping', 5)         # refreshes a2
active_sessions('alice')  -> 1
record('bruno', 'b1', 'ping', 9)         # T becomes 9; a2's ping at 5 is still inside [3, 9]
active_sessions('alice')  -> 1
record('bruno', 'b2', 'ping', 12)        # T becomes 12; a2's ping at 5 has now fallen out of [6, 12]
active_sessions('alice')  -> 0
```

### Part 3 — Out-of-order arrival

Drop the non-decreasing assumption: `record` may now be called in any order, independent of `timestamp`. To compensate, `active_sessions` takes an explicit second argument, `now`.

```py
class OutOfOrderSessionTracker:
    def __init__(self, window: int): ...
    def record(self, user_id: str, chat_id: str, event_type: str, timestamp: int) -> None: ...
    def active_sessions(self, user_id: str, now: int) -> int: ...
```

Three rules pin down the semantics:

- `now` is never smaller than any `timestamp` already passed to `record` by that point, and, across successive calls to `active_sessions`, `now` never decreases either — it is a forward-moving wall clock, not a value chosen after the fact.
- Activeness is decided by the *timestamps* the events carry, not by the order they arrived in: track, per chat, `last_ping` and `last_close`, each the largest `timestamp` seen so far for that `event_type` (`-∞` if none has arrived yet). The chat is active as of `now` iff `last_ping >= now - window` and `last_close <= last_ping` (a tie again goes to `ping`).
- Write $C$ for the largest value seen so far among every `now` passed to `active_sessions` and every `timestamp` passed to `record`; by rule 1 the clock has already reached $C$, and $C$ only ever grows. A `record` call with `timestamp < C - window` can therefore no longer change the answer to any future query, and changes nothing. The same memory rule as Parts 1–2 still holds, with $C$ in the role Part 1 gave `T`: a chat is remembered only while its most recent known timestamp, ping or close, is still within `window` of $C$.

Example, with `window = 4`:

```text
record('alice', 'a1', 'ping', 10)
active_sessions('alice', 10)  -> 1
record('alice', 'a1', 'close', 8)        # a close, but earlier than the ping -> no effect
active_sessions('alice', 11)  -> 1
record('alice', 'a1', 'close', 12)       # a close later than the ping
active_sessions('alice', 12)  -> 0
record('alice', 'a1', 'ping', 11)        # a late ping, but still earlier than that close -> still closed
active_sessions('alice', 13)  -> 0
record('alice', 'a1', 'ping', 14)        # a ping later than the close -> active again
active_sessions('alice', 14)  -> 1
record('alice', 'a2', 'ping', 5)         # C = 14, so C - window = 10; 5 < 10 -> already stale, dropped
active_sessions('alice', 14)  -> 1       # unchanged: a2 never entered the state
```

### Part 4 — Scaling up (discussed, not coded)

No code is required for this part; answer each question out loud, in a few sentences. The event volume grows to millions of events per second, spread across many machines, and no single machine can hold the whole state.

- How would you partition the stream across machines so each machine keeps running Part 3's algorithm unchanged, without talking to any other machine?
- Part 3 assumed `now` only moves forward and no event is outrageously late. What must the upstream delivery system guarantee for that to hold, and what should happen to an event that violates it?
- One `user_id`, or one `chat_id`, can produce far more traffic than the others. What breaks when that happens, and how would you change the partitioning to cope?
- A machine crashes and restarts. What is the minimum history it must replay to reconstruct correct state, and why does the memory bound from Parts 1–3 make that amount small?
