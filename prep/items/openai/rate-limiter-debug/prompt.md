The class below is meant to enforce per-user request limits under three tiers. A *rule* is a `(max_requests, period_seconds)` pair, and a request fits under a rule when fewer than `max_requests` of that user's already-admitted requests still count against it: an admitted request made at time `t` counts while `now - t <= period_seconds` and stops counting after that. A request is admitted only when it fits under all three rules at once; an admitted request is then recorded against all three rules, and a rejected request is recorded against none. Two calls for the same user running on different threads must come out the same as if one had run after the other.

`rate_limited` is a decorator standing in for a web framework's middleware — it wraps a view function and turns a rejected request into a `429` response instead of calling the view. `clock` is a zero-argument callable returning the current time in seconds, so a test can move time forward by hand instead of waiting on the wall clock. `_within_window` calls `checkpoint` once it has decided and before it records anything; `checkpoint` does nothing unless a test replaces it, so a test can hold several threads at that exact point.

```python
import time
from collections import defaultdict
from functools import wraps


class FakeClock:
    """A clock a test can move by hand instead of waiting on the wall clock."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RateLimiter:
    """Three tiers per user: minute_rule, hour_rule and day_rule, each a
    (max_requests, period_seconds) pair."""

    def __init__(self, minute_rule, hour_rule, day_rule, clock=time.time, checkpoint=lambda: None):
        self.minute_rule = minute_rule
        self.hour_rule = hour_rule
        self.day_rule = day_rule
        self.clock = clock
        self.checkpoint = checkpoint
        self.recent = defaultdict(list)     # user_id -> timestamps
        self.day_count = defaultdict(int)   # user_id -> request count

    def _within_window(self, user_id, now, max_requests, period):
        timestamps = self.recent[user_id]
        while timestamps and now - timestamps[0] > period:
            timestamps = timestamps[1:]
        would_pass = len(timestamps) < max_requests
        self.checkpoint()
        if not would_pass:
            return False
        timestamps.append(now)
        return True

    def should_allow_request(self, user_id):
        now = self.clock()
        if not self._within_window(user_id, now, *self.minute_rule):
            return False
        if not self._within_window(user_id, now, *self.hour_rule):
            return False
        max_requests, _period = self.day_rule
        self.day_count[user_id] += 1
        return self.day_count[user_id] <= max_requests


def rate_limited(limiter):
    """Stands in for a web framework's middleware: returns a 429 instead of calling the view."""
    def decorator(view):
        @wraps(view)
        def wrapped(user_id, *args, **kwargs):
            if not limiter.should_allow_request(user_id):
                return {"status": 429, "body": {"error": "rate limit exceeded"}}
            return view(user_id, *args, **kwargs)
        return wrapped
    return decorator


limiter = RateLimiter(minute_rule=(5, 60), hour_rule=(20, 3600), day_rule=(50, 86400))


@rate_limited(limiter)
def post_message(user_id, text):
    return {"status": 200, "body": {"echo": text}}


def test_allows_requests_under_the_limit():
    responses = [post_message("alice", "hi") for _ in range(3)]
    assert all(r["status"] == 200 for r in responses)


test_allows_requests_under_the_limit()
```

### Find and fix the four bugs

`RateLimiter` does not behave as described above: it has four bugs, all of them inside the class itself. `test_allows_requests_under_the_limit` passes because three requests never get close to any of the three caps; it says nothing about whether the limiter limits anything. Find each bug, write a test that fails against the class above and passes once the bug is fixed, and say what the bug does to the limiter's behaviour and why. Keep the class's surface — the constructor arguments, `should_allow_request` and the decorator — rather than replacing it with a limiter of your own design. Assume that once you claim a bug is fixed, more test cases will keep arriving that probe it from another angle.
