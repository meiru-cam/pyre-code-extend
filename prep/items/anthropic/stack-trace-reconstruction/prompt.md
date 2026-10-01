A profiler observes a running program by taking periodic *samples*: at certain instants it records the full call stack — the ordered list of functions currently on the stack, from the outermost (the one that has been running the longest, called by nothing else in the trace) to the innermost (the one actually executing at that instant). A `Sample` pairs a timestamp with this list; an `Event` reports one function beginning or ending its time on the stack.

```py
from typing import Literal, NamedTuple


class Sample(NamedTuple):
    ts: float
    stack: list[str]   # outermost (root) first, innermost (currently executing) last


class Event(NamedTuple):
    kind: Literal["start", "end"]
    ts: float
    name: str
    depth: int   # 0-based position in the stack
```

Two samples can report the same function name at two different depths at once — direct or indirect recursion — so an event's *frame* is identified by its `(depth, name)` pair, not by `name` alone: `depth` is part of every event below, not a diagnostic extra.

### Part 1 — Samples to events

`convert_samples_to_events(samples, end_time)` takes `samples`, sorted by strictly increasing `ts`, and returns the list of events that explains how the stack evolved from the first sample up to `end_time` (a timestamp at least as large as the last sample's `ts`; when `samples` is empty, `end_time` is unused and the result is `[]`).

At the timestamp of the first sample, every frame in its stack starts, listed outermost to innermost.

Between one sample and the next, compare the two stacks position by position from the outside in and find their *longest common prefix* (LCP): the largest `p` such that the two stacks agree on every position `0, 1, ..., p - 1`. A frame at a position at or beyond `p` in the earlier stack has left the stack; a frame at a position at or beyond `p` in the later stack has entered it. Both kinds of change are stamped with the later sample's `ts`: frames that left end first, in order from the innermost position outward (the call that was deepest returns first), and only then do the frames that appeared start, in order from `p` outward to the innermost (a caller has to already be on the stack before anything it calls can start). Two adjacent samples with identical stacks (`p` equal to both their lengths) contribute no events at all — this is what compresses a run of unchanged samples down to nothing.

At `end_time`, every frame still in the last sample's stack ends, innermost first; `end_time` is the only source of closing events for those frames — `convert_samples_to_events` never closes a frame at an earlier time just because the input runs out.

```py
def convert_samples_to_events(samples: list[Sample], end_time: float) -> list[Event]: ...
```

```text
samples = [
    Sample(0.0, ["main"]),
    Sample(1.0, ["main", "solve"]),
    Sample(2.0, ["main", "solve", "solve"]),           # solve calls itself: two active frames named "solve"
    Sample(2.5, ["main", "solve", "solve"]),           # identical to the previous stack -> no events at 2.5
    Sample(4.0, ["main", "solve", "solve", "solve"]),  # solve recurses again: three active "solve" frames
    Sample(6.0, ["main", "solve", "helper"]),          # the two innermost "solve" frames return; helper is called
]
convert_samples_to_events(samples, end_time=9.0) == [
    Event("start", 0.0, "main", 0),
    Event("start", 1.0, "solve", 1),
    Event("start", 2.0, "solve", 2),      # a second, distinct "solve" frame, one level deeper than the first
    # at 2.5 the common prefix with the previous stack is 3 (the whole stack) -> no events
    Event("start", 4.0, "solve", 3),      # a third, distinct "solve" frame
    # at 6.0 the common prefix with ["main","solve","solve","solve"] is 2: main and solve@1 are unaffected
    Event("end", 6.0, "solve", 3),        # depth 3 leaves first (innermost)
    Event("end", 6.0, "solve", 2),        # then depth 2
    Event("start", 6.0, "helper", 2),     # only then does the new depth-2 frame start
    # end_time = 9.0: main, solve@1 and helper@2 are all still active in the last sample -> close them
    Event("end", 9.0, "helper", 2),
    Event("end", 9.0, "solve", 1),
    Event("end", 9.0, "main", 0),
]
```

### Part 2 — Streaming and noise

A real profiler delivers samples one at a time over what may be a long run; `stream_debounced_events` must not hold state proportional to the number of samples processed so far. It consumes `samples` as an iterator — it may only be safe to read once, front to back — and yields events one at a time as they become certain rather than returning a finished list. Unlike Part 1, it takes no `end_time`: when the iterator is exhausted, a frame that has not yet been confirmed closed simply produces no further event, exactly as Part 1 never closes a frame still active in its last sample, generalised here to "not yet confirmed closed."

A sampled profiler also has noise: an instrument momentarily misreporting a frame, or a call so short-lived that only a single sample happens to catch it. `stream_debounced_events(samples, k)` filters this with a debounce rule. Identify frames exactly as Part 1 does, by the position reached along the longest-common-prefix chain from each sample to the next, so a break in that chain always starts a brand-new frame at every position it reaches or passes, even if the same name reoccupies a position immediately afterward (see the example below). A frame produces no event until it has held its position for `k` consecutive samples (`k >= 1`; `k = 1` needs no debounce and reduces to Part 1's own transitions, minus the final closing at `end_time`, which Part 2 does not have). Its `start` event is stamped with the `k`-th of those consecutive samples. If the frame's position later becomes vacant and stays vacant for `k` consecutive samples in a row, its `end` event is stamped with the `k`-th of *those*. A frame that never reaches `k` consecutive occupying samples produces neither event; a frame that does reach `k`, but whose vacancy never reaches `k` consecutive samples before the input ends, produces a `start` but no `end`.

For example, with `k = 1` (every single sample already counts on its own), `["req", "validate"]` followed by `["req", "validate", "persist"]` has a common prefix of 2 (the whole earlier stack): `req` and `validate` are unbroken, and only `persist` starts. But `["req", "validate"]` followed by `["worker", "validate", "req"]` has a common prefix of 0 — position 0 already disagrees, `req` against `worker` — so `req` and `validate` both end there and a brand-new `req` starts at depth 2, even though the name `validate` still happens to sit at depth 1 and the name `req` happens to reappear one level deeper: neither is connected to the earlier occupant of its position, because the chain already broke before reaching it.

```text
stream_debounced_events([Sample(1, ["req", "validate"]), Sample(2, ["req", "validate", "persist"])], k=1)
== [Event("start", 1, "req", 0), Event("start", 1, "validate", 1), Event("start", 2, "persist", 2)]

stream_debounced_events([Sample(1, ["req", "validate"]), Sample(2, ["worker", "validate", "req"])], k=1)
== [Event("start", 1, "req", 0), Event("start", 1, "validate", 1),
    Event("end", 2, "validate", 1), Event("end", 2, "req", 0),
    Event("start", 2, "worker", 0), Event("start", 2, "validate", 1), Event("start", 2, "req", 2)]
```

```py
from collections.abc import Iterable, Iterator


def stream_debounced_events(samples: Iterable[Sample], k: int) -> Iterator[Event]: ...
```

```text
samples = [Sample(1, ["svc"]), Sample(2, ["svc"]), Sample(3, ["svc", "handle"]), Sample(4, ["svc"]),
           Sample(5, ["svc", "flush"]), Sample(6, ["svc", "flush"]), Sample(7, ["svc"]),
           Sample(8, ["svc"]), Sample(9, ["svc"])]
list(stream_debounced_events(samples, k=2)) == [
    Event("start", 2, "svc", 0),     # depth 0 holds "svc" at samples 1 and 2: 2 consecutive -> confirmed at sample 2
    # depth 1 holds "handle" only at sample 3, then reverts at sample 4: 1 consecutive -> never confirmed, no event
    Event("start", 6, "flush", 1),   # depth 1 holds "flush" at samples 5 and 6: 2 consecutive -> confirmed at sample 6
    Event("end", 8, "flush", 1),     # depth 1 is vacant at samples 7 and 8: 2 consecutive -> closed at sample 8
    # "svc" is still active when the input ends and its vacancy never starts -> no end event for it
]
```

### Part 3 — Suffix matching against known traces

A *complete* stack trace records every frame from the entry point (outermost) to the point of the error (innermost). `TraceIndex` is built once from a fixed collection of complete, known traces, each identified by a unique `trace_id`, and then answers queries: given only the innermost frames of an incomplete trace — a *suffix* — report every known trace whose own last frames, taken to the same length, are exactly that suffix, frame for frame.

`TraceIndex(traces)` takes `traces`, a list of `(trace_id, frames)` pairs with `frames` ordered entry point first, error point last, and non-empty. `index.match_suffix(frames)` takes a non-empty query in the same order and returns the `trace_id`s of every known trace with at least `len(frames)` frames whose own last `len(frames)` frames equal `frames` exactly, position for position, sorted lexicographically ascending; if no known trace matches — including every known trace that is simply shorter than the query — it returns the string `"UNKNOWN"` instead of an empty list.

```py
class TraceIndex:
    def __init__(self, traces: list[tuple[str, list[str]]]) -> None: ...
    def match_suffix(self, frames: list[str]) -> list[str] | str: ...
```

```text
traces = [
    ("T1", ["boot", "ingest", "parse", "oom"]),
    ("T2", ["worker", "parse", "oom"]),
    ("T3", ["boot", "cache", "oom"]),
    ("T4", ["boot", "ingest", "oom"]),
]
index = TraceIndex(traces)
index.match_suffix(["parse", "oom"])                   == ["T1", "T2"]              # both end ..., parse, oom
index.match_suffix(["oom"])                            == ["T1", "T2", "T3", "T4"]  # every trace ends in oom
index.match_suffix(["ingest", "oom"])                  == ["T4"]                    # only T4 ends ..., ingest, oom
index.match_suffix(["boot", "ingest", "parse", "oom"]) == ["T1"]                    # matches only the whole trace
index.match_suffix(["cache", "parse", "oom"])          == "UNKNOWN"                 # no known trace ends this way
```
