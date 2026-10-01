"""Turn profiler samples into call events, then debounce them as a stream, then match crash traces."""

from ._interview import interview

# A model that names every frame first, then derives events from each frame's first and last
# sample. It never walks two stacks side by side the way the reference does.
_HELPERS = r"""
import itertools, random, time, weakref

def frame_ids(samples):
    ids, prev, prev_ids, fresh = [], [], [], itertools.count()
    for _, stack in samples:
        same = 0
        while same < min(len(prev), len(stack)) and prev[same] == stack[same]:
            same += 1
        cur = [prev_ids[d] if d < same else next(fresh) for d in range(len(stack))]
        ids.append(cur)
        prev, prev_ids = stack, cur
    return ids

def spans(samples):
    # frame id -> [name, depth, first sample index, last sample index]
    out = {}
    for i, (ids, (_, stack)) in enumerate(zip(frame_ids(samples), samples)):
        for depth, fid in enumerate(ids):
            out.setdefault(fid, [stack[depth], depth, i, i])[3] = i
    return out.values()

def ordered(raw):
    # raw: (sample index, kind, ts, name, depth); ends deepest first, then starts shallowest first
    raw.sort(key=lambda e: (e[0], e[1] == "start", -e[4] if e[1] == "end" else e[4]))
    return [e[1:] for e in raw]

def slow_events(samples, end_time):
    raw = []
    for name, depth, first, last in spans(samples):
        raw.append((first, "start", samples[first][0], name, depth))
        if last + 1 < len(samples):
            raw.append((last + 1, "end", samples[last + 1][0], name, depth))
        else:
            raw.append((len(samples), "end", end_time, name, depth))
    return ordered(raw)

def slow_debounced(samples, k):
    raw = []
    for name, depth, first, last in spans(samples):
        if last - first + 1 >= k:
            raw.append((first + k - 1, "start", samples[first + k - 1][0], name, depth))
            if last + k < len(samples):
                raw.append((last + k, "end", samples[last + k][0], name, depth))
    return ordered(raw)

def random_samples(rng, n, names="abc", depth=4):
    out, stack = [], []
    for i in range(n):
        roll = rng.random()
        if roll < 0.3 and stack:
            stack = stack[:rng.randrange(len(stack))]
        elif roll < 0.6 and len(stack) < depth:
            stack = stack + [rng.choice(names)]
        elif roll < 0.75:
            stack = [rng.choice(names) for _ in range(rng.randint(0, depth))]
        out.append((float(i), list(stack)))
    return out

def as_tuples(events):
    return [tuple(e) for e in events]
"""

TASK = {
    "title": "Call Stacks from Profiler Samples",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "StackProfiler",
    "description_en": r"""Build `StackProfiler`, which turns a sampling profiler's call stacks into start and end events for each function call.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `StackProfiler` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A sample is a tuple `(ts, stack)`: a timestamp and a list of function names, outermost call first. Samples arrive in strictly increasing `ts`.
- An event is a tuple `(kind, ts, name, depth)`: `kind` is `"start"` or `"end"`, and `depth` is the 0-based position in the stack.
- Frames are tracked by position. From one sample to the next, find the longest common prefix `p` of the two stacks: positions below `p` hold the same frames, while every frame at position `p` or deeper has ended, and every name there in the new stack is a new frame. The same name at two depths is two frames.
- Within one sample, end events come first, deepest first; then start events, shallowest first.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the conversion is a short loop once the common-prefix rule is clear, and the later parts test streaming and indexing.

**Where it is used:** sampling profilers such as py-spy and perf, and the flame graphs and trace viewers built on them.

Adapted from the call stacks from profiler samples question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Samples to events",
            "description_en": r"""**Signature:** `StackProfiler()`, `events(samples, end_time) -> list[tuple]`

- Every frame of the first sample starts at its `ts`.
- At each later sample, the frames that left end and the frames that entered start, all stamped with that sample's `ts`. Identical consecutive stacks give no events.
- At `end_time`, which is at least the last `ts`, every frame still on the last stack ends, deepest first.
- No samples give `[]`.

**Example:** samples `(0, ["main"])`, `(1, ["main", "run"])`, `(2, ["main", "run", "run"])`, `(3, ["main", "io"])` with `end_time = 5` give:
- `("start", 0, "main", 0)`, `("start", 1, "run", 1)`, `("start", 2, "run", 2)`
- `("end", 3, "run", 2)`, `("end", 3, "run", 1)`, `("start", 3, "io", 1)`
- `("end", 5, "io", 1)`, `("end", 5, "main", 0)`""",
        },
        {
            "title": "Stream and debounce",
            "description_en": r"""**Signature:** `debounced_events(samples, k) -> iterator of tuples`

Keep Part 1 and add a streaming method that filters noise. `k >= 1`.

- `samples` may be a one-pass iterator, possibly endless. Yield each event as soon as it is certain, and keep state proportional to the deepest stack, not to the samples seen.
- A frame produces events only once it has held its position for `k` consecutive samples. Its start is stamped with the `k`-th of them.
- A confirmed frame ends at the `k`-th sample after the last sample it appeared in.
- There is no `end_time`: a frame not ended when the input runs out gets no end event. With `k = 1` the result is Part 1's events without the final ones at `end_time`.

**Example:** with `k = 2`, samples at `ts` 1 to 7 holding `["svc"]`, `["svc"]`, `["svc", "log"]`, `["svc", "db"]`, `["svc", "db"]`, `["svc"]`, `["svc"]` yield `("start", 2, "svc", 0)`, `("start", 5, "db", 1)`, `("end", 7, "db", 1)`. `log` held its position once, so it gives nothing.""",
        },
        {
            "title": "Match crash traces",
            "description_en": r"""**Signature:** `StackProfiler(known_traces=())`, `match_suffix(frames) -> list[str] | str`

Keep Parts 1–2 and add an index of known crash traces:

- `known_traces` is a list of `(trace_id, frames)` pairs, with frames outermost first and never empty.
- `match_suffix(frames)` takes the innermost frames of a partial trace, never empty and outermost of them first, and returns the ids of every known trace whose last `len(frames)` frames equal them, sorted ascending.
- A trace shorter than the query does not match. With no match, return the string `"UNKNOWN"`.
- A query costs time proportional to its length and its answer, not to the number of known traces.

**Example:** with `("A", ["boot", "load", "parse", "crash"])`, `("B", ["serve", "parse", "crash"])` and `("C", ["boot", "crash"])`:
- `match_suffix(["parse", "crash"])` is `["A", "B"]`
- `match_suffix(["load", "crash"])` is `"UNKNOWN"`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What do you need to remember from the previous sample to process the next one? How far do two consecutive stacks agree, and what does that tell you about every deeper position? In which order must a caller and its callee end?"},
        {"level": 2, "kind": "analysis", "content": "Keep the previous stack. For each new sample, count the length p of the common prefix; emit ends for the previous stack's positions from its deepest down to p, then starts for the new stack's positions from p up. At end_time, end everything left, deepest first."},
    ],
    "model_connections": [
        "Sampling profilers such as py-spy and Linux perf record stacks at intervals, and tools like speedscope rebuild call spans from them to draw flame charts.",
        "Crash reporting services such as Sentry group incoming stack traces by their innermost frames to find known issues.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Diffing consecutive stacks by common prefix needs only the previous stack and runs in time linear in the total stack sizes.",
            "Debouncing hides frames seen in a single sample, which are often noise or calls too short to measure.",
            "A trie over reversed traces answers a suffix query in time proportional to the query, however many traces are indexed.",
        ],
        "cons": [
            "Every timestamp is when a change was observed, not when the call actually started or returned.",
            "Debouncing delays every event by k - 1 samples and drops genuinely short calls.",
            "Storing every trace id at every trie node costs memory proportional to the total length of the traces.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "events.ordering", "code": _HELPERS + r"""
samples = [(0, ["main"]), (1, ["main", "run"]), (2, ["main", "run", "run"]), (3, ["main", "io"])]
assert as_tuples({fn}().events(samples, 5)) == [
    ("start", 0, "main", 0), ("start", 1, "run", 1), ("start", 2, "run", 2),
    ("end", 3, "run", 2), ("end", 3, "run", 1), ("start", 3, "io", 1),
    ("end", 5, "io", 1), ("end", 5, "main", 0),
]
"""},
        {"name": "Part 1: empty input and unchanged stacks", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "No samples give [], identical consecutive stacks give no events, and empty stacks are allowed.",
         "code": _HELPERS + r"""
profiler = {fn}()
assert as_tuples(profiler.events([], 10)) == []
assert as_tuples(profiler.events([(1, []), (2, [])], 3)) == []
assert as_tuples(profiler.events([(1, ["a"]), (2, ["a"]), (3, ["a"])], 3)) == [("start", 1, "a", 0), ("end", 3, "a", 0)]
assert as_tuples(profiler.events([(1, ["a"]), (2, []), (3, ["a"])], 4)) == [
    ("start", 1, "a", 0), ("end", 2, "a", 0), ("start", 3, "a", 0), ("end", 4, "a", 0)]
"""},
        {"name": "Part 1: a broken prefix restarts deeper frames", "part": 1, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Once two stacks differ at a position, every deeper frame is new even when its name and depth repeat.",
         "code": _HELPERS + r"""
samples = [(1, ["req", "check"]), (2, ["job", "check", "req"])]
assert as_tuples({fn}().events(samples, 2)) == [
    ("start", 1, "req", 0), ("start", 1, "check", 1),
    ("end", 2, "check", 1), ("end", 2, "req", 0),
    ("start", 2, "job", 0), ("start", 2, "check", 1), ("start", 2, "req", 2),
    ("end", 2, "req", 2), ("end", 2, "check", 1), ("end", 2, "job", 0),
]
"""},
        {"name": "Part 1: random samples match a reference", "part": 1, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On random samples, the events or their order differed from tracking each frame's first and last sample.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    samples = random_samples(rng, rng.randint(0, 12))
    end_time = len(samples) + rng.random()
    assert as_tuples({fn}().events(samples, end_time)) == slow_events(samples, end_time), (seed, samples)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "events.ordering", "code": _HELPERS + r"""
stacks = [["svc"], ["svc"], ["svc", "log"], ["svc", "db"], ["svc", "db"], ["svc"], ["svc"]]
samples = [(ts, stack) for ts, stack in enumerate(stacks, start=1)]
assert as_tuples({fn}().debounced_events(samples, 2)) == [
    ("start", 2, "svc", 0), ("start", 5, "db", 1), ("end", 7, "db", 1)]
assert as_tuples({fn}().debounced_events(iter(samples), 1)) == as_tuples({fn}().events(samples, 99))[:-1]
"""},
        {"name": "Part 2: events stream out of an endless input", "part": 2, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "debounced_events must read its input once, lazily, and yield each event as soon as it is certain.",
         "code": _HELPERS + r"""
def endless():
    for i in itertools.count():
        assert i < 1000, "read far past the events that were asked for"
        yield (i, ["loop", "tick"] if i % 4 < 2 else ["loop"])
first = list(itertools.islice({fn}().debounced_events(endless(), 2), 4))
assert as_tuples(first) == [("start", 1, "loop", 0), ("start", 1, "tick", 1), ("end", 3, "tick", 1), ("start", 5, "tick", 1)]
"""},
        {"name": "Part 2: old samples are not kept", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After many samples, the stacks of old samples were still alive; keep only state the size of the deepest stack.",
         "code": _HELPERS + r"""
class Stack(list):
    pass
refs = []
def samples():
    for i in range(5000):
        stack = Stack(["main", "loop"] + (["work"] if i % 3 else []))
        refs.append(weakref.ref(stack))
        yield (i, stack)
events = {fn}().debounced_events(samples(), 2)
count = sum(1 for _ in itertools.islice(events, 3000))
alive = sum(ref() is not None for ref in refs)
assert count == 3000
assert alive <= 4, f"{alive} sample stacks are still referenced"
"""},
        {"name": "Part 2: random samples match a reference", "part": 2, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On random samples, the debounced events differed from confirming each frame after k samples and ending it k samples after it left.",
         "code": _HELPERS + r"""
for seed in range(400):
    rng = random.Random(seed)
    samples = random_samples(rng, rng.randint(0, 16), names="ab", depth=3)
    k = rng.randint(1, 4)
    got = as_tuples({fn}().debounced_events(iter(samples), k))
    assert got == slow_debounced(samples, k), (seed, k, samples, got)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
profiler = {fn}(known_traces=[("A", ["boot", "load", "parse", "crash"]), ("B", ["serve", "parse", "crash"]), ("C", ["boot", "crash"])])
assert profiler.match_suffix(["parse", "crash"]) == ["A", "B"]
assert profiler.match_suffix(["crash"]) == ["A", "B", "C"]
assert profiler.match_suffix(["load", "crash"]) == "UNKNOWN"
assert profiler.match_suffix(["boot", "load", "parse", "crash"]) == ["A"]
assert profiler.match_suffix(["x", "boot", "load", "parse", "crash"]) == "UNKNOWN"
"""},
        {"name": "Part 3: random traces match a scan", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random traces, match_suffix differed from checking each trace's last frames directly.",
         "code": r"""
import random
for seed in range(100):
    rng = random.Random(seed)
    traces = [(f"t{i:02d}", [rng.choice("abc") for _ in range(rng.randint(1, 5))]) for i in range(rng.randint(0, 12))]
    rng.shuffle(traces)
    profiler = {fn}(known_traces=traces)
    for _ in range(20):
        query = [rng.choice("abc") for _ in range(rng.randint(1, 4))]
        expected = sorted(tid for tid, frames in traces if len(frames) >= len(query) and frames[-len(query):] == query)
        assert profiler.match_suffix(query) == (expected or "UNKNOWN"), (seed, traces, query)
"""},
        {"name": "Part 3: queries do not scan every trace", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "match_suffix on 20,000 known traces was much slower than on 20; index the traces by their innermost frames.",
         "code": r"""
import time
def index(n):
    return {fn}(known_traces=[(f"t{i}", ["main", f"f{i}", "crash"]) for i in range(n)])
small, big = index(20), index(20000)
assert big.match_suffix(["f7", "crash"]) == ["t7"]
def best(profiler):
    out = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for i in range(500):
            profiler.match_suffix([f"f{i % 20}", "crash"])
        out = min(out, time.perf_counter() - start)
    return out
ratio = best(big) / best(small)
assert ratio < 20, f"queries on 20,000 traces took {ratio:.0f}x as long"
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).


def _common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


class StackProfiler:
    def __init__(self, known_traces=()):
        # A trie over reversed traces; every node lists the ids of all traces that pass through it.
        self._trie = {}
        for trace_id, frames in known_traces:
            node = self._trie
            for frame in reversed(frames):
                node = node.setdefault(frame, {})
                node.setdefault("$ids", []).append(trace_id)
        stack = [self._trie]
        while stack:
            node = stack.pop()
            for key, child in node.items():
                if key == "$ids":
                    child.sort()
                else:
                    stack.append(child)

    def events(self, samples, end_time):
        out, prev = [], []
        for ts, stack in samples:
            p = _common_prefix(prev, stack)
            for depth in range(len(prev) - 1, p - 1, -1):
                out.append(("end", ts, prev[depth], depth))
            for depth in range(p, len(stack)):
                out.append(("start", ts, stack[depth], depth))
            prev = stack
        for depth in range(len(prev) - 1, -1, -1):
            out.append(("end", end_time, prev[depth], depth))
        return out

    def debounced_events(self, samples, k):
        prev, streak = [], []  # streak[d]: consecutive samples position d has held its current frame
        confirmed = 0          # positions below this have had their start emitted
        closing = {}           # depth -> [name, samples since a confirmed frame there left]
        for ts, stack in samples:
            p = _common_prefix(prev, stack)
            for entry in closing.values():
                entry[1] += 1
            for depth in range(len(prev) - 1, p - 1, -1):
                if depth < confirmed:
                    closing[depth] = [prev[depth], 1]
            confirmed = min(confirmed, p)
            for depth in sorted((d for d, (_, gone) in closing.items() if gone >= k), reverse=True):
                yield ("end", ts, closing.pop(depth)[0], depth)
            streak = [streak[d] + 1 if d < p else 1 for d in range(len(stack))]
            while confirmed < len(stack) and streak[confirmed] >= k:
                yield ("start", ts, stack[confirmed], confirmed)
                confirmed += 1
            prev = stack

    def match_suffix(self, frames):
        node = self._trie
        for frame in reversed(frames):
            if frame not in node:
                return "UNKNOWN"
            node = node[frame]
        return list(node["$ids"])
''',
    "interview_questions": interview(
        concept=[
            "Why are frames identified by their position along the common prefix rather than by name?",
            "In which order do you emit the ends and the starts at one sample, and why?",
        ],
        deep_dive=[
            "What does a sample's timestamp tell you about when a call really started or returned?",
        ],
        tradeoffs=[
            "What state does a streaming, debounced converter need, and why does it stay bounded by the stack depth?",
            "What does debouncing cost you, and how would you choose k?",
            "How does reversing the traces turn suffix matching into a trie lookup, and what does the index cost in memory?",
        ],
    ),
}
