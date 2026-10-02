"""An iterator whose position can be saved and restored: over a list, then over rows that may be empty, then over lists nested to any depth."""

from ._interview import interview

# A model that lists every (index tuple, item) up front, and random data and calls to compare against it.
_MODEL = r"""
import json, random, time

def flatten(items, depth, prefix=()):
    if depth == 1:
        return [(prefix + (i,), x) for i, x in enumerate(items)]
    out = []
    for i, sub in enumerate(items):
        out += flatten(sub, depth - 1, prefix + (i,))
    return out

class Model:
    def __init__(self, items, depth):
        self.flat = flatten(items, depth)
        self.end = (len(items),) + (0,) * (depth - 1)
        self.index = {t: k for k, (t, _) in enumerate(self.flat)}
        self.depth, self.at = depth, 0
    def next(self):
        if self.at == len(self.flat):
            return ("stop",)
        self.at += 1
        return ("ok", self.flat[self.at - 1][1])
    def get(self):
        return self.flat[self.at][0] if self.at < len(self.flat) else self.end
    def set(self, state):
        if not isinstance(state, (tuple, list)) or len(state) != self.depth or not all(type(i) is int for i in state):
            return ("raise", "TypeError")
        state = tuple(state)
        if state == self.end:
            self.at = len(self.flat)
        elif state in self.index:
            self.at = self.index[state]
        else:
            return ("raise", "ValueError")
        return ("ok", None)

def nested(rng, depth, width, empty):
    if depth == 0:
        return rng.randint(0, 99)
    if rng.random() < empty:
        return []
    return [nested(rng, depth - 1, width, empty) for _ in range(rng.randint(1, width))]

def attempt(call):
    try:
        return ("ok", call())
    except StopIteration:
        return ("stop",)
    except Exception as e:
        return ("raise", type(e).__name__)

def compare(fn, items, depth, rng, steps, label):
    it = fn(items, depth) if depth != 1 or rng.random() < 0.5 else fn(items)
    m = Model(items, depth)
    states = [t for t, _ in m.flat] + [m.end]
    for step in range(steps):
        r = rng.random()
        if r < 0.45:
            got, want = attempt(lambda: next(it)), m.next()
        elif r < 0.65:
            got, want = ("ok", it.get_state()), ("ok", m.get())
            assert type(got[1]) is tuple, (label, step, got)
        else:
            if rng.random() < 0.6:
                s = rng.choice(states)
            else:
                s = tuple(rng.randint(-1, 4) for _ in range(depth))
            if rng.random() < 0.3:
                s = json.loads(json.dumps(s))
            got, want = attempt(lambda: it.set_state(s)), m.set(s)
        assert got == want, (label, step, got, want, items)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "checkpoint.recovery", "code": r"""
import json
def kind(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__
    return None
it = {fn}(["ox", "elk", "yak"])
start = it.get_state()
assert start == (0,)
assert list(it) == ["ox", "elk", "yak"]
assert it.get_state() == (3,)
it.set_state(json.loads(json.dumps((1,))))
assert next(it) == "elk"
assert kind(lambda: it.set_state((4,))) == "ValueError"
assert kind(lambda: it.set_state((True,))) == "TypeError"
assert kind(lambda: it.set_state("2")) == "TypeError"
assert next(it) == "yak"
it.set_state(start)
assert next(it) == "ox"
"""},
    {"name": "Part 1: checks, order and failed calls", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "TypeError comes first, for a state that is not a tuple or list, has the wrong length, or holds a non-int or a bool; then ValueError for an index outside 0..len(items); a rejected state leaves the position alone; an empty list starts at (0,); iter(it) is it; states may move backward after StopIteration.",
     "code": _MODEL + r"""
it = {fn}([])
assert it.get_state() == (0,) and attempt(lambda: next(it)) == ("stop",)
it = {fn}([[1], [2]])
assert iter(it) is it
assert next(it) == [1]
assert attempt(lambda: it.set_state((5, "x"))) == ("raise", "TypeError")
assert attempt(lambda: it.set_state((-1,))) == ("raise", "ValueError")
assert attempt(lambda: it.set_state((1.0,))) == ("raise", "TypeError")
assert attempt(lambda: it.set_state(None)) == ("raise", "TypeError")
assert attempt(lambda: it.set_state(())) == ("raise", "TypeError")
assert attempt(lambda: it.set_state((3,))) == ("raise", "ValueError")
assert it.get_state() == (1,) and next(it) == [2]
assert attempt(lambda: next(it)) == ("stop",) and attempt(lambda: next(it)) == ("stop",)
it.set_state([0])
assert [x for x in it] == [[1], [2]]
state = it.get_state()
it.set_state((2,))
assert it.get_state() == state == (2,)
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random lists and random next, get_state and set_state calls, including states sent through JSON, a result or error differed from a model that lists every position up front.",
     "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(seed)
    compare({fn}, [rng.randint(0, 9) for _ in range(rng.randint(0, 6))], 1, rng, 40, seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "edge.empty_or_boundary", "code": r"""
def kind(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__
    return None
it = {fn}([[], [4, 8], [], [6]], 2)
assert it.get_state() == (1, 0)
assert next(it) == 4 and it.get_state() == (1, 1)
assert next(it) == 8 and it.get_state() == (3, 0)
saved = it.get_state()
assert next(it) == 6 and it.get_state() == (4, 0)
for bad in [(0, 0), (1, 2), (2, 0)]:
    assert kind(lambda: it.set_state(bad)) == "ValueError", bad
it.set_state(saved)
assert list(it) == [6]
assert {fn}([[], []], 2).get_state() == (2, 0)
"""},
    {"name": "Part 2: states across instances", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
     "failure_message": "A state is checked only against the receiving iterator's own data: on data of the same shape iteration continues at the same index tuple, on another shape it is accepted only if it names an item or the end there; the exhausted state is (len(items), 0); a 2-int state is required.",
     "code": _MODEL + r"""
a = {fn}([[1, 2], [], [3]], 2)
next(a)
s = a.get_state()
b = {fn}([["x", "y"], [], ["z"]], 2)
b.set_state(s)
assert list(b) == ["y", "z"]
c = {fn}([[0], [5, 6, 7]], 2)
assert attempt(lambda: c.set_state(s)) == ("raise", "ValueError")
c.set_state((1, 2))
assert list(c) == [7] and c.get_state() == (2, 0)
assert attempt(lambda: c.set_state((2, 1))) == ("raise", "ValueError")
assert attempt(lambda: c.set_state((1,))) == ("raise", "TypeError")
c.set_state((2, 0))
assert attempt(lambda: next(c)) == ("stop",)
"""},
    {"name": "Part 2: random rows", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random rows, many of them empty, a result or error differed from a model that lists every (row, column) position up front.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(1000 + seed)
    compare({fn}, nested(rng, 2, 4, 0.4), 2, rng, 40, seed)
"""},
    {"name": "Part 2: long runs of empty rows", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "Walking 100,000 rows, most of them empty, with get_state after every item took too long: keep the position and move forward from it, never rescan from the first row.",
     "code": r"""
import time
rows = [[i] if i % 4 == 0 else [] for i in range(100000)]
start = time.perf_counter()
it = {fn}(rows, 2)
total = 0
for x in it:
    total += x
    it.get_state()
elapsed = time.perf_counter() - start
assert total == sum(range(0, 100000, 4)) and it.get_state() == (100000, 0)
assert elapsed < 2.0, f"{elapsed:.2f}s for 100,000 rows"
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "edge.empty_or_boundary", "code": r"""
data = [[], [[], [[2, 4]]], [[], [[8]]]]
it = {fn}(data, 3)
assert it.get_state() == (1, 1, 0)
assert next(it) == [2, 4]
assert it.get_state() == (2, 1, 0)
assert next(it) == [8] and it.get_state() == (3, 0, 0)
it.set_state([1, 1, 0])
assert list(it) == [[2, 4], [8]]
deep = {fn}(data, 4)
assert list(deep) == [2, 4, 8]
"""},
    {"name": "Part 3: random nesting", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random data nested 1 to 5 levels, with empty lists at any level, a result or error differed from a model that lists every index tuple up front.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(2000 + seed)
    depth = rng.randint(1, 5)
    compare({fn}, nested(rng, depth, 3, 0.3), depth, rng, 40, seed)
"""},
    {"name": "Part 3: very deep data", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "Data nested 3,000 levels deep failed or was too slow: walk levels with a loop and keep the lists along the current position instead of recursing or walking from the top on every call.",
     "code": r"""
import time
depth = 3000
data = ["leaf"]
for _ in range(depth - 1):
    data = [[], data, []]
start = time.perf_counter()
it = {fn}(data, depth)
state = it.get_state()
assert len(state) == depth and state[:3] == (1, 1, 1) and state[-1] == 0
assert next(it) == "leaf"
assert it.get_state() == (3,) + (0,) * (depth - 1)
for _ in range(200):
    it.set_state(state)
    assert next(it) == "leaf"
assert time.perf_counter() - start < 3.0
"""},
]

TASK = {
    "title": "Resumable Iterator",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ResumableIterator",
    "description_en": r"""Build `ResumableIterator`, an iterator that can report its position and later continue from a saved one: first over a list, then over rows that may be empty, then over lists nested to any depth.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ResumableIterator` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `ResumableIterator(items, depth=1)` walks data nested `depth` levels deep: whatever `depth` indexing steps reach is an item, even if it is a list. Items come in order of their index tuples.
- It is a Python iterator: `iter(it)` returns `it`, and `next(it)` returns the next item or raises `StopIteration`.
- `get_state()` returns a tuple of `depth` ints: the index tuple of the item `next` would return, or `(len(items), 0, ..., 0)` once nothing is left.
- `set_state(state)` accepts exactly those tuples for its own data, and lists too, since JSON turns tuples into lists. It may move backward or forward at any time.
- `set_state` raises `TypeError` for anything that is not a tuple or list of `depth` ints (a `bool` is not an int), and otherwise `ValueError` for a state `get_state` could never return. A rejected state changes nothing.
- The data is never changed while an iterator walks it.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it looks like an easy iterator until empty lists and saved states meet. Each later part adds one requirement: rows that may be empty force a single meaning for every position, and any depth forces a loop where recursion and rescans were enough before.

**Where it is used:** data loaders that resume an epoch after a crash or preemption, streaming jobs that checkpoint their offset in each partition, and paginated APIs that hand back a cursor.

Adapted from the resumable iterators question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as one class with a `depth` argument instead of an abstract base class and three subclasses. The source's first part, the abstract interface, becomes the Rules above, and Parts 2 and 3 add speed checks.""",
    "parts": [
        {
            "title": "A flat list",
            "description_en": r"""**Signature:** `ResumableIterator(items)` with `get_state() -> tuple` and `set_state(state)`, where `items` is a list, possibly empty.

- The state is `(i,)`, the index of the next item; it is `(len(items),)` at the end.
- `set_state` accepts `(i,)` or `[i]` for `0 <= i <= len(items)`.

**Example:** `it = ResumableIterator(["ox", "elk", "yak"])`:
- `get_state()` is `(0,)`; `list(it)` is `["ox", "elk", "yak"]`, and then `get_state()` is `(3,)`
- after `set_state([1])`, as it comes back from JSON, `next(it)` is `"elk"`
- `set_state((4,))` raises `ValueError`; `set_state((True,))` and `set_state("2")` raise `TypeError`""",
        },
        {
            "title": "Rows that may be empty",
            "description_en": r"""Keep Part 1. `ResumableIterator(rows, 2)` walks a list of rows, any of which may be empty, row by row.

- A state names an item that exists, so it skips ahead past empty rows and ended rows at once: it never points into an empty row or one past the end of a row.
- The state at the end is `(len(rows), 0)`.
- A state is checked only against this iterator's own rows, so a state saved on other rows of the same shape continues at the same position.
- 100,000 rows with a `get_state()` after every item must take well under a second.

**Example:** `it = ResumableIterator([[], [4, 8], [], [6]], 2)`:
- `get_state()` is `(1, 0)` before any `next`; after `4` it is `(1, 1)`, and after `8` it is already `(3, 0)`
- after `6` it is `(4, 0)`
- `(0, 0)`, `(1, 2)` and `(2, 0)` all raise `ValueError`""",
        },
        {
            "title": "Any depth",
            "description_en": r"""Keep Parts 1–2. `depth` may be any positive integer, and a list at any level may be empty.

- The same rules hold at every level: a state never points into an empty list or one past the end of a list, and the end is `(len(items), 0, ..., 0)`.
- Data nested 3,000 levels deep must work, so do not recurse once per level.

**Example:** `data = [[], [[], [[2, 4]]], [[], [[8]]]]` with `depth=3`:
- `get_state()` starts at `(1, 1, 0)`; `next` gives `[2, 4]`, then the state is `(2, 1, 0)`
- `next` gives `[8]`, and the state is `(3, 0, 0)`
- with `depth=4`, the same data gives `2`, `4` and `8`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What single number tells you where you are in a list, and what value should it have once every item is used? Which inputs to set_state are the wrong kind of value, and which are the right kind but out of range?"},
        {"level": 2, "kind": "analysis", "content": "Keep one index of the next item. next returns items[index] and adds one, or raises StopIteration at len(items). get_state returns (index,). set_state first checks the type (tuple or list, one element, an int that is not a bool), then the range 0..len(items), and only then stores the index."},
    ],
    "model_connections": [
        "Training data loaders save their position with each checkpoint so a preempted job resumes mid-epoch without repeating or skipping samples.",
        "Streaming evaluation and data pipelines keep offsets per shard so a restart continues where each shard left off.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A state is a plain tuple of ints, so it survives JSON and can be stored with a checkpoint.",
            "Normalising every position to the next real item gives each state exactly one spelling.",
            "Keeping the lists along the current path makes next amortised O(1) and set_state O(depth).",
        ],
        "cons": [
            "The state says nothing about the data, so a state from different data is accepted whenever it happens to fit.",
            "Moving past a long run of empty lists costs time proportional to the run, once.",
            "Changing the data while iterating breaks the contract with no detection.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).


class ResumableIterator:
    def __init__(self, items, depth=1):
        self._items = items
        self._depth = depth
        self._pos = [0] * depth
        self._path = [items]  # the list at each level along the current position
        self._settle()

    def _settle(self):
        """Move forward past ended and empty lists until the position names an item or the end."""
        d, pos, path = self._depth, self._pos, self._path
        while True:
            k = len(path) - 1
            if pos[k] < len(path[k]):
                if k == d - 1:
                    return
                path.append(path[k][pos[k]])  # go one level down
                continue
            if k == 0:
                return  # exhausted: (len(items), 0, ..., 0)
            path.pop()  # this list is used up: go up and on to the next sibling
            pos[k] = 0
            pos[k - 1] += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos[0] >= len(self._items):
            raise StopIteration
        item = self._path[-1][self._pos[-1]]
        self._pos[-1] += 1
        self._settle()
        return item

    def get_state(self):
        return tuple(self._pos)

    def set_state(self, state):
        if (not isinstance(state, (tuple, list)) or len(state) != self._depth
                or not all(isinstance(i, int) and not isinstance(i, bool) for i in state)):
            raise TypeError(f"expected {self._depth} ints, got {state!r}")
        state = list(state)  # JSON turns tuples into lists
        if state[0] == len(self._items) and not any(state[1:]):
            self._pos, self._path = state, [self._items]
            return
        path = [self._items]
        for k, i in enumerate(state):
            if not 0 <= i < len(path[k]):
                raise ValueError(f"{tuple(state)!r} does not name an item")
            if k < self._depth - 1:
                path.append(path[k][i])
        self._pos, self._path = state, path  # checked fully before changing anything
''',
    "interview_questions": interview(
        concept=[
            "What should get_state return before the first next, and what once every item is used?",
            "Why must set_state check the type before the range, and why is a bool rejected?",
        ],
        deep_dive=[
            "Why must set_state accept a list as well as a tuple, and what has to stay unchanged when it rejects a state?",
        ],
        tradeoffs=[
            "Why should a position always point at the next real item instead of the place where iteration stopped?",
            "Why does a state from other data of the same shape continue correctly, and what can go wrong on a different shape?",
            "How do you make next fast after a long run of empty lists, and what do you keep between calls?",
            "How would you detect that a saved state belongs to different data?",
        ],
    ),
}
