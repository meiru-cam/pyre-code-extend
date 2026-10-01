"""Shard ranges capped at an overlap limit, holes closed, at scale; then keys routed by hashing."""

from ._interview import interview

# A key-by-key oracle written straight from the rules; only fit for small coordinates.
_ORACLE = r"""
import builtins, collections, itertools, random, time

def slow_rebalance(limit, shards, fill_holes=True):
    kept = []
    for sid, start, end in sorted(shards, key=lambda s: (s[1], s[2], s[0])):
        for k in range(start, end + 1):
            if sum(1 for _, a, b in kept if a <= k <= b) < limit:
                kept.append([sid, k, end])
                break
    if fill_holes and kept:
        lo, hi = min(s[1] for s in shards), max(s[2] for s in shards)
        for k in range(lo, hi + 1):
            if not any(a <= k <= b for _, a, b in kept):
                owner = min((s for s in kept if s[2] == k - 1), key=lambda s: (s[1], s[0]))
                owner[2] = k
    return sorted((tuple(s) for s in kept), key=lambda s: (s[1], s[2], s[0]))

def has_hole(shards):
    if not shards:
        return False
    lo, hi = min(s[1] for s in shards), max(s[2] for s in shards)
    return any(not any(a <= k <= b for _, a, b in shards) for k in range(lo, hi + 1))

def random_shards(rng, n, span):
    ids = rng.sample("abcdefghijklmnop", n)
    return [(ids[i], *sorted((rng.randint(0, span), rng.randint(0, span)))) for i in range(n)]

def as_tuples(result):
    return [tuple(s) for s in result]
"""

# Key routing checks shared by the Part 4 cases.
_ROUTING = r"""
import builtins, collections, random

def owners(router, keys):
    return {k: router.locate(k) for k in keys}

def router_with(ids):
    router = {fn}()
    for sid in ids:
        router.add_shard(sid)
    return router
"""

TASK = {
    "title": "Shard Rebalancing",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ShardManager",
    "description_en": r"""Build `ShardManager`, which decides which shard owns which integer keys.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ShardManager` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A shard is a tuple `(id, start, end)`: a string id, distinct within one call, and the closed range of keys `start` to `end`, both included, with `start <= end`.
- A key's coverage is the number of shards whose range contains it.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the rule reads like a simulation over keys, and the work is seeing that it is a greedy over intervals that never needs to look at a single key. Each later part adds one requirement.

**Where it is used:** range-sharded databases such as Bigtable, HBase and CockroachDB split, merge and move key ranges between servers.

Adapted from the shard rebalancing question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, with the rule split into parts on one class.""",
    "parts": [
        {
            "title": "Cap the overlap",
            "description_en": r"""**Signature:** `ShardManager().rebalance(limit, shards) -> list[tuple]`, with `limit >= 1`

Go through the shards one at a time, sorted by `start`, then `end`, then `id`:
- Count coverage over the shards kept so far, with the ranges they were kept with.
- Move the shard's start forward to the first key in its range whose coverage is below `limit`. If its own start qualifies, it stays.
- If no key in its range qualifies, drop the shard.
- Otherwise keep it with range `(new start, end)`.

Return the kept shards as `(id, start, end)` tuples, sorted by `(start, end, id)`. Do not modify the input.

In this part the input shards together cover every key from the smallest `start` to the largest `end`.

**Example**, `limit = 2`, input `a (0, 30)`, `b (0, 32)`, `c (0, 34)`, `d (0, 90)`, `e (1, 32)`, `f (91, 100)`:
- `a` and `b` are kept as they are
- `c` starts at `31`: `a` and `b` already cover `0` to `30` twice
- `d` starts at `33`: every key up to `32` is covered twice
- `e` is dropped: every key from `1` to `32` is covered twice
- result: `[("a", 0, 30), ("b", 0, 32), ("c", 31, 34), ("d", 33, 90), ("f", 91, 100)]`""",
        },
        {
            "title": "Close the holes",
            "description_en": r"""Keep Part 1. The input may now leave holes: keys between the smallest `start` and the largest `end` that no input shard covers.

- After every shard has been processed, take each maximal run of keys in that span that no kept shard covers.
- Extend the end of the kept shard that ends at the key just before the run, up to the run's last key.
- If several kept shards end there, extend the one with the smallest `start`, then the smallest `id`.
- Sort the result by the final `(start, end, id)`.

**Example**, `limit = 2`, input `a (0, 4)`, `b (0, 4)`, `c (2, 9)`, `f (3, 6)`, `d (5, 9)`, `e (14, 16)`:
- after Part 1's rule: `a (0, 4)`, `b (0, 4)`, `c (5, 9)`, `f (5, 6)`, `d (7, 9)`, `e (14, 16)`
- keys `10` to `13` are a hole; `c` and `d` both end at `9`, and `c` has the smaller start
- result: `[("a", 0, 4), ("b", 0, 4), ("f", 5, 6), ("c", 5, 13), ("d", 7, 9), ("e", 14, 16)]`""",
        },
        {
            "title": "Large keys",
            "description_en": r"""Keep Parts 1–2. Keys may now be anywhere from `-10**9` to `10**9`, and there may be 20,000 shards.

- `rebalance` must run in `O(n log n)` time for `n` shards, however wide the ranges are. It cannot visit keys one by one.

**Example**, `limit = 1`: `[("x", -10**9, 10**9), ("y", 0, 10**9), ("z", 5, 10)]` returns `[("x", -10**9, 10**9)]` at once.""",
        },
        {
            "title": "Route keys by hashing",
            "description_en": r"""Keep Parts 1–3. Rebalancing ranges can move many shards' boundaries when one shard changes. Add a second way to assign keys, where the shard set changes one shard at a time and each key goes to exactly one shard. It does not use `rebalance`.

**Signature:** `add_shard(shard_id) -> None`, `remove_shard(shard_id) -> None`, `locate(key) -> str`

- `add_shard` raises `ValueError` if the id is already present; `remove_shard` raises `ValueError` if it is absent. `locate(key)` returns the id of the shard that owns the integer `key`, and raises `LookupError` when there are no shards.
- `locate` depends only on the key and the current set of shard ids: not on the order they were added, and not on the process. Do not use the built-in `hash()`, whose result for a `str` changes between processes; `hashlib` and `zlib.crc32` are fine.
- Only keys that must move do: after `add_shard(x)`, every key whose owner changed is now owned by `x`. After `remove_shard(x)`, every key whose owner changed was owned by `x`.
- Keys spread about evenly: with `N` shards, each owns roughly `1/N` of them.

**Example**, over the keys `0` to `29,999`:
- add `"red"`, `"green"` and `"blue"`: each owns roughly a third of the keys
- add `"gold"`: roughly a quarter of the keys change owner, and all of them now belong to `"gold"`
- remove `"green"`: exactly the keys `"green"` owned change owner""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Think of limit lanes, each holding kept shards that never overlap. When a shard arrives in start order, which lane frees up first, and from which key? Is the first key where coverage drops below limit always where that lane frees up, or the shard's own start?"},
        {"level": 2, "kind": "analysis", "content": "Keep a min-heap of at most limit keys: the key from which each lane is free. For each shard in (start, end, id) order, take the smallest; an unused lane counts as free at the shard's start. The new start is max(start, that key). If it is past end, put the key back and drop the shard; otherwise keep it and push end + 1."},
    ],
    "model_connections": [
        "Range-partitioned stores for training data and features rebalance key ranges when shards grow uneven.",
        "Inference fleets route a user or session to a replica with consistent or rendezvous hashing, so adding a replica moves few sessions and their KV caches.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A heap of at most limit free keys turns the shift rule into O(n log n), with no key ever visited.",
            "Shifting starts forward keeps every kept shard's end where it was, so no new holes appear.",
            "Consistent hashing with virtual nodes moves only about 1/N of the keys when a shard joins or leaves.",
        ],
        "cons": [
            "The processing order decides who yields: long shards that sort later absorb all the shifts.",
            "Range rebalancing balances coverage, not load; a hot range still needs splitting.",
            "Hash routing balances key counts, not traffic, and loses range scans over adjacent keys.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
shards = [("a", 0, 30), ("b", 0, 32), ("c", 0, 34), ("d", 0, 90), ("e", 1, 32), ("f", 91, 100)]
original = list(shards)
result = [tuple(s) for s in {fn}().rebalance(2, shards)]
assert result == [("a", 0, 30), ("b", 0, 32), ("c", 31, 34), ("d", 33, 90), ("f", 91, 100)], result
assert shards == original, "the input must not change"
"""},
        {"name": "Part 1: every small input without holes", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On small inputs without holes, the kept shards differ from applying the rule key by key: sort by (start, end, id), move each start to the first key covered fewer than limit times by the shards kept so far, drop the shard if there is none.",
         "code": _ORACLE + r"""
checked = 0
for n in (1, 2, 3):
    for combo in itertools.product(itertools.combinations_with_replacement(range(4), 2), repeat=n):
        shards = [("ABC"[i], *combo[i]) for i in range(n)]
        if has_hole(shards):
            continue
        for limit in (1, 2, 3):
            got = as_tuples({fn}().rebalance(limit, shards))
            assert got == slow_rebalance(limit, shards), (limit, shards, got)
            checked += 1
rng = random.Random(1)
while checked < 6000:
    shards = random_shards(rng, rng.randint(1, 9), rng.choice([12, 40]))
    if has_hole(shards):
        continue
    limit = rng.randint(1, 4)
    got = as_tuples({fn}().rebalance(limit, shards))
    assert got == slow_rebalance(limit, shards), (limit, shards, got)
    checked += 1
"""},
        {"name": "Part 1: limit 1, ties and nested ranges", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Ties in start and end are processed in id order, a range inside a fuller one is dropped, limit 1 leaves no overlap, and an empty input gives an empty list.",
         "code": _ORACLE + r"""
manager = {fn}()
assert as_tuples(manager.rebalance(3, [])) == []
assert as_tuples(manager.rebalance(1, [("solo", 7, 7)])) == [("solo", 7, 7)]
same = [("q", 0, 5), ("p", 0, 5), ("r", 0, 5)]
assert as_tuples(manager.rebalance(2, same)) == [("p", 0, 5), ("q", 0, 5)], "ties go to the smaller id"
assert as_tuples(manager.rebalance(1, [("a", 0, 4), ("b", 2, 6), ("c", 7, 9)])) == [("a", 0, 4), ("b", 5, 6), ("c", 7, 9)]
assert as_tuples(manager.rebalance(1, [("outer", 0, 20), ("inner", 5, 9)])) == [("outer", 0, 20)]
assert as_tuples(manager.rebalance(2, [("outer", 0, 20), ("inner", 5, 9)])) == [("outer", 0, 20), ("inner", 5, 9)]
assert as_tuples(manager.rebalance(10, [("a", 0, 1), ("b", 0, 1)])) == [("a", 0, 1), ("b", 0, 1)]
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
shards = [("a", 0, 4), ("b", 0, 4), ("c", 2, 9), ("f", 3, 6), ("d", 5, 9), ("e", 14, 16)]
result = [tuple(s) for s in {fn}().rebalance(2, shards)]
assert result == [("a", 0, 4), ("b", 0, 4), ("f", 5, 6), ("c", 5, 13), ("d", 7, 9), ("e", 14, 16)], result
"""},
        {"name": "Part 2: random inputs with holes", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Every uncovered run of keys must go to the kept shard ending just before it, the one with the smallest start and then id among ties, and the result must be sorted by the final ranges.",
         "code": _ORACLE + r"""
rng = random.Random(2)
holes = 0
for trial in range(6000):
    shards = random_shards(rng, rng.randint(1, 8), rng.choice([12, 40]))
    holes += has_hole(shards)
    limit = rng.randint(1, 4)
    got = as_tuples({fn}().rebalance(limit, shards))
    assert got == slow_rebalance(limit, shards), (limit, shards, got)
assert holes > 500
tie = [("y", 0, 3), ("x", 0, 3), ("w", 6, 8)]
assert as_tuples({fn}().rebalance(2, tie)) == [("y", 0, 3), ("x", 0, 5), ("w", 6, 8)]
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "performance.complexity", "code": r"""
result = [tuple(s) for s in {fn}().rebalance(1, [("x", -10**9, 10**9), ("y", 0, 10**9), ("z", 5, 10)])]
assert result == [("x", -10**9, 10**9)], result
result = [tuple(s) for s in {fn}().rebalance(2, [("x", -10**9, 10**9), ("y", 0, 10**9), ("z", 5, 10)])]
assert result == [("x", -10**9, 10**9), ("y", 0, 10**9)], result
"""},
        {"name": "Part 3: 20,000 wide shards", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "20,000 shards took far more than ten times as long as 2,000, or the result broke the limit or left a key uncovered; use a heap of at most limit free keys instead of comparing against every kept shard.",
         "code": _ORACLE + r"""
def workload(n, seed):
    rng = random.Random(seed)
    shards = []
    for i in range(n):
        lo = rng.randint(-10**9, 10**9)
        shards.append((f"s{i}", lo, lo + rng.randint(0, 10**6)))
    return shards

def check(limit, shards, result):
    lo, hi = min(s[1] for s in shards), max(s[2] for s in shards)
    events = sorted([(s[1], 1) for s in result] + [(s[2] + 1, -1) for s in result])
    assert events[0][0] == lo and events[-1][0] == hi + 1
    active, position = 0, lo
    for key, delta in events:
        assert key == position or active >= 1, f"keys {position}..{key - 1} are uncovered"
        active, position = active + delta, key
        assert active <= limit, f"coverage {active} above {limit} at {key}"
    starts = {s[0]: s for s in shards}
    for sid, start, end in result:
        assert start >= starts[sid][1] and end >= starts[sid][2]

def best_of_three(limit, shards):
    best = float("inf")
    for _ in range(3):
        begin = time.perf_counter()
        result = as_tuples({fn}().rebalance(limit, shards))
        best = min(best, time.perf_counter() - begin)
    check(limit, shards, result)
    return best

small, big = workload(2000, 1), workload(20000, 2)
ratio = best_of_three(3, big) / best_of_three(3, small)
assert ratio < 30, f"10x the shards took {ratio:.0f}x as long"
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "routing.selection", "code": _ROUTING + r"""
keys = range(30000)
router = router_with(["red", "green", "blue"])
before = owners(router, keys)
share = collections.Counter(before.values())
assert all(0.25 < share[s] / len(keys) < 0.42 for s in ["red", "green", "blue"]), share
router.add_shard("gold")
after_add = owners(router, keys)
moved = [k for k in keys if before[k] != after_add[k]]
assert all(after_add[k] == "gold" for k in moved), "a key moved to a shard other than the new one"
assert 0.17 < len(moved) / len(keys) < 0.33, len(moved)
router.remove_shard("green")
after_remove = owners(router, keys)
moved = {k for k in keys if after_add[k] != after_remove[k]}
assert moved == {k for k in keys if after_add[k] == "green"}
"""},
        {"name": "Part 4: errors", "part": 4, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "add_shard of a present id and remove_shard of an absent id raise ValueError, and locate with no shards raises LookupError.",
         "code": _ROUTING + r"""
router = {fn}()
try:
    router.locate(1)
    raise AssertionError("locate with no shards should raise LookupError")
except LookupError:
    pass
router.add_shard("a")
assert router.locate(12345) == "a"
for bad in [lambda: router.add_shard("a"), lambda: router.remove_shard("b")]:
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
router.remove_shard("a")
try:
    router.locate(1)
    raise AssertionError("locate after removing the last shard should raise LookupError")
except LookupError:
    pass
router.add_shard("a")
assert router.locate(-7) == "a"
"""},
        {"name": "Part 4: only keys that must move do", "part": 4, "visibility": "unshown", "behavior": "routing.selection",
         "failure_message": "After add_shard(x) every key that changed owner must belong to x, after remove_shard(x) every key that changed owner must have belonged to x, and the owners must not depend on the order shards were added.",
         "code": _ROUTING + r"""
keys = range(-2000, 10000)
rng = random.Random(4)
router, present, previous = {fn}(), [], None
for step in range(25):
    if present and (len(present) > 6 or rng.random() < 0.4):
        x = rng.choice(present)
        present.remove(x)
        router.remove_shard(x)
    else:
        x = f"shard-{step}"
        present.append(x)
        router.add_shard(x)
    now = owners(router, keys) if present else None
    if previous and now:
        for k in keys:
            if now[k] != previous[k]:
                assert (now[k] == x) if x in present else (previous[k] == x), (step, x, k)
    previous = now
ids = [f"n{i}" for i in range(8)]
forward, backward = router_with(ids), router_with(ids[::-1])
assert owners(forward, keys) == owners(backward, keys), "owners depend on the order shards were added"
"""},
        {"name": "Part 4: keys spread about evenly", "part": 4, "visibility": "unshown", "behavior": "routing.normalization",
         "failure_message": "With 20 shards, some shard owned under half or over 1.5x its fair share of 40,000 keys; one hash point per shard spreads too unevenly, so give each shard many points.",
         "code": _ROUTING + r"""
ids = [f"shard-{i}" for i in range(20)]
router = router_with(ids)
count = collections.Counter(router.locate(k) for k in range(40000))
fair = [count[s] * len(ids) / 40000 for s in ids]
assert min(fair) > 0.5 and max(fair) < 1.5, [round(f, 2) for f in fair]
"""},
        {"name": "Part 4: the same owners in a new process", "part": 4, "visibility": "unshown", "behavior": "routing.selection",
         "failure_message": "The owners changed when str hashing was seeded differently, as it is in a new process; use hashlib or zlib.crc32 instead of the built-in hash().",
         "code": _ROUTING + r"""
real_hash = builtins.hash

def reseeded(value):
    if value is None or isinstance(value, (bool, int, float)):
        return real_hash(value)
    return real_hash((value, "another process"))

ids = ["red", "green", "blue", "gold", "plum"]
keys = range(5000)
here = owners(router_with(ids), keys)
builtins.hash = reseeded
try:
    there = owners(router_with(ids), keys)
finally:
    builtins.hash = real_hash
assert here == there, "owners depend on the process's string hash seed"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import bisect
import hashlib
import heapq

VIRTUAL_NODES = 160


def _point(text):
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "big")  # stable across processes


class ShardManager:
    def __init__(self):
        self._ring = []  # sorted (point, shard id): VIRTUAL_NODES points per shard
        self._shards = set()

    def rebalance(self, limit, shards):
        # Up to `limit` lanes, each holding kept shards that never overlap; the heap holds the key
        # from which each lane is free. A shard takes the lane that frees up first.
        free, kept = [], []
        for sid, start, end in sorted(shards, key=lambda s: (s[1], s[2], s[0])):
            lane = heapq.heappop(free) if len(free) == limit else start  # an unused lane is free now
            new_start = max(start, lane)
            if new_start > end:
                heapq.heappush(free, lane)  # hand the lane back unused
                continue
            kept.append([sid, new_start, end])
            heapq.heappush(free, end + 1)

        # Shifts never open a hole, so the only holes are those already in the input.
        kept.sort(key=lambda s: (s[1], s[2], s[0]))
        reach = owner = None
        for shard in kept:
            if owner is not None and shard[1] > reach + 1:
                owner[2] = shard[1] - 1
            if owner is None or shard[2] > reach:  # strict: ties keep the smaller (start, id)
                reach, owner = shard[2], shard
        kept.sort(key=lambda s: (s[1], s[2], s[0]))
        return [tuple(s) for s in kept]

    def add_shard(self, shard_id):
        if shard_id in self._shards:
            raise ValueError(f"shard already present: {shard_id!r}")
        self._shards.add(shard_id)
        points = [(_point(f"{shard_id}#{v}"), shard_id) for v in range(VIRTUAL_NODES)]
        self._ring = sorted(self._ring + points)

    def remove_shard(self, shard_id):
        if shard_id not in self._shards:
            raise ValueError(f"no such shard: {shard_id!r}")
        self._shards.remove(shard_id)
        self._ring = [entry for entry in self._ring if entry[1] != shard_id]

    def locate(self, key):
        if not self._ring:
            raise LookupError("no shards")
        i = bisect.bisect_left(self._ring, (_point(str(key)),))  # the first point clockwise
        return self._ring[i % len(self._ring)][1]                 # wrap past the largest
''',
    "interview_questions": interview(
        concept=[
            "Why is the first key where coverage drops below limit exactly where the earliest-free lane frees up?",
            "Why does a kept shard's range never need checking again after its start is moved?",
        ],
        deep_dive=[
            "How does a heap of at most limit keys replace counting coverage key by key?",
        ],
        tradeoffs=[
            "Can moving starts forward ever open a hole, and where can holes come from?",
            "With one hash point per shard, why are the shares uneven, and how do virtual nodes fix it?",
            "Why must the hash not be Python's built-in hash() for strings?",
            "How does rendezvous hashing compare with a hash ring for locate cost and key movement?",
        ],
    ),
}
