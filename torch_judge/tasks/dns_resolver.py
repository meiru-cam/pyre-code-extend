"""Resolve host names through alias chains, then add a fallback zone, a TTL cache and coalesced concurrent lookups."""

from ._interview import interview

# Record builders, error checks and a plain model of chain resolution with a fallback.
_HELPERS = r"""
import random, threading, time

def A(*addresses, ttl=60):
    return ("A", list(addresses), ttl)

def CNAME(target, ttl=60):
    return ("CNAME", target, ttl)

def raises(kind, call, name=None):
    try:
        call()
    except Exception as e:
        assert type(e).__name__ == kind, f"expected {kind}, got {type(e).__name__}: {e}"
        if name is not None:
            assert getattr(e, "name", None) == name, f"{kind}.name should be {name!r}, got {getattr(e, 'name', None)!r}"
        return e
    raise AssertionError(f"expected {kind}, nothing was raised")

def norm(name):
    return name.lower().rstrip(".") + "."

def model(zone, fallback, limit, name):
    # (addresses, ttl) or (error class name, error name)
    def walk(z):
        cur, seen, chain, ttl = name, {name}, 0, None
        while True:
            if cur not in z:
                return ("NameNotFoundError", cur)
            kind, value, t = z[cur]
            ttl = t if ttl is None else min(ttl, t)
            if kind == "A":
                return (list(value), ttl)
            nxt = norm(value)
            if nxt in seen:
                return ("ResolutionCycleError", nxt)
            seen.add(nxt)
            chain += 1
            if chain > limit:
                return ("ChainTooLongError", name)
            cur = nxt
    first = walk(zone)
    if first[0] == "NameNotFoundError" and fallback is not None:
        return walk(fallback)
    return first

NAMES = [f"n{i}.lab.test." for i in range(7)]

def random_zone(rng, size):
    zone = {}
    for name in rng.sample(NAMES, size):
        if rng.random() < 0.45:
            zone[name] = A(*[f"10.0.{rng.randint(0, 9)}.{k}" for k in range(rng.randint(1, 3))], ttl=rng.randint(1, 50))
        else:
            target = rng.choice(NAMES)
            zone[name] = CNAME(rng.choice([target, target.upper(), target.rstrip("."), target + ".."]), ttl=rng.randint(1, 50))
    return zone

def check(resolver, zone, fallback, limit, name, label):
    want = model(zone, fallback, limit, norm(name))
    if isinstance(want[0], list):
        assert resolver.resolve(name) == want[0], (label, name, want)
    else:
        raises(want[0], lambda: resolver.resolve(name), want[1])
    return want
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
zone = {"web.lab.test.": A("192.0.2.4", ttl=120), "mail.lab.test.": A("192.0.2.8", "192.0.2.9", ttl=120)}
r = {fn}(zone)
assert r.resolve("WEB.Lab.test") == ["192.0.2.4"]
assert r.resolve("mail.lab.test...") == ["192.0.2.8", "192.0.2.9"]
raises("NameNotFoundError", lambda: r.resolve("ftp.lab.test"), "ftp.lab.test.")
"""},
    {"name": "Part 1: normalisation and copies", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Names lower-case and end in exactly one dot before lookup; the error's name attribute is the normalised name; address order is kept and every returned list is a copy that never changes the zone.",
     "code": _HELPERS + r"""
zone = {"a.b.": A("1.1.1.1", "0.0.0.0", ttl=5)}
r = {fn}(zone)
for spelling in ["a.b", "A.B.", "a.B....", "A.b"]:
    assert r.resolve(spelling) == ["1.1.1.1", "0.0.0.0"], spelling
got = r.resolve("a.b")
got.append("junk")
assert r.resolve("a.b") == ["1.1.1.1", "0.0.0.0"] and zone["a.b."][1] == ["1.1.1.1", "0.0.0.0"]
fresh = {fn}(zone)
got = fresh.resolve("A.B")
got.append("junk")
assert zone["a.b."][1] == ["1.1.1.1", "0.0.0.0"], "the first answer must not be the zone's own list"
assert fresh.resolve("a.b") == ["1.1.1.1", "0.0.0.0"]
raises("NameNotFoundError", lambda: r.resolve("A.B.C"), "a.b.c.")
raises("NameNotFoundError", lambda: {fn}({}).resolve("x"), "x.")
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
zone = {"web.lab.test.": A("192.0.2.4", ttl=120),
        "www.lab.test.": CNAME("Front.lab.test", ttl=300),
        "front.lab.test.": CNAME("web.lab.test.", ttl=45)}
assert {fn}(zone).resolve("www.lab.test") == ["192.0.2.4"]
loop = {"x.lab.test.": CNAME("y.lab.test."), "y.lab.test.": CNAME("z.lab.test."), "z.lab.test.": CNAME("x.lab.test.")}
raises("ResolutionCycleError", lambda: {fn}(loop).resolve("x.lab.test."), "x.lab.test.")
raises("ChainTooLongError", lambda: {fn}(zone, max_chain_length=1).resolve("www.lab.test."), "www.lab.test.")
"""},
    {"name": "Part 2: chain edges", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "A chain of exactly max_chain_length CNAMEs resolves and one more raises ChainTooLongError naming the query; a cycle is reported (naming the repeated name) even when it would also exceed the limit; a self-loop is a cycle; a missing target raises NameNotFoundError naming that target.",
     "code": _HELPERS + r"""
chain = {f"c{i}.": CNAME(f"c{i + 1}.") for i in range(4)}
chain["c4."] = A("10.9.9.9")
assert {fn}(chain, max_chain_length=4).resolve("c0") == ["10.9.9.9"]
raises("ChainTooLongError", lambda: {fn}(chain, max_chain_length=3).resolve("c0"), "c0.")
assert {fn}(chain, max_chain_length=0).resolve("c4") == ["10.9.9.9"]
raises("ChainTooLongError", lambda: {fn}(chain, max_chain_length=0).resolve("c3"), "c3.")
cyc = {"p.": CNAME("q."), "q.": CNAME("r."), "r.": CNAME("q.")}
raises("ResolutionCycleError", lambda: {fn}(cyc, max_chain_length=2).resolve("p"), "q.")
raises("ResolutionCycleError", lambda: {fn}({"s.": CNAME("S..")}).resolve("s"), "s.")
raises("NameNotFoundError", lambda: {fn}({"m.": CNAME("Gone")}).resolve("m"), "gone.")
"""},
    {"name": "Part 2: random zones", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random zones with CNAME chains, the result or the raised error (type and name) differed from following the chain with a visited set and a length limit.",
     "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    zone, limit = random_zone(rng, rng.randint(1, 7)), rng.randint(0, 4)
    r = {fn}(zone, max_chain_length=limit)
    for name in NAMES:
        check(r, zone, None, limit, name, seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": _HELPERS + r"""
primary = {"img.lab.test.": CNAME("cdn.lab.test.")}
fallback = {"img.lab.test.": CNAME("edge.lab.test.", ttl=30), "edge.lab.test.": A("10.0.0.7", ttl=30),
            "cdn.lab.test.": A("10.9.9.9", ttl=30)}
assert {fn}(primary, fallback).resolve("img.lab.test") == ["10.0.0.7"]
raises("NameNotFoundError", lambda: {fn}({"m.lab.test.": CNAME("n.lab.test.")}, {}).resolve("m.lab.test"), "m.lab.test.")
long_chain = {"a.lab.test.": CNAME("b.lab.test."), "b.lab.test.": CNAME("c.lab.test."), "c.lab.test.": A("10.0.0.1")}
raises("ChainTooLongError", lambda: {fn}(long_chain, {"a.lab.test.": A("10.0.0.2")}, 1).resolve("a.lab.test"), "a.lab.test.")
"""},
    {"name": "Part 3: fallback rules", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Only NameNotFoundError falls through; the fallback starts fresh from the query name, never from the missing target; its own NameNotFoundError names its own missing name; a too-long chain in the primary never falls back.",
     "code": _HELPERS + r"""
primary = {"q.": CNAME("mid.")}
fallback = {"mid.": A("10.1.1.1"), "q.": CNAME("other.")}
raises("NameNotFoundError", lambda: {fn}(primary, fallback).resolve("q"), "other.")
assert {fn}({}, {"q.": A("10.2.2.2")}).resolve("Q") == ["10.2.2.2"]
long = {"a.": CNAME("b."), "b.": CNAME("c."), "c.": A("10.3.3.3")}
raises("ChainTooLongError", lambda: {fn}(long, {"a.": A("10.4.4.4")}, max_chain_length=1).resolve("a"), "a.")
assert {fn}({"a.": CNAME("x.")}, {"a.": CNAME("b."), "b.": A("10.5.5.5")}, max_chain_length=1).resolve("a") == ["10.5.5.5"]
"""},
    {"name": "Part 3: random zones with a fallback", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random primary and fallback zones, the result or the raised error differed from resolving in the primary and, only on NameNotFoundError, fresh in the fallback.",
     "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(1000 + seed)
    zone, fallback, limit = random_zone(rng, rng.randint(0, 5)), random_zone(rng, rng.randint(0, 5)), rng.randint(0, 3)
    r = {fn}(zone, fallback, max_chain_length=limit)
    for name in NAMES:
        check(r, zone, fallback, limit, name, seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": _HELPERS + r"""
zone = {"web.lab.test.": A("192.0.2.4", ttl=120),
        "www.lab.test.": CNAME("Front.lab.test", ttl=300),
        "front.lab.test.": CNAME("web.lab.test.", ttl=45)}
r = {fn}(zone)
assert r.resolve("www.lab.test") == ["192.0.2.4"]
zone["web.lab.test."] = A("192.0.2.99", ttl=120)
r.advance_clock(44)
assert r.resolve("www.lab.test") == ["192.0.2.4"], "cached until 0 + min(300, 45, 120) = 45"
assert r.resolve("web.lab.test") == ["192.0.2.99"], "only the query name is cached"
r.advance_clock(1)
assert r.resolve("www.lab.test") == ["192.0.2.99"], "at 45 the entry has expired"
"""},
    {"name": "Part 4: what is cached", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Errors are never cached; the expiry uses the smallest ttl of the records read in the attempt that succeeded, counted from the clock when stored; a hit returns a copy; the cache key is the normalised name.",
     "code": _HELPERS + r"""
zone = {}
r = {fn}(zone)
raises("NameNotFoundError", lambda: r.resolve("late"))
zone["late."] = A("10.0.0.1", ttl=10)
assert r.resolve("late") == ["10.0.0.1"], "a failure must not be cached"
r.advance_clock(5)
zone["late."] = A("10.0.0.2", ttl=10)
assert r.resolve("LATE..") == ["10.0.0.1"], "same name after normalisation"
got = r.resolve("late")
got.clear()
assert r.resolve("late") == ["10.0.0.1"]
r.advance_clock(5)
assert r.resolve("late") == ["10.0.0.2"]
zone["late."] = A("10.0.0.5", ttl=10)
r.advance_clock(9)
assert r.resolve("late") == ["10.0.0.2"], "stored at 10 with ttl 10, so valid until 20"
primary = {"f.": CNAME("missing.", ttl=1)}
fallback = {"f.": A("10.0.0.3", ttl=30)}
r2 = {fn}(primary, fallback)
assert r2.resolve("f") == ["10.0.0.3"]
fallback["f."] = A("10.0.0.4", ttl=30)
r2.advance_clock(29)
assert r2.resolve("f") == ["10.0.0.3"], "the failed primary attempt's ttl does not count"
"""},
    {"name": "Part 4: random zones with a clock", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "With the zone changing between calls and the clock moving, a result differed from a model that caches each query name until stored time plus the smallest ttl read.",
     "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(2000 + seed)
    zone, limit = random_zone(rng, 5), 3
    r, clock, cache = {fn}(zone, max_chain_length=limit), 0, {}
    for step in range(40):
        if rng.random() < 0.3:
            zone.update(random_zone(rng, 2))
        if rng.random() < 0.4:
            dt = rng.randint(0, 20)
            r.advance_clock(dt)
            clock += dt
        name = rng.choice(NAMES)
        key = norm(name)
        if key in cache and clock < cache[key][1]:
            assert r.resolve(name) == cache[key][0], (seed, step)
            continue
        want = check(r, zone, None, limit, name, (seed, step))
        if isinstance(want[0], list):
            cache[key] = (want[0], clock + want[1])
"""},
    {"name": "Part 5: concurrent callers share one resolution", "part": 5, "behavior": "scheduler.concurrency", "code": _HELPERS + r"""
class SlowZone(dict):
    def __init__(self, items, slow):
        super().__init__(items)
        self.slow, self.hits, self.lock = slow, {}, threading.Lock()
    def _touch(self, key):
        with self.lock:
            self.hits[key] = self.hits.get(key, 0) + 1
        if key in self.slow:
            time.sleep(0.3)
    def __getitem__(self, key):
        self._touch(key)
        return super().__getitem__(key)
    def __contains__(self, key):
        self._touch(key)
        return super().__contains__(key)
    def get(self, key, default=None):
        self._touch(key)
        return super().get(key, default)
zone = SlowZone({"feed.lab.test.": CNAME("cache.lab.test."), "img.lab.test.": CNAME("cache.lab.test."),
                 "cache.lab.test.": A("10.1.1.1"), "loop.lab.test.": CNAME("loop.lab.test.")},
                slow={"feed.lab.test.", "img.lab.test.", "loop.lab.test."})
r = {fn}(zone)
def run(names):
    results, gate = [], threading.Barrier(len(names))
    def call(name):
        gate.wait()
        try:
            results.append((name, r.resolve(name)))
        except Exception as e:
            results.append((name, type(e).__name__))
    threads = [threading.Thread(target=call, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return sorted(results)
got = run(["feed.lab.test"] * 6 + ["img.lab.test"])
assert got == sorted([("feed.lab.test", ["10.1.1.1"])] * 6 + [("img.lab.test", ["10.1.1.1"])]), got
assert zone.hits["feed.lab.test."] <= 3, f"feed.lab.test. was looked up {zone.hits['feed.lab.test.']} times: the six callers resolved it separately"
got = run(["loop.lab.test"] * 4)
assert got == [("loop.lab.test", "ResolutionCycleError")] * 4, got
assert zone.hits["loop.lab.test."] <= 4, f"loop.lab.test. was looked up {zone.hits['loop.lab.test.']} times: the four callers resolved it separately"
"""},
    {"name": "Part 5: errors, copies and parallel names", "part": 5, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "Callers waiting on a shared resolution must all get the same error when it fails, each caller gets its own list, a later call after a failure resolves again, and resolutions of different names must run at the same time rather than one after another.",
     "code": _HELPERS + r"""
class SlowZone(dict):
    def __init__(self, items, delay):
        super().__init__(items)
        self.delay, self.hits, self.lock = delay, 0, threading.Lock()
    def _touch(self, key):
        with self.lock:
            self.hits += 1
        time.sleep(self.delay)
    def __getitem__(self, key):
        self._touch(key)
        return super().__getitem__(key)
    def __contains__(self, key):
        self._touch(key)
        return super().__contains__(key)
    def get(self, key, default=None):
        self._touch(key)
        return super().get(key, default)
def run_all(r, names):
    out, gate = {}, threading.Barrier(len(names))
    def call(i, name):
        gate.wait()
        try:
            out[i] = r.resolve(name)
        except Exception as e:
            out[i] = type(e).__name__
    threads = [threading.Thread(target=call, args=(i, n)) for i, n in enumerate(names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return out
broken = SlowZone({"a.": CNAME("b."), "b.": CNAME("a.")}, 0.1)
r = {fn}(broken)
out = run_all(r, ["a"] * 6)
assert list(out.values()) == ["ResolutionCycleError"] * 6, out
first = broken.hits
assert first <= 8, f"{first} lookups for six callers: they did not share one resolution"
raises("ResolutionCycleError", lambda: r.resolve("a"))
assert broken.hits > first, "a failure is not cached, so a later call resolves again"
good = {fn}(SlowZone({"x.": A("10.7.7.7")}, 0.1))
lists = run_all(good, ["x"] * 4)
for i in range(4):
    lists[i].append(i)
assert all(lists[i] == ["10.7.7.7", i] for i in range(4)), "each caller must get its own list"
assert good.resolve("x") == ["10.7.7.7"], "a caller's list must not be the cached one"
wide = SlowZone({f"h{i}.": A(f"10.8.0.{i}") for i in range(6)}, 0.25)
r3 = {fn}(wide)
start = time.perf_counter()
out = run_all(r3, [f"h{i}" for i in range(6)])
elapsed = time.perf_counter() - start
assert [out[i] for i in range(6)] == [[f"10.8.0.{i}"] for i in range(6)]
assert elapsed < 1.2, f"six different names took {elapsed:.2f}s: resolutions of different names must overlap"
"""},
]

TASK = {
    "title": "DNS Resolver With Cache",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "DnsResolver",
    "description_en": r"""Build `DnsResolver`, which turns a host name into IPv4 addresses by following alias records, then adds a fallback zone, a TTL cache and safe concurrent use.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `DnsResolver` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A zone is a `dict` from a name to one record: `("A", [address, ...], ttl)` or `("CNAME", target, ttl)`. `ttl` is a positive integer number of seconds.
- Zone keys are already normalised. To normalise any other name, lower-case it and end it with exactly one `.`, so `"Web.Lab.test.."` becomes `"web.lab.test."`.
- Read the zone dicts when `resolve` runs. Do not copy them in `__init__`: callers may change a zone between calls.
- Errors are your own classes, recognised by class name, each with an attribute `name`. `resolve` returns a new list each time, with addresses in zone order.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it starts as a dict lookup and turns into a chain walk with cycle detection, a cache with expiry, and request coalescing under threads. Each later part adds one requirement.

**Where it is used:** stub resolvers and DNS caches, service discovery with aliases, and any cache that must stop a burst of identical misses from hitting the backend at once.

Adapted from the DNS resolver online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class renamed `DnsResolver`, with no required common base class for the errors. Records are tuples instead of record classes, zone keys arrive normalised, and zones are read at resolve time so updates and coalescing can be observed.""",
    "parts": [
        {
            "title": "Normalise and look up",
            "description_en": r"""**Signature:** `DnsResolver(zone, fallback_zone=None, max_chain_length=8)`, `resolve(name) -> list[str]`

- In this part every record is an `A` record. Normalise `name`, look it up, and return its addresses.
- If the name is not in the zone, raise `NameNotFoundError` with `name` set to the normalised name.
- The other two arguments matter from later parts on.

**Example:** `zone = {"web.lab.test.": ("A", ["192.0.2.4"], 120), "mail.lab.test.": ("A", ["192.0.2.8", "192.0.2.9"], 120)}`:
- `resolve("WEB.Lab.test")` is `["192.0.2.4"]`
- `resolve("mail.lab.test...")` is `["192.0.2.8", "192.0.2.9"]`
- `resolve("ftp.lab.test")` raises `NameNotFoundError` with `name == "ftp.lab.test."`""",
        },
        {
            "title": "Alias chains",
            "description_en": r"""Keep Part 1. Records may now be `CNAME` records.

- From the normalised query name, follow `CNAME` records, normalising each target, until an `A` record. The chain length is the number of `CNAME` records followed.
- Keep the set of names visited, starting with the query. If a target is already in it, raise `ResolutionCycleError` with `name` set to that target. Check this before the length.
- Count each `CNAME` as you follow it. If the count passes `max_chain_length`, raise `ChainTooLongError` with `name` set to the normalised query name, before looking up the target.
- A missing name anywhere on the chain raises `NameNotFoundError` with that name.

**Example:** add `"www.lab.test.": ("CNAME", "Front.lab.test", 300)` and `"front.lab.test.": ("CNAME", "web.lab.test.", 45)`:
- `resolve("www.lab.test")` follows two aliases and returns `["192.0.2.4"]`
- with `max_chain_length=1` it raises `ChainTooLongError` naming `"www.lab.test."`
- in a zone where `x` points to `y`, `y` to `z` and `z` to `x`, resolving `x` raises `ResolutionCycleError` naming `"x.lab.test."`""",
        },
        {
            "title": "Fallback zone",
            "description_en": r"""Keep Parts 1–2 and use `fallback_zone`.

- First resolve in `zone`. If that raises `NameNotFoundError` and a fallback zone was given, resolve the normalised query name again from the start in `fallback_zone` alone, with the same length limit, and return its result or raise its error.
- `ResolutionCycleError` and `ChainTooLongError` from `zone` are raised at once; the fallback is not tried.

**Example:**
- `zone = {"img.lab.test.": ("CNAME", "cdn.lab.test.", 60)}` and a fallback holding `"img.lab.test.": ("CNAME", "edge.lab.test.", 30)`, `"edge.lab.test.": ("A", ["10.0.0.7"], 30)` and `"cdn.lab.test.": ("A", ["10.9.9.9"], 30)`: `resolve("img.lab.test")` is `["10.0.0.7"]`, because the fallback starts again from `img.lab.test.` instead of picking up at `cdn.lab.test.`
- if neither zone has the query, the fallback's error is raised: `zone = {"m.lab.test.": ("CNAME", "n.lab.test.", 60)}` with an empty fallback raises `NameNotFoundError` naming `"m.lab.test."`, not `"n.lab.test."`
- a chain in `zone` longer than `max_chain_length` raises `ChainTooLongError` even if the fallback could answer the query""",
        },
        {
            "title": "TTL cache",
            "description_en": r"""Keep Parts 1–3 and cache answers on a manual clock.

**Signature:** `advance_clock(seconds)` adds a non-negative integer to a clock that starts at `0`.

- After a successful resolution, cache the addresses under the normalised query name, with expiry equal to the clock now plus the smallest `ttl` of every record read in the attempt that succeeded.
- While the clock is before the expiry, `resolve` for that name returns the cached addresses without reading any zone. At the expiry or later, it resolves again and replaces the entry.
- Only the query name is cached, never the names passed through on the way. Errors are never cached.

**Example**, the zone from Part 2:
- `resolve("www.lab.test")` caches `["192.0.2.4"]` until `0 + min(300, 45, 120) = 45`
- the zone's `web.lab.test.` then changes to `["192.0.2.99"]`, and the clock advances to `44`: `resolve("www.lab.test")` still returns `["192.0.2.4"]`, while `resolve("web.lab.test")` returns `["192.0.2.99"]`
- at `45`, `resolve("www.lab.test")` returns `["192.0.2.99"]`""",
        },
        {
            "title": "Concurrent callers",
            "description_en": r"""Keep Parts 1–4. Many threads now call one resolver at once.

- Each call must behave as in Part 4.
- While a resolution for a normalised name is under way, other calls for the same name wait for it and return its result, or raise its error, instead of resolving again.
- Calls for different names never wait for each other's resolution, even when their chains share records.
- The cache, the clock and any bookkeeping are safe to use from many threads without locks of the caller's own.

**Example:** `feed.lab.test.` and `img.lab.test.` are both aliases of `cache.lab.test.`, and `loop.lab.test.` is an alias of itself:
- six threads call `resolve("feed.lab.test")` at once while a seventh calls `resolve("img.lab.test")`: all seven get `["10.1.1.1"]`, and `feed.lab.test.` is read by one resolution only
- four threads call `resolve("loop.lab.test")` at once: all four raise `ResolutionCycleError`, again from one resolution""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which two edits turn \"Web.Lab.test..\" into \"web.lab.test.\"? Once the name is normalised, what single dict operation finds its record, and what should happen when the name is not there?"},
        {"level": 2, "kind": "analysis", "content": "def normalise(n): return n.lower().rstrip('.') + '.'. In resolve: key = normalise(name); if key not in self._zone: raise NameNotFoundError(key); kind, addresses, ttl = self._zone[key]; return list(addresses). Give the error class an __init__ that stores name."},
    ],
    "model_connections": [
        "Inference gateways coalesce identical in-flight requests the same way, so a burst of equal prompts or embedding lookups costs one backend call.",
        "Model and feature registries resolve aliases such as latest or prod to a concrete version through a chain, with the same cycle and depth checks.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A visited set catches any alias cycle on its first repeat.",
            "Caching with the smallest TTL on the chain never serves an answer longer than any record allows.",
            "Per-name in-flight entries collapse a burst of identical misses into one resolution without serialising different names.",
        ],
        "cons": [
            "Lazy expiry keeps dead entries in memory until their name is queried again.",
            "Not caching errors lets a burst of lookups for a missing name hit the zone every time; real resolvers add negative caching.",
            "Waiting callers inherit the leader's error, so one slow or failing lookup fails everyone waiting on it.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import threading


class DNSError(Exception):
    def __init__(self, name):
        super().__init__(name)
        self.name = name


class NameNotFoundError(DNSError):
    pass


class ResolutionCycleError(DNSError):
    pass


class ChainTooLongError(DNSError):
    pass


def _normalise(name):
    return name.lower().rstrip(".") + "."  # ASCII case-fold, exactly one trailing dot


class _Flight:
    def __init__(self):
        self.done = threading.Event()
        self.result = None
        self.error = None


class DnsResolver:
    def __init__(self, zone, fallback_zone=None, max_chain_length=8):
        self._zone = zone  # read at resolve time, so later zone updates are seen on a miss
        self._fallback = fallback_zone
        self._max = max_chain_length
        self._clock = 0
        self._cache = {}    # query name -> (addresses, expiry)
        self._flights = {}  # query name -> _Flight for the resolution under way
        self._lock = threading.Lock()

    def _walk(self, zone, start):
        """Resolve start inside one zone: (addresses, smallest ttl read)."""
        name, seen, chain, ttl = start, {start}, 0, None
        while True:
            if name not in zone:
                raise NameNotFoundError(name)
            kind, value, record_ttl = zone[name]
            ttl = record_ttl if ttl is None else min(ttl, record_ttl)
            if kind == "A":
                return list(value), ttl
            target = _normalise(value)
            if target in seen:
                raise ResolutionCycleError(target)  # a cycle wins over the length limit
            seen.add(target)
            chain += 1
            if chain > self._max:
                raise ChainTooLongError(start)
            name = target

    def _lookup(self, name):
        try:
            return self._walk(self._zone, name)
        except NameNotFoundError:
            if self._fallback is None:
                raise
            return self._walk(self._fallback, name)  # a fresh start, never resumed mid-chain

    def advance_clock(self, seconds):
        with self._lock:
            self._clock += seconds

    def resolve(self, name):
        name = _normalise(name)
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None and self._clock < hit[1]:
                return list(hit[0])
            flight = self._flights.get(name)
            leader = flight is None
            if leader:
                flight = self._flights[name] = _Flight()
        if not leader:
            flight.done.wait()  # join the resolution already under way for this name
            if flight.error is not None:
                raise flight.error
            return list(flight.result)
        try:
            addresses, ttl = self._lookup(name)
            flight.result = addresses
            with self._lock:
                self._cache[name] = (addresses, self._clock + ttl)
            return list(addresses)
        except Exception as error:
            flight.error = error  # waiting callers re-raise it
            raise
        finally:
            with self._lock:
                del self._flights[name]
            flight.done.set()
''',
    "interview_questions": interview(
        concept=[
            "Why normalise the query before the lookup instead of storing every spelling of a name in the zone?",
            "Why should a missing name raise an error with the name attached instead of returning an empty list?",
        ],
        deep_dive=[
            "Why must resolve return a new list on every call, and what breaks if it returns the zone's own list?",
        ],
        tradeoffs=[
            "Why must a cycle be checked before the chain length, and what would the caller see otherwise?",
            "Why fall back only on a missing name and not on a cycle or an overlong chain?",
            "Why use the smallest TTL on the chain for the cache entry, and why not cache errors?",
            "How do you stop eight callers from resolving the same name at once without making different names wait for each other?",
        ],
    ),
}
