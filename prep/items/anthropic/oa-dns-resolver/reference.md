Worth confirming up front: whether a failed resolution should itself be cached, for instance with a short negative-TTL (assumed not here — only a successful answer is cached, stated in Level 4); and whether concurrent coalescing should extend to names that merely share part of a chain rather than being the same queried name (assumed not, kept to exactly the queried name, stated in Level 5).

### Level 1

`ARecord` and `CNAMERecord` store their fields as given; `DNSResolutionError` and its Level 1 subclass carry the offending name as `.name`. `_normalize` implements case-folding and the trailing-dot rule in one expression; `_normalized_zone` runs it over every key of an input zone once, at construction, so `resolve` itself only ever has to normalise the one name it is called with.

```python
class DNSResolutionError(Exception):
    """Base class for every error resolve can raise."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class NameNotFoundError(DNSResolutionError):
    """name has no record anywhere it was looked up."""


class ARecord:
    def __init__(self, addresses: list[str], ttl: int) -> None:
        self.addresses = list(addresses)
        self.ttl = ttl


class CNAMERecord:
    def __init__(self, target: str, ttl: int) -> None:
        self.target = target
        self.ttl = ttl


def _normalize(name: str) -> str:
    return name.lower().rstrip(".") + "."   # NOTE: case-fold, then collapse to exactly one trailing dot


def _normalized_zone(zone):
    out = {}
    for name, record in zone.items():
        out[_normalize(name)] = record       # NOTE: a later duplicate normalised key wins, like a dict literal
    return out


class Resolver:
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        self._zone = _normalized_zone(zone)
        self._fallback_zone = _normalized_zone(fallback_zone) if fallback_zone is not None else None
        self.max_chain_length = max_chain_length

    def resolve(self, name: str) -> list[str]:
        current = _normalize(name)           # NOTE: normalise before ever touching the zone (or, later, the cache)
        record = self._zone.get(current)
        if record is None:
            raise NameNotFoundError(current)
        return list(record.addresses)        # NOTE: a fresh copy -- the caller must not be able to mutate the zone
```

Every Level 1 operation is $O(1)$: one normalisation and one dict lookup.

### Level 2

`_follow_chain` implements the whole chain-following rule once, over an explicit `zone` argument rather than always `self._zone`, so that Level 3 can reuse it against `self._fallback_zone` too without duplicating it. It keeps `visited`, a set seeded with the origin, and a running minimum `ttl` — computed here even though nothing reads it before Level 4, because computing it later would mean walking the same chain a second time. The cycle check comes before the length check on every step, exactly as the Problem section orders them, and `origin` is kept in its own variable so that `ChainTooLongError` can still name the original query after `current` has moved on.

```python
class ResolutionCycleError(DNSResolutionError):
    """a step's target is a name already visited earlier in this resolution."""


class ChainTooLongError(DNSResolutionError):
    """reaching an A record needed more than max_chain_length CNAME hops."""


class Resolver(Resolver):
    def _follow_chain(self, zone: dict, name: str) -> tuple[list[str], int]:
        origin = _normalize(name)
        current = origin
        visited = {current}
        ttl = None
        hops = 0
        while True:
            record = zone.get(current)
            if record is None:
                raise NameNotFoundError(current)
            ttl = record.ttl if ttl is None else min(ttl, record.ttl)   # NOTE: running min ttl, needed at Level 4
            if isinstance(record, ARecord):
                return list(record.addresses), ttl
            target = _normalize(record.target)
            if target in visited:                # NOTE: cycle is checked before the length cap, on every step
                raise ResolutionCycleError(target)
            hops += 1
            if hops > self.max_chain_length:
                raise ChainTooLongError(origin)   # NOTE: names the query, not the hop where the cap was hit
            visited.add(target)
            current = target

    def resolve(self, name: str) -> list[str]:
        addresses, _ttl = self._follow_chain(self._zone, name)
        return addresses
```

`resolve` is now $O(k)$ in the chain length $k \le$ `max_chain_length`: one dict lookup, one `ttl` comparison and one set membership check per hop.

### Level 3

`resolve` is reopened again to try `_follow_chain` against `self._zone` and catch only `NameNotFoundError`; `ResolutionCycleError` and `ChainTooLongError` are not named in the `except` clause, so Python re-raises them on its own without a fallback ever being consulted. The fallback attempt calls `_follow_chain(self._fallback_zone, name)` with the original `name`, not `current` (which this method no longer even has in scope) — it is a fresh chain from the top, never a resumption.

```python
class Resolver(Resolver):
    def resolve(self, name: str) -> list[str]:
        try:
            addresses, _ttl = self._follow_chain(self._zone, name)
            return addresses
        except NameNotFoundError:
            if self._fallback_zone is None:
                raise
            # NOTE: ResolutionCycleError and ChainTooLongError from the try block above are not caught
            # here -- they already propagated out of resolve before this except clause could run.
            addresses, _ttl = self._follow_chain(self._fallback_zone, name)   # NOTE: name, not current: fresh chain
            return addresses
```

Still $O(k)$: `_follow_chain` runs at most twice, once per zone.

### Level 4

Level 3's fallback logic moves out of `resolve` and into a new helper, `_resolve_uncached`, unchanged except for its name and its return type: it now hands back the `ttl` `_follow_chain` was already computing, alongside the addresses, instead of discarding it. `resolve` itself becomes a thin cache check wrapped around that helper. The comparison `self._now < entry.expires_at`, not `<=`, matches the Problem section's example: an entry stored with total `ttl` $t$ at clock reading $c$ is alive up to, but not including, clock reading $c + t$.

```python
class _CacheEntry:
    __slots__ = ("addresses", "expires_at")

    def __init__(self, addresses: list[str], expires_at: int) -> None:
        self.addresses = addresses
        self.expires_at = expires_at


class Resolver(Resolver):
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        super().__init__(zone, fallback_zone, max_chain_length=max_chain_length)
        self._now = 0
        self._cache: dict[str, _CacheEntry] = {}

    def advance_clock(self, seconds: int) -> None:
        self._now += seconds

    def _resolve_uncached(self, name: str) -> tuple[list[str], int]:
        try:
            return self._follow_chain(self._zone, name)
        except NameNotFoundError:
            if self._fallback_zone is None:
                raise
            return self._follow_chain(self._fallback_zone, name)

    def resolve(self, name: str) -> list[str]:
        key = _normalize(name)                                     # NOTE: cache is keyed by the normalised name
        entry = self._cache.get(key)
        if entry is not None and self._now < entry.expires_at:     # NOTE: lazy expiry -- only checked on a read
            return list(entry.addresses)
        addresses, ttl = self._resolve_uncached(key)
        self._cache[key] = _CacheEntry(list(addresses), self._now + ttl)
        return list(addresses)
```

A cache hit is $O(1)$; a miss costs Level 3's $O(k)$ plus one dict write, and nothing ever scans the whole cache, so an expired entry that is never queried again simply sits there harmlessly.

### Level 5

Everything mutable that more than one thread can touch — `_cache`, `_in_flight`, `_now` — is only ever read or written inside `with self._lock:`; `_zone`, `_fallback_zone` and `max_chain_length` need no lock, because `__init__` is the only place that ever sets them. The cache check and the leader/follower decision happen inside one `with` block, so no two calls for the same key can both see a cache miss and both create a `Future`: whichever one runs that block first leaves a `Future` behind for every later call to find. `_resolve_uncached` — like `future.result()` on a follower's side — runs with the lock released, so one name's resolution never blocks another's, and a follower waiting on a slow lookup never holds up `advance_clock` or an unrelated `resolve` call meanwhile.

```python
import threading
from concurrent.futures import Future


class Resolver(Resolver):
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        super().__init__(zone, fallback_zone, max_chain_length=max_chain_length)
        self._lock = threading.Lock()
        self._in_flight: dict[str, Future] = {}

    def advance_clock(self, seconds: int) -> None:
        with self._lock:
            self._now += seconds

    def resolve(self, name: str) -> list[str]:
        key = _normalize(name)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and self._now < entry.expires_at:
                return list(entry.addresses)
            future = self._in_flight.get(key)
            if future is None:
                future = self._in_flight[key] = Future()
                leader = True
            else:
                leader = False
        if not leader:
            return list(future.result())    # NOTE: re-raises the leader's own exception here too, if it failed
        try:
            addresses, ttl = self._resolve_uncached(key)
        except Exception as exc:
            with self._lock:
                del self._in_flight[key]     # NOTE: not cached (Level 4) -- the next caller starts a fresh attempt
            future.set_exception(exc)
            raise
        with self._lock:
            self._cache[key] = _CacheEntry(list(addresses), self._now + ttl)
            del self._in_flight[key]
        future.set_result(list(addresses))   # NOTE: wakes every follower parked in future.result() above
        return list(addresses)
```

On failure, the leader deletes its own `_in_flight` entry *before* calling `future.set_exception`, so a call arriving just afterwards sees neither a cache entry nor an in-flight one and starts a fresh attempt of its own, exactly as Level 4 already required for an uncached failure — and `set_exception` then wakes every follower that did arrive in time, with `future.result()` re-raising the same exception object in each of them. `resolve` costs $O(1)$ of locking plus, for whichever call ends up the leader, Level 4's own cost; every other concurrent call for the same name pays only the cost of waiting on a `Future`.

### Follow-ups

- Negative caching — storing "no such name" itself, usually under a short TTL of its own — would need `resolve` to catch `NameNotFoundError` before it propagates and cache a marker that a later call re-raises, plus its own eviction so a name that starts existing afterwards is not shadowed forever.
- Extending coalescing to share work across two different queried names that pass through the same `CNAMERecord` would need a lookup keyed by every name touched, not only the one first asked about, and a rule for what happens when their fallback outcomes differ.
- A `threading.Condition` guarding a plain dict of "done" flags would work as well as `Future` for coalescing, at the cost of writing the wait/notify bookkeeping by hand instead of reusing `Future`'s own thread-safe result/exception plumbing.
- More than two zones — a chain of fallbacks tried in order — generalises Level 3's `except NameNotFoundError: ...` into a loop over the zones, stopping at the first success or the last zone's own error.
- A zone large enough that the cache must not grow without bound would need eviction, such as LRU, on top of — not instead of — the lazy, read-time expiry Level 4 already performs.

```python
def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__} from a call with args {args!r}")


# ---- Level 1 example, replayed exactly ----
zone = {
    "api.acme.com.": ARecord(["203.0.113.10"], ttl=60),
    "db.acme.com.": ARecord(["203.0.113.20", "203.0.113.21"], ttl=60),
}
r = Resolver(zone)
assert r.resolve("API.acme.com") == ["203.0.113.10"]
assert r.resolve("db.acme.com.") == ["203.0.113.20", "203.0.113.21"]
assert expect(NameNotFoundError, r.resolve, "nope.acme.com.").name == "nope.acme.com."
print("Level 1 example replayed")

# ---- Level 2 example, replayed exactly ----
zone["www.acme.com."] = CNAMERecord("edge.acme.com.", ttl=60)
zone["edge.acme.com."] = CNAMERecord("api.acme.com.", ttl=30)
r = Resolver(zone)
assert r.resolve("www.acme.com.") == ["203.0.113.10"]

cycle_zone = {"a.acme.com.": CNAMERecord("b.acme.com.", ttl=60),
              "b.acme.com.": CNAMERecord("a.acme.com.", ttl=60)}
assert expect(ResolutionCycleError, Resolver(cycle_zone).resolve, "a.acme.com.").name == "a.acme.com."

long_zone = {
    "start.acme.com.": CNAMERecord("mid1.acme.com.", ttl=60),
    "mid1.acme.com.": CNAMERecord("mid2.acme.com.", ttl=60),
    "mid2.acme.com.": CNAMERecord("mid3.acme.com.", ttl=60),
    "mid3.acme.com.": ARecord(["203.0.113.99"], ttl=60),
}
assert expect(ChainTooLongError, Resolver(long_zone, max_chain_length=2).resolve,
              "start.acme.com.").name == "start.acme.com."
assert Resolver(long_zone, max_chain_length=3).resolve("start.acme.com.") == ["203.0.113.99"]
print("Level 2 example replayed")

# ---- Level 3 example, replayed exactly ----
primary = {"svc.acme.com.": CNAMERecord("internal.acme.com.", ttl=60)}
fallback = {"svc.acme.com.": ARecord(["198.51.100.5"], ttl=30)}
assert Resolver(primary, fallback).resolve("svc.acme.com.") == ["198.51.100.5"]

looping = {"loop.acme.com.": CNAMERecord("loop.acme.com.", ttl=60)}
has_answer = {"loop.acme.com.": ARecord(["198.51.100.9"], ttl=60)}
assert expect(ResolutionCycleError, Resolver(looping, has_answer).resolve,
              "loop.acme.com.").name == "loop.acme.com."

assert expect(NameNotFoundError, Resolver({}, {}).resolve, "nowhere.acme.com.").name == "nowhere.acme.com."

# a differing fallback chain names ITS OWN missing target, not the primary's
primary3 = {"x.acme.com.": CNAMERecord("y.acme.com.", ttl=10)}
fallback3 = {"x.acme.com.": CNAMERecord("z.acme.com.", ttl=10)}
assert expect(NameNotFoundError, Resolver(primary3, fallback3).resolve, "x.acme.com.").name == "z.acme.com."
print("Level 3 example replayed")

# ---- Level 4 example, replayed exactly, with the zone instrumented to prove caching actually happens ----
class _CountingZone(dict):
    """A dict that counts every .get() call, standing in for Resolver's own _zone after construction,
    to prove whether a resolve() call touched the zone at all."""

    def __init__(self, data):
        super().__init__(data)
        self.reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)


r4 = Resolver(zone)
assert r4.resolve("www.acme.com.") == ["203.0.113.10"]     # cached with expiry 0 + min(60, 30, 60) = 30
r4._zone = _CountingZone(r4._zone)
r4.advance_clock(29)
assert r4.resolve("www.acme.com.") == ["203.0.113.10"]
assert r4._zone.reads == 0, "a live cache entry must not touch the zone at all"
r4.advance_clock(1)
assert r4.resolve("www.acme.com.") == ["203.0.113.10"]
assert r4._zone.reads > 0, "an expired entry must be resolved again, not served stale"
print("Level 4 example replayed")

# ---- edge cases the examples do not reach ----
# a failed resolution is never cached: each retry re-touches the zone
rb = Resolver({"only.acme.com.": ARecord(["203.0.113.1"], ttl=5)})
rb._zone = _CountingZone(rb._zone)
expect(NameNotFoundError, rb.resolve, "missing.acme.com.")
first_reads = rb._zone.reads
expect(NameNotFoundError, rb.resolve, "missing.acme.com.")
assert rb._zone.reads > first_reads

# a zone key that only differs by case or trailing dots collapses to one entry: the later one wins
dup_zone = {"Dup.acme.com.": ARecord(["203.0.113.200"], ttl=10), "dup.acme.com": ARecord(["203.0.113.201"], ttl=10)}
assert Resolver(dup_zone).resolve("dup.acme.com.") == ["203.0.113.201"]

# a chain of exactly max_chain_length hops is allowed; one more is not
edge_zone = {"h0.t.": CNAMERecord("h1.t.", ttl=9), "h1.t.": ARecord(["203.0.113.7"], ttl=9)}
assert Resolver(edge_zone, max_chain_length=1).resolve("h0.t.") == ["203.0.113.7"]
expect(ChainTooLongError, Resolver(edge_zone, max_chain_length=0).resolve, "h0.t.")
print("edge cases OK")
```

```python
import random
from collections import Counter


def _ref_normalize(name: str) -> str:
    return name.lower().rstrip(".") + "."


class _SlowResolver:
    """Independent reference model, written from the statement alone: no cache, no coalescing, and no
    code shared with Resolver -- it re-walks the chain (and, on failure, the fallback chain) from
    scratch on every call."""

    def __init__(self, zone, fallback_zone=None, max_chain_length=8):
        self.zone = {_ref_normalize(k): v for k, v in zone.items()}
        self.fallback_zone = ({_ref_normalize(k): v for k, v in fallback_zone.items()}
                               if fallback_zone is not None else None)
        self.max_chain_length = max_chain_length

    def _chain(self, zone, name):
        origin = _ref_normalize(name)
        current = origin
        visited = {current}
        ttl = None
        hops = 0
        while True:
            if current not in zone:
                raise NameNotFoundError(current)
            record = zone[current]
            ttl = record.ttl if ttl is None else min(ttl, record.ttl)
            if isinstance(record, ARecord):
                return list(record.addresses), ttl
            target = _ref_normalize(record.target)
            if target in visited:
                raise ResolutionCycleError(target)
            hops += 1
            if hops > self.max_chain_length:
                raise ChainTooLongError(origin)
            visited.add(target)
            current = target

    def resolve(self, name):
        try:
            return self._chain(self.zone, name)
        except NameNotFoundError:
            if self.fallback_zone is None:
                raise
            return self._chain(self.fallback_zone, name)


def _random_zone(rng, n_names=10, prefix="n"):
    names = [f"{prefix}{i}.test." for i in range(n_names)]
    zone = {}
    for nm in names:
        if rng.random() < 0.6:
            zone[nm] = CNAMERecord(rng.choice(names), ttl=rng.randint(1, 50))   # may chain, loop or self-cycle
        else:
            addrs = [f"203.0.113.{rng.randint(1, 254)}" for _ in range(rng.randint(1, 3))]
            zone[nm] = ARecord(addrs, ttl=rng.randint(1, 50))
    if rng.random() < 0.3:
        del zone[rng.choice(names)]        # leave a gap: some name becomes entirely absent
    return zone, names


def _run_trial(seed):
    rng = random.Random(seed)
    prim, prim_names = _random_zone(rng, prefix="p")
    use_fallback = rng.random() < 0.6
    fb, fb_names = _random_zone(rng, n_names=6, prefix="f") if use_fallback else (None, [])
    max_len = rng.choice([1, 2, 3, 5, 8])
    real = Resolver(prim, fb, max_chain_length=max_len)
    slow = _SlowResolver(prim, fb, max_chain_length=max_len)
    candidates = prim_names + fb_names + ["ghost.test."]
    outcomes = Counter()

    for _ in range(15):
        base = rng.choice(candidates)
        spelling = rng.choice([base, base.upper(), base[:-1]])   # exercises case-fold and trailing-dot rules too
        try:
            want_addrs, want_ttl = slow.resolve(spelling)
        except DNSResolutionError as want_exc:
            got_exc = expect(type(want_exc), real.resolve, spelling)
            assert got_exc.name == want_exc.name, (seed, spelling, got_exc.name, want_exc.name)
            outcomes[type(want_exc).__name__] += 1
            continue
        got_addrs = real.resolve(spelling)
        assert got_addrs == want_addrs, (seed, spelling, got_addrs, want_addrs)
        outcomes["ok"] += 1

        # cross-check caching around the independently computed expiry, on a resolver of its own
        fresh = Resolver(prim, fb, max_chain_length=max_len)
        assert fresh.resolve(spelling) == want_addrs
        fresh.advance_clock(want_ttl - 1)
        assert fresh.resolve(spelling) == want_addrs        # one tick before the computed expiry: still cached
        fresh.advance_clock(1)
        assert fresh.resolve(spelling) == want_addrs         # at the expiry: recomputed, same static answer
    return outcomes


totals = Counter()
for seed in range(300):
    totals += _run_trial(seed)
assert min(totals[k] for k in ("ok", "NameNotFoundError", "ResolutionCycleError", "ChainTooLongError")) > 50, totals
print(f"cross-validated 300 random zones against an independent reference resolver: {dict(totals)}")
```

```python
class _GatedResolver(Resolver):
    """Test-only subclass: the upstream resolution of one chosen name blocks on an Event until
    released, so at least one concurrent caller is guaranteed to find it still in flight."""

    def __init__(self, *args, gate_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream_calls = Counter()
        self._count_lock = threading.Lock()
        self._gate_key = _normalize(gate_name) if gate_name is not None else None
        self.entered = threading.Event()
        self._release = threading.Event()

    def _resolve_uncached(self, name):
        with self._count_lock:
            self.upstream_calls[name] += 1
        if name == self._gate_key:
            self.entered.set()
            assert self._release.wait(timeout=5.0), "test never released the gate (deadlock?)"
        return super()._resolve_uncached(name)


def _start_calls(resolver, names):
    """Starts one thread per name, each recording ("ok", addresses) or ("err", exception); returns
    (threads, results) without waiting, so the caller can act while the threads are still running."""
    results = [None] * len(names)

    def call(i, nm):
        try:
            results[i] = ("ok", resolver.resolve(nm))
        except DNSResolutionError as e:
            results[i] = ("err", e)

    threads = [threading.Thread(target=call, args=(i, nm)) for i, nm in enumerate(names)]
    for t in threads:
        t.start()
    return threads, results


def _join_all(threads, timeout=5.0):
    for t in threads:
        t.join(timeout=timeout)
    assert all(not t.is_alive() for t in threads), "a thread never returned (deadlock?)"


# ---- many concurrent callers of the SAME name coalesce into exactly one upstream lookup ----
chain_zone = {"chain.acme.com.": CNAMERecord("mid.acme.com.", ttl=40),
              "mid.acme.com.": ARecord(["203.0.113.55"], ttl=40)}
gated = _GatedResolver(chain_zone, gate_name="chain.acme.com.")
leader_threads, leader_results = _start_calls(gated, ["chain.acme.com."])
assert gated.entered.wait(timeout=5.0), "no call ever reached the gated upstream lookup"
# NOTE: entered is set only from inside _resolve_uncached, which resolve() reaches only after already
# creating this name's in-flight Future under the lock -- so every follower started from here on is
# guaranteed, deterministically and without relying on any sleep, to find that Future already in place.
follower_threads, follower_results = _start_calls(gated, ["CHAIN.acme.com"] * 9)
gated._release.set()
_join_all(leader_threads + follower_threads)

key = _normalize("chain.acme.com.")
assert gated.upstream_calls == Counter({key: 1}), gated.upstream_calls
results = leader_results + follower_results
assert len(results) == 10 and all(kind == "ok" and value == ["203.0.113.55"] for kind, value in results)
for _, value in results:
    value.append("corrupted")                                     # mutate every returned list
assert gated.resolve("chain.acme.com.") == ["203.0.113.55"]        # unaffected -- each was an independent copy
print("single-name coalescing: 10 concurrent callers, 1 upstream lookup")

# ---- the same rule for a name that fails: every waiter gets the identical error, and it is not cached ----
gated_fail = _GatedResolver({"only.acme.com.": ARecord(["203.0.113.1"], ttl=30)}, gate_name="missing.acme.com.")
leader_threads, leader_results = _start_calls(gated_fail, ["missing.acme.com."])
assert gated_fail.entered.wait(timeout=5.0)
follower_threads, follower_results = _start_calls(gated_fail, ["missing.acme.com."] * 6)
gated_fail._release.set()
_join_all(leader_threads + follower_threads)

fail_key = _normalize("missing.acme.com.")
results = leader_results + follower_results
assert gated_fail.upstream_calls[fail_key] == 1
assert len(results) == 7
assert all(kind == "err" and isinstance(exc, NameNotFoundError) and exc.name == fail_key for kind, exc in results)
expect(NameNotFoundError, gated_fail.resolve, "missing.acme.com.")    # not cached: a later call retries
assert gated_fail.upstream_calls[fail_key] == 2
assert fail_key not in gated_fail._cache
print("failure coalescing: every waiter sees the same error, and a failed lookup is never cached")

# ---- two DIFFERENT queried names are never coalesced together, even when they share a CNAME target ----
shared_zone = {"www.acme.com.": CNAMERecord("edge.acme.com.", ttl=40),
               "shop.acme.com.": CNAMERecord("edge.acme.com.", ttl=40),
               "edge.acme.com.": ARecord(["203.0.113.77"], ttl=40)}
threads, results = _start_calls(Resolver(shared_zone), ["www.acme.com.", "shop.acme.com."] * 5)
_join_all(threads)
assert len(results) == 10 and all(kind == "ok" and value == ["203.0.113.77"] for kind, value in results)


# ---- broad randomized stress: many threads, many rounds, checked against the independent reference ----
def _stress(seed, n_threads=20, n_rounds=60):
    rng = random.Random(seed)
    prim, prim_names = _random_zone(rng, n_names=8, prefix="s")
    fb, fb_names = _random_zone(rng, n_names=4, prefix="g")
    max_len = 4
    resolver = Resolver(prim, fb, max_chain_length=max_len)
    slow = _SlowResolver(prim, fb, max_chain_length=max_len)
    candidates = prim_names + fb_names + ["ghost.test."]
    expected = {}
    for nm in candidates:
        try:
            addrs, _ttl = slow.resolve(nm)
            expected[nm] = ("ok", addrs)
        except DNSResolutionError as e:
            expected[nm] = (type(e), e.name)

    failures = []
    failures_lock = threading.Lock()
    done = [False] * n_threads

    def worker(idx):
        local_rng = random.Random(f"{seed}:{idx}")
        for _ in range(n_rounds):
            name = local_rng.choice(candidates)
            spelling = local_rng.choice([name, name.upper(), name[:-1]])
            kind, value = expected[name]
            try:
                got = resolver.resolve(spelling)
                ok = kind == "ok" and got == value
            except DNSResolutionError as e:
                ok = kind != "ok" and type(e) is kind and e.name == value
            if not ok:
                with failures_lock:
                    failures.append((seed, idx, spelling, kind, value))
            if local_rng.random() < 0.1:
                resolver.advance_clock(local_rng.randint(0, 3))
        done[idx] = True

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert all(done), f"seed {seed}: a worker never finished within the timeout (deadlock?)"
    assert not failures, failures[:5]


for seed in range(8):
    _stress(seed)
print("concurrent stress: 8 zones, 20 threads x 60 rounds each, every result matched the reference resolver")

print("all checks passed")
```
