`Memoize` wraps a callable `func` and caches the value it returns, so a later call with the same arguments returns the cached value instead of calling `func` again — in the style of `functools.lru_cache`. It holds at most `capacity` distinct calls, a positive integer; adding one beyond that evicts the *least recently used* entry first — the cached call that has gone longest without being made again. Both a call already cached (a *hit*) and one that is not (a *miss*) count as using the entry they touch when deciding which one is least recently used.

Two calls resolve to the same cache entry exactly when they pass equal positional arguments in the same positions and equal keyword arguments — compared by name and value, regardless of the order the keywords were written in at the call site. A call is never the same as one that passes, as a keyword, an argument the other passed positionally, even for the same-named parameter: `generate_key` builds a key from the `*args`/`**kwargs` it receives and never inspects `func`'s signature, so `f(1, 2)` and `f(1, b=2)` are different calls that may occupy separate entries. Every positional and keyword argument value must be hashable, since it becomes part of a cache key; passing one that is not (a `list`, for instance) fails exactly as it would from using that value as a `dict` key directly.

The skeleton below is provided, written in the style of `functools.lru_cache`:

```py
from collections import OrderedDict


class Memoize:
    """Wraps func so that repeated calls with the same arguments return a cached result instead of
    calling func again. Holds at most capacity entries; a call that would add a new one beyond that
    evicts the least recently used entry first. Both a hit and a miss count as a "use" of the key
    they touch, for the purpose of deciding which entry is least recently used."""

    def __init__(self, func, capacity):
        self.func = func
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def generate_key(self, *args, **kwargs):
        key = args + tuple(kwargs.items())
        if len(key) == 1:
            return key[0]
        return key

    def __call__(self, *args, **kwargs):
        key = self.generate_key(*args, **kwargs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        result = self.func(*args, **kwargs)
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result
```

### Part 1 — Find and fix three bugs

The code above has three bugs: two in `generate_key`, one in `__call__`. Two are wrong lines; one is a missing line. With all three fixed, `generate_key` and `__call__` together satisfy every rule stated above: kwargs given in a different order still resolve to the same entry; a call is never confused with an unrelated one that flattens to the same argument sequence through a different split between positional and keyword arguments — `f(1, 2)` (two positional arguments) and `f((1, 2))` (one positional argument, itself a two-element tuple) must never collide — and a hit updates recency exactly as a miss does. For each bug, give its location, state which calls it makes wrongly collide, wrongly fail to collide, or (for the third) what it does to eviction order, and why; then fix it.

```text
cache = Memoize(lambda *args, **kwargs: (args, kwargs), capacity=2)
cache(1, b=2, c=3)   # miss                                       cache (LRU -> MRU): [f(1,b=2,c=3)]
cache(1, c=3, b=2)   # hit -- kwargs reordered: the same call, the same entry
cache((1, 2))        # miss -- ONE positional argument, itself a two-element tuple
                      #                                    cache: [f(1,b=2,c=3), f((1,2))]
cache(1, 2)           # miss -- TWO positional arguments, not the same call as f((1,2)); capacity
                      # exceeded, so f(1,b=2,c=3) (unused since step 1) is evicted
                      #                                    cache: [f((1,2)), f(1,2)]
cache((1, 2))          # hit -- touches f((1,2)) again, moving it to MRU
                        #                                    cache: [f(1,2), f((1,2))]
cache(1, 3)             # miss -- capacity exceeded again, so f(1,2) (never touched again after
                        # insertion) is evicted, not the more recently touched f((1,2)): a hit
                        # updates recency exactly like a miss
                        #                                    cache: [f((1,2)), f(1,3)]
```

### Part 2 — Durable cache

From here on `Memoize` is a given, with all three bugs fixed. `DurableMemoize(func, capacity, path=None)` extends it so a crash — the process killed at an arbitrary point, with no chance to run further code — loses no call that finished beforehand, and recovers a cache indistinguishable from the one the process held at that instant. `path`, if given, is the base path this cache persists to and recovers from; `path=None` disables persistence entirely, identical to `Memoize`. When `path` names an existing store, recovery happens inside `__init__`, before it returns.

Persisting every call by rewriting the whole cache is disallowed: recording one call must cost time and space proportional to that call alone, never to `capacity` or to the calls before it. This requires an *append-only log*: a file only ever added to, never rewritten in place, until it is explicitly compacted by `compact()` below.

Before `__call__` returns, the call's effect — a new entry on a miss, a recency update on a hit — is *durable*: its record has been appended to the log, flushed, and `fsync`ed. A crash after that instant can never lose it; a crash before it loses it completely, never partially, as if the call had not been made. Constructing `DurableMemoize(func, capacity, path)` against such a `path` recovers a cache whose entries, values, and recency order — least to most recently used — match exactly what the earlier instance held after its last durable call, except that a call whose record never finished being written counts as never made: recovery must detect a crash mid-record — from the record's declared length and a checksum — and discard it and everything after it in the log, without raising or trusting any byte of it. `hits` and `misses` are bookkeeping only, never persisted, and start at 0 in every new instance.

`compact()` may be called at any time; the cache never calls it on its own. After it returns, the amount of data stored no longer depends on how many calls have been made since `path` was first used, only on the (at most `capacity`) entries currently cached. Replacing the old stored state with the new one is atomic: whichever instant the process crashes at during `compact()`, a subsequent recovery reproduces either the state from just before it was called or the state it was about to commit, never a mixture, and never loses an entry that was durable beforehand.

A positional or keyword argument that takes part in a key, and the value `func` returns, must be *JSON-serialisable*: `None`, a `bool`, an `int`, a `float`, a `str`, or a `list`/`dict` built from these, recursively (a `dict`'s keys must themselves be `str`). An argument's value round-trips exactly, including a `tuple` nested inside it; a returned value is under the weaker restriction alone, so a `tuple` inside it survives a crash and a restart as a `list` with the same elements instead, since JSON keeps no separate record of which one it was.

```py
class DurableMemoize(Memoize):
    def __init__(self, func, capacity, path: str | None = None) -> None:
        """path is the base path this cache persists to and recovers from; None (the default)
        disables persistence entirely. Recovery, if path names an existing store, happens here,
        before __init__ returns."""

    def compact(self) -> None:
        """Rewrites the stored state so that its size no longer depends on call history, only on
        the entries currently cached. Safe to call at any time, including right before a crash."""
```

For example:

```text
w = DurableMemoize(lambda n: n * n, capacity=2, path="cache.log")
w(3)                     # miss -- durably records (3,) -> 9 before returning
w(4)                     # miss -- durably records (4,) -> 16;   cache (LRU -> MRU): [(3,), (4,)]
w(3)                     # hit -- durably records that (3,) was used again;   cache: [(4,), (3,)]

# the process is killed here, mid-way through durably recording the next call, w(5) -- the log
# on disk ends in a fragment: a well-formed record's declared length, followed by fewer bytes
# than that length promises

r = DurableMemoize(lambda n: n * n, capacity=2, path="cache.log")
# r replays the log: the records for w(3), w(4), and the touch of 3 are intact and applied; the
# trailing fragment fails its length/checksum check and is discarded. r's cache now maps n=4 to 16
# and n=3 to 9, in exactly that order from least to most recently used -- the order right before the crash
```

### Part 3 — Thread-safe cache

`ThreadSafeMemoize` extends `DurableMemoize` with the same observable contract under any number of threads calling it concurrently — for the same key, for different keys, in any interleaving, under any scheduling — plus one more guarantee: `func` is never running more than once at a time for a given key, even when several threads call it with that key while no cached entry for it yet exists. A thread that arrives while another is already computing that key's value waits for that result instead of starting a second, redundant computation; it must not do so while holding any lock a call for an unrelated key needs, since `func` may take an arbitrary, unknown amount of time (a network call, a slow computation), and a cache that serialises unrelated keys behind it stops behaving like a cache under load. An exception `func` raises propagates to every thread waiting on that call, and to none of the others; nothing is cached or persisted for that call.

```py
class ThreadSafeMemoize(DurableMemoize):
    def __init__(self, func, capacity, path: str | None = None) -> None:
        """Same contract as DurableMemoize.__init__."""
```

For example:

```text
sf = ThreadSafeMemoize(fetch_profile, capacity=100, path="cache.log")

# Thread A: sf(user_id=7)   # miss -- becomes responsible for key (user_id=7,); calls fetch_profile
#                           # without holding any lock this cache keeps
# Thread B: sf(user_id=7)   # arrives while A's call is still running; finds (user_id=7,) already
#                           # being computed, and waits for A's result instead of calling it itself
# Thread C: sf(user_id=9)   # a different key -- proceeds immediately, never waiting on A or B

# fetch_profile(user_id=7) runs exactly once, however many threads called sf(user_id=7) while it
# was in flight; every one of them returns its result, and fetch_profile(user_id=9) runs freely
# at the same time as it
```
