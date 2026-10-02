Implement the five levels below in order: a level's tests must pass before the next level's tests run, and every level extends the same `Resolver` class rather than replacing it. A *zone* is a `dict[str, ARecord | CNAMERecord]` mapping a domain name to one *record*:

```py
class ARecord:
    def __init__(self, addresses: list[str], ttl: int) -> None:
        """addresses is one or more IPv4 addresses, each a non-empty string such as "203.0.113.7",
        kept in the order given. ttl is a positive integer number of seconds; see Level 4."""

class CNAMERecord:
    def __init__(self, target: str, ttl: int) -> None:
        """target is another name -- not necessarily normalised (below), and not required to be a
        key of any zone. ttl is a positive integer number of seconds; see Level 4."""
```

*Name normalisation* case-folds a name (every ASCII letter to lower case) and collapses its trailing dots to exactly one: zero or more trailing `.` characters are replaced by a single one, so `"Example.COM"`, `"example.com."` and `"EXAMPLE.com.."` all normalise to `"example.com."`. Every name this problem looks up — the name `resolve` is called with, and every `CNAMERecord.target` reached while following a chain — is normalised first. A zone's own keys are normalised once, when a `Resolver` is built from it; if two of a zone's keys normalise to the same name, the one that appears later when the `dict` is iterated wins.

A `Resolver` is built once and answers many calls to `resolve` afterwards:

```py
class DNSResolutionError(Exception):
    """Base class for every error resolve can raise; name is whichever name each subclass below names."""
    name: str

class Resolver:
    def __init__(self, zone: dict[str, ARecord | CNAMERecord],
                 fallback_zone: dict[str, ARecord | CNAMERecord] | None = None,
                 *, max_chain_length: int = 8) -> None: ...

    def resolve(self, name: str) -> list[str]:
        """Returns the IPv4 addresses name resolves to, in the order Level 2 defines."""
```

`fallback_zone` and `max_chain_length` matter from Level 3 on; until then, treat every `Resolver` as if `fallback_zone` were omitted. Every record's `ttl` matters from Level 4 on; before that it is carried on the record but never read.

### Level 1 — Name normalisation and direct A lookup

`resolve` normalises `name`, looks it up directly in the zone, and returns its addresses. It raises `NameNotFoundError(name)` (`name` the normalised name) if no record for it exists in the zone. Level 1's zone holds only `ARecord`s; Level 2 below adds `CNAMERecord`s and what `resolve` does with one.

```py
class NameNotFoundError(DNSResolutionError): ...
```

Example:

```text
zone = {
    "api.acme.com.": ARecord(["203.0.113.10"], ttl=60),
    "db.acme.com.": ARecord(["203.0.113.20", "203.0.113.21"], ttl=60),
}
r = Resolver(zone)
r.resolve("API.acme.com")      # -> ["203.0.113.10"]                    -- case-folded, dot appended
r.resolve("db.acme.com.")      # -> ["203.0.113.20", "203.0.113.21"]
r.resolve("nope.acme.com.")    # -> raises NameNotFoundError("nope.acme.com.")
```

### Level 2 — CNAME chains, cycles and the chain-length limit

`resolve` is redefined to handle a `CNAMERecord`, which never appeared in Level 1's zone. Resolving `name` starts at its normalised form and follows a chain of `CNAMERecord`s, one per step — each step moves to the normalised form of the current record's `target` — until it reaches a name whose record is an `ARecord`, whose addresses are then returned; the chain's length is the number of `CNAMERecord`s followed this way, so a name that is itself an `ARecord` has chain length $0$. `resolve` tracks every name visited in the call, starting with the normalised query name, and raises `ResolutionCycleError(target)` the moment a step's target is a name already in that set (`target` the repeated name) — checked before anything else on that step, so a cycle always raises `ResolutionCycleError`, even where continuing would also exceed the length limit. It raises `ChainTooLongError(name)` (`name` the original, normalised query name) if reaching an `ARecord` needs a chain longer than `max_chain_length`. As in Level 1, `NameNotFoundError(name)` is raised whenever a name reached this way — the original query or any later target — is not a key of the zone at all.

```py
class ResolutionCycleError(DNSResolutionError): ...
class ChainTooLongError(DNSResolutionError): ...
```

Example, extending the zone above with a chain:

```text
zone["www.acme.com."] = CNAMERecord("edge.acme.com.", ttl=60)
zone["edge.acme.com."] = CNAMERecord("api.acme.com.", ttl=30)
r = Resolver(zone)
r.resolve("www.acme.com.")
# www.acme.com. -> CNAME edge.acme.com. -> CNAME api.acme.com. -> A ["203.0.113.10"]
# -> ["203.0.113.10"]                      -- chain length 2, well under the default max_chain_length=8

cycle_zone = {"a.acme.com.": CNAMERecord("b.acme.com.", ttl=60),
              "b.acme.com.": CNAMERecord("a.acme.com.", ttl=60)}
Resolver(cycle_zone).resolve("a.acme.com.")
# a -> CNAME b (visited={a, b}) -> CNAME a: a already visited -> raises ResolutionCycleError("a.acme.com.")

long_zone = {
    "start.acme.com.": CNAMERecord("mid1.acme.com.", ttl=60),
    "mid1.acme.com.": CNAMERecord("mid2.acme.com.", ttl=60),
    "mid2.acme.com.": CNAMERecord("mid3.acme.com.", ttl=60),
    "mid3.acme.com.": ARecord(["203.0.113.99"], ttl=60),
}
Resolver(long_zone, max_chain_length=2).resolve("start.acme.com.")
# start -> mid1 (length 1) -> mid2 (length 2) -> mid3 (length 3, exceeds max_chain_length=2)
# -> raises ChainTooLongError("start.acme.com.")
```

### Level 3 — Fallback zone

`resolve` is redefined again to make use of `fallback_zone`, which Levels 1 and 2 have ignored so far. It first attempts Level 2's resolution entirely within `zone`. If that attempt raises `NameNotFoundError` and `fallback_zone` was given, `resolve` makes one more attempt: a fresh Level 2 resolution of the *original*, normalised query name within `fallback_zone` alone — never resuming the first attempt's chain partway through, and using the same `max_chain_length` — and returns its result. If that second attempt also raises `NameNotFoundError`, `resolve` raises that error (naming whichever name inside `fallback_zone`'s own chain was missing); the first attempt's error is discarded once a fallback exists to try. If `fallback_zone` was not given, or if the first attempt raises `ResolutionCycleError` or `ChainTooLongError`, `resolve` raises that error immediately and never consults `fallback_zone` — only a plain "no such name" falls through, never a malformed chain.

Example:

```text
primary = {"svc.acme.com.": CNAMERecord("internal.acme.com.", ttl=60)}    # internal.acme.com. is not in primary
fallback = {"svc.acme.com.": ARecord(["198.51.100.5"], ttl=30)}
Resolver(primary, fallback).resolve("svc.acme.com.")
# primary: svc -> CNAME internal.acme.com., which is missing -> NameNotFoundError -- falls through
# fallback: svc.acme.com. -> A ["198.51.100.5"]            -- resolved fresh, not resumed from internal
# -> ["198.51.100.5"]

looping = {"loop.acme.com.": CNAMERecord("loop.acme.com.", ttl=60)}
has_answer = {"loop.acme.com.": ARecord(["198.51.100.9"], ttl=60)}
Resolver(looping, has_answer).resolve("loop.acme.com.")
# raises ResolutionCycleError("loop.acme.com.") -- fallback is never tried, even though it holds an answer

Resolver({}, {}).resolve("nowhere.acme.com.")
# -> raises NameNotFoundError("nowhere.acme.com.")          -- absent from both zones
```

### Level 4 — A TTL cache on a manual clock

`Resolver` gains a manual clock, moved only by `advance_clock`; it starts at $0$.

```py
    def advance_clock(self, seconds: int) -> None:
        """Adds seconds (a non-negative integer) to the resolver's clock."""
```

`resolve` is redefined to cache a successful result under the normalised query name: the first call for a name performs Level 3's resolution and stores its addresses together with an expiry, and every later call for the same name returns the stored addresses directly, touching neither zone, *while the clock has not reached that expiry*. A cached answer's expiry is the clock reading at the moment it was stored, plus the *minimum* `ttl` of every record read while producing it — every `CNAMERecord` followed and the final `ARecord`, across whichever one of Level 3's (up to two) attempts succeeded. Expiry is checked lazily, only when a name is looked up: nothing scans the cache for expired entries on its own, and an entry left behind by a later `advance_clock` call simply stops being treated as a hit the next time its name is queried, whereupon it is resolved and overwritten like any other miss. A resolution that raises `NameNotFoundError`, `ResolutionCycleError` or `ChainTooLongError` is not cached; the next call for that name repeats the full resolution. Caching applies only to the name `resolve` was originally called with — resolving `"www.acme.com."` never creates a cache entry for `"edge.acme.com."` or any other name visited only as part of its chain.

Example, continuing Level 2's chain (`www.acme.com.` at ttl $60$ $\to$ `edge.acme.com.` at ttl $30$ $\to$ `api.acme.com.` at ttl $60$, an `ARecord`):

```text
r = Resolver(zone)              # zone as extended in Level 2; clock starts at 0
r.resolve("www.acme.com.")      # -> ["203.0.113.10"]; cached with expiry 0 + min(60, 30, 60) = 30
r.advance_clock(29)             # clock = 29
r.resolve("www.acme.com.")      # -> ["203.0.113.10"]     -- 29 < 30: cache hit, zone untouched
r.advance_clock(1)              # clock = 30
r.resolve("www.acme.com.")      # -> ["203.0.113.10"]     -- 30 == 30: expired, resolved fresh again
```

### Level 5 — Concurrent resolution

`resolve` is called from many threads at once, all sharing one `Resolver`. It must behave exactly as Level 4 specifies from each caller's point of view, plus one more rule: while a name's Level 3 resolution (the work behind a cache miss) is under way, started by whichever call for that name arrived first, every other call for the *same normalised name* that arrives before it finishes waits for that one resolution instead of starting a redundant one of its own, then returns its result — the same addresses, or the same exception raised in every waiting call — exactly as if it had performed the resolution itself. A `resolve` call is never charged for another name's resolution: two different names that arrive at the same time, even ones whose chains pass through the same `CNAMERecord`, are resolved independently and may run concurrently. Every list `resolve` returns is a call's own copy: mutating it never affects the cache or any other call's result. The cache, the clock and the bookkeeping this level adds are all safe to use from multiple threads without any lock of the caller's own.

For example: nothing has been resolved yet, and threads $t_1, \dots, t_5$ all call `resolve("www.acme.com.")` before any of them returns. However these five calls interleave, the chain in `zone` is walked at most once for `"www.acme.com."` while any of them is outstanding, and all five return `["203.0.113.10"]`. A sixth thread, $t_6$, calling `resolve("shop.acme.com.")` at the same moment for a name that also happens to chain through `"edge.acme.com."`, does not join $t_1, \dots, t_5$'s wait and may resolve concurrently with them.
