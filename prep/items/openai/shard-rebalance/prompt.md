A *shard* owns a closed range of integer keys `[start, end]` (`start <= end`, both ends included); the ranges of different shards may overlap, and keys can be as large in magnitude as $10^9$. For a key $k$, its *coverage* is the number of shards whose range contains $k$.

```py
class Shard:
    id: str        # distinct among the shards passed to one call
    start: int
    end: int
```

### Part 1 — Capping overlap

Implement `rebalance(limit, shards)`, which narrows and drops shards so that no key's coverage exceeds `limit`, while every key from the smallest original start to the largest original end stays covered by at least one shard.

```py
def rebalance(limit: int, shards: list[Shard]) -> list[Shard]:
    """limit >= 1. Returns the surviving shards. A surviving shard's start is >= its original
    start and its end is >= its original end -- never the other way around."""
```

Apply the following rule, in order:

- Sort the input by `start` ascending, then `end` ascending, then `id` ascending (ids are distinct strings, so this is a total order), and process the shards one at a time in that order.
- When shard $s$ is processed, let *kept* be the shards already decided to be kept earlier in this order, each with the range this same rule gave it when it was processed (never yet extended by step 5). Shift $s$'s start forward to the smallest key in $[s.start, s.end]$ at which the coverage counted over *kept* alone is strictly less than `limit`; if $s.start$ itself already qualifies, nothing is shifted.
- If no key in $[s.start, s.end]$ qualifies, drop $s$ -- it does not appear in the output.
- Otherwise $s$ is kept with range $[\text{new start}, s.end]$ and joins *kept* for the shards processed afterwards.
- Once every shard has been through steps 2-4, let $E = [\min(start), \max(end)]$ be the envelope over *every* input shard, kept or not. A *hole* is a maximal run of keys in $E$ that no kept shard covers, using the ranges from step 4. Close each hole by extending the end of the kept shard that ends at the key just before the hole up to the hole's last key; if several kept shards end there, extend the one with the smallest start, then the smallest id. ($E$'s lowest key is always covered, so such a shard always exists.)

Return the survivors sorted by their final `(start, end, id)`.

Example, with `limit = 2`:

```text
north:  [5, 40]
south:  [5, 42]
east:   [5, 44]
west:   [5, 120]
inland: [6, 42]
coast:  [130, 150]
->
north:  [5, 40]
south:  [5, 42]
east:  [41, 44]     # shifted: north and south already fill [5, 40] to coverage 2
west:  [43, 129]    # shifted: coverage is already 2 on all of [5, 42]; extended over the hole [121, 129]
coast: [130, 150]
                    # inland is dropped: every key of [6, 42] already has coverage 2
```

Write your own test cases for the rule as well, covering at least a hole between two shards, one shard's range contained in another's, several shards with identical ranges, and `limit = 1`.

### Part 2 — Incremental shards and key routing

Rebalancing a fixed batch, as in Part 1, does not fit a shard being added or removed one at a time: narrowing one range in the middle of the key space can shift several neighbours' boundaries. This part starts over rather than building on Part 1: a shard no longer owns a range of keys, and the mapping from key to shard is yours to design. Implement `ShardRouter`, which assigns every integer key to one shard of a set that changes one shard at a time.

```py
class ShardRouter:
    def __init__(self): ...
    def add_shard(self, shard_id: str) -> None: ...     # ValueError if already present
    def remove_shard(self, shard_id: str) -> None: ...  # ValueError if absent
    def locate(self, key: int) -> str: ...              # owner's id; LookupError if there are no shards
```

Requirements:

- `locate` depends only on the key and the current set of shard ids: not on the order in which the shards were added, and not on the process, so a key maps to the same shard after a restart.
- Only the keys that must move do: after `add_shard(x)`, every key whose owner changed is now owned by `x`; after `remove_shard(x)`, every key whose owner changed was owned by `x`.
- Keys spread about evenly: with $N$ shards, each owns roughly $1/N$ of them.

Example, over the keys `0..29_999`:

```text
r = ShardRouter()
r.add_shard("amber"); r.add_shard("cobalt"); r.add_shard("jade")
    # each of the three owns roughly a third of the keys
r.add_shard("slate")
    # roughly a quarter of the keys change owner, and every one of them now belongs to "slate"
r.remove_shard("cobalt")
    # exactly the keys "cobalt" owned (again roughly a quarter) change owner
```
