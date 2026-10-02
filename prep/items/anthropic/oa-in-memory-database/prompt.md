Implement the four levels below in order: a level's tests must pass before the next level's tests run, and every level extends the same class rather than replacing it. A *record* is identified by a string `key` and holds a set of *fields*, each a string field name mapped to a string value; a key, a field name, and a value are each non-empty strings, and a field name or a value never contains `(`, `)`, or `,`. Every method returns a string; one with nothing useful to report returns `""`.

### Level 1 — Core operations

```py
class InMemoryDatabase:
    def set(self, key: str, field: str, value: str) -> str:
        """Sets field of the record key to value, creating the record if key is new and
        overwriting any value already held there. Returns ""."""

    def get(self, key: str, field: str) -> str:
        """Returns the value held in field of key, or "" if key does not exist or holds no
        such field."""

    def delete(self, key: str, field: str) -> str:
        """Removes field from key if present. Returns "true" if a field was removed, "false"
        if key did not exist or held no such field."""
```

Example:

```text
db = InMemoryDatabase()
db.set("order-1", "status", "packed")     # -> ""
db.set("order-1", "carrier", "fedex")     # -> ""
db.get("order-1", "status")               # -> "packed"
db.get("order-1", "weight")               # -> ""            -- field never set
db.get("order-9", "status")               # -> ""            -- key never created
db.delete("order-1", "carrier")           # -> "true"
db.delete("order-1", "carrier")           # -> "false"        -- already gone
db.set("order-1", "status", "shipped")    # -> ""             -- overwrites
db.get("order-1", "status")               # -> "shipped"
```

### Level 2 — Scan

```py
class InMemoryDatabase:
    def scan(self, key: str) -> str:
        """Returns every field of key formatted as "name1(value1), name2(value2), ...", with
        fields sorted lexicographically by name (Python's default string ordering); "" if key
        does not exist or holds no fields."""

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        """Same as scan, restricted to the fields of key whose name starts with prefix
        (prefix="" behaves exactly like scan)."""
```

Example:

```text
db = InMemoryDatabase()
db.set("order-1", "carrier", "fedex")
db.set("order-1", "cost", "12.50")
db.set("order-1", "status", "packed")
db.scan_by_prefix("order-1", "c")   # -> "carrier(fedex), cost(12.50)"
db.scan("order-1")                  # -> "carrier(fedex), cost(12.50), status(packed)"
db.scan_by_prefix("order-9", "c")   # -> ""                              -- key never created
```

### Level 3 — Timestamps and TTL

Every call from this level on happens at an integer *timestamp*; across the whole sequence of calls made to one instance, timestamps never decrease, whether or not they are given explicitly. Level 1's and Level 2's five methods keep working exactly as specified above, but internally they now happen at the current value of a clock that starts at $0$ and advances by $1$ after every such call. Each of the five also gains a twin whose name ends in `_at` and which takes an explicit `timestamp` as its first argument instead of reading the clock; a call to one of these sets the clock to `timestamp`, which is never smaller than the clock's current value. Both kinds of call share this one clock, so a field written by a plain `set` can be read by `get_at`, and one written by `set_at_with_ttl` can expire under a later plain `get`.

```py
class InMemoryDatabase:
    def set_at(self, timestamp: int, key: str, field: str, value: str) -> str:
        """Same as set, at timestamp. Clears any TTL field previously held, even one that has
        not expired yet -- the field becomes permanent."""

    def set_at_with_ttl(self, timestamp: int, key: str, field: str, value: str, ttl: int) -> str:
        """Same as set_at, but field is visible from timestamp up to, but not including,
        timestamp + ttl (ttl is a positive integer). A later set_at_with_ttl call on the same
        field replaces this window outright, computed fresh from the new call; it does not
        extend the old one."""

    def get_at(self, timestamp: int, key: str, field: str) -> str:
        """Same as get, at timestamp: a field whose TTL window has elapsed by timestamp counts
        as absent."""

    def delete_at(self, timestamp: int, key: str, field: str) -> str:
        """Same as delete, at timestamp; a field whose TTL window has elapsed by timestamp
        counts as already absent, so this returns "false" for it."""

    def scan_at(self, timestamp: int, key: str) -> str:
        """Same as scan, at timestamp."""

    def scan_by_prefix_at(self, timestamp: int, key: str, prefix: str) -> str:
        """Same as scan_by_prefix, at timestamp."""
```

Example:

```text
db = InMemoryDatabase()
db.set_at_with_ttl(10, "order-1", "carrier", "fedex", 5)  # carrier alive on [10, 15)
db.set_at_with_ttl(12, "order-1", "carrier", "ups", 6)    # replaced: window becomes [12, 18), not extended to 21
db.set_at(13, "order-1", "cost", "12.50")                  # no ttl -> cost is permanent
db.scan_at(17, "order-1")                # -> "carrier(ups), cost(12.50)"   17 < 18: carrier still alive
db.scan_at(18, "order-1")                # -> "cost(12.50)"                 18 == 18: carrier just expired
db.delete_at(18, "order-1", "carrier")   # -> "false"    already expired -- nothing to delete
db.set_at_with_ttl(19, "order-1", "carrier", "usps", 3)    # carrier alive again, on [19, 22)
db.set_at(20, "order-1", "carrier", "dhl")                 # no ttl -> clears the TTL, now permanent
db.scan_at(22, "order-1")                # -> "carrier(dhl), cost(12.50)"   22 is the old expiry, but it was cleared
```

### Level 4 — Backup and restore

```py
class InMemoryDatabase:
    def backup(self, timestamp: int) -> str:
        """Deep-snapshots every field currently live at timestamp under Level 3's rules. A
        field with a TTL is stored in the snapshot as the *remaining* time until its expiry
        (its own expiry minus timestamp), not as an absolute expiry; a permanent field is
        stored as permanent. A second backup at the same timestamp replaces the first.
        Returns ""."""

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        """Finds the backup with the largest own timestamp that is <= timestamp_to_restore (at
        least one such backup is guaranteed to have been made) and makes it the database's
        entire live state as of timestamp, discarding every write and delete made since --
        including ones made by an earlier restore. Every field of the restored snapshot that
        carried a remaining-TTL delta d gets a new expiry of timestamp + d: its countdown
        resumes from the moment of this restore, not from the moment of the original backup or
        of the field's original write. No backup is ever deleted, so a later restore call may
        still reach a backup made after this one. Returns ""."""
```

Example:

```text
db = InMemoryDatabase()
db.set_at_with_ttl(0, "order-1", "carrier", "fedex", 10)  # alive on [0, 10)
db.set_at(0, "order-1", "cost", "12.50")                   # permanent
db.backup(4)                              # -> ""   snapshot: carrier has 6 remaining (10 - 4), cost permanent
db.set_at(5, "order-1", "cost", "15.00")  # the live database changes after the backup; the snapshot does not
db.delete_at(6, "order-1", "carrier")     # -> "true"   carrier was alive at 6 (6 < 10)
db.restore(100, 4)                        # -> ""   restores the ts=4 backup as of timestamp 100
db.get_at(100, "order-1", "cost")         # -> "12.50"   back to the pre-backup value
db.get_at(100, "order-1", "carrier")      # -> "fedex"    restored, with 6 ticks left, resumed from 100
db.get_at(105, "order-1", "carrier")      # -> "fedex"    100 + 6 = 106: still alive at 105
db.get_at(106, "order-1", "carrier")      # -> ""         106 is the new expiry -- exclusive again
```
