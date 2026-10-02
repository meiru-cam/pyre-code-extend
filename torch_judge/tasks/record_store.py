"""A key-field record store that grows timestamps, expiring fields, and backups part by part."""

from ._interview import interview

# A slow, plain model of every rule, and a random call generator that keeps timestamps in order.
_MODEL = r"""
import random

class Model:
    def __init__(self):
        self.data, self.clock, self.saved = {}, 0, {}
    def now(self, ts):
        if ts is None:
            ts, self.clock = self.clock, self.clock + 1
        else:
            self.clock = ts
        return ts
    def live(self, key, t):
        return {f: v for f, (v, end) in self.data.get(key, {}).items() if end is None or t < end}
    def show(self, live, prefix):
        return ", ".join(name + "(" + live[name] + ")" for name in sorted(live) if name.startswith(prefix))
    def call(self, op, ts, *args):
        t = self.now(ts)
        if op == "set":
            key, field, value = args
            self.data.setdefault(key, {})[field] = (value, None)
            return ""
        if op == "set_ttl":
            key, field, value, ttl = args
            self.data.setdefault(key, {})[field] = (value, t + ttl)
            return ""
        if op == "get":
            return self.live(args[0], t).get(args[1], "")
        if op == "delete":
            key, field = args
            if field in self.live(key, t):
                del self.data[key][field]
                return "true"
            return "false"
        if op == "scan":
            return self.show(self.live(args[0], t), "")
        if op == "prefix":
            return self.show(self.live(args[0], t), args[1])
        if op == "backup":
            self.saved[t] = {k: {f: (v, None if end is None else end - t) for f, (v, end) in fs.items() if end is None or t < end}
                             for k, fs in self.data.items()}
            return ""
        if op == "restore":
            best = max(b for b in self.saved if b <= args[0])
            self.data = {k: {f: (v, None if left is None else t + left) for f, (v, left) in fs.items()}
                         for k, fs in self.saved[best].items()}
            return ""
        raise ValueError(op)

PLAIN = {"set": "set", "get": "get", "delete": "delete", "scan": "scan", "prefix": "scan_by_prefix"}
TIMED = {"set": "set_at", "set_ttl": "set_at_with_ttl", "get": "get_at", "delete": "delete_at", "scan": "scan_at",
         "prefix": "scan_by_prefix_at", "backup": "backup", "restore": "restore"}

def random_calls(rng, n, timed, saves):
    keys, fields, values = ["k1", "k2", "K"], ["ab", "a", "b", "B", "abc", "x"], ["1", "v", "zz", "q q"]
    model, out = Model(), []
    for _ in range(n):
        ops = ["set", "get", "delete", "scan", "prefix"] + (["set_ttl", "set_ttl"] if timed else [])
        if saves:
            ops += ["backup"] + (["restore"] if model.saved else [])
        op = rng.choice(ops)
        plain = op in PLAIN and (not timed or rng.random() < 0.3)
        ts = None if plain else model.clock + rng.choice([0, 0, 1, 1, 2, 5])
        key, field = rng.choice(keys), rng.choice(fields)
        args = {"set": (key, field, rng.choice(values)), "set_ttl": (key, field, rng.choice(values), rng.randint(1, 6)),
                "get": (key, field), "delete": (key, field), "scan": (key,), "prefix": (key, rng.choice(["", "a", "ab", "B", "y"])),
                "backup": (), "restore": ()}[op]
        if op == "restore":
            args = (rng.randint(min(model.saved), ts),)
        expected = model.call(op, ts, *args)
        out.append((op, ts, args, expected))
    return out

def replay(store, calls, label):
    for i, (op, ts, args, expected) in enumerate(calls):
        if ts is None:
            got = getattr(store, PLAIN[op])(*args)
        else:
            got = getattr(store, TIMED[op])(ts, *args)
        assert got == expected, (label, i, op, ts, args, got, expected)
"""

TASK = {
    "title": "Record Store With TTL and Backups",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "RecordStore",
    "description_en": r"""Build `RecordStore`, an in-memory store of records. Each record has a key and holds named fields with string values.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `RecordStore` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Keys, field names and values are non-empty strings. Field names and values never contain `(`, `)` or `,`.
- Every method returns a string. A method with nothing to report returns `""`.
- A record with no fields left behaves exactly like a key that was never written.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment that rewards a data layout which survives new requirements. Each later part adds one requirement, and an early shortcut such as storing absolute expiry times in a backup breaks a later part.

**Where it is used:** Redis hashes with per-field expiry, caches with time-to-live, and snapshot and restore in key-value stores.

Adapted from the in-memory database online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Set, get and delete fields",
            "description_en": r"""**Signatures:**
- `RecordStore().set(key, field, value) -> str` stores `value` in that field of that record, replacing any old value, and returns `""`.
- `get(key, field) -> str` returns the stored value, or `""` if the key or the field does not exist.
- `delete(key, field) -> str` removes the field and returns `"true"`, or returns `"false"` if there was nothing to remove.

**Example:**
- `set("lamp-2", "color", "amber")` and `set("lamp-2", "watts", "40")` both return `""`
- `get("lamp-2", "color")` is `"amber"`; `get("lamp-2", "lumens")` and `get("lamp-5", "color")` are `""`
- `delete("lamp-2", "watts")` is `"true"`, then the same call again is `"false"`
- `set("lamp-2", "color", "teal")` replaces the value, so `get("lamp-2", "color")` is `"teal"`""",
        },
        {
            "title": "Scan a record",
            "description_en": r"""Keep Part 1 and add two read methods.

**Signatures:**
- `scan(key) -> str` lists every field of the record as `"name(value)"` items joined by `", "`, sorted by field name with Python's default string order. A missing or empty record gives `""`.
- `scan_by_prefix(key, prefix) -> str` does the same for the fields whose name starts with `prefix`. An empty `prefix` matches every field.

**Example:** after `set("lamp-2", "watts", "9")`, `set("lamp-2", "bulb", "led")` and `set("lamp-2", "base", "e27")`:
- `scan_by_prefix("lamp-2", "b")` is `"base(e27), bulb(led)"`
- `scan("lamp-2")` is `"base(e27), bulb(led), watts(9)"`
- `scan_by_prefix("lamp-2", "x")` and `scan("lamp-5")` are `""`""",
        },
        {
            "title": "Timestamps and expiring fields",
            "description_en": r"""Keep Parts 1–2. Every call now happens at an integer time.

**The clock:**
- The store keeps a clock that starts at `0`.
- Each of the five methods from Parts 1–2 runs at the current clock value, then adds `1` to the clock.
- Each method below takes a `timestamp` first. It sets the clock to `timestamp` and runs at that time. A `timestamp` is never smaller than the clock.
- Both kinds of call read and write the same fields.

**Signatures:**
- `set_at(timestamp, key, field, value)`, `get_at(timestamp, key, field)`, `delete_at(timestamp, key, field)`, `scan_at(timestamp, key)` and `scan_by_prefix_at(timestamp, key, prefix)` behave like the methods without `_at`, at `timestamp`.
- `set_at_with_ttl(timestamp, key, field, value, ttl) -> str` stores a field that exists at times `t` with `timestamp <= t < timestamp + ttl`. `ttl` is a positive integer.
- An expired field counts as absent everywhere. `delete_at` returns `"false"` for it.
- Writing a field again starts over. `set_at_with_ttl` gives a new window counted from its own `timestamp`. A write without a ttl, plain or `_at`, makes the field permanent.

**Example:**
- `set_at_with_ttl(3, "lamp-2", "mode", "dim", 4)`: `mode` exists from 3 to 6
- `set_at_with_ttl(5, "lamp-2", "mode", "bright", 3)`: the window is now 5 to 7
- `set_at(6, "lamp-2", "room", "den")`, then `scan_at(7, "lamp-2")` is `"mode(bright), room(den)"`
- `scan_at(8, "lamp-2")` is `"room(den)"`, and `delete_at(8, "lamp-2", "mode")` is `"false"`
- `set_at_with_ttl(9, "lamp-2", "mode", "off", 2)`, then `set_at(10, "lamp-2", "mode", "auto")`: at 11, `scan_at` is `"mode(auto), room(den)"`
- a plain `get("lamp-2", "mode")` now runs at 11 and returns `"auto"`; the clock becomes 12""",
        },
        {
            "title": "Backup and restore",
            "description_en": r"""Keep Parts 1–3 and add two methods. Both set the clock like the other `_at` methods.

**Signatures:**
- `backup(timestamp) -> str` saves a copy of every field that exists at `timestamp`. An expiring field is saved with its time left, `expiry - timestamp`. A second backup at the same `timestamp` replaces the first. Returns `""`.
- `restore(timestamp, timestamp_to_restore) -> str` picks the backup with the largest time that is `<= timestamp_to_restore`. One always exists. Every current field is replaced by that backup's fields, and a field saved with time left `d` now expires at `timestamp + d`. Returns `""`.
- Later writes never change a saved backup, and a restore deletes no backup. Restoring the same backup twice gives the same state both times.

**Example:**
- `set_at_with_ttl(2, "lamp-2", "mode", "dim", 8)` exists from 2 to 9; `set_at(2, "lamp-2", "room", "den")` is permanent
- `backup(5)` saves `room` and `mode` with 5 ticks left
- `set_at(6, "lamp-2", "room", "hall")`, then `delete_at(7, "lamp-2", "mode")` is `"true"`
- `restore(50, 5)`: `get_at(50, "lamp-2", "room")` is `"den"`
- `get_at(54, "lamp-2", "mode")` is `"dim"` and `get_at(55, "lamp-2", "mode")` is `""`, because `50 + 5 = 55`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which two-level structure lets you reach one field of one record in O(1)? What should get return for a key that was never written, and what should delete return when the field is already gone?"},
        {"level": 2, "kind": "analysis", "content": "Keep a dict from key to a dict from field to value. set writes self.data.setdefault(key, {})[field] = value. get chains .get(key, {}).get(field, \"\"). delete checks membership first, deletes and returns \"true\", otherwise \"false\". Every method returns a string."},
    ],
    "model_connections": [
        "Feature stores and KV caches for inference keep per-key fields with a time-to-live so stale entries drop out without a sweep.",
        "Checkpoints of a training job save remaining budgets, such as steps left in a warmup, rather than wall-clock deadlines, for the same reason backups here save time left.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A dict of dicts gives O(1) set, get and delete on one field.",
            "Checking expiry when a field is read needs no background sweep.",
            "Saving time left makes a restored field behave as if the gap between backup and restore never happened.",
        ],
        "cons": [
            "Lazy expiry keeps dead fields in memory until something reads or overwrites them.",
            "A scan sorts every live field of the record, so it costs O(f log f) on each call.",
            "Every backup is a full copy, so memory grows with the number of backups times the store size.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
store = {fn}()
assert store.set("lamp-2", "color", "amber") == ""
assert store.set("lamp-2", "watts", "40") == ""
assert store.get("lamp-2", "color") == "amber"
assert store.get("lamp-2", "lumens") == ""
assert store.get("lamp-5", "color") == ""
assert store.delete("lamp-2", "watts") == "true"
assert store.delete("lamp-2", "watts") == "false"
assert store.set("lamp-2", "color", "teal") == ""
assert store.get("lamp-2", "color") == "teal"
"""},
        {"name": "Part 1: missing keys and fields", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "get must return the empty string, not None, for a missing key or field; delete returns the string \"false\" when the key or field is missing and \"true\" only when it removed something.",
         "code": r"""
store = {fn}()
assert store.get("none", "f") == ""
assert store.delete("none", "f") == "false"
store.set("a", "f", "1")
assert store.delete("a", "g") == "false"
assert store.delete("a", "f") == "true"
assert store.get("a", "f") == ""
store.set("a", "f", "2")
assert store.get("a", "f") == "2"
store.set("b", "f", "3")
assert store.get("a", "f") == "2", "records with different keys are separate"
"""},
        {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random sequence of set, get and delete calls, a return value differed from a plain dictionary model.",
         "code": _MODEL + r"""
for seed in range(150):
    rng = random.Random(seed)
    calls = [c for c in random_calls(rng, 60, False, False) if c[0] in ("set", "get", "delete")]
    replay({fn}(), calls, seed)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
store = {fn}()
store.set("lamp-2", "watts", "9")
store.set("lamp-2", "bulb", "led")
store.set("lamp-2", "base", "e27")
assert store.scan_by_prefix("lamp-2", "b") == "base(e27), bulb(led)"
assert store.scan("lamp-2") == "base(e27), bulb(led), watts(9)"
assert store.scan_by_prefix("lamp-2", "x") == ""
assert store.scan("lamp-5") == ""
"""},
        {"name": "Part 2: ordering, prefixes and emptied records", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "scan sorts by field name with Python's default order (uppercase before lowercase), joins items with \", \", and a prefix must match the start of the name, not anywhere in it; a record whose fields were all deleted scans as \"\".",
         "code": r"""
store = {fn}()
for field, value in [("beta", "2"), ("Alpha", "1"), ("ab", "3"), ("xab", "4"), ("a", "5")]:
    store.set("r", field, value)
assert store.scan("r") == "Alpha(1), a(5), ab(3), beta(2), xab(4)"
assert store.scan_by_prefix("r", "ab") == "ab(3)"
assert store.scan_by_prefix("r", "a") == "a(5), ab(3)"
assert store.scan_by_prefix("r", "") == store.scan("r")
for field in ["beta", "Alpha", "ab", "xab", "a"]:
    store.delete("r", field)
assert store.scan("r") == "" and store.scan_by_prefix("r", "") == ""
"""},
        {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random sequence of calls including scan and scan_by_prefix, a return value differed from a plain dictionary model.",
         "code": _MODEL + r"""
for seed in range(150):
    rng = random.Random(seed)
    replay({fn}(), random_calls(rng, 60, False, False), seed)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
store = {fn}()
assert store.set_at_with_ttl(3, "lamp-2", "mode", "dim", 4) == ""
assert store.set_at_with_ttl(5, "lamp-2", "mode", "bright", 3) == ""
assert store.set_at(6, "lamp-2", "room", "den") == ""
assert store.scan_at(7, "lamp-2") == "mode(bright), room(den)"
assert store.scan_at(8, "lamp-2") == "room(den)"
assert store.delete_at(8, "lamp-2", "mode") == "false"
store.set_at_with_ttl(9, "lamp-2", "mode", "off", 2)
store.set_at(10, "lamp-2", "mode", "auto")
assert store.scan_at(11, "lamp-2") == "mode(auto), room(den)"
assert store.get("lamp-2", "mode") == "auto"
"""},
        {"name": "Part 3: window edges and the shared clock", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A field set with ttl at time t exists at t + ttl - 1 and is gone at t + ttl; rewriting it with a ttl restarts the window from the new time; plain calls run at the clock and move it on by one, and _at calls set the clock.",
         "code": r"""
store = {fn}()
store.set_at_with_ttl(4, "k", "f", "v", 3)
assert store.get_at(6, "k", "f") == "v"
assert store.get_at(7, "k", "f") == ""
assert store.scan_by_prefix_at(7, "k", "") == ""
store.set_at_with_ttl(10, "k", "f", "v", 4)
store.set_at_with_ttl(12, "k", "f", "w", 1)
assert store.get_at(12, "k", "f") == "w"
assert store.get_at(13, "k", "f") == "", "the second ttl replaces the window, it does not extend it"
store.set_at_with_ttl(20, "k", "g", "1", 2)
assert store.get("k", "g") == "1", "a plain call right after an _at call runs at that same time"
assert store.get("k", "g") == "1"
assert store.get("k", "g") == "", "two plain calls later the clock is at 22"
store.set_at_with_ttl(30, "k", "h", "1", 5)
store.set("k", "h", "2")
assert store.get_at(100, "k", "h") == "2", "a plain set makes the field permanent"
store.set("k", "p", "x")
store.set_at_with_ttl(200, "k", "p", "y", 1)
assert store.delete_at(201, "k", "p") == "false"
assert store.get_at(201, "k", "p") == ""
"""},
        {"name": "Part 3: random timed calls", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random mix of plain and timed calls with expiring fields, a return value differed from a model that runs plain calls at the clock and drops a field at timestamp + ttl.",
         "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(seed)
    replay({fn}(), random_calls(rng, 80, True, False), seed)
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "checkpoint.recovery", "code": r"""
store = {fn}()
store.set_at_with_ttl(2, "lamp-2", "mode", "dim", 8)
store.set_at(2, "lamp-2", "room", "den")
assert store.backup(5) == ""
store.set_at(6, "lamp-2", "room", "hall")
assert store.delete_at(7, "lamp-2", "mode") == "true"
assert store.restore(50, 5) == ""
assert store.get_at(50, "lamp-2", "room") == "den"
assert store.get_at(54, "lamp-2", "mode") == "dim"
assert store.get_at(55, "lamp-2", "mode") == ""
"""},
        {"name": "Part 4: choosing and reusing backups", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "restore must pick the latest backup at or before timestamp_to_restore, keep every backup for later restores, replace the whole current state, and leave the backup unchanged by writes made after it is taken or restored.",
         "code": r"""
store = {fn}()
store.set_at(1, "a", "f", "one")
store.backup(2)
store.set_at(3, "a", "f", "two")
store.set_at(3, "b", "g", "new")
store.backup(4)
store.set_at(5, "a", "f", "three")
store.restore(6, 3)
assert store.get_at(6, "a", "f") == "one" and store.scan_at(6, "b") == "", "the latest backup at or before 3 is the one from 2"
store.set_at(7, "a", "f", "dirty")
store.set_at(7, "a", "x", "extra")
store.restore(8, 2)
assert store.scan_at(8, "a") == "f(one)", "writes after a restore must not leak into the backup"
store.restore(9, 100)
assert store.get_at(9, "a", "f") == "two" and store.get_at(9, "b", "g") == "new", "a later restore can still reach the backup from 4"
store.set_at(10, "c", "h", "first")
store.backup(10)
store.set_at(10, "c", "h", "second")
store.backup(10)
store.set_at(11, "c", "h", "third")
store.restore(12, 10)
assert store.get_at(12, "c", "h") == "second", "a second backup at the same time replaces the first"
"""},
        {"name": "Part 4: time left resumes at restore", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "A backup must skip fields already expired at its time and save time left, not an absolute expiry, so a restored field expires at restore time plus the time it had left.",
         "code": r"""
store = {fn}()
store.set_at_with_ttl(0, "k", "gone", "1", 3)
store.set_at_with_ttl(0, "k", "left", "2", 10)
store.set_at(0, "k", "keep", "3")
store.backup(3)
store.restore(1000, 3)
assert store.scan_at(1000, "k") == "keep(3), left(2)"
assert store.get_at(1006, "k", "left") == "2"
assert store.get_at(1007, "k", "left") == ""
store.restore(2000, 3)
assert store.get_at(2006, "k", "left") == "2", "restoring the same backup twice gives the same state"
assert store.get_at(5000, "k", "keep") == "3"
"""},
        {"name": "Part 4: random calls with backups", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "On a random mix of timed calls, backups and restores, a return value differed from a model that saves time left and restarts each countdown at the restore time.",
         "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(seed)
    replay({fn}(), random_calls(rng, 80, True, True), seed)
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
class RecordStore:
    def __init__(self):
        self._records = {}  # key -> field -> (value, expiry); expiry None means permanent
        self._clock = 0
        self._backups = {}  # backup time -> key -> field -> (value, time left or None)

    def _tick(self):
        now = self._clock  # a plain call runs at the clock, then moves it on by one
        self._clock += 1
        return now

    def _move_to(self, timestamp):
        self._clock = max(self._clock, timestamp)  # a plain call has already moved past its own time

    @staticmethod
    def _alive(entry, now):
        return entry is not None and (entry[1] is None or now < entry[1])  # the expiry time itself is already gone

    def _live(self, key, now):
        return {f: e[0] for f, e in self._records.get(key, {}).items() if self._alive(e, now)}

    @staticmethod
    def _format(live, names):
        return ", ".join(f"{name}({live[name]})" for name in names)

    def set(self, key, field, value):
        return self.set_at(self._tick(), key, field, value)

    def get(self, key, field):
        return self.get_at(self._tick(), key, field)

    def delete(self, key, field):
        return self.delete_at(self._tick(), key, field)

    def scan(self, key):
        return self.scan_at(self._tick(), key)

    def scan_by_prefix(self, key, prefix):
        return self.scan_by_prefix_at(self._tick(), key, prefix)

    def set_at(self, timestamp, key, field, value):
        self._move_to(timestamp)
        self._records.setdefault(key, {})[field] = (value, None)  # a write without a ttl clears any expiry
        return ""

    def set_at_with_ttl(self, timestamp, key, field, value, ttl):
        self._move_to(timestamp)
        self._records.setdefault(key, {})[field] = (value, timestamp + ttl)  # a new window, never extended
        return ""

    def get_at(self, timestamp, key, field):
        self._move_to(timestamp)
        entry = self._records.get(key, {}).get(field)
        return entry[0] if self._alive(entry, timestamp) else ""

    def delete_at(self, timestamp, key, field):
        self._move_to(timestamp)
        fields = self._records.get(key, {})
        if self._alive(fields.get(field), timestamp):
            del fields[field]
            return "true"
        return "false"

    def scan_at(self, timestamp, key):
        self._move_to(timestamp)
        live = self._live(key, timestamp)
        return self._format(live, sorted(live))

    def scan_by_prefix_at(self, timestamp, key, prefix):
        self._move_to(timestamp)
        live = self._live(key, timestamp)
        return self._format(live, sorted(f for f in live if f.startswith(prefix)))

    def backup(self, timestamp):
        self._move_to(timestamp)
        snapshot = {}
        for key, fields in self._records.items():
            kept = {f: (v, None if end is None else end - timestamp)  # time left, not an absolute expiry
                    for f, (v, end) in fields.items() if self._alive((v, end), timestamp)}
            if kept:
                snapshot[key] = kept
        self._backups[timestamp] = snapshot
        return ""

    def restore(self, timestamp, timestamp_to_restore):
        self._move_to(timestamp)
        chosen = max(t for t in self._backups if t <= timestamp_to_restore)
        self._records = {key: {f: (v, None if left is None else timestamp + left) for f, (v, left) in fields.items()}
                         for key, fields in self._backups[chosen].items()}  # new dicts, so the backup stays intact
        return ""
''',
    "interview_questions": interview(
        concept=[
            "Why does a dict of dicts fit this store better than one dict keyed by (key, field) pairs?",
            "Why does every method return a string, and what should get and delete return when nothing is there?",
        ],
        deep_dive=[
            "What can go wrong if delete removes a field without first checking that it exists?",
        ],
        tradeoffs=[
            "How would you keep scan fast if records held millions of fields and scans were frequent?",
            "Lazy expiry checks a field when it is read; what does that cost in memory, and when would you add an active sweep?",
            "Why must a backup store time left instead of an absolute expiry, and what breaks if it stores the expiry?",
            "How would you make backups cheaper than a full copy each time?",
        ],
    ),
}
