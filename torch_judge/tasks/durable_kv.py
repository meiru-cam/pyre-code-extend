"""A string key-value store that persists to a simulated disk: own format, crash-safe saves, then a log."""

from ._interview import interview

# The simulated disk the learner's store is handed, plus crash and counting variants of it.
_FS = r"""
import random, time, types

class FileSystem:
    def __init__(self, max_file_size=None):
        self.max_file_size = max_file_size
        self._files = {}
    def _check(self, name, size):
        if self.max_file_size is not None and size > self.max_file_size:
            raise ValueError(f"{name}: {size} bytes exceeds max_file_size={self.max_file_size}")
    def write(self, name, data):
        data = bytes(data)
        self._check(name, len(data))
        self._files[name] = bytearray(data)
    def append(self, name, data):
        data = bytes(data)
        old = self._files.get(name, bytearray())
        self._check(name, len(old) + len(data))
        old += data
        self._files[name] = old
    def read(self, name):
        data = self._files.get(name)
        return None if data is None else bytes(data)
    def list(self):
        return list(self._files)
    def delete(self, name):
        self._files.pop(name, None)
    def total_size(self):
        return sum(len(data) for data in self._files.values())

class Crash(BaseException):
    pass

class CrashingFileSystem(FileSystem):
    # The process dies just before mutating call number crash_at (write, append or delete), counted from 0.
    def __init__(self, max_file_size=None):
        super().__init__(max_file_size)
        self.calls, self.crash_at = 0, None
    def _tick(self):
        if self.calls == self.crash_at:
            raise Crash()
        self.calls += 1
    def write(self, name, data):
        self._tick()
        super().write(name, data)
    def append(self, name, data):
        self._tick()
        super().append(name, data)
    def delete(self, name):
        self._tick()
        super().delete(name)

WEIRD = ["", "a", "a:b", "a,b", "a=b", "key\nwith\nnewlines", "\x00null\x00byte", "\x00" * 12, "\\", '"q"',
         "emoji \U0001F600 \U0001F4A9", "long" * 3000, "mixed 中文 and English", ":,=\n\x00", chr(0xD800), "x" + chr(0xDFFF)]
SMALL = [w for w in WEIRD if len(w) < 50]

def reopen(fs, **kwargs):
    store = {fn}(fs, **kwargs)
    store.load()
    return store

def contents(store, keys):
    return {k: store.get(k) for k in keys if store.get(k) is not None}
"""

TASK = {
    "title": "Durable Key-Value Store",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "KVStore",
    "description_en": r"""Build `KVStore(fs)`, a store of string keys and string values that persists itself to `fs`, a simulated disk. A new `KVStore` on the same `fs` must be able to recover the data.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `KVStore` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**The disk `fs`** is built by the tests; your code only calls it:
- `fs.write(name, data)` creates or replaces the file `name` with the bytes `data`. It is atomic: a crash leaves the file either untouched or fully replaced.
- `fs.read(name)` returns the file's bytes, or `None` if it does not exist. `fs.list()` returns every file name. `fs.delete(name)` removes a file, atomically, and ignores a missing one.
- A new process is modelled by a new `KVStore` on the same `fs`, followed by `load()`.

**Rules for every part:**
- Keys and values are any `str`: empty, containing `:`, `,`, `=`, newlines or `\x00`, emoji, even a lone surrogate such as `chr(0xD800)`. All of them round-trip unchanged.
- Design the byte format yourself. Do not use `json`, `pickle`, `marshal`, `shelve`, `eval` or `ast.literal_eval`. `str.encode`, `bytes.decode`, `int.to_bytes`, `int.from_bytes` and `zlib` are allowed.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** a format that works for ordinary strings breaks on the first key that contains its delimiter, and each later part adds one requirement.

**Where it is used:** every database and configuration store needs a byte format it can always parse back, and a way to recover after the process dies.

Adapted from the durable KV store question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Save and load",
            "description_en": r"""**Signature:** `KVStore(fs)`, `put(key, value) -> None`, `get(key) -> str | None`, `save() -> None`, `load() -> None`

- `put` sets `key` to `value` in memory, replacing any earlier value. `get` returns the value, or `None` if the key has none.
- A new `KVStore` starts empty.
- `save()` writes the whole in-memory store to `fs`.
- `load()` replaces the in-memory store with what the last `save()` on this `fs` wrote, from any instance. If nothing was ever saved, the store becomes empty.
- A later `save()` fully replaces an earlier one.

**Example:**
- `put("a=b", "x,y")`, `put("", "nothing")`, `put("poem", "line one\nline two ✓")`, then `save()`
- `store2 = KVStore(fs)`: `store2.get("a=b")` is `None`, because `load()` has not run yet
- after `store2.load()`: `"x,y"`, `"nothing"` and `"line one\nline two ✓"` for those keys, `None` for `"poem "` or any other key""",
        },
        {
            "title": "Size-capped files and interrupted saves",
            "description_en": r"""Keep Part 1. Now `fs.max_file_size` may be set, to any value of `64` or more. It is `None` or a limit in bytes, and `fs.write` raises `ValueError` and changes nothing when a file would exceed it.

- A single key or value longer than `fs.max_file_size` must still round-trip.
- The process can die between any two `fs.write` or `fs.delete` calls of a `save()`, or before the first. Then `load()` on a fresh instance returns one of two stores, complete: the one being saved, or the one saved before it (empty if there was none). It must not raise, and must not combine pieces of both.
- After a `save()` completes, every file on `fs` is one that `load()` reads. Files from older saves, including saves that died partway, are gone: `fs` holds as many files as saving the same store to an empty disk would.

**Example:** `fs = FileSystem(max_file_size=64)`; 30 keys `key0` to `key29`, each with a value of up to 36 characters, add up to far more than 64 bytes. `save()` must spread them over several files, none over 64 bytes, and `load()` on a fresh instance gets all 30 back.""",
        },
        {
            "title": "Append-only log",
            "description_en": r"""Keep Parts 1–2 and add a log mode. `KVStore(fs)` with no flag behaves exactly as before.

**Signature:** `KVStore(fs, log=False)`, `delete(key) -> None`, `compact() -> None`

- `fs.append(name, data)` adds `data` to the end of `name`, creating it if absent. It is not atomic: a crash can leave any prefix of `data` behind, and those bytes may be damaged.

- `delete(key)` removes `key`: `get(key)` returns `None` until it is `put` again. A missing key is fine. Without `log=True`, the removal is in memory until the next `save()`.
- With `log=True`, `put` and `delete` are on disk by the time they return. Each one calls `fs.append` once and makes no other `fs` call, and how many bytes it appends is fixed by its key and value alone, not by what is already stored. `get` never calls `fs`. `save()` is not used.
- With `log=True`, `load()` replays a log of `N` operations in `O(N)` time.
- If the process dies inside an `fs.append`, the file ends in a damaged fragment. The next `load()` returns the store as it was after the last operation that completed, ignores the fragment entirely, and does not raise. A `put` or `delete` made after that `load()` is still there after a further `load()`.
- `compact()` rewrites the stored data. Afterwards, two stores holding the same keys and values occupy the same number of bytes on `fs`, however many overwrites and deletes led there, and `load()` still gives the same store. If the process dies during `compact()`, nothing is lost.
- Tests of `log=True` use an `fs` with no `max_file_size`.

**Example:** with `log=True`, `put("a", "1")`, `put("b", "2")`, `delete("a")`, `put("", "")`; a new `KVStore(fs, log=True)` returns `None`, `"2"` and `""` for `"a"`, `"b"` and `""` after `load()`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What goes wrong with key + \":\" + value + \"\\n\" when a key contains \":\" or a newline? If the reader knew how long each string is before reading it, would it ever need to search for a separator? Is that length counted in characters or in bytes, and does it matter for an emoji?"},
        {"level": 2, "kind": "analysis", "content": "Encode each string as its UTF-8 bytes with a fixed 8-byte length in front (int.to_bytes(8, \"big\")), using the \"surrogatepass\" error handler so a lone surrogate survives. A store is key, value, key, value. Reading takes 8 bytes, then that many bytes, and never looks for a delimiter. save() writes the whole blob with one fs.write; load() treats a missing file as an empty store."},
    ],
    "model_connections": [
        "Checkpoint files for model weights and optimizer state use length-prefixed records with checksums so a reader can tell a complete file from a torn one.",
        "Write-ahead logs in databases and in training job schedulers recover after a crash by replaying records up to the last valid one.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A length prefix makes every string parseable without escaping, so no character is special.",
            "Writing new files first and switching to them with one atomic write makes a multi-file save all-or-nothing.",
            "An append-only log makes each update cost one small append, independent of the store size.",
        ],
        "cons": [
            "A fixed 8-byte length adds 8 bytes to every string; a varint would shrink short ones.",
            "Rewriting the whole store on each save costs time proportional to its size, however little changed.",
            "A log grows with every overwrite and delete until it is compacted, and replay time grows with it.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _FS + r"""
fs = FileSystem()
store = {fn}(fs)
store.put("a=b", "x,y")
store.put("", "nothing")
store.put("poem", "line one\nline two \u2713")
store.save()
fresh = {fn}(fs)
assert fresh.get("a=b") is None, "a new store starts empty until load()"
fresh.load()
assert fresh.get("a=b") == "x,y"
assert fresh.get("") == "nothing"
assert fresh.get("poem") == "line one\nline two \u2713"
assert fresh.get("poem ") is None
assert fresh.get("other") is None
"""},
        {"name": "Part 1: awkward strings round-trip", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A key or value containing a delimiter-like character, a null byte, an emoji, a lone surrogate or a long run did not come back unchanged after save() and load().",
         "code": _FS + r"""
for seed in range(200):
    rng = random.Random(seed)
    pairs = {}
    fs = FileSystem()
    store = {fn}(fs)
    for _ in range(rng.randint(0, 12)):
        key, value = rng.choice(WEIRD), rng.choice(WEIRD)
        store.put(key, value)
        pairs[key] = value
    store.save()
    assert contents(reopen(fs), WEIRD) == pairs, seed
for text in WEIRD:
    fs = FileSystem()
    store = {fn}(fs)
    store.put(text, text[::-1])
    store.put(text + "!", text)
    store.save()
    back = reopen(fs)
    assert back.get(text) == text[::-1] and back.get(text + "!") == text, repr(text[:20])
"""},
        {"name": "Part 1: empty disks, unsaved changes and later saves", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "load() must give an empty store when nothing was saved, discard unsaved puts, and reproduce only the latest save(), from whichever instance made it.",
         "code": _FS + r"""
fs = FileSystem()
assert reopen(fs).get("x") is None
store = {fn}(fs)
store.put("unsaved", "1")
store.load()
assert store.get("unsaved") is None, "nothing was saved, so load() empties the store"
first = {fn}(fs)
first.put("a", "1")
first.put("b", "2")
first.save()
second = {fn}(fs)
second.put("a", "3")
second.save()
back = reopen(fs)
assert back.get("a") == "3" and back.get("b") is None, "the later save replaces the earlier one"
back.put("c", "4")
back.load()
assert back.get("c") is None and back.get("a") == "3"
{fn}(fs).save()
assert reopen(fs).get("a") is None, "saving an empty store leaves an empty store"
again = {fn}(fs)
again.put("k", "v")
again.save()
again.save()
assert reopen(fs).get("k") == "v"
"""},
        {"name": "Part 1: no ready-made serializer", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "The store must define its own byte format: json, pickle, marshal, shelve, eval and ast.literal_eval are not allowed.",
         "code": _FS + r"""
BANNED = {"json", "pickle", "_pickle", "marshal", "shelve", "eval", "literal_eval"}
BANNED_MODULES = {"json", "pickle", "_pickle", "marshal", "shelve", "ast"}

def names_in(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= names_in(const)
    return names

def functions_of(value):
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, property):
        return [f for f in (value.fget, value.fset, value.fdel) if f is not None]
    if isinstance(value, types.FunctionType):
        return [value]
    if isinstance(value, type):
        return [f for attr in vars(value).values() for f in functions_of(attr)]
    return []

fs = FileSystem()
store = {fn}(fs)
store.put("k:1", "v\n")
store.save()
assert reopen(fs).get("k:1") == "v\n", "the store must round-trip before its format is checked"

own = [f for f in functions_of({fn}) if f.__code__.co_filename == "<submitted-solution>"]
assert own, "KVStore defines no methods"
used = set()
for value in list(own[0].__globals__.values()) + [{fn}]:
    if isinstance(value, types.ModuleType):
        used.add(value.__name__.split(".")[0])
    elif callable(value) and (getattr(value, "__module__", None) or "").split(".")[0] in BANNED_MODULES:
        used.add(value.__module__)
    for f in functions_of(value):
        if f.__code__.co_filename == "<submitted-solution>":
            used |= names_in(f.__code__)
assert not used & (BANNED | BANNED_MODULES), sorted(used & (BANNED | BANNED_MODULES))
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _FS + r"""
fs = FileSystem(max_file_size=64)
store = {fn}(fs)
pairs = {f"key{i}": "value-" * (i % 7) for i in range(30)}
for key, value in pairs.items():
    store.put(key, value)
store.save()
assert len(fs.list()) > 1, "several hundred bytes cannot fit in one 64-byte file"
assert all(len(fs.read(name)) <= 64 for name in fs.list())
assert contents(reopen(fs), pairs) == pairs
"""},
        {"name": "Part 2: every cap, long values and repeated saves", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With any max_file_size of 64 or more, including values longer than the cap and stores that shrink or empty between saves, load() must return exactly the last save.",
         "code": _FS + r"""
for cap in [64, 65, 97, 128, 1000, 4096, None]:
    rng = random.Random(cap or 0)
    fs = FileSystem(max_file_size=cap)
    store = {fn}(fs)
    pairs = {}
    for i in range(25):
        key = f"key_{i}_" + "x" * rng.randint(0, 150)
        pairs[key] = rng.choice(WEIRD) if i % 3 == 0 else "y" * rng.randint(0, 300)
    pairs["huge"] = "\U0001F600" * 2000
    for key, value in pairs.items():
        store.put(key, value)
    store.save()
    assert contents(reopen(fs), list(pairs) + ["missing"]) == pairs, cap
    shrunk = {"only": "one"}
    other = {fn}(fs)
    other.put("only", "one")
    other.save()
    assert contents(reopen(fs), list(pairs) + ["only"]) == shrunk, cap
    {fn}(fs).save()
    assert contents(reopen(fs), list(pairs) + ["only"]) == {}, cap
for key_len in range(30, 70):
    fs = FileSystem(max_file_size=64)
    store = {fn}(fs)
    store.put("k" * key_len, "\U0001F600" * 3)
    store.save()
    assert reopen(fs).get("k" * key_len) == "\U0001F600" * 3, key_len
"""},
        {"name": "Part 2: a crash at any point of a save", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "After a save() interrupted before any of its write or delete calls, load() must return exactly the previous store or exactly the new one, without raising, and the next completed save() must leave no extra files.",
         "code": _FS + r"""
def attempt(fs, state, crash_at):
    store = {fn}(fs)
    for key, value in state.items():
        store.put(key, value)
    fs.calls, fs.crash_at = 0, crash_at
    try:
        store.save()
        return True
    except Crash:
        return False
    finally:
        fs.crash_at = None

def baseline(state):
    fs = FileSystem(max_file_size=64)
    store = {fn}(fs)
    for key, value in state.items():
        store.put(key, value)
    store.save()
    return len(fs.list()), fs.total_size()

S0 = {f"k{i}": "v" * 30 for i in range(5)}
S1 = {"big": "z" * 300, "\U0001F600": "", "k0": "new"}
S2 = {"tiny": "t"}
KEYS = set(S0) | set(S1) | set(S2)
OPTIONS = [S0, S1, S2, {}]

first, done1, scenarios = 0, False, 0
while not done1:
    fs = CrashingFileSystem(max_file_size=64)
    if first % 2:
        assert attempt(fs, S0, None)
    before = S0 if first % 2 else {}
    done1 = attempt(fs, S1, first // 2)
    after1 = contents(reopen(fs), KEYS)
    assert after1 in (before, S1), (first, after1)
    if done1:
        assert after1 == S1
    second, done2 = 0, False
    while not done2 and second < 400:
        trial = CrashingFileSystem(max_file_size=64)
        trial._files = {name: bytearray(data) for name, data in fs._files.items()}
        done2 = attempt(trial, S2, second)
        after2 = contents(reopen(trial), KEYS)
        assert after2 in (after1, S2), (first, second, after2)
        assert attempt(trial, S0, None)
        assert contents(reopen(trial), KEYS) == S0
        count, size = baseline(S0)
        assert len(trial.list()) == count and trial.total_size() <= size + 16, "files left over from an interrupted save"
        scenarios, second = scenarios + 1, second + 1
    assert done2, "save() never finished"
    first += 1
    assert first < 800, "save() never finished"
assert scenarios > 20
"""},
        {"name": "Part 2: no leftover files", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After save() returns, fs must hold only the files load() needs for that save: as many files and about as many bytes as saving the same store to an empty disk.",
         "code": _FS + r"""
def baseline(state, cap):
    fs = FileSystem(max_file_size=cap)
    store = {fn}(fs)
    for key, value in state.items():
        store.put(key, value)
    store.save()
    return len(fs.list()), fs.total_size()

rng = random.Random(7)
for cap in [64, 200, None]:
    fs = FileSystem(max_file_size=cap)
    for round_ in range(12):
        state = {f"k{i}": "v" * rng.randint(0, 120) for i in range(rng.choice([0, 1, 3, 20, 40]))}
        store = {fn}(fs)
        for key, value in state.items():
            store.put(key, value)
        store.save()
        count, size = baseline(state, cap)
        assert len(fs.list()) == count, (cap, round_, sorted(fs.list()))
        assert fs.total_size() <= size + 16, (cap, round_)
        assert contents(reopen(fs), [f"k{i}" for i in range(40)]) == state
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": _FS + r"""
fs = FileSystem()
store = {fn}(fs, log=True)
store.put("a", "1")
store.put("b", "2")
store.delete("a")
store.put("", "")
store.delete("missing")
assert store.get("a") is None and store.get("b") == "2" and store.get("") == ""
again = reopen(fs, log=True)
assert again.get("a") is None
assert again.get("b") == "2"
assert again.get("") == ""
"""},
        {"name": "Part 3: random operations survive reloads", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With log=True, a sequence of puts and deletes, with reloads by new instances in between, ended in a store different from applying the same operations to a dict.",
         "code": _FS + r"""
for seed in range(120):
    rng = random.Random(seed)
    fs = FileSystem()
    store, model = {fn}(fs, log=True), {}
    for step in range(40):
        key = rng.choice(SMALL)
        if rng.random() < 0.3:
            store.delete(key)
            model.pop(key, None)
        else:
            model[key] = rng.choice(SMALL)
            store.put(key, model[key])
        assert contents(store, SMALL) == model, (seed, step)
        if rng.random() < 0.15:
            store = reopen(fs, log=True)
            assert contents(store, SMALL) == model, (seed, step)
    assert contents(reopen(fs, log=True), SMALL) == model, seed
"""},
        {"name": "Part 3: delete without the log", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Without log=True, delete must remove the key from memory at once and from the disk at the next save(), and nothing else.",
         "code": _FS + r"""
fs = FileSystem(max_file_size=64)
store = {fn}(fs)
store.put("a", "1")
store.put("b", "2")
store.save()
store.delete("a")
store.delete("never")
assert store.get("a") is None and store.get("b") == "2"
assert reopen(fs).get("a") == "1", "not saved yet"
store.save()
back = reopen(fs)
assert back.get("a") is None and back.get("b") == "2"
store.put("a", "again")
assert store.get("a") == "again"
"""},
        {"name": "Part 3: a torn append is discarded", "part": 3, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "When the last append was cut short, zeroed or had a byte damaged, load() must restore exactly the operations before it, without raising, and a put made after that load() must survive the next one.",
         "code": _FS + r"""
class TearingFileSystem(FileSystem):
    # Append number tear_at keeps only damage(data) and the process dies.
    def __init__(self):
        super().__init__()
        self.sizes, self.tear_at, self.damage = [], None, None
    def append(self, name, data):
        data = bytes(data)
        if len(self.sizes) == self.tear_at:
            super().append(name, self.damage(data))
            raise Crash()
        self.sizes.append(len(data))
        super().append(name, data)

rng = random.Random(3)
ops = []
for _ in range(12):
    key = rng.choice(SMALL)
    ops.append(("delete", (key,)) if rng.random() < 0.3 else ("put", (key, rng.choice(SMALL))))
ops[0] = ("put", ("first", "value"))
KEYS = SMALL + ["first", "after"]

dry = TearingFileSystem()
store, model, states = {fn}(dry, log=True), {}, [{}]
for name, args in ops:
    getattr(store, name)(*args)
    if name == "put":
        model[args[0]] = args[1]
    else:
        model.pop(args[0], None)
    states.append(dict(model))
assert len(dry.sizes) == len(ops), "each put and delete makes exactly one append"

def damages(size):
    for cut in range(size):
        yield lambda data, cut=cut: data[:cut]
    for cut in range(1, size + 1):
        yield lambda data, cut=cut: bytes(cut)
    for i in range(size):
        yield lambda data, i=i: data[:i] + bytes([data[i] ^ 0xFF]) + data[i + 1:]

checked = 0
for k in range(len(ops)):
    for damage in damages(dry.sizes[k]):
        fs = TearingFileSystem()
        fs.tear_at, fs.damage = k, damage
        store = {fn}(fs, log=True)
        try:
            for name, args in ops:
                getattr(store, name)(*args)
            raise AssertionError("the tear did not happen")
        except Crash:
            pass
        fs.tear_at = None
        recovered = reopen(fs, log=True)
        assert contents(recovered, KEYS) == states[k], (k, checked)
        recovered.put("after", "recovery")
        assert contents(reopen(fs, log=True), KEYS) == {**states[k], "after": "recovery"}, (k, checked)
        checked += 1
assert checked > 300
"""},
        {"name": "Part 3: one append per operation", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "With log=True, each put and delete must make exactly one fs.append, of a size that depends only on its own key and value, and get must not touch fs.",
         "code": _FS + r"""
class CountingFileSystem(FileSystem):
    def __init__(self):
        super().__init__()
        self.calls = []
    def write(self, name, data):
        self.calls.append(("write", len(data)))
        super().write(name, data)
    def append(self, name, data):
        self.calls.append(("append", len(data)))
        super().append(name, data)
    def read(self, name):
        self.calls.append(("read", 0))
        return super().read(name)
    def list(self):
        self.calls.append(("list", 0))
        return super().list()
    def delete(self, name):
        self.calls.append(("delete", 0))
        super().delete(name)

fs = CountingFileSystem()
store = {fn}(fs, log=True)
put_sizes, delete_sizes = set(), set()
for i in range(3000):
    fs.calls.clear()
    store.put(f"k{i % 50:02d}", f"v{i % 10}")
    assert [c[0] for c in fs.calls] == ["append"], (i, fs.calls)
    put_sizes.add(fs.calls[0][1])
    fs.calls.clear()
    assert store.get(f"k{i % 50:02d}") == f"v{i % 10}"
    assert fs.calls == [], "get must not touch fs"
    if i % 3 == 0:
        fs.calls.clear()
        store.delete(f"k{(i + 7) % 50:02d}")
        assert [c[0] for c in fs.calls] == ["append"], (i, fs.calls)
        delete_sizes.add(fs.calls[0][1])
assert len(put_sizes) == 1, f"put sizes varied with the history: {sorted(put_sizes)}"
assert len(delete_sizes) == 1, f"delete sizes varied with the history: {sorted(delete_sizes)}"
"""},
        {"name": "Part 3: load is linear in the log", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Replaying 10x as many log records took far more than 10x as long; load() must read the log once, without re-slicing or re-reading it per record.",
         "code": _FS + r"""
def build(n):
    fs = FileSystem()
    store = {fn}(fs, log=True)
    for i in range(n):
        store.put(f"k{i % 500}", f"value {i}")
        if i % 5 == 0:
            store.delete(f"k{(i * 7) % 500}")
    return fs, {f"k{j}": store.get(f"k{j}") for j in range(500)}

def best_of_three(fs, expected):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        store = reopen(fs, log=True)
        best = min(best, time.perf_counter() - start)
    assert {k: store.get(k) for k in expected} == expected
    return best

small, big = build(1500), build(15000)
ratio = best_of_three(*big) / best_of_three(*small)
assert ratio < 40, f"10x the records took {ratio:.0f}x as long to load"
"""},
        {"name": "Part 3: compaction", "part": 3, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "After compact(), the stored size must depend only on the current keys and values and load() must give the same store; a crash at any point of compact() must lose nothing.",
         "code": _FS + r"""
FINAL = {"x": "final", "y": "keep", "": ""}
KEYS = ["x", "y", "", "after"] + [f"tmp{i}" for i in range(7)]

def history(fs, n):
    store = {fn}(fs, log=True)
    for i in range(n):
        store.put("x", str(i))
        store.put(f"tmp{i % 7}", "y" * 20)
        store.delete(f"tmp{i % 7}")
    store.put("x", "final")
    store.put("y", "keep")
    store.put("", "")
    return store

short, long_ = FileSystem(), FileSystem()
a, b = history(short, 3), history(long_, 400)
grown = long_.total_size()
a.compact()
b.compact()
assert contents(b, KEYS) == FINAL
assert long_.total_size() < grown / 20, "compact() did not shrink the log"
assert long_.total_size() == short.total_size(), "the compacted size still depends on the history"
assert contents(reopen(long_, log=True), KEYS) == FINAL
b.put("after", "compact")
b.delete("y")
assert contents(reopen(long_, log=True), KEYS) == {"x": "final", "": "", "after": "compact"}

crash_at, done = 0, False
while not done:
    fs = CrashingFileSystem()
    store = history(fs, 30)
    fs.calls, fs.crash_at = 0, crash_at
    try:
        store.compact()
        done = True
    except Crash:
        pass
    fs.crash_at = None
    assert contents(reopen(fs, log=True), KEYS) == FINAL, crash_at
    crash_at += 1
    assert crash_at < 200, "compact() never finished"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import zlib

PUT, DELETE = 1, 0


def _encode_str(s):
    raw = s.encode("utf-8", "surrogatepass")  # a str may hold a lone surrogate such as chr(0xD800)
    return len(raw).to_bytes(8, "big") + raw  # the length counts bytes, not characters


def _read_str(data, pos):
    end = pos + 8 + int.from_bytes(data[pos:pos + 8], "big")
    return data[pos + 8:end].decode("utf-8", "surrogatepass"), end


def _serialize(data):
    return b"".join(_encode_str(k) + _encode_str(v) for k, v in data.items())


def _deserialize(blob):
    store, pos = {}, 0
    while pos < len(blob):
        key, pos = _read_str(blob, pos)
        store[key], pos = _read_str(blob, pos)
    return store


def _encode_record(op, key, value=""):
    payload = bytes([op]) + _encode_str(key) + (_encode_str(value) if op == PUT else b"")
    header = len(payload).to_bytes(8, "big")
    return header + zlib.crc32(header + payload).to_bytes(4, "big") + payload  # the checksum covers the length


def _read_record(data, pos):
    """Returns (op, key, value, end), or None if no complete, valid record starts at pos."""
    if pos + 12 > len(data):
        return None
    end = pos + 12 + int.from_bytes(data[pos:pos + 8], "big")
    if end > len(data):
        return None  # cut off, or a damaged (huge) length
    payload = data[pos + 12:end]
    if zlib.crc32(data[pos:pos + 8] + payload) != int.from_bytes(data[pos + 8:pos + 12], "big"):
        return None
    key, kpos = _read_str(payload, 1)
    value = _read_str(payload, kpos)[0] if payload[0] == PUT else ""
    return payload[0], key, value, end


class KVStore:
    MANIFEST = "manifest"
    LOG = "log"

    def __init__(self, fs, log=False):
        self.fs = fs
        self.log = log
        self._data = {}

    def put(self, key, value):
        if self.log:
            self.fs.append(self.LOG, _encode_record(PUT, key, value))  # disk first, memory second
        self._data[key] = value

    def delete(self, key):
        if self.log:
            self.fs.append(self.LOG, _encode_record(DELETE, key))
        self._data.pop(key, None)

    def get(self, key):
        return self._data.get(key)

    def save(self):
        old = self.fs.read(self.MANIFEST)
        generation = int.from_bytes(old[:8], "big") + 1 if old is not None else 0  # never the live one
        blob = _serialize(self._data)
        cap = self.fs.max_file_size or max(len(blob), 1)
        names = set()
        for i in range(0, len(blob), cap):  # a cut may fall inside a length or a character
            name = f"chunk_{generation}_{len(names)}"
            self.fs.write(name, blob[i:i + cap])
            names.add(name)
        # The commit point: one atomic write switches load() to the new generation.
        self.fs.write(self.MANIFEST, generation.to_bytes(8, "big") + len(names).to_bytes(8, "big"))
        for name in self.fs.list():  # also chunks of interrupted saves, not only the previous one
            if name.startswith("chunk_") and name not in names:
                self.fs.delete(name)

    def load(self):
        if self.log:
            self._replay()
            return
        manifest = self.fs.read(self.MANIFEST)
        if manifest is None:
            self._data = {}
            return
        generation, count = int.from_bytes(manifest[:8], "big"), int.from_bytes(manifest[8:16], "big")
        self._data = _deserialize(b"".join(self.fs.read(f"chunk_{generation}_{i}") for i in range(count)))

    def _replay(self):
        data = self.fs.read(self.LOG) or b""
        store, pos = {}, 0
        while (record := _read_record(data, pos)) is not None:  # stop at the first bad record
            op, key, value, pos = record
            if op == PUT:
                store[key] = value
            else:
                store.pop(key, None)
        if pos < len(data):  # cut the bad tail off, or later appends land behind it
            self.fs.write(self.LOG, data[:pos])
        self._data = store

    def compact(self):
        self.fs.write(self.LOG, b"".join(_encode_record(PUT, k, v) for k, v in self._data.items()))
''',
    "interview_questions": interview(
        concept=[
            "Why does a format like key + \":\" + value + newline break, and what does a length prefix change?",
            "Should a string's length prefix count characters or bytes, and what goes wrong with the other choice?",
        ],
        deep_dive=[
            "Which strings fail to encode with plain UTF-8, and how do you make them round-trip?",
        ],
        tradeoffs=[
            "Only single writes are atomic. How do you make a save that spans several files all-or-nothing?",
            "Which files can an interrupted save leave behind, and how does the next save find and remove them?",
            "How does a log record let load() tell a complete record from a torn or damaged one, and why should the checksum cover the length?",
            "Why must load() cut a damaged tail off the log instead of just skipping it?",
        ],
    ),
}
