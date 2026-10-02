Worth confirming first: which `fs` calls are atomic, since that decides what a crash can leave behind, and that `get` on a missing key returns `None` rather than raising.

### Part 1

A delimiter-based format such as `key + ":" + value + "\n"` is ambiguous the moment a key or value contains the delimiter. Escaping fixes that in principle, but only if the escape character is itself escaped everywhere it occurs, which is easy to get subtly wrong. A *length-prefixed* encoding removes the problem: every string is preceded by the number of bytes it occupies, so reading never searches the data for a separator and no byte sequence is forbidden. The code puts a fixed 8-byte big-endian length in front of every string.

```python
def _encode_str(s):
    raw = s.encode("utf-8", "surrogatepass")  # a str may hold a lone surrogate such as "\ud800"
    return len(raw).to_bytes(8, "big") + raw  # NOTE: the length counts BYTES; len(s) would count code points


def _read_str(data, pos):
    length = int.from_bytes(data[pos:pos + 8], "big")
    raw = data[pos + 8:pos + 8 + length]
    return raw.decode("utf-8", "surrogatepass"), pos + 8 + length


def _serialize_store(data):
    return b"".join(_encode_str(k) + _encode_str(v) for k, v in data.items())


def _deserialize_store(blob):
    store, pos = {}, 0
    while pos < len(blob):
        key, pos = _read_str(blob, pos)
        value, pos = _read_str(blob, pos)
        store[key] = value
    return store


class KVStore:
    FILE_NAME = "store"

    def __init__(self, fs):
        self.fs = fs
        self._data = {}

    def put(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def save(self):
        self.fs.write(self.FILE_NAME, _serialize_store(self._data))  # NOTE: one write replaces the whole file

    def load(self):
        blob = self.fs.read(self.FILE_NAME)
        self._data = _deserialize_store(blob) if blob is not None else {}
```

### Part 2

The serialized bytes of Part 1 know nothing about files, and the simplest way to respect the cap is to keep it that way: serialize first, then slice the byte string at fixed offsets. A key or value longer than `max_file_size` needs no special case, because the slicing never looks at record boundaries; `load()` joins the pieces and hands them to the `_deserialize_store` of Part 1.

What needs care is the interrupted `save()`. If a save overwrote `chunk_0`, `chunk_1`, … in place, dying halfway would leave new chunks next to old ones, and `load()` would rebuild a store that never existed. So every chunk name carries a *generation* number, and a small *manifest* file records which generation is current, together with its chunk count. `save()` reads the current generation $g$ from the manifest, writes all its chunks under $g + 1$, then writes the manifest, and only then deletes. Every call on `fs` is atomic, so a crash falls between two calls, and there are two cases. Before the manifest write, the manifest still names generation $g$, whose files this save has not touched: `load()` returns the previous save. After it, every chunk the new manifest names is already in place: `load()` returns the new save, however far the deletions got. The manifest write is the commit point.

An interrupted save leaves chunk files that no manifest names, and the next save uses the same number $g + 1$ again. That cannot mislead `load()`, because a save rewrites every chunk its manifest will name before it commits; but leftovers with higher indices would stay forever, and so would chunks of generation $g$ if the crash came during the deletions. The cleanup therefore deletes every chunk file outside the new generation, not only those of generation $g$.

The manifest also stores the length and the CRC-32 of the serialized bytes. No crash can make them disagree with the chunks; they guard against a chunk that something else deleted or altered, for which `load()` raises `CorruptStoreError` instead of returning a silently wrong store. The manifest is four 8-byte integers, 32 bytes: with a `max_file_size` below 32 its write raises `ValueError`, which `save()` lets propagate. That happens before the commit point, so `load()` still returns the previous save.

```python
import zlib


class CorruptStoreError(RuntimeError):
    pass


def _encode_manifest(*fields):  # generation, num_chunks, total_length, checksum
    return b"".join(field.to_bytes(8, "big") for field in fields)


def _decode_manifest(data):
    return [int.from_bytes(data[i:i + 8], "big") for i in range(0, 32, 8)]


class ChunkedKVStore(KVStore):
    MANIFEST = "manifest"

    def save(self):
        old = self.fs.read(self.MANIFEST)
        generation = _decode_manifest(old)[0] + 1 if old is not None else 0  # NOTE: never the live generation

        blob = _serialize_store(self._data)
        cap = self.fs.max_file_size or len(blob)  # no cap: a single chunk
        chunks = [blob[i:i + cap] for i in range(0, len(blob), cap)] if blob else []
        names = [f"chunk_{generation}_{i}" for i in range(len(chunks))]
        for name, chunk in zip(names, chunks):  # NOTE: a cut may fall inside a length header or a character;
            self.fs.write(name, chunk)          # load() re-joins the chunks before parsing
        # NOTE: the commit point -- one atomic write switches load() to the new generation
        self.fs.write(self.MANIFEST, _encode_manifest(generation, len(chunks), len(blob), zlib.crc32(blob)))

        keep = set(names)
        for name in self.fs.list():  # NOTE: not only the previous generation -- interrupted saves leave others
            if name.startswith("chunk_") and name not in keep:
                self.fs.delete(name)

    def load(self):
        manifest = self.fs.read(self.MANIFEST)
        if manifest is None:
            self._data = {}
            return
        generation, num_chunks, total_length, checksum = _decode_manifest(manifest)
        blob = b"".join(self.fs.read(f"chunk_{generation}_{i}") or b"" for i in range(num_chunks))
        if len(blob) != total_length or zlib.crc32(blob) != checksum:
            raise CorruptStoreError("chunk contents do not match the manifest")
        self._data = _deserialize_store(blob)
```

### Part 3

Each `put` or `delete` appends one *record* to a single log file, and `load()` replays the records in order. A deletion is a record type of its own (a *tombstone*), not a put of an empty value, because `put(key, "")` is a legal and different operation.

A torn append leaves a bad tail, and the record layout — length, checksum, payload — has to expose it. A tail that is too short is caught by comparing the announced length with the bytes actually present; a length field damaged into a huge number fails the same comparison. A tail that is long enough but wrong is caught by the CRC-32. The checksum covers the length field as well as the payload: otherwise twelve zero bytes, a typical torn tail, would pass both checks as a record of length 0 (`crc32(b"") == 0`), and parsing its empty payload would raise. Replay stops at the first bad record, because nothing after it can be located reliably. `load()` then cuts the bad tail off; if it stayed, later records would be appended behind it and no `load()` would ever reach them.

`compact()` writes one put record per live key; tombstones are no longer needed, since the new log holds no older record for them to cancel. `fs.write` is atomic, so an interrupted `compact()` leaves the old log in place.

```python
PUT, DELETE = 1, 0


def _encode_record(op, key, value=""):
    payload = bytes([op]) + _encode_str(key) + (_encode_str(value) if op == PUT else b"")
    header = len(payload).to_bytes(8, "big")
    return header + zlib.crc32(header + payload).to_bytes(4, "big") + payload  # NOTE: the checksum covers the length


def _read_record(data, pos):
    """Returns (op, key, value, end), or None if no complete, checksum-valid record starts at pos."""
    if pos + 12 > len(data):
        return None  # NOTE: not even a full header remains
    length = int.from_bytes(data[pos:pos + 8], "big")
    end = pos + 12 + length
    if end > len(data):
        return None  # NOTE: cut off mid-record, or a damaged (huge) length
    payload = data[pos + 12:end]
    if zlib.crc32(data[pos:pos + 8] + payload) != int.from_bytes(data[pos + 8:pos + 12], "big"):
        return None  # NOTE: enough bytes, but not the ones that were written
    op = payload[0]
    key, kpos = _read_str(payload, 1)
    value = _read_str(payload, kpos)[0] if op == PUT else ""
    return op, key, value, end


class LogKVStore(KVStore):
    LOG = "log"

    def put(self, key, value):
        self.fs.append(self.LOG, _encode_record(PUT, key, value))  # NOTE: log first, memory second
        super().put(key, value)

    def delete(self, key):
        self.fs.append(self.LOG, _encode_record(DELETE, key))
        self._data.pop(key, None)

    def load(self):
        data = self.fs.read(self.LOG) or b""
        store, pos = {}, 0
        while (record := _read_record(data, pos)) is not None:  # NOTE: stop at the first bad record, for good
            op, key, value, pos = record
            if op == PUT:
                store[key] = value
            else:
                store.pop(key, None)
        if pos < len(data):  # NOTE: cut the bad tail off, or later appends land behind it and are never replayed
            self.fs.write(self.LOG, data[:pos])
        self._data = store

    def compact(self):
        self.fs.write(self.LOG, b"".join(_encode_record(PUT, k, v) for k, v in self._data.items()))
```

### Follow-ups

- On a real file system, overwriting a file in place is not atomic. The equivalent of this `fs.write` is to write a temporary file, `fsync` it and `rename` it over the old name; an append is durable only after `fsync`.
- A varint length (7 bits per byte) shrinks the overhead of a short string from 8 bytes to 1, at the cost of a decode loop instead of one slice.
- Bitcask keeps only an index from key to file offset in memory and reads values from the log on demand, so the values need not fit in RAM.

```python
import random

# --- Part 1: round-trip with adversarial strings ---
WEIRD = ["", "a", "a:b", "a,b", "a=b", "key\nwith\nnewlines", "\x00null\x00byte", "\x00" * 12,
         "emoji \U0001F600 \U0001F4A9", "long" * 5000, "mixed 中文 and English", ":,=\n\x00", "\ud800"]

rng = random.Random(0)
for _ in range(300):
    n = rng.randint(0, len(WEIRD))
    pairs = {k: rng.choice(WEIRD) for k in rng.sample(WEIRD, k=n)}
    fs = FileSystem()
    writer = KVStore(fs)
    for k, v in pairs.items():
        writer.put(k, v)
    writer.save()
    reader = KVStore(fs)
    reader.load()
    assert reader._data == pairs
    assert reader.get("definitely-not-a-key") is None

assert _encode_str("\U0001F600")[:8] == (4).to_bytes(8, "big")     # one code point, four bytes

fs_empty = FileSystem()
assert KVStore(fs_empty).get("x") is None      # nothing put yet
fresh = KVStore(fs_empty)
fresh.put("unsaved", "1")
fresh.load()                                    # nothing ever saved: load() leaves the store empty
assert fresh._data == {}
writer = KVStore(fs_empty)
writer.put("a", "1")
writer.put("b", "2")
writer.save()
other = KVStore(fs_empty)
other.put("a", "3")
other.save()                                    # a later save fully replaces the earlier one
reader = KVStore(fs_empty)
reader.load()
assert reader._data == {"a": "3"}

# a naive delimiter scheme really does fail once a key contains the delimiter
naive_serialize = lambda data: "".join(f"{k}:{v}\n" for k, v in data.items())
def naive_deserialize(text):
    return {line.split(":", 1)[0]: line.split(":", 1)[1] for line in text.splitlines()}
original = {"time:now": "value"}
assert naive_deserialize(naive_serialize(original)) != original

# --- Part 2: round-trip across a range of caps, including the smallest that can work and no cap ---
def saved(fs):
    reader = ChunkedKVStore(fs)
    reader.load()                                # must never raise after a crash
    return reader._data


def needed_files(fs):
    generation, num_chunks, _, _ = _decode_manifest(fs.read("manifest"))
    return {"manifest"} | {f"chunk_{generation}_{i}" for i in range(num_chunks)}


MANIFEST_SIZE = len(_encode_manifest(0, 0, 0, 0))
assert MANIFEST_SIZE == 32
for cap in [MANIFEST_SIZE, MANIFEST_SIZE + 5, 40, 64, 100, 1024, 10_000, None]:
    fs = FileSystem(max_file_size=cap)
    writer = ChunkedKVStore(fs)
    pairs = {}
    for i in range(30):
        k = f"key_{i}_" + "x" * rng.randint(0, 200)
        pairs[k] = rng.choice(WEIRD) if i % 3 == 0 else "y" * rng.randint(0, 300)
        writer.put(k, pairs[k])
    writer.save()
    assert cap is None or all(len(fs.read(name)) <= cap for name in fs.list())
    size = len(_serialize_store(pairs))
    assert len(fs.list()) == 1 + (1 if cap is None else -(-size // cap))      # the manifest plus ceil(size / cap) chunks
    assert saved(fs) == pairs and set(fs.list()) == needed_files(fs)

# a chunk boundary (byte 64) inside the value's length header, which occupies bytes [8 + key_len, 16 + key_len),
# or inside one of the 4-byte characters that follow it
assert 8 + 52 < 64 < 16 + 52 and 16 + 42 + 4 < 64 < 16 + 42 + 8
for key_len in range(30, 60):
    fs = FileSystem(max_file_size=64)
    writer = ChunkedKVStore(fs)
    writer.put("k" * key_len, "\U0001F600" * 3)
    writer.save()
    assert saved(fs) == {"k" * key_len: "\U0001F600" * 3}

for cap in range(MANIFEST_SIZE):                 # too small for the manifest: ValueError, nothing committed
    too_small = FileSystem(max_file_size=cap)
    writer = ChunkedKVStore(too_small)
    writer.put("a", "b")
    try:
        writer.save()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert saved(too_small) == {}


class Crash(Exception):
    pass


class CrashingFileSystem(FileSystem):
    """The process dies (Crash) just before its mutating call number crash_at, counted from 0."""

    def __init__(self, max_file_size=None):
        super().__init__(max_file_size)
        self.calls, self.crash_at = 0, None

    def _tick(self):
        if self.calls == self.crash_at:
            raise Crash
        self.calls += 1

    def write(self, name, data):
        self._tick()
        super().write(name, data)

    def delete(self, name):
        self._tick()
        super().delete(name)


def try_save(fs, state, crash_at):
    """save() of state from a new instance; returns False if the process died before call crash_at."""
    store = ChunkedKVStore(fs)
    for k, v in state.items():
        store.put(k, v)
    fs.calls, fs.crash_at = 0, crash_at
    try:
        store.save()
        return True
    except Crash:
        return False
    finally:
        fs.crash_at = None


# every crash point of one save, followed by every crash point of the next, followed by a clean save
S0 = {f"k{i}": "v" * 30 for i in range(5)}
S1 = {"big": "z" * 300, "\U0001F600": ""}
S2 = {"tiny": "t"}
chunks_of = lambda state: -(-len(_serialize_store(state)) // 48)
scenarios, first, done1 = 0, 0, False
while not done1:
    second, done2 = 0, False
    while not done2:
        fs = CrashingFileSystem(max_file_size=48)
        assert try_save(fs, S0, None)
        done1 = try_save(fs, S1, first)
        after1 = saved(fs)
        assert after1 == (S1 if first > chunks_of(S1) else S0)     # committed iff the manifest write happened
        done2 = try_save(fs, S2, second)
        after2 = saved(fs)
        assert after2 == (S2 if second > chunks_of(S2) else after1)
        assert not done2 or set(fs.list()) == needed_files(fs)
        assert try_save(fs, S0, None) and saved(fs) == S0
        assert set(fs.list()) == needed_files(fs)                  # leftovers of both crashes are gone
        scenarios, second = scenarios + 1, second + 1
    first += 1
assert scenarios > 100

# damage that no crash can cause is reported, not silently returned
for damage in ["flip", "delete"]:
    fs_corrupt = FileSystem(max_file_size=64)
    victim = ChunkedKVStore(fs_corrupt)
    for i in range(10):
        victim.put(f"k{i}", "v" * 40)
    victim.save()
    if damage == "flip":                         # NOTE: reaches into the mock to simulate bit rot
        fs_corrupt._files["chunk_0_0"] = bytes([fs_corrupt.read("chunk_0_0")[0] ^ 0xFF]) + fs_corrupt.read("chunk_0_0")[1:]
    else:
        fs_corrupt.delete("chunk_0_1")
    try:
        ChunkedKVStore(fs_corrupt).load()
        raise AssertionError("expected CorruptStoreError")
    except CorruptStoreError:
        pass

# --- Part 3: replay against a plain dict, a cut at every byte, damaged tails, compaction ---
SMALL = [w for w in WEIRD if len(w) < 50]
fs_log = FileSystem()
log_writer = LogKVStore(fs_log)
model, states, ends = {}, [{}], [0]              # states[n] / ends[n]: the store / the log size after n operations
for _ in range(40):
    key = rng.choice(SMALL)
    if rng.random() < 0.3:
        log_writer.delete(key)
        model.pop(key, None)
    else:
        model[key] = rng.choice(SMALL)
        log_writer.put(key, model[key])
    assert log_writer._data == model
    states.append(dict(model))
    ends.append(len(fs_log.read("log")))
full_log = fs_log.read("log")


def recover(log_bytes):
    fs = FileSystem()
    fs._files["log"] = log_bytes                 # NOTE: reaches into the mock to plant what a crash left
    store = LogKVStore(fs)
    store.load()
    return store, fs


assert recover(b"")[0]._data == {} and LogKVStore(FileSystem()).get("x") is None
for cut in range(len(full_log) + 1):
    n = max(i for i in range(len(ends)) if ends[i] <= cut)         # complete records before the cut
    store, fs = recover(full_log[:cut])
    assert store._data == states[n], cut
    store.put("after", "recovery")                                  # must survive the next load()
    reloaded = LogKVStore(fs)
    reloaded.load()
    assert reloaded._data == {**states[n], "after": "recovery"}, cut

for i in range(ends[-2], len(full_log)):         # the last record, damaged: one byte flipped, or zeroed from byte i on
    flipped = full_log[:i] + bytes([full_log[i] ^ 0xFF]) + full_log[i + 1:]
    assert recover(flipped)[0]._data == states[-2], i
    zeroed = full_log[:i] + bytes(len(full_log) - i)
    assert zeroed == full_log or recover(zeroed)[0]._data == states[-2], i
huge = full_log[:ends[-2]] + b"\xff" * 8 + full_log[ends[-2] + 8:]  # the last length field reads 2**64 - 1
assert recover(huge)[0]._data == states[-2]
assert zlib.crc32(b"") == 0
for n in range(1, 40):                           # zero-filled or random garbage after the last complete record
    for tail in [bytes(n), bytes(rng.randrange(256) for _ in range(n))]:
        store, fs = recover(full_log + tail)
        assert store._data == states[-1] and fs.read("log") == full_log

# compaction: same store, size independent of the history, and an interrupted compact() loses nothing
fs_compact = CrashingFileSystem()
compactor = LogKVStore(fs_compact)
for i in range(50):
    compactor.put("x", str(i))
compactor.put("y", "keep")
compactor.delete("y")
compactor.put("", "")
size_before = len(fs_compact.read("log"))
fs_compact.crash_at = fs_compact.calls
try:
    compactor.compact()
    raise AssertionError("expected Crash")
except Crash:
    pass
assert len(fs_compact.read("log")) == size_before
fs_compact.crash_at = None
compactor.compact()
direct = LogKVStore(FileSystem())
direct.put("x", "49")
direct.put("", "")
assert fs_compact.read("log") == direct.fs.read("log") and fs_compact.list() == ["log"]
verifier = LogKVStore(fs_compact)
verifier.load()
assert verifier._data == {"x": "49", "": ""}
```
