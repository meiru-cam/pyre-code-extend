`FileSystem` is an in-memory stand-in for a disk: a store of named byte blobs that outlives every store object built on it, so a new process is modelled by a new instance constructed on the same `fs`. Every part uses it as given, unmodified. On top of it, a *key-value store* lets a caller associate string keys with string values in memory and persist them, so that a new instance can recover exactly the same mapping. Keys and values are arbitrary `str`: any length, including the empty string, and any characters — delimiter-like characters such as `:`, `,`, `=`, newlines, the null character, and characters outside the Basic Multilingual Plane such as emoji must all round-trip unchanged. Neither `json` nor `pickle`, nor any other ready-made serialization module, may be used; `str.encode`, `bytes.decode`, `int.to_bytes` and `int.from_bytes` are allowed.

```python
class FileSystem:
    """An in-memory mock file system that stores named byte blobs. Files are independent: writing
    "b" never touches "a". write and delete are atomic: if the process dies, the call has either
    taken full effect or had no effect at all. append is not atomic."""

    def __init__(self, max_file_size: int | None = None):
        self.max_file_size = max_file_size
        self._files: dict[str, bytes] = {}

    def write(self, name: str, data: bytes) -> None:
        """Creates or overwrites the file called name with exactly data. Raises ValueError if
        max_file_size is set and len(data) exceeds it; name is then left exactly as it was before
        this call (unchanged, or still absent)."""
        if self.max_file_size is not None and len(data) > self.max_file_size:
            raise ValueError(f"{name}: {len(data)} bytes exceeds max_file_size={self.max_file_size}")
        self._files[name] = data

    def append(self, name: str, data: bytes) -> None:
        """Adds data to the end of the file called name, creating it if absent; raises ValueError,
        changing nothing, if the file would then exceed max_file_size. NOT atomic: if the process
        dies during this call, the file may be left ending in any prefix of data, and the bytes of
        that prefix may be damaged (zeroed, for instance)."""
        old = self._files.get(name, b"")
        if self.max_file_size is not None and len(old) + len(data) > self.max_file_size:
            raise ValueError(f"{name}: {len(old) + len(data)} bytes exceeds max_file_size={self.max_file_size}")
        self._files[name] = old + data

    def read(self, name: str) -> bytes | None:
        """Returns the current bytes of name, or None if no such file exists."""
        return self._files.get(name)

    def list(self) -> list[str]:
        """Names of every file that currently exists, in arbitrary order."""
        return list(self._files)

    def delete(self, name: str) -> None:
        """Removes name if present; a no-op if it does not exist."""
        self._files.pop(name, None)
```

### Part 1 — Basic durable store

Implement `KVStore`. `save()` persists the entire in-memory mapping to `fs`; a fresh `KVStore` constructed on the same `fs` must, after `load()`, have `get` return exactly what was `put` before that `save()`. `get` returns `None` for a key that has no value, and `load()` leaves the store empty if nothing was ever saved to `fs`. A later `save()` fully replaces an earlier one: after it, `load()` reproduces only the mapping that the later call saved.

```py
class KVStore:
    def __init__(self, fs: FileSystem):
        """fs is the FileSystem this store persists to and restores from."""

    def put(self, key: str, value: str) -> None:
        """Sets key -> value in memory, overwriting any previous value for key."""

    def get(self, key: str) -> str | None:
        """Returns the current value of key, or None if key has no value."""

    def save(self) -> None:
        """Persists the entire in-memory store to fs."""

    def load(self) -> None:
        """Replaces the in-memory store with what fs currently holds."""
```

For example, after

```py
fs = FileSystem()
store = KVStore(fs)
store.put("time:now", "12:00")
store.put("", "empty key")
store.put("emoji", "🙂\n")
store.save()
```

a new `KVStore(fs)` on which `load()` is called must return `"12:00"` for `"time:now"`, `"empty key"` for `""`, `"🙂\n"` for `"emoji"`, and `None` for any other key.

### Part 2 — Size-capped files and interrupted saves

Now `fs` is a `FileSystem(max_file_size=...)`, and `fs.write` raises whenever `data` is longer than the cap. `ChunkedKVStore` has the same interface and the same guarantees as `KVStore`, plus:

- A single key or value that is itself longer than `max_file_size` must still round-trip.
- When `save()` returns, `fs.list()` contains no file that `load()` does not need for the data just saved — nothing left over from earlier saves, whether they completed or were interrupted.
- `save()` may be interrupted: the process can die before or after any call it makes on `fs` (never in the middle of one, since `write` and `delete` are atomic). A `load()` on a new instance over the same `fs` must then reproduce, completely, either the mapping of the interrupted `save()` or whatever `load()` would have reproduced just before that `save()` began (an empty store if nothing was ever saved) — never a mixture of two saves, and never an error.
- Every `max_file_size` of at least 64 bytes must be supported. With a smaller cap `save()` may raise `ValueError` instead, which counts as an interrupted save.

```py
class ChunkedKVStore:
    def __init__(self, fs: FileSystem):
        """fs is the FileSystem this store persists to and restores from; fs.max_file_size may be set."""

    def put(self, key: str, value: str) -> None:
        """Same contract as KVStore.put."""

    def get(self, key: str) -> str | None:
        """Same contract as KVStore.get."""

    def save(self) -> None:
        """Persists the entire in-memory store to fs, never writing more than fs.max_file_size
        bytes to any single file."""

    def load(self) -> None:
        """Replaces the in-memory store with what fs currently holds."""
```

For example, with `max_file_size=64` and enough key-value pairs that the store takes several hundred bytes, `save()` must produce more than one file, and `load()` on a fresh `ChunkedKVStore(fs)` must still return every pair unchanged.

### Part 3 — Append-only log with crash recovery

`LogKVStore` has no `save()`: `put` and `delete` are durable as soon as they return. Neither may rewrite what is already stored — each persists itself with a single `fs.append` whose size depends only on its own key and value, so `put`, `delete`, and `get` must each take time independent of how many records the log already holds. A new instance recovers by calling `load()`, which must replay a log of $N$ records in $O(N)$ time, not $O(N^2)$. Here `fs` is a plain `FileSystem()`, with no size cap.

`delete(key)` removes `key`: from then on `get(key)` returns `None`, also on a new instance after `load()`, until `key` is `put` again.

The process may die at any moment, including in the middle of an `fs.append`, which leaves the file ending in an incomplete or damaged fragment of what was being appended. `load()` must restore the effect of every `put` and `delete` persisted completely before that fragment; it must not raise, and must take nothing from the fragment. Operations performed after such a recovery must survive the next `load()` like any others.

`compact()` stops the stored data from growing with the history of overwritten and deleted keys: after it, the total size of the files on `fs` depends only on the keys and values currently in the store, and a `load()`, on this instance or a new one, still reconstructs exactly that store. An interrupted `compact()` must lose nothing.

```py
class LogKVStore:
    def __init__(self, fs: FileSystem):
        """fs is the FileSystem this store appends to and recovers from."""

    def put(self, key: str, value: str) -> None:
        """Durably records key -> value before returning."""

    def delete(self, key: str) -> None:
        """Durably records the removal of key before returning."""

    def get(self, key: str) -> str | None:
        """Same contract as KVStore.get."""

    def load(self) -> None:
        """Rebuilds the in-memory store from what fs currently holds."""

    def compact(self) -> None:
        """Rewrites what is stored so that its size no longer depends on the history of overwritten
        or deleted keys; a subsequent load() reconstructs the same in-memory store."""
```
