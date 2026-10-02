Implement an in-memory file system as a single class `FileSystem`, built up over four levels: a level's tests must pass before the next level's tests run, and each level adds methods to the same class rather than replacing it.

A path is an absolute, non-empty string that starts with `/`, made of *components* separated by `/` — for example `/a/b/c` names the directory or file `c` inside `b` inside `a`, all reached from the root `/`, which always exists and is a directory. Normalization collapses consecutive `/` characters into one and drops one trailing `/`, so `/a//b/` and `/a/b` name the same node, and `/` alone names the root. Neither `.` nor `..` acts as a navigation shortcut: a path is *invalid* if it does not start with `/`, or if, after normalization, one of its components is exactly `.` or `..`. An invalid path is treated exactly like a path that does not currently exist: every method below returns whatever value it uses to signal failure. Components are arbitrary non-empty strings that contain no `/`; they are compared case-sensitively, and the same component name can appear under two different directories without naming the same node.

Wherever a method creates something at a path that does not yet exist — a new file, or the destination of a move or a copy — it also creates any missing ancestor directory of that path, the same way `mkdir` does. A method that only reads, removes, or moves *from* a path never creates anything on that side: a missing ancestor there simply makes the call fail. Whenever a call returns the value that signals failure, the file system — and, from Level 3 on, every user's stored usage — is left exactly as it was before the call: nothing is partially created, moved, deleted, or billed.

### Level 1 — Files and directories

A fresh `FileSystem()` starts with nothing but the root directory `/`.

```py
class FileSystem:
    def mkdir(self, path: str) -> bool:
        """Creates a directory at path, creating every missing ancestor directory along the way
        (like mkdir -p). Returns True once the directory exists, whether it already existed or was
        just created. Returns False, without effect, for an invalid path, or if path or any
        ancestor of it already exists as a file."""

    def write(self, path: str, content: str) -> bool:
        """Creates a file at path holding content, or overwrites its content if a file is already
        there. A file's *size* is the number of characters in its content. Returns True on success.
        Returns False, without effect, for an invalid path, for path == "/", if path already names
        a directory, or if any ancestor of path already exists as a file."""

    def read(self, path: str) -> str | None:
        """Returns the content of the file at path. Returns None for an invalid path, if nothing
        exists at path, or if path names a directory."""

    def ls(self, path: str) -> list[str] | None:
        """Returns the names of path's immediate children, files and subdirectories together in one
        list, sorted lexicographically (Python's default string order: comparing Unicode code
        points, so every uppercase letter sorts before every lowercase one). Returns None for an
        invalid path, if nothing exists at path, or if path names a file."""
```

Example:

```text
fs = FileSystem()
fs.mkdir("/reports/2025")                   # True  -- creates /reports and /reports/2025
fs.mkdir("/reports/2025")                   # True  -- already a directory: no-op
fs.write("/reports/2025/q1.txt", "draft")   # True
fs.write("/reports/2025/q1.txt", "final")   # True  -- overwrites the content
fs.read("/reports/2025/q1.txt")             # "final"
fs.ls("/reports")                           # ["2025"]
fs.ls("/reports/2025")                      # ["q1.txt"]
fs.read("/reports/2025")                    # None  -- a directory, not a file
fs.write("/reports/2025", "oops")           # False -- a directory already exists there
fs.ls("/reports/2025/q1.txt")               # None  -- a file, not a directory
fs.mkdir("/reports/2025/q1.txt")            # False -- a file already exists there
```

### Level 2 — Move, copy, delete

```py
class FileSystem:
    def mv(self, src: str, dst: str) -> bool:
        """Moves src to dst -- src's entire subtree too, if it is a directory -- so that src no
        longer exists afterwards. If dst does not currently exist, src is placed there in full. If
        dst exists and is the same kind (file or directory) as src, a file overwrites a file's
        content in place, and a directory discards its whole subtree and replaces it with src's.
        Returns True on success. Returns False, without effect, for an invalid src or dst, if
        nothing exists at src, if src and dst name the same path or lie on the same ancestry chain
        (one a prefix of the other's components, so neither can move into its own subtree or over
        its own ancestor), if some ancestor of dst already exists as a file, or if dst exists but
        is not the same kind as src."""

    def cp(self, src: str, dst: str) -> bool:
        """Same as mv, but src is left untouched and the copy at dst is independent of it
        afterwards: writing to either one never changes the other."""

    def rm(self, path: str) -> bool:
        """Deletes whatever is at path; deleting a directory recursively deletes its entire
        subtree. Returns True if something was deleted. Returns False, without effect, for an
        invalid path, if nothing exists at path, or if path is the root "/" (the root can never be
        deleted)."""

    def size(self, path: str) -> int | None:
        """Returns a file's size as defined in Level 1, or, for a directory, the sum of the sizes
        of every file anywhere in its subtree (0 for an empty directory, or one that holds only
        empty files). Returns None for an invalid path or if nothing exists at path."""
```

Example:

```text
fs.mkdir("/reports/2025")
fs.write("/reports/2025/q1.txt", "Q1 results")
fs.write("/reports/2025/q2.txt", "Q2 results here")
fs.mkdir("/archive")

fs.cp("/reports/2025", "/archive/2025")                # True -- deep copy
fs.write("/archive/2025/q1.txt", "changed")             # only the copy changes
fs.read("/reports/2025/q1.txt")                          # "Q1 results" -- original untouched
fs.size("/reports/2025/q1.txt")                           # 10
fs.size("/reports/2025")                                  # 25  -- 10 + 15, recursive
fs.mv("/reports/2025/q2.txt", "/archive/2025/q2.txt")     # True
fs.ls("/reports/2025")                                     # ["q1.txt"]
fs.ls("/archive/2025")                                     # ["q1.txt", "q2.txt"]
fs.mv("/reports", "/reports/2025/nested")                  # False -- destination is inside source's own subtree
fs.rm("/reports/2025/q1.txt")                               # True
fs.rm("/reports")                                            # True  -- recursive
fs.ls("/reports")                                             # None  -- no longer exists
```

### Level 3 — Users and storage quotas

Every user is identified by a *user id*, a string. There is always an implicit user `"admin"` with unlimited storage; `write`'s `owner` defaults to it, and `"admin"` can never be registered or looked up as a regular user. Every file has an *owner*, the user id it was created under, fixed for that file's whole lifetime. A file's size counts against its owner's quota unless the owner is `"admin"`. Ownership only constrains `write`, `mv` and `cp`; `read`, `ls` and `size` behave exactly as before, for any file regardless of its owner. From this level on, `mv` and `cp` preserve each moved or copied file's owner — the owner it already had, never the caller — and, like `write`, fail if an existing destination file's owner does not match the incoming file's owner. `cp` also counts every copied file's size against its owner's capacity exactly as `write` does: if copying a directory would push any one of the several owners whose files it contains over that owner's capacity, the entire `cp` call fails and nothing is copied, even the files belonging to owners who did have room; `mv` never fails on capacity, since relocating a file changes no owner's total. `rm`, and any overwrite performed by `write`, `mv` or `cp`, free the removed size from its former owner's used total (unless that owner is `"admin"`).

```py
class FileSystem:
    def add_user(self, user_id: str, capacity: int) -> bool:
        """Registers user_id with a storage *capacity* -- the largest total size, summed over
        every file it owns anywhere in the tree, it may hold at once -- and a *used* total that
        starts at 0. Returns True on success. Returns False, without effect, if user_id is "admin"
        or already registered, or if capacity < 0."""

    def write(self, path: str, content: str, owner: str = "admin") -> bool:
        """Grows Level 1's write with three further rules. owner must be "admin" or a registered
        user id, or the call fails. If a file already exists at path, owner must equal its current
        owner, or the call fails -- a file's owner never changes across a write. If owner is not
        "admin", the call also fails whenever it would push owner's used total over its capacity
        (the size after this write, counting every other file owner already has, compared against
        capacity)."""

    def update_capacity(self, user_id: str, capacity: int) -> int | None:
        """Sets user_id's capacity. If its used total is already at or under the new capacity,
        nothing else happens and the call returns 0. Otherwise deletes that user's files, largest
        first, ties broken by comparing the two files' full paths as strings in the same order as
        ls, until the used total is at or under capacity, and returns the number of files deleted.
        Returns None, without effect, if user_id is not registered (including "admin") or
        capacity < 0."""
```

Example:

```text
fs.add_user("alice", 20)                                 # True
fs.add_user("alice", 50)                                  # False -- already exists
fs.write("/data/a.txt", "hello world", owner="alice")      # True  -- size 11, 11 <= 20
fs.write("/data/b.txt", "hi", owner="alice")                # True  -- size 2, 13 <= 20
fs.write("/data/c.txt", "0123456789", owner="alice")         # False -- size 10, 23 > 20
fs.write("/data/a.txt", "hey", owner="bob")                   # False -- a.txt is owned by alice, not bob
fs.update_capacity("alice", 5)                                  # 1     -- evicts a.txt (11), the larger file
fs.read("/data/a.txt")                                            # None  -- evicted
fs.read("/data/b.txt")                                             # "hi"  -- kept, still fits
```

### Level 4 — Snapshots

A *snapshot* is a saved copy of the entire current state — every directory and file, every file's content and owner, and every user's capacity and used total — identified by a `snapshot_id` chosen by the caller.

```py
class FileSystem:
    def snapshot(self, snapshot_id: str) -> bool:
        """Captures the entire current state under snapshot_id. Returns True on success. Returns
        False, without effect, if snapshot_id was already used by an earlier snapshot."""

    def restore(self, snapshot_id: str) -> bool:
        """Replaces the entire current state with an independent copy of the one captured under
        snapshot_id: later changes to the file system, or a later restore of a different snapshot,
        must never affect a previously captured snapshot, and restoring the same snapshot_id twice
        must give the same result both times. Returns True on success. Returns False, without
        effect, if no snapshot was ever captured under snapshot_id."""
```

Example:

```text
fs.mkdir("/work")
fs.write("/work/draft.txt", "v1")
fs.snapshot("before-edit")             # True
fs.write("/work/draft.txt", "v2")
fs.snapshot("before-edit")             # False -- id already used
fs.read("/work/draft.txt")             # "v2"
fs.restore("before-edit")              # True
fs.read("/work/draft.txt")             # "v1"
fs.restore("nope")                     # False -- no such snapshot

fs.add_user("carol", 10)
fs.write("/work/notes.txt", "abcde", owner="carol")   # True
fs.snapshot("with-carol")               # True
fs.update_capacity("carol", 2)          # 1 -- evicts notes.txt
fs.read("/work/notes.txt")              # None
fs.restore("with-carol")                # True
fs.read("/work/notes.txt")              # "abcde" -- restored
```
