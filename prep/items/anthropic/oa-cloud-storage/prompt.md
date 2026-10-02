Implement a `CloudStorage` class that tracks files by name and size only: the bytes of a file are never stored or read, only how many of them it has. The class is built up over four levels, in order; each level's tests must pass before the next level's methods are exercised, and every level adds methods to the same `CloudStorage` class the levels before it defined, never a new one.

A *file name* is a nonempty string of ASCII letters, digits, `.`, `_` and `-`; names are case-sensitive, so `"Notes.txt"` and `"notes.txt"` name different files. At most one file may exist under a given name at any moment, across the whole store, regardless of who — if anyone — owns it (ownership starts at Level 3). A file's size is a positive integer number of bytes that never changes while that name exists: copying, compressing and decompressing (Level 4) each create a differently named file instead of changing an existing one in place. A *user id* is a nonempty string built from the same characters as a file name; user ids and file names are separate namespaces, so the same string may serve as both without conflict. Every call in this problem passes a name, id or size that already satisfies these rules.

### Level 1 — Files, copies, sizes

```py
class CloudStorage:
    def add_file(self, name: str, size: int) -> bool:
        """Creates a file called name with this size. Fails, returning False and changing nothing, if
        a file called name already exists; otherwise creates it and returns True."""

    def copy_file(self, source: str, destination: str) -> bool:
        """Creates a file called destination with the same size as source; source is unaffected and
        continues to exist as before. Fails, returning False and changing nothing, if source does not
        currently exist or if destination already exists; otherwise creates destination and returns
        True."""

    def get_file_size(self, name: str) -> int | None:
        """Returns the current size of name, or None if no file called name currently exists."""
```

For example:

```text
cs = CloudStorage()
cs.add_file("report.txt", 200)                   # -> True
cs.add_file("report.txt", 10)                    # -> False  -- "report.txt" already exists
cs.copy_file("report.txt", "report_backup.txt")  # -> True   -- report_backup.txt now exists, size 200
cs.copy_file("missing.txt", "x.txt")             # -> False  -- source does not exist
cs.copy_file("report.txt", "report_backup.txt")  # -> False  -- destination already exists
cs.get_file_size("report_backup.txt")            # -> 200
cs.get_file_size("nope.txt")                     # -> None
```

### Level 2 — Search by prefix and suffix

`find_files(prefix, suffix)` searches every file currently in the store, owned or not, for those whose name starts with `prefix` and ends with `suffix`, in the sense of `str.startswith` and `str.endswith`: either argument may be `""`, which matches every name on that side, and a short name can satisfy both at once, as `"ab"` does for `prefix="ab"` and `suffix="ab"`. Each match is formatted as `f"{name}({size})"` — for instance `"report.txt(200)"` — and the returned list is sorted by size descending, then by name ascending among files of the same size.

```py
    def find_files(self, prefix: str, suffix: str) -> list[str]:
        """Returns "name(size)" for every file currently in the store whose name starts with prefix
        and ends with suffix, sorted by size descending and, among files of the same size, by name
        ascending."""
```

Continuing the store above:

```text
cs.add_file("report_draft.txt", 200)
cs.add_file("report_final.txt", 500)
cs.add_file("summary.md", 50)
cs.find_files("report", ".txt")
# ["report_final.txt(500)", "report.txt(200)", "report_backup.txt(200)", "report_draft.txt(200)"]
# size 500 first; the three files tied at 200 follow in name order -- "." sorts before "_" in ASCII,
# so "report.txt" < "report_backup.txt" < "report_draft.txt"; "summary.md" does not start with "report"
cs.find_files("summary", "")
# ["summary.md(50)"]   -- suffix="" matches every name, and prefix "summary" rules out every report_*.txt
```

### Level 3 — User accounts and storage quotas

A *user* has a *capacity*, a positive integer number of bytes fixed when `add_user` creates the account, and owns zero or more files; a file's size counts against its owner's capacity for as long as the file exists, and a user's *remaining capacity* is its capacity minus the sizes of every file it currently owns. A file created by `add_file` has no owner and never counts against anyone's capacity. Every call to a method introduced from here on passes a `user_id` and a `capacity` that already satisfy these rules.

```py
    def add_user(self, user_id: str, capacity: int) -> bool:
        """Registers user_id with this capacity and no files. Fails, returning False, if user_id is
        already a user; otherwise returns True."""

    def add_file_by(self, user_id: str, name: str, size: int) -> int | None:
        """Creates a file called name with this size, owned by user_id. Fails, returns None, changing
        nothing, if user_id is not currently a user, if a file called name already exists (owned or
        not), or if size exceeds user_id's remaining capacity; otherwise creates the file and returns
        user_id's remaining capacity after this addition."""

    def merge_user(self, target: str, source: str) -> int | None:
        """Moves every file source currently owns to target (names and sizes unchanged) and adds
        source's capacity to target's. Fails, returns None, changing nothing, if target equals source
        or if target or source is not currently a user. Otherwise source is removed as a user --
        afterwards, user id source refers to no one, as if add_user had never been called for it,
        until a later add_user gives that id a fresh account -- and the call returns target's
        remaining capacity after the merge."""
```

From this level on, `copy_file` gives `destination` the same owner as `source` (no owner, if `source` has none). When `source` has an owner, `destination`'s size counts against that owner's remaining capacity: `copy_file` now also fails, returns `False`, changing nothing, if the owner's remaining capacity is less than `source`'s size, even when `source` exists and `destination` is free.

Continuing the store above:

```text
cs.add_user("alice", 500)                          # -> True
cs.add_user("alice", 100)                          # -> False  -- "alice" already exists
cs.add_file_by("alice", "alice_a.txt", 300)        # -> 200  -- remaining: 500 - 300
cs.add_file_by("alice", "report.txt", 10)          # -> None  -- "report.txt" already exists, unowned
cs.add_file_by("carol", "c.txt", 10)               # -> None  -- "carol" is not a user
cs.add_file_by("alice", "alice_b.txt", 250)        # -> None  -- 250 exceeds alice's remaining 200
cs.get_file_size("alice_a.txt")                    # -> 300   -- unaffected by the three failed calls above
cs.copy_file("alice_a.txt", "alice_a_copy.txt")    # -> False -- would need 300 more, alice has 200 left
cs.copy_file("report_final.txt", "report_final_copy.txt")  # -> True -- unowned source: no capacity check
cs.add_file_by("alice", "alice_small.txt", 100)    # -> 100   -- remaining: 200 - 100
cs.copy_file("alice_small.txt", "alice_small_copy.txt")     # -> True -- needs exactly 100, alice has exactly 100
cs.get_file_size("alice_small_copy.txt")           # -> 100
cs.add_file_by("alice", "alice_c.txt", 1)          # -> None  -- alice now has 0 bytes left
cs.add_user("bob", 50)                             # -> True
cs.add_file_by("bob", "bob_notes.txt", 20)         # -> 30
cs.merge_user("alice", "bob")                      # -> 30    -- alice: capacity 500+50, used 500+20
cs.add_file_by("bob", "bob_notes2.txt", 5)         # -> None  -- "bob" is no longer a user
cs.merge_user("alice", "bob")                      # -> None  -- source "bob" no longer exists
cs.get_file_size("bob_notes.txt")                  # -> 20    -- the file still exists, now owned by alice
cs.merge_user("alice", "alice")                    # -> None  -- target equals source
cs.merge_user("dave", "alice")                     # -> None  -- "dave" is not a user
```

### Level 4 — Compression, backup and restore

Level 4 adds four methods; the first two only ever act on a file `user_id` currently owns. Let `COMPRESSED_SUFFIX` denote the fixed string `".cmp"`.

```py
    def compress_file(self, user_id: str, name: str) -> str | None:
        """Replaces name, owned by user_id, with a smaller file: removes name and creates a file
        called name + COMPRESSED_SUFFIX, still owned by user_id, whose size is name's previous size
        divided by two and rounded up. Fails, returns None, changing nothing, if user_id is not
        currently a user, if name does not currently exist or is not currently owned by user_id, or
        if name + COMPRESSED_SUFFIX already names an existing file; otherwise returns the new file's
        name. May be called on a name that already ends with COMPRESSED_SUFFIX, which compresses it
        again."""

    def decompress_file(self, user_id: str, name: str) -> str | None:
        """The reverse of compress_file: removes name and creates a file called name with its
        trailing COMPRESSED_SUFFIX removed, still owned by user_id, whose size is exactly double
        name's previous size. Fails, returns None, changing nothing, if user_id is not currently a
        user, if name does not currently exist or is not currently owned by user_id, if name does
        not currently end with COMPRESSED_SUFFIX, if the name with COMPRESSED_SUFFIX removed already
        names an existing file, or if name's current size exceeds user_id's remaining capacity;
        otherwise returns the new file's name."""

    def backup_user(self, user_id: str) -> bool:
        """Records, as user_id's one backup, the name and size of every file user_id currently owns
        and user_id's current capacity, replacing any earlier backup. Fails, returns False, if
        user_id is not currently a user; otherwise returns True."""

    def restore_user(self, user_id: str) -> int | None:
        """Resets user_id's files and capacity to its most recent backup_user(user_id): a name
        user_id owns now but the backup does not list is deleted; a name the backup lists that
        user_id does not currently own is re-created at its backed-up size; a name owned at both
        times is left exactly as it is now. Fails, returns None, changing nothing, if user_id is not
        a user, has never been backed up, or a name to be created would collide with a file this
        restore does not itself remove; otherwise resets user_id's capacity to the backed-up value
        and returns the resulting remaining capacity."""
```

The rounding in `compress_file` is the previous size divided by two, rounded up ($\lceil \text{size} / 2 \rceil$), so a file of size 1 compresses to a file of size 1, never to size
- `decompress_file` doubles exactly, with no rounding, so it does not always recover the exact size from before a matching `compress_file` when that size was odd. `decompress_file`'s capacity check compares `name`'s current, compressed size against `user_id`'s remaining capacity computed with `name` still counted at that size — not against the doubled size the call is about to produce.

`restore_user` compares by name only, not by size: a name `user_id` owned both at backup time and now is left untouched even if its size has changed, which can happen in the unusual case where that name was freed — by `compress_file` or `decompress_file` — and then given to an unrelated file by a later call; `restore_user` does not notice, and the unrelated file is what remains. A backup belongs to a user id only for as long as that id keeps referring to the same account: merging a user away, or later giving its freed id a new account with `add_user`, leaves nothing for `restore_user` to find. Restoring does not consume the backup: the same backup can be restored more than once.

Continuing the store above:

```text
cs.add_user("erin", 100)                     # -> True
cs.add_file_by("erin", "photo.raw", 41)      # -> 59
cs.backup_user("erin")                       # -> True  -- snapshot: {photo.raw: 41}, capacity 100
cs.compress_file("erin", "photo.raw")        # -> "photo.raw.cmp"  -- size: ceil(41 / 2) = 21
cs.get_file_size("photo.raw")                # -> None  -- freed by the compression
cs.get_file_size("photo.raw.cmp")            # -> 21
cs.add_file_by("erin", "logo.png", 30)       # -> 49    -- remaining: 100 - 21 - 30
cs.restore_user("erin")                      # -> 59    -- back to {photo.raw: 41}, capacity 100
cs.get_file_size("photo.raw")                # -> 41    -- re-created by the restore
cs.get_file_size("photo.raw.cmp")            # -> None  -- not in the snapshot, so deleted
cs.get_file_size("logo.png")                 # -> None  -- added after the backup, so deleted

cs.add_user("frank", 60)
cs.add_file_by("frank", "video.raw", 25)          # -> 35
cs.compress_file("frank", "video.raw")            # -> "video.raw.cmp"  -- size: ceil(25 / 2) = 13
cs.decompress_file("frank", "video.raw.cmp")      # -> "video.raw"  -- size: 13 * 2 = 26, not the original 25
cs.get_file_size("video.raw")                     # -> 26

cs.add_user("gina", 25)
cs.add_file_by("gina", "clip.raw", 25)            # -> 0
cs.compress_file("gina", "clip.raw")              # -> "clip.raw.cmp"  -- size: ceil(25 / 2) = 13, remaining 12
cs.decompress_file("gina", "clip.raw.cmp")        # -> None  -- needs 13 more bytes of capacity, only 12 remain
```
