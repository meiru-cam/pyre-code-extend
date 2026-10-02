"""A file store that tracks names and sizes, then adds quotas, compression and per-user backups."""

from ._interview import interview

# A model that recounts usage from the file table on every call, and a random call generator.
_MODEL = r"""
import random

class Model:
    # Recomputes usage from the file table on every call, unlike a running total.
    def __init__(self):
        self.files, self.cap, self.backup = {}, {}, {}
    def used(self, u):
        return sum(s for s, o in self.files.values() if o == u)
    def left(self, u):
        return self.cap[u] - self.used(u)
    def add_file(self, name, size):
        if name in self.files:
            return False
        self.files[name] = (size, None)
        return True
    def copy_file(self, src, dst):
        if src not in self.files or dst in self.files:
            return False
        size, owner = self.files[src]
        if owner is not None and self.left(owner) < size:
            return False
        self.files[dst] = (size, owner)
        return True
    def get_file_size(self, name):
        return self.files[name][0] if name in self.files else None
    def find_files(self, prefix, suffix):
        hits = sorted((-s, n) for n, (s, _) in self.files.items() if n.startswith(prefix) and n.endswith(suffix))
        return [f"{n}({-s})" for s, n in hits]
    def add_user(self, u, cap):
        if u in self.cap:
            return False
        self.cap[u] = cap
        return True
    def add_file_by(self, u, name, size):
        if u not in self.cap or name in self.files or self.left(u) < size:
            return None
        self.files[name] = (size, u)
        return self.left(u)
    def merge_user(self, t, s):
        if t == s or t not in self.cap or s not in self.cap:
            return None
        self.files = {n: (sz, t if o == s else o) for n, (sz, o) in self.files.items()}
        self.cap[t] += self.cap.pop(s)
        self.backup.pop(s, None)
        return self.left(t)
    def compress_file(self, u, name):
        if u not in self.cap or self.files.get(name, (0, None))[1] != u or name + ".zip" in self.files:
            return None
        size = self.files.pop(name)[0]
        self.files[name + ".zip"] = (-(-size // 2), u)
        return name + ".zip"
    def decompress_file(self, u, name):
        if u not in self.cap or self.files.get(name, (0, None))[1] != u or not name.endswith(".zip"):
            return None
        plain, size = name[:-4], self.files[name][0]
        if plain in self.files or size > self.left(u):
            return None
        del self.files[name]
        self.files[plain] = (2 * size, u)
        return plain
    def backup_user(self, u):
        if u not in self.cap:
            return False
        self.backup[u] = (self.cap[u], {n: s for n, (s, o) in self.files.items() if o == u})
        return True
    def restore_user(self, u):
        if u not in self.cap or u not in self.backup:
            return None
        cap, saved = self.backup[u]
        mine = {n for n, (_, o) in self.files.items() if o == u}
        if any(n in self.files and n not in mine for n in saved):
            return None
        for n in mine - set(saved):
            del self.files[n]
        for n, s in saved.items():
            if n not in mine:
                self.files[n] = (s, u)
        self.cap[u] = cap
        return self.left(u)

def random_calls(rng, n, level):
    names = ["a", "a.zip", "a.zip.zip", "b.txt", "b.txt.zip", "c-1", "c_1", "c.1"]
    users = ["u", "v", "w"]
    ops = ["add_file", "add_file", "copy_file", "get_file_size"]
    if level >= 2:
        ops += ["find_files"]
    if level >= 3:
        ops += ["add_user", "add_file_by", "add_file_by", "add_file_by", "merge_user"]
    if level >= 4:
        ops += ["compress_file", "compress_file", "decompress_file", "decompress_file", "backup_user", "restore_user"]
    model, out = Model(), []
    for _ in range(n):
        op = rng.choice(ops)
        u, v, name, other = rng.choice(users), rng.choice(users), rng.choice(names), rng.choice(names)
        args = {"add_file": (name, rng.randint(1, 30)), "copy_file": (name, other), "get_file_size": (name,),
                "find_files": (rng.choice(["", "a", "c", "b.t"]), rng.choice(["", ".zip", "1", "txt"])),
                "add_user": (u, rng.randint(1, 80)), "add_file_by": (u, name, rng.randint(1, 30)), "merge_user": (u, v),
                "compress_file": (u, name), "decompress_file": (u, name), "backup_user": (u,), "restore_user": (u,)}[op]
        out.append((op, args, getattr(model, op)(*args)))
    return out

def replay(store, calls, label):
    for i, (op, args, want) in enumerate(calls):
        got = getattr(store, op)(*args)
        assert got == want, (label, i, op, args, got, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
cs = {fn}()
assert cs.add_file("a.log", 70) is True
assert cs.copy_file("a.log", "b.log") is True and cs.copy_file("b.log", "c.log") is True
assert cs.get_file_size("c.log") == 70 and cs.get_file_size("A.log") is None
assert cs.add_file("b.log", 9) is False and cs.get_file_size("b.log") == 70
assert cs.copy_file("c.log", "a.log") is False
assert cs.copy_file("zz.log", "d.log") is False and cs.get_file_size("d.log") is None
"""},
    {"name": "Part 1: case and failed calls", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Names are case-sensitive; a failed add or copy must not create or resize anything; a copy is a separate file that survives as its own name.",
     "code": r"""
cs = {fn}()
assert cs.add_file("Notes", 7) and cs.add_file("notes", 9)
assert cs.get_file_size("Notes") == 7 and cs.get_file_size("notes") == 9
assert cs.add_file("Notes", 1) is False and cs.get_file_size("Notes") == 7
assert cs.copy_file("Notes", "notes") is False and cs.get_file_size("notes") == 9
assert cs.copy_file("Notes", "Notes") is False
assert cs.copy_file("missing", "new") is False and cs.get_file_size("new") is None
assert cs.copy_file("Notes", "n2") and cs.copy_file("n2", "n3") and cs.get_file_size("n3") == 7
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of add_file, copy_file and get_file_size calls, a return value differed from a simple model.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(seed), 50, 1), seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
cs = {fn}()
cs.add_file("a.log", 70)
cs.copy_file("a.log", "b.log")
cs.copy_file("b.log", "c.log")
cs.add_file("ab.log", 200)
cs.add_file("a.txt", 5)
assert cs.find_files("", ".log") == ["ab.log(200)", "a.log(70)", "b.log(70)", "c.log(70)"]
assert cs.find_files("a", "") == ["ab.log(200)", "a.log(70)", "a.txt(5)"]
assert cs.find_files("ab", "b.log") == ["ab.log(200)"]
assert cs.find_files("z", "") == []
"""},
    {"name": "Part 2: overlapping prefix and suffix", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "find_files must use startswith and endswith, so a short name can match both at once; empty strings match everything; ties sort by name in Python order (uppercase first); an empty store gives [].",
     "code": r"""
cs = {fn}()
assert cs.find_files("", "") == []
for name, size in [("ab", 3), ("abab", 3), ("aXb", 3), ("Ab", 3), ("b", 9), ("ba", 1)]:
    cs.add_file(name, size)
assert cs.find_files("ab", "ab") == ["ab(3)", "abab(3)"]
assert cs.find_files("", "") == ["b(9)", "Ab(3)", "aXb(3)", "ab(3)", "abab(3)", "ba(1)"]
assert cs.find_files("a", "b") == ["aXb(3)", "ab(3)", "abab(3)"]
assert cs.find_files("x", "") == []
"""},
    {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of calls including find_files, a return value differed from a simple model.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(seed), 50, 2), seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "budget.enforcement", "code": r"""
cs = {fn}()
cs.add_file("a.log", 70)
assert cs.add_user("u-ana", 100) is True and cs.add_user("u-ben", 50) is True and cs.add_user("u-ben", 9) is False
assert cs.add_file_by("u-ben", "b1", 50) == 0
assert cs.copy_file("b1", "b2") is False
assert cs.merge_user("u-ana", "u-ben") == 100
assert cs.copy_file("b1", "b2") is True
assert cs.add_file_by("u-ana", "big", 51) is None and cs.add_file_by("u-ana", "fits", 50) == 0
assert cs.add_file_by("u-ben", "x", 1) is None
assert cs.add_file_by("u-ana", "a.log", 1) is None
"""},
    {"name": "Part 3: exact fits and merges", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "A size equal to the remaining capacity fits; copies count against the source's owner; merge_user fails for equal or unknown ids, moves files and capacity, and frees the source id for a fresh add_user.",
     "code": r"""
cs = {fn}()
cs.add_user("a", 10)
assert cs.add_file_by("a", "f", 4) == 6
assert cs.copy_file("f", "g") is True and cs.add_file_by("a", "h", 2) == 0
assert cs.copy_file("h", "i") is False and cs.add_file_by("a", "j", 1) is None
assert cs.merge_user("a", "a") is None and cs.merge_user("a", "nobody") is None and cs.merge_user("nobody", "a") is None
cs.add_user("b", 5)
cs.add_file_by("b", "k", 5)
assert cs.merge_user("b", "a") == 0
assert cs.add_file_by("a", "z", 1) is None
assert cs.add_user("a", 3) is True and cs.add_file_by("a", "z", 3) == 0, "a new account under the old id starts empty"
assert cs.copy_file("f", "f2") is False, "f now belongs to b, which has no room"
"""},
    {"name": "Part 3: random calls", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "On a random sequence of calls with users, quotas, owned copies and merges, a return value differed from a model that recounts usage from the file table.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 70, 3), seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "checkpoint.recovery", "code": r"""
cs = {fn}()
cs.add_user("u-cy", 60)
assert cs.add_file_by("u-cy", "scan.tif", 41) == 19
assert cs.compress_file("u-cy", "scan.tif") == "scan.tif.zip" and cs.get_file_size("scan.tif.zip") == 21
assert cs.backup_user("u-cy") is True
assert cs.add_file_by("u-cy", "x.tif", 30) == 9
assert cs.decompress_file("u-cy", "scan.tif.zip") is None and cs.get_file_size("scan.tif.zip") == 21
assert cs.restore_user("u-cy") == 39 and cs.get_file_size("x.tif") is None
assert cs.decompress_file("u-cy", "scan.tif.zip") == "scan.tif" and cs.get_file_size("scan.tif") == 42
cs.add_file("scan.tif.zip", 1)
assert cs.restore_user("u-cy") is None
assert cs.get_file_size("scan.tif") == 42 and cs.get_file_size("scan.tif.zip") == 1
"""},
    {"name": "Part 4: compression rules", "part": 4, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "compress rounds up (size 1 stays 1) and may run on a .zip file; both directions need the user to own the file; decompress needs the .zip suffix, a free plain name and room for the extra bytes, and doubles exactly.",
     "code": r"""
cs = {fn}()
cs.add_user("u", 100)
cs.add_user("v", 100)
cs.add_file_by("u", "one", 1)
assert cs.compress_file("u", "one") == "one.zip" and cs.get_file_size("one.zip") == 1
assert cs.compress_file("u", "one.zip") == "one.zip.zip"
assert cs.compress_file("v", "one.zip.zip") is None, "v does not own it"
assert cs.compress_file("ghost", "one.zip.zip") is None
cs.add_file("free", 8)
assert cs.compress_file("u", "free") is None, "a file with no owner belongs to no user"
cs.add_file_by("u", "x", 9)
cs.add_file_by("u", "x.zip", 2)
assert cs.compress_file("u", "x") is None, "x.zip already exists"
assert cs.decompress_file("u", "x") is None, "no .zip suffix"
cs.add_file_by("u", "y.zip", 3)
cs.add_file_by("u", "y", 1)
assert cs.decompress_file("u", "y.zip") is None, "y already exists"
assert cs.decompress_file("u", "x.zip") is None, "x already exists"
cs.add_file_by("u", "w.zip", 7)
assert cs.decompress_file("v", "w.zip") is None, "v does not own it"
assert cs.decompress_file("u", "w.zip") == "w" and cs.get_file_size("w") == 14
"""},
    {"name": "Part 4: backups across merges and blocked restores", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
     "failure_message": "restore needs a backup for the current account and fails, changing nothing, when a name to re-create is held by another file; the same backup restores more than once; merging deletes the source's backup, and backups replace earlier ones.",
     "code": r"""
cs = {fn}()
cs.add_user("u", 50)
assert cs.restore_user("u") is None and cs.backup_user("nobody") is False
cs.add_file_by("u", "doc", 10)
cs.backup_user("u")
cs.compress_file("u", "doc")
cs.add_file("doc", 3)
assert cs.restore_user("u") is None, "doc is now an unrelated file"
assert cs.get_file_size("doc.zip") == 5 and cs.get_file_size("doc") == 3
cs.add_file_by("u", "late", 4)
cs.backup_user("u")
cs.add_file_by("u", "later", 4)
assert cs.restore_user("u") == 41 and cs.get_file_size("later") is None
cs.add_file_by("u", "again", 1)
assert cs.restore_user("u") == 41 and cs.get_file_size("again") is None
cs.add_user("t", 10)
cs.merge_user("t", "u")
cs.add_user("u", 5)
assert cs.restore_user("u") is None, "the merged-away account's backup is gone"
"""},
    {"name": "Part 4: random calls", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
     "failure_message": "On a random sequence of calls with compression, backups and restores, a return value differed from a model that recounts usage from the file table.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 80, 4), seed)
"""},
]

TASK = {
    "title": "Cloud Storage With Quotas",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "CloudStorage",
    "description_en": r"""Build `CloudStorage`, which tracks files by name and size only, then adds owners with quotas, compression and per-user backups.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `CloudStorage` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A file has a unique, case-sensitive name and a positive integer size. File contents are never stored.
- A size never changes while its name exists. Copying, compressing and decompressing create a file under another name.
- A call that fails changes nothing and returns the failure value given for it. No method raises.
- Every name, user id and size passed in is well formed.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment where each part reuses the last. Each later part adds one requirement, and the quota rules only stay simple if every file change goes through one place that also updates usage.

**Where it is used:** object stores and shared drives with per-user quotas, account merges, and per-user restore points.

Adapted from the cloud storage online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The compressed suffix is `.zip` instead of `.cmp`.""",
    "parts": [
        {
            "title": "Files, copies and sizes",
            "description_en": r"""**Signatures:**
- `CloudStorage().add_file(name, size) -> bool` creates the file. It returns `False` if the name is taken.
- `copy_file(source, destination) -> bool` creates `destination` with `source`'s size and keeps `source`. It returns `False` if `source` does not exist or `destination` does.
- `get_file_size(name) -> int | None` returns the size, or `None` if there is no such file.

**Example:**
- `add_file("a.log", 70)` is `True`; `copy_file("a.log", "b.log")` and then `copy_file("b.log", "c.log")` are `True`
- `get_file_size("c.log")` is `70`, and `get_file_size("A.log")` is `None`: names are case-sensitive
- `add_file("b.log", 9)` is `False`, and `copy_file("c.log", "a.log")` is `False`: both names are taken""",
        },
        {
            "title": "Search by prefix and suffix",
            "description_en": r"""Keep Part 1 and add a search.

**Signature:** `find_files(prefix, suffix) -> list[str]`

- Return every file whose name starts with `prefix` and ends with `suffix`, as `str.startswith` and `str.endswith` decide. An empty string matches every name.
- Each match is `"{name}({size})"`. Sort by size, largest first, then by name in Python's default string order.

**Example:** with the three `.log` files from Part 1 (70 each), add `ab.log` (200) and `a.txt` (5):
- `find_files("", ".log")` is `["ab.log(200)", "a.log(70)", "b.log(70)", "c.log(70)"]`
- `find_files("a", "")` is `["ab.log(200)", "a.log(70)", "a.txt(5)"]`
- `find_files("ab", "b.log")` is `["ab.log(200)"]`: the prefix and the suffix may overlap""",
        },
        {
            "title": "Users and quotas",
            "description_en": r"""Keep Parts 1–2 and add users who own files.

- A user has a fixed capacity in bytes. Its remaining capacity is the capacity minus the sizes of the files it owns.
- Files made by `add_file` have no owner and count against nobody.

**Signatures:**
- `add_user(user_id, capacity) -> bool` returns `False` if the id is taken.
- `add_file_by(user_id, name, size) -> int | None` creates a file owned by the user and returns the user's remaining capacity. It returns `None` if the user does not exist, the name is taken, or `size` is more than the remaining capacity.
- `merge_user(target, source) -> int | None` gives all of `source`'s files to `target`, adds `source`'s capacity to `target`'s, and deletes the user `source`. It returns `target`'s remaining capacity, or `None` if the two ids are equal or either user does not exist.
- `copy_file` now gives the copy the source's owner. If there is an owner, the copy counts against it, and `copy_file` returns `False` when the owner's remaining capacity is less than the size.

**Example:** with `a.log` (70 bytes, no owner) from Part 1:
- `add_user("u-ana", 100)` and `add_user("u-ben", 50)` are `True`; `add_file_by("u-ben", "b1", 50)` is `0`
- `copy_file("b1", "b2")` is `False`: `u-ben` has no room for a 50-byte copy
- `merge_user("u-ana", "u-ben")` is `100`: capacity `150`, used `50`
- now `copy_file("b1", "b2")` is `True` and the copy belongs to `u-ana`, leaving `50`
- after the merge, `add_file_by("u-ben", "x", 1)` is `None`; `add_file_by("u-ana", "a.log", 1)` is `None` too, since that name is taken""",
        },
        {
            "title": "Compression, backup and restore",
            "description_en": r"""Keep Parts 1–3 and add four methods.

**Signatures:**
- `compress_file(user_id, name) -> str | None` replaces the user's file `name` with `name + ".zip"`, same owner, size `ceil(size / 2)`, and returns the new name. It returns `None` if the user does not exist, does not own `name`, or `name + ".zip"` exists. A `.zip` file can be compressed again.
- `decompress_file(user_id, name) -> str | None` replaces the user's file `name`, which must end in `.zip`, with the name minus that suffix and double the size, and returns the new name. It returns `None` if the user does not exist, does not own `name`, the name has no `.zip` suffix, the shorter name exists, or `name`'s size is more than the remaining capacity, since doubling adds exactly that many bytes.
- `backup_user(user_id) -> bool` saves the user's capacity and the names and sizes of its files, replacing any earlier backup. It returns `False` if the user does not exist.
- `restore_user(user_id) -> int | None` returns the user to its backup: files it owns now that the backup lacks are deleted, files in the backup it does not own now are created again, and files in both are left as they are; the capacity is restored. It returns the remaining capacity, which may be negative. It returns `None` if the user does not exist, has no backup, or a file to create again is held by someone else or by nobody.
- A backup can be restored many times. `merge_user` deletes `source`'s backup with the user, so a new user under that id starts without one.

**Example:**
- `add_user("u-cy", 60)` and `add_file_by("u-cy", "scan.tif", 41)` is `19`; `compress_file("u-cy", "scan.tif")` is `"scan.tif.zip"`, size `21`
- `backup_user("u-cy")`, then `add_file_by("u-cy", "x.tif", 30)` is `9`, and `decompress_file("u-cy", "scan.tif.zip")` is `None`: doubling needs `21` more bytes
- `restore_user("u-cy")` is `39` and `x.tif` is gone; now `decompress_file("u-cy", "scan.tif.zip")` is `"scan.tif"`, size `42`
- `add_file("scan.tif.zip", 1)`, then `restore_user("u-cy")` is `None`: the backup needs `scan.tif.zip`, which now has no owner""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Since contents are never stored, what is the whole state of one file? Which single dict answers both \"does this name exist?\" and \"how big is it?\", and what must copy_file check before it writes anything?"},
        {"level": 2, "kind": "analysis", "content": "Keep files: name -> size. add_file fails if name in files, else stores it. copy_file fails if source not in files or destination in files, else files[destination] = files[source]. get_file_size returns files.get(name). Return the exact True/False/None values the part asks for."},
    ],
    "model_connections": [
        "GPU clusters charge jobs against per-team quotas and refuse a job that would exceed them, the same check add_file_by makes.",
        "Checkpoint stores keep per-run snapshots and restore one without touching other runs, as restore_user does per user.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A running used total per user makes every quota check O(1).",
            "Routing every create and delete through two helpers keeps the totals right without recounting.",
            "Backups that store names and sizes, not objects, cannot be changed by later edits.",
        ],
        "cons": [
            "find_files scans and sorts every file, O(n log n) per call.",
            "merge_user rewrites the owner of every file, O(n) unless files are grouped by owner.",
            "Restoring by name only cannot tell a file that was replaced by an unrelated one under the same name.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
SUFFIX = ".zip"


class CloudStorage:
    def __init__(self):
        self._files = {}     # name -> (size, owner or None)
        self._capacity = {}  # user_id -> capacity
        self._used = {}      # user_id -> bytes owned, kept in step with every file change
        self._backups = {}   # user_id -> (capacity, {name: size})

    def _create(self, name, size, owner):
        self._files[name] = (size, owner)
        if owner is not None:
            self._used[owner] += size

    def _remove(self, name):
        size, owner = self._files.pop(name)
        if owner is not None:
            self._used[owner] -= size

    def _left(self, user_id):
        return self._capacity[user_id] - self._used[user_id]

    def add_file(self, name, size):
        if name in self._files:
            return False
        self._create(name, size, None)
        return True

    def copy_file(self, source, destination):
        if source not in self._files or destination in self._files:
            return False
        size, owner = self._files[source]
        if owner is not None and size > self._left(owner):
            return False  # the copy counts against the source's owner
        self._create(destination, size, owner)
        return True

    def get_file_size(self, name):
        entry = self._files.get(name)
        return None if entry is None else entry[0]

    def find_files(self, prefix, suffix):
        hits = [(size, name) for name, (size, _) in self._files.items()
                if name.startswith(prefix) and name.endswith(suffix)]
        hits.sort(key=lambda hit: (-hit[0], hit[1]))
        return [f"{name}({size})" for size, name in hits]

    def add_user(self, user_id, capacity):
        if user_id in self._capacity:
            return False
        self._capacity[user_id], self._used[user_id] = capacity, 0
        return True

    def add_file_by(self, user_id, name, size):
        if user_id not in self._capacity or name in self._files or size > self._left(user_id):
            return None
        self._create(name, size, user_id)
        return self._left(user_id)

    def merge_user(self, target, source):
        if target == source or target not in self._capacity or source not in self._capacity:
            return None
        for name, (size, owner) in list(self._files.items()):
            if owner == source:
                self._files[name] = (size, target)
        self._capacity[target] += self._capacity.pop(source)
        self._used[target] += self._used.pop(source)
        self._backups.pop(source, None)  # the id no longer names this account
        return self._left(target)

    def _owned(self, user_id, name):
        return user_id in self._capacity and self._files.get(name, (0, None))[1] == user_id

    def compress_file(self, user_id, name):
        packed = name + SUFFIX
        if not self._owned(user_id, name) or packed in self._files:
            return None
        size = self._files[name][0]
        self._remove(name)
        self._create(packed, (size + 1) // 2, user_id)
        return packed

    def decompress_file(self, user_id, name):
        if not self._owned(user_id, name) or not name.endswith(SUFFIX):
            return None
        plain = name[:-len(SUFFIX)]
        size = self._files[name][0]
        if plain in self._files or size > self._left(user_id):  # the extra bytes equal the current size
            return None
        self._remove(name)
        self._create(plain, size * 2, user_id)
        return plain

    def backup_user(self, user_id):
        if user_id not in self._capacity:
            return False
        owned = {name: size for name, (size, owner) in self._files.items() if owner == user_id}
        self._backups[user_id] = (self._capacity[user_id], owned)
        return True

    def restore_user(self, user_id):
        if user_id not in self._capacity or user_id not in self._backups:
            return None
        capacity, saved = self._backups[user_id]
        owned = {name for name, (_, owner) in self._files.items() if owner == user_id}
        drop = owned - set(saved)
        if any(name in self._files and name not in drop for name in saved if name not in owned):
            return None  # a file someone else holds blocks the restore
        for name in drop:
            self._remove(name)
        for name, size in saved.items():
            if name not in owned:
                self._create(name, size, user_id)
        self._capacity[user_id] = capacity
        return self._left(user_id)
''',
    "interview_questions": interview(
        concept=[
            "What is the whole state of a file here, and which data structure holds it?",
            "Why must a failed call change nothing, and how do you make sure copy_file never leaves a half-made file?",
        ],
        deep_dive=[
            "Which checks must copy_file make, and in which order, before it creates the destination?",
        ],
        tradeoffs=[
            "Would you keep a running used total per user or recompute it from the files, and what does each cost?",
            "How would you make find_files fast for millions of files with frequent prefix searches?",
            "Why does decompress_file compare the current size, not the doubled size, against the remaining capacity?",
            "What can go wrong when restore matches files by name only, and how would you avoid it?",
        ],
    ),
}
