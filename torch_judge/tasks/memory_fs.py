"""An in-memory tree of directories and files, then moves and copies, per-user quotas and snapshots."""

from ._interview import interview

# A flat path-table model of every rule, independent of the tree in the reference, and a random call generator.
_MODEL = r"""
import copy, random

def parts_of(path):
    if not path.startswith("/"):
        return None
    parts = tuple(p for p in path.split("/") if p)
    return None if any(p in (".", "..") for p in parts) else parts

class Model:
    # A flat table: path tuple -> None for a directory, or [content, owner] for a file.
    def __init__(self):
        self.nodes, self.cap, self.snaps = {(): None}, {}, {}
    def is_file(self, p):
        return self.nodes.get(p) is not None
    def blocked(self, p):
        return any(self.is_file(p[:i]) for i in range(1, len(p)))
    def used(self, u):
        return sum(len(n[0]) for n in self.nodes.values() if n is not None and n[1] == u)
    def under(self, p):
        return [k for k in self.nodes if k[:len(p)] == p]
    def make_dirs(self, p):
        for i in range(1, len(p) + 1):
            self.nodes.setdefault(p[:i], None)
    def mkdir(self, path):
        p = parts_of(path)
        if p is None or any(self.is_file(p[:i]) for i in range(1, len(p) + 1)):
            return False
        self.make_dirs(p)
        return True
    def write(self, path, content, owner="admin"):
        p = parts_of(path)
        if not p or self.blocked(p) or (p in self.nodes and self.nodes[p] is None):
            return False
        if owner != "admin" and owner not in self.cap:
            return False
        old = self.nodes.get(p)
        if old is not None and old[1] != owner:
            return False
        if owner != "admin" and self.used(owner) - (len(old[0]) if old else 0) + len(content) > self.cap[owner]:
            return False
        self.make_dirs(p[:-1])
        self.nodes[p] = [content, owner]
        return True
    def read(self, path):
        p = parts_of(path)
        return self.nodes[p][0] if p is not None and self.is_file(p) else None
    def ls(self, path):
        p = parts_of(path)
        if p is None or p not in self.nodes or self.is_file(p):
            return None
        return sorted(k[-1] for k in self.nodes if len(k) == len(p) + 1 and k[:len(p)] == p)
    def transfer(self, src, dst, keep):
        s, d = parts_of(src), parts_of(dst)
        if s is None or d is None or s not in self.nodes:
            return False
        n = min(len(s), len(d))
        if s[:n] == d[:n] or self.blocked(d):
            return False
        if d in self.nodes and self.is_file(d) != self.is_file(s):
            return False
        if self.is_file(d) and self.nodes[d][1] != self.nodes[s][1]:
            return False
        moved = {d + k[len(s):]: copy.deepcopy(self.nodes[k]) for k in self.under(s)}
        after = {k: v for k, v in self.nodes.items() if k[:len(d)] != d and (keep or k[:len(s)] != s)}
        after.update(moved)
        if keep:
            for u in self.cap:
                mine = sum(len(v[0]) for v in after.values() if v is not None and v[1] == u)
                if mine > self.used(u) and mine > self.cap[u]:
                    return False
        self.nodes = after
        self.make_dirs(d[:-1])
        return True
    def mv(self, src, dst):
        return self.transfer(src, dst, False)
    def cp(self, src, dst):
        return self.transfer(src, dst, True)
    def rm(self, path):
        p = parts_of(path)
        if not p or p not in self.nodes:
            return False
        for k in self.under(p):
            del self.nodes[k]
        return True
    def size(self, path):
        p = parts_of(path)
        if p is None or p not in self.nodes:
            return None
        return sum(len(self.nodes[k][0]) for k in self.under(p) if self.nodes[k] is not None)
    def add_user(self, u, cap):
        if u == "admin" or u in self.cap or cap < 0:
            return False
        self.cap[u] = cap
        return True
    def update_capacity(self, u, cap):
        if u not in self.cap or cap < 0:
            return None
        self.cap[u] = cap
        mine = sorted((-len(v[0]), "/" + "/".join(k)) for k, v in self.nodes.items() if v is not None and v[1] == u)
        gone = 0
        for _, path in mine:
            if self.used(u) <= cap:
                break
            self.rm(path)
            gone += 1
        return gone
    def snapshot(self, sid):
        if sid in self.snaps:
            return False
        self.snaps[sid] = copy.deepcopy((self.nodes, self.cap))
        return True
    def restore(self, sid):
        if sid not in self.snaps:
            return False
        self.nodes, self.cap = copy.deepcopy(self.snaps[sid])
        return True

PATHS = ["/", "/a", "/a/", "//a//b", "/a/b", "/a/b/c", "/a/c", "/b", "/b/a", "/A", "/a/./b", "a/b", "/a/..", "/b/c/d"]

def random_calls(rng, n, level):
    ops = ["mkdir", "write", "write", "read", "ls"]
    if level >= 2:
        ops += ["mv", "cp", "cp", "rm", "size"]
    if level >= 3:
        ops += ["add_user", "write_by", "write_by", "update_capacity"]
    if level >= 4:
        ops += ["snapshot", "restore"]
    model, out = Model(), []
    for _ in range(n):
        op = rng.choice(ops)
        p, q = rng.choice(PATHS), rng.choice(PATHS)
        text = rng.choice(["", "x", "hello", "0123456789"])
        user = rng.choice(["u", "v", "admin"])
        args = {"mkdir": (p,), "write": (p, text), "read": (p,), "ls": (p,), "mv": (p, q), "cp": (p, q), "rm": (p,),
                "size": (p,), "add_user": (user, rng.randint(-1, 20)), "write_by": (p, text, user),
                "update_capacity": (user, rng.randint(-1, 12)), "snapshot": (rng.choice("st"),), "restore": (rng.choice("stx"),)}[op]
        name = "write" if op == "write_by" else op
        out.append((name, args, getattr(model, name)(*args)))
    return out

def replay(fs, calls, label):
    for i, (op, args, want) in enumerate(calls):
        got = getattr(fs, op)(*args)
        assert got == want, (label, i, op, args, got, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
fs = {fn}()
assert fs.mkdir("/proj//docs/") is True
assert fs.write("/proj/src/main.py", "print(1)") is True and fs.read("/proj/src/main.py") == "print(1)"
assert fs.ls("/proj") == ["docs", "src"] and fs.ls("/") == ["proj"]
assert fs.write("/proj/docs", "x") is False
assert fs.mkdir("/proj/src/main.py/x") is False
assert fs.read("/proj/../proj/src/main.py") is None
assert fs.ls("/proj/src") == ["main.py"]
"""},
    {"name": "Part 1: paths and kinds", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Paths must start with /, collapse repeated slashes and one trailing slash, and reject . and .. components; the root is a directory that cannot be written; mkdir fails when any part of the path is a file; ls sorts in Python order (uppercase first).",
     "code": r"""
fs = {fn}()
assert fs.ls("/") == [] and fs.write("/", "x") is False and fs.read("/") is None
for bad in ["a/b", "", "/a/../b", "/..", "/./x"]:
    assert fs.mkdir(bad) is False and fs.write(bad, "x") is False and fs.ls(bad) is None and fs.read(bad) is None, bad
assert fs.write("//x///y", "1") and fs.read("/x/y") == "1" and fs.read("/x/y/") == "1"
assert fs.mkdir("/x/y/z") is False and fs.mkdir("/x/y") is False
assert fs.write("/x/y", "22") and fs.read("/x/y") == "22", "write overwrites a file"
for name in ["b", "B", "a", "_", "Z"]:
    fs.write("/x/" + name, "")
assert fs.ls("/x") == ["B", "Z", "_", "a", "b", "y"]
assert fs.read("/x/a") == "", "an empty file is still a file"
assert fs.mkdir("/") is True
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of mkdir, write, read and ls calls, a return value differed from a flat path-table model.",
     "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_calls(random.Random(seed), 50, 1), seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
fs = {fn}()
fs.write("/lab/notes.md", "hi")
fs.write("/lab/run/log.txt", "step 1")
assert fs.mv("/lab/run", "/archive/run1") is True
assert fs.ls("/lab") == ["notes.md"] and fs.read("/archive/run1/log.txt") == "step 1"
assert fs.cp("/lab/notes.md", "/archive/run1/log.txt") is True
assert fs.read("/archive/run1/log.txt") == "hi" and fs.size("/archive") == 2
fs.write("/archive/run1/log.txt", "edited")
assert fs.read("/lab/notes.md") == "hi", "a copy is independent"
assert fs.cp("/lab/notes.md", "/archive") is False
assert fs.mv("/archive", "/archive/run1/old") is False
assert fs.rm("/lab/notes.md") is True and fs.size("/lab") == 0 and fs.ls("/lab") == []
assert fs.rm("/") is False
"""},
    {"name": "Part 2: move and copy rules", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "mv and cp fail on a missing source, the same path, a path inside the other, a file ancestor of the destination, or a kind mismatch; a file over a file replaces it, a directory over a directory replaces the whole subtree; a failed call creates no directories; copies are independent.",
     "code": r"""
fs = {fn}()
fs.write("/d/f", "12")
fs.write("/d/g", "345")
fs.write("/e/old", "zzzz")
fs.write("/e/keep", "k")
assert fs.mv("/d", "/d") is False and fs.cp("/d", "/d/sub") is False and fs.mv("/d/f", "/d") is False
assert fs.cp("/missing", "/n/x") is False and fs.ls("/n") is None, "a failed call must not create /n"
assert fs.cp("/d/f", "/d/g/x") is False, "/d/g is a file"
assert fs.mv("/d", "/e/keep") is False and fs.mv("/d/f", "/e") is False, "kinds must match"
assert fs.cp("/d/g", "/d/f") is True and fs.read("/d/f") == "345"
assert fs.cp("/d", "/e") is True
assert fs.ls("/e") == ["f", "g"], "a directory over a directory replaces its whole subtree"
fs.write("/e/f", "new")
assert fs.read("/d/f") == "345"
assert fs.mv("/e", "/deep/er/e") is True and fs.ls("/") == ["d", "deep"] and fs.read("/deep/er/e/f") == "new"
assert fs.size("/deep") == 6 and fs.size("/nope") is None and fs.size("/") == 12
assert fs.rm("/nope") is False and fs.rm("/d/./f") is False
assert fs.mv("/", "/x") is False and fs.cp("/d", "/") is False
"""},
    {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random sequence of calls including mv, cp, rm and size, a return value differed from a flat path-table model.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 60, 2), seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "budget.enforcement", "code": r"""
fs = {fn}()
assert fs.add_user("ml", 10) is True and fs.add_user("ops", 3) is True
assert fs.add_user("admin", 5) is False and fs.add_user("ml", 1) is False
assert fs.write("/team/ml/w.bin", "aaaa", owner="ml") is True and fs.write("/team/ops/cfg", "bb", owner="ops") is True
assert fs.cp("/team", "/team2") is False and fs.ls("/") == ["team"]
assert fs.mv("/team", "/team2") is True
assert fs.write("/team2/ops/cfg", "ccc") is False
assert fs.write("/x/p1", "zz", owner="ml") is True and fs.write("/x/p0", "zz", owner="ml") is True
assert fs.update_capacity("ml", 2) == 2
assert fs.read("/team2/ml/w.bin") is None and fs.read("/x/p0") is None and fs.read("/x/p1") == "zz"
assert fs.ls("/team2/ml") == [], "eviction leaves the directory in place"
"""},
    {"name": "Part 3: owners, quotas and eviction order", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "Unknown owners fail; an overwrite frees the old size first; cp of a directory fails as a whole if any owner would go over; mv keeps owners and never checks quotas; rm frees bytes; update_capacity evicts largest first, ties by path, and returns 0 when nothing must go and None for unknown users or negative capacity.",
     "code": r"""
fs = {fn}()
assert fs.write("/x", "a", owner="ghost") is False
assert fs.add_user("p", 6) and fs.add_user("q", 6) and fs.add_user("r", -1) is False
assert fs.write("/p/1", "aaaa", owner="p") and fs.write("/p/1", "bbbbbb", owner="p"), "overwrite frees the old 4 bytes"
assert fs.write("/p/2", "c", owner="p") is False
fs.write("/mix/q1", "qq", owner="q")
fs.write("/mix/a1", "admin-data")
fs.mkdir("/mix/empty")
assert fs.cp("/mix", "/mix2") is True
assert fs.cp("/mix", "/mix3") is True and fs.cp("/mix", "/mix4") is False, "q would need 8 bytes"
assert fs.ls("/") == ["mix", "mix2", "mix3", "p"], "a failed cp creates nothing"
assert fs.mv("/p/1", "/q/moved") is True and fs.write("/q/moved", "x", owner="p") is True, "mv keeps the owner"
assert fs.write("/q/moved", "y", owner="q") is False
fs.rm("/mix2")
assert fs.write("/q/q2", "qq", owner="q") is True, "rm freed q's bytes"
assert fs.update_capacity("q", 6) == 0 and fs.update_capacity("admin", 1) is None and fs.update_capacity("q", -1) is None
assert fs.update_capacity("q", 2) == 2
assert fs.read("/mix/q1") is None and fs.read("/mix3/q1") is None and fs.read("/q/q2") == "qq", "ties go by path: /mix/q1, then /mix3/q1"
assert fs.add_user("t", 4) and fs.write("/b/z1", "tt", owner="t") and fs.write("/a/z2", "tt", owner="t")
assert fs.update_capacity("t", 2) == 1
assert fs.read("/a/z2") is None and fs.read("/b/z1") == "tt", "the tie goes to the smaller full path, /a/z2"
"""},
    {"name": "Part 3: random calls", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "On a random sequence of calls with owners and quotas, a return value differed from a flat path-table model that recounts usage.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 70, 3), seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "checkpoint.recovery", "code": r"""
fs = {fn}()
assert fs.add_user("bo", 6) is True and fs.write("/a.txt", "1") is True
assert fs.snapshot("base") is True
assert fs.add_user("cy", 3) is True and fs.write("/c.txt", "xyz", owner="cy") is True
assert fs.restore("base") is True
assert fs.read("/c.txt") is None and fs.add_user("cy", 1) is True
assert fs.write("/a.txt", "2") is True
assert fs.restore("base") is True and fs.read("/a.txt") == "1"
assert fs.add_user("bo", 1) is False and fs.write("/b.txt", "123456", owner="bo") is True
assert fs.snapshot("base") is False and fs.restore("nope") is False
"""},
    {"name": "Part 4: snapshots stay independent", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
     "failure_message": "Changes after a snapshot or after a restore must never reach the stored snapshot, restoring twice gives the same state, and users added after a snapshot disappear on restore.",
     "code": r"""
fs = {fn}()
fs.write("/a/f", "1")
fs.snapshot("s")
fs.write("/a/f", "2")
fs.restore("s")
fs.write("/a/f", "3")
fs.write("/a/g", "4")
fs.restore("s")
assert fs.ls("/a") == ["f"] and fs.read("/a/f") == "1"
fs.add_user("late", 5)
fs.restore("s")
assert fs.write("/a/h", "x", owner="late") is False and fs.add_user("late", 5) is True
"""},
    {"name": "Part 4: random calls", "part": 4, "visibility": "unshown", "behavior": "checkpoint.recovery",
     "failure_message": "On a random sequence of calls with snapshots and restores, a return value differed from a flat path-table model.",
     "code": _MODEL + r"""
for seed in range(200):
    replay({fn}(), random_calls(random.Random(seed), 80, 4), seed)
"""},
]

TASK = {
    "title": "In-Memory File System",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "MemoryFS",
    "description_en": r"""Build `MemoryFS`, an in-memory tree of directories and text files, then add moves and copies, per-user quotas and snapshots.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `MemoryFS` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A path starts with `/`. Repeated slashes count as one and a trailing slash is ignored, so `"/a//b/"` is `"/a/b"`, and `"/"` is the root directory, which always exists.
- A path is invalid if it does not start with `/` or has a component equal to `.` or `..`. An invalid path behaves like a missing one.
- Names are case-sensitive. A file's size is the length of its content.
- A call that creates something at a path also creates its missing parent directories. A call that only reads, removes, or moves away from a path creates nothing.
- A call that fails changes nothing and returns the failure value given for it. No method raises.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment that rewards a tree with one place for path handling. Each later part adds one requirement, and the quota and snapshot parts break any shortcut where state lives in more than one copy.

**Where it is used:** in-memory file systems for tests (pyfakefs, memfs), container layers with quotas, and copy-on-write snapshots in ZFS or btrfs.

Adapted from the file system online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class with a new name.""",
    "parts": [
        {
            "title": "Directories and files",
            "description_en": r"""**Signatures:**
- `MemoryFS().mkdir(path) -> bool` creates the directory and any missing parents, like `mkdir -p`. It returns `True` if the directory exists afterwards, and `False` if the path is invalid or any part of it is a file.
- `write(path, content) -> bool` creates or overwrites the file. It returns `False` if the path is invalid, is `"/"`, is a directory, or has a file as a parent.
- `read(path) -> str | None` returns the content, or `None` if there is no file there.
- `ls(path) -> list[str] | None` returns the names directly inside a directory, files and directories together, sorted in Python's default string order. It returns `None` if there is no directory there.

**Example:**
- `mkdir("/proj//docs/")` is `True`; `write("/proj/src/main.py", "print(1)")` is `True` and creates `/proj/src`
- `ls("/proj")` is `["docs", "src"]` and `ls("/")` is `["proj"]`
- `write("/proj/docs", "x")` is `False` because `/proj/docs` is a directory, and `mkdir("/proj/src/main.py/x")` is `False` because a parent is a file
- `read("/proj/../proj/src/main.py")` is `None`: `..` makes the path invalid""",
        },
        {
            "title": "Move, copy, delete and size",
            "description_en": r"""Keep Part 1 and add four methods.

**Signatures:**
- `mv(src, dst) -> bool` moves `src`, with its whole subtree, to `dst`. If `dst` is missing, it is created there. If `dst` is the same kind, a file replaces the file and a directory replaces the whole directory.
- `mv` returns `False` if either path is invalid, `src` is missing, the two paths are equal or one is inside the other, a parent of `dst` is a file, or `dst` exists as the other kind.
- `cp(src, dst) -> bool` follows the same rules but keeps `src`. The copy is independent: writing to one never changes the other.
- `rm(path) -> bool` deletes a file, or a directory with everything in it. It returns `False` for an invalid or missing path and for `"/"`.
- `size(path) -> int | None` is a file's size, or the total size of all files under a directory. It returns `None` for an invalid or missing path.

**Example:** after `write("/lab/notes.md", "hi")` and `write("/lab/run/log.txt", "step 1")`:
- `mv("/lab/run", "/archive/run1")` is `True` and moves the whole directory, so `ls("/lab")` is `["notes.md"]`
- `cp("/lab/notes.md", "/archive/run1/log.txt")` is `True`: the file replaces the file, and `size("/archive")` is `2`
- `cp("/lab/notes.md", "/archive")` is `False`: a file cannot replace a directory
- `mv("/archive", "/archive/run1/old")` is `False`; `rm("/lab/notes.md")` is `True`, leaving `size("/lab")` at `0`""",
        },
        {
            "title": "Owners and quotas",
            "description_en": r"""Keep Parts 1–2 and give every file an owner.

- The user `"admin"` always exists and has no limit. Every file has an owner fixed when it is created. A file counts against its owner's capacity unless the owner is `"admin"`.
- `add_user(user_id, capacity) -> bool` registers a user with no files. It returns `False` if the id is `"admin"` or taken, or `capacity < 0`.
- `write(path, content, owner="admin")` also returns `False` if `owner` is not `"admin"` or a registered user, if a file exists at `path` with a different owner, or if the owner's total after the write, the old content no longer counted, would be over capacity.
- `mv` and `cp` keep each file's owner. They fail if `dst` is a file whose owner differs from `src`'s owner. A directory that replaces a directory discards everything under it, whatever the owners. `cp` counts the copies like writes and fails as a whole if any owner would go over capacity; `mv` never fails on capacity.
- `rm` and every overwrite free the removed bytes from their owner.
- `update_capacity(user_id, capacity) -> int | None` sets the capacity. While the user is over it, delete its largest file, ties going to the smaller full path string, and return how many files were deleted. Their directories stay, even if empty. It returns `None` for an unknown user, `"admin"`, or `capacity < 0`.

**Example:** `add_user("ml", 10)` and `add_user("ops", 3)`, then `write("/team/ml/w.bin", "aaaa", owner="ml")` and `write("/team/ops/cfg", "bb", owner="ops")`:
- `cp("/team", "/team2")` is `False` and creates nothing: `ops` would need `4` bytes
- `mv("/team", "/team2")` is `True`, since moves never check quotas; `write("/team2/ops/cfg", "ccc")` is `False`, since the default owner is `"admin"`
- writing `"zz"` as `ml` to `/x/p1` and then `/x/p0` brings `ml` to `8`; `update_capacity("ml", 2)` is `2`: it deletes `w.bin`, the largest, then `/x/p0`, which wins the tie with `/x/p1` by path order""",
        },
        {
            "title": "Snapshots",
            "description_en": r"""Keep Parts 1–3 and save and restore the whole state.

- `snapshot(snapshot_id) -> bool` saves every directory, file and owner, and every registered user with its capacity, under the id. It returns `False` if the id was used before.
- `restore(snapshot_id) -> bool` replaces the whole state with a copy of that snapshot. It returns `False` for an unknown id.
- Nothing done after a snapshot, including after restoring it, may change it, so restoring the same id twice gives the same state.

**Example:** `add_user("bo", 6)`, `write("/a.txt", "1")`, then `snapshot("base")` is `True`:
- `add_user("cy", 3)` and a write of `/c.txt` as `cy`, then `restore("base")`: `/c.txt` is gone and `cy` is no longer a user, so `add_user("cy", 1)` is `True`
- `write("/a.txt", "2")`, then `restore("base")` again: `read("/a.txt")` is `"1"`
- `snapshot("base")` is `False` and `restore("nope")` is `False`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If every directory is a dict from child name to child, what is a file? Which one helper turns \"/a//b/\" into the list of names to walk, and what should it return for \"/a/../b\" so every method can treat that path as missing?"},
        {"level": 2, "kind": "analysis", "content": "Write parts(path): reject paths not starting with \"/\", split on \"/\", drop empty pieces, reject \".\" and \"..\". Walk from the root dict through each name. mkdir: first check nothing on the way is a file, then setdefault each dict. write: check every parent is a dict or missing and the target is not a dict, then create parents and store the file."},
    ],
    "model_connections": [
        "Dataset and checkpoint stores lay files out in trees with per-team quotas, and evict the largest files first when a quota shrinks.",
        "Experiment tracking snapshots a run's whole state so a later restore reproduces it exactly, the same contract as snapshot and restore here.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A tree of dicts makes every path operation proportional to its depth.",
            "Checking every failure condition before mutating makes each call all or nothing.",
            "Deep-copying for snapshots is simple and keeps them independent of later changes.",
        ],
        "cons": [
            "size and update_capacity walk whole subtrees, O(n) per call unless totals are cached per directory.",
            "Full-copy snapshots cost memory proportional to the whole tree each time.",
            "Owner totals must be updated on every create, overwrite, move and delete, and one missed path leaves them wrong.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import copy

ADMIN = "admin"


class _File:
    def __init__(self, content, owner):
        self.content = content
        self.owner = owner


def _parts(path):
    """Components of a valid path, [] for the root, or None for an invalid path."""
    if not path.startswith("/"):
        return None
    parts = [p for p in path.split("/") if p]  # collapses repeated and trailing slashes
    return None if any(p in (".", "..") for p in parts) else parts


def _files(node, prefix):
    """(path, file) for every file under node; prefix is node's own path."""
    if isinstance(node, _File):
        yield prefix, node
        return
    for name, child in node.items():
        yield from _files(child, prefix.rstrip("/") + "/" + name)


class MemoryFS:
    def __init__(self):
        self._root = {}
        self._capacity = {}  # user -> capacity; admin is never here
        self._used = {}      # user -> bytes owned
        self._snapshots = {}

    # -- lookups ---------------------------------------------------------
    def _get(self, parts):
        node = self._root
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                return None
            node = node[p]
        return node

    def _blocked(self, parts):
        """True if some ancestor of parts exists as a file."""
        node = self._root
        for p in parts[:-1]:
            node = node.get(p)
            if node is None:
                return False
            if isinstance(node, _File):
                return True
        return False

    def _parent(self, parts):
        node = self._root
        for p in parts[:-1]:
            node = node.setdefault(p, {})  # like mkdir -p; callers check _blocked first
        return node

    def _bill(self, owner, delta):
        if owner != ADMIN:
            self._used[owner] += delta

    # -- part 1 ----------------------------------------------------------
    def mkdir(self, path):
        parts = _parts(path)
        if parts is None:
            return False
        node = self._root
        for p in parts:
            if isinstance(node.get(p), _File):
                return False
            node = node.get(p, {})
        node = self._root
        for p in parts:
            node = node.setdefault(p, {})
        return True

    def write(self, path, content, owner=ADMIN):
        parts = _parts(path)
        if not parts or self._blocked(parts) or isinstance(self._get(parts), dict):
            return False
        if owner != ADMIN and owner not in self._capacity:
            return False
        old = self._get(parts)
        if old is not None and old.owner != owner:
            return False  # a file's owner never changes
        freed = len(old.content) if old is not None else 0
        if owner != ADMIN and self._used[owner] - freed + len(content) > self._capacity[owner]:
            return False
        if old is not None:
            self._bill(owner, -freed)
            old.content = content
        else:
            self._parent(parts)[parts[-1]] = _File(content, owner)
        self._bill(owner, len(content))
        return True

    def read(self, path):
        parts = _parts(path)
        node = None if parts is None else self._get(parts)
        return node.content if isinstance(node, _File) else None

    def ls(self, path):
        parts = _parts(path)
        node = None if parts is None else self._get(parts)
        return sorted(node) if isinstance(node, dict) else None

    # -- part 2 ----------------------------------------------------------
    def _transfer(self, src, dst, keep_source):
        sp, dp = _parts(src), _parts(dst)
        if sp is None or dp is None:
            return False
        node = self._get(sp)
        if node is None:
            return False
        shorter = min(len(sp), len(dp))
        if sp[:shorter] == dp[:shorter]:
            return False  # same path, or one inside the other
        if self._blocked(dp):
            return False
        target = self._get(dp)
        if target is not None and isinstance(target, _File) != isinstance(node, _File):
            return False
        if isinstance(target, _File) and target.owner != node.owner:
            return False
        moved = copy.deepcopy(node) if keep_source else node
        delta = {}
        if target is not None:
            for _, f in _files(target, "/"):
                delta[f.owner] = delta.get(f.owner, 0) - len(f.content)  # overwritten files free their bytes
        if keep_source:
            for _, f in _files(moved, "/"):
                delta[f.owner] = delta.get(f.owner, 0) + len(f.content)
            for owner, change in delta.items():
                if owner != ADMIN and change > 0 and self._used[owner] + change > self._capacity[owner]:
                    return False  # all or nothing
        else:
            del self._get(sp[:-1])[sp[-1]]
        for owner, change in delta.items():
            self._bill(owner, change)
        self._parent(dp)[dp[-1]] = moved
        return True

    def mv(self, src, dst):
        return self._transfer(src, dst, keep_source=False)

    def cp(self, src, dst):
        return self._transfer(src, dst, keep_source=True)

    def rm(self, path):
        parts = _parts(path)
        if not parts:
            return False  # invalid, or the root
        node = self._get(parts)
        if node is None:
            return False
        for _, f in _files(node, "/"):
            self._bill(f.owner, -len(f.content))
        del self._get(parts[:-1])[parts[-1]]
        return True

    def size(self, path):
        parts = _parts(path)
        node = None if parts is None else self._get(parts)
        if node is None:
            return None
        return sum(len(f.content) for _, f in _files(node, "/"))

    # -- part 3 ----------------------------------------------------------
    def add_user(self, user_id, capacity):
        if user_id == ADMIN or user_id in self._capacity or capacity < 0:
            return False
        self._capacity[user_id], self._used[user_id] = capacity, 0
        return True

    def update_capacity(self, user_id, capacity):
        if user_id not in self._capacity or capacity < 0:
            return None
        self._capacity[user_id] = capacity
        owned = [(path, f) for path, f in _files(self._root, "/") if f.owner == user_id]
        owned.sort(key=lambda item: (-len(item[1].content), item[0]))  # largest first, then path order
        evicted = 0
        for path, _ in owned:
            if self._used[user_id] <= capacity:
                break
            self.rm(path)
            evicted += 1
        return evicted

    # -- part 4 ----------------------------------------------------------
    def snapshot(self, snapshot_id):
        if snapshot_id in self._snapshots:
            return False
        self._snapshots[snapshot_id] = copy.deepcopy((self._root, self._capacity, self._used))
        return True

    def restore(self, snapshot_id):
        if snapshot_id not in self._snapshots:
            return False
        self._root, self._capacity, self._used = copy.deepcopy(self._snapshots[snapshot_id])  # the snapshot stays intact
        return True
''',
    "interview_questions": interview(
        concept=[
            "How do you represent directories and files so that both mkdir and write can walk the same path?",
            "Why normalise and validate a path in one helper instead of in each method?",
        ],
        deep_dive=[
            "What must write check, and in which order, so that a failed call creates no parent directories?",
        ],
        tradeoffs=[
            "How do you stop mv from moving a directory into its own subtree?",
            "How would you keep size fast on a deep tree that changes often?",
            "Why must cp check every owner's quota before copying anything?",
            "How would you make snapshots cheaper than a full copy of the tree?",
        ],
    ),
}
