"""Group files with identical bytes, first correctly, then cheaply, then in parallel."""

from ._interview import interview

# Every case builds a real tree in a temporary directory. ReadMonitor replaces the built-in
# open while a call runs, so a case can count the bytes read, refuse a file, or slow reads down,
# without depending on file permissions or on the machine's disk. The swap is process-wide; it is
# safe because the grader runs one case at a time under its stdout lock.
_HELPERS = r"""
import builtins, io, os, random, tempfile, threading, time

REAL_OPEN = builtins.open

def make_tree(root, files):
    for rel, data in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with REAL_OPEN(path, "wb") as out:
            out.write(data)

def p(root, rel):
    return os.path.join(root, *rel.split("/"))

def brute_force(root):
    # Compare every pair of regular files byte for byte and merge the matches.
    paths = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path) and os.path.isfile(path):
                paths.append(path)
    data = {}
    for path in paths:
        with REAL_OPEN(path, "rb") as f:
            data[path] = f.read()
    groups = []
    for path in sorted(paths):
        for group in groups:
            if data[group[0]] == data[path]:
                group.append(path)
                break
        else:
            groups.append([path])
    return sorted([g for g in groups if len(g) >= 2], key=lambda g: g[0])

class _CountingFile(io.FileIO):
    def __init__(self, path, monitor):
        super().__init__(path, "r")
        self._monitor = monitor
    def _track(self, read):
        monitor = self._monitor
        with monitor.lock:
            monitor.active += 1
            monitor.peak = max(monitor.peak, monitor.active)
        try:
            if monitor.delay:
                time.sleep(monitor.delay)
            if os.path.realpath(self.name) in monitor.broken:
                raise OSError(5, "Input/output error", self.name)
            data = read()
        finally:
            with monitor.lock:
                monitor.active -= 1
        with monitor.lock:
            count = data if isinstance(data, int) else len(data or b"")
            monitor.bytes += count
            monitor.largest = max(monitor.largest, count)
        return data
    def readinto(self, buffer):
        return self._track(lambda: super(_CountingFile, self).readinto(buffer))
    def readall(self):
        return self._track(lambda: super(_CountingFile, self).readall())

class ReadMonitor:
    def __init__(self, root, delay=0.0, deny=(), broken=()):
        # deny: opening raises PermissionError. broken: opening works, reading raises OSError.
        self.root = os.path.realpath(root)
        self.delay, self.deny = delay, {os.path.realpath(d) for d in deny}
        self.broken = {os.path.realpath(b) for b in broken}
        self.bytes = self.active = self.peak = self.largest = 0
        self.lock = threading.Lock()
    def open(self, file, mode="r", *args, **kwargs):
        if isinstance(file, int) or not os.path.realpath(os.fspath(file)).startswith(self.root):
            return REAL_OPEN(file, mode, *args, **kwargs)
        if os.path.realpath(os.fspath(file)) in self.deny:
            raise PermissionError(13, "Permission denied", os.fspath(file))
        if any(flag in mode for flag in "wax+"):
            return REAL_OPEN(file, mode, *args, **kwargs)
        buffered = io.BufferedReader(_CountingFile(os.fspath(file), self), buffer_size=1)
        if "b" in mode:
            return buffered
        return io.TextIOWrapper(buffered, encoding=kwargs.get("encoding"), errors=kwargs.get("errors"))
    def __enter__(self):
        builtins.open = io.open = self.open
        return self
    def __exit__(self, *exc):
        builtins.open = io.open = REAL_OPEN
"""

TASK = {
    "title": "File Deduplication",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "find_duplicate_files",
    "description_en": r"""Write `find_duplicate_files(root_path)`, which walks a directory tree and returns every group of files whose bytes are identical.

The requirement arrives in parts. Each part keeps every earlier behavior, so one function passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Walk every real directory under `root_path`, at any depth. Two files are duplicates exactly when they hold the same bytes; there is no text decoding.
- Return a list of groups. Each group lists at least two paths, sorted ascending; the groups are sorted by their first path. Files with no duplicate are left out, so a tree without duplicates gives `[]`.
- Build each path with `os.path.join` under `root_path`, as `os.walk` produces it.
- All empty files form one group, like any other identical content.
- A symbolic link is never followed and never reported, whatever it points to, and a directory reached only through a link is not walked.
- A file that cannot be opened or read is skipped: it is in no group, and the rest of the walk still completes.
- Read file content only with the built-in `open`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first version fits in a few lines, and each later part adds one requirement.

**Where it is used:** backup tools, storage deduplication and cleanup utilities such as fdupes and rmlint.

Adapted from the file deduplication question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one function. The process-pool option was dropped, and the chunk size and sample size are now requirements.""",
    "parts": [
        {
            "title": "Correct grouping",
            "description_en": r"""**Signature:** `find_duplicate_files(root_path) -> list[list[str]]`

- Return the groups exactly as the rules above define them. Reading every file in full is fine for this part.

**Example:** for a tree holding
- `a.txt` and `docs/b.txt`, both `hello world`
- `docs/c.bin`, eleven other bytes
- `docs/old/d.txt` and `docs/old/e.txt`, both `bye`
- `z.txt` and `docs/y.txt`, both empty

the result is `[[a.txt, docs/b.txt], [docs/old/d.txt, docs/old/e.txt], [docs/y.txt, z.txt]]`, each path joined under `root_path`.""",
        },
        {
            "title": "Read only what you must",
            "description_en": r"""Trees now hold large files, and most file sizes are unique. Keep Part 1 and add:

- A file whose size no other file shares is never opened. On a tree where all sizes differ, no file content is read at all.
- A sample of at most the first 4,096 bytes may be read from files that share a size.
- A file is read in full only when another file has both its size and the same sample.
- Files are read in chunks of at most 1 MiB, never into memory whole.

**Example:** two 1 MB files that differ in their first byte cost at most 8,192 bytes of reading in total.""",
        },
        {
            "title": "Read in parallel",
            "description_en": r"""**Signature:** `find_duplicate_files(root_path, max_workers=None) -> list[list[str]]`

Keep Parts 1–2 and add:

- Reads of different files, samples and full reads alike, run concurrently on a thread pool with `max_workers` workers. `None` lets the pool choose.
- With `max_workers=1` at most one read is ever in progress; with more workers, slow reads overlap.
- The result is exactly the same as with one worker, whatever order the reads finish in.

**Example:** when each read takes 0.1 seconds, eight files sharing one size are read with up to four reads in flight at once under `max_workers=4`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What can stand in for a file's whole content as a dictionary key? Which call lists every file at any depth, and how do you spot a symbolic link among them? Where can opening or reading a file fail, and what should happen then?"},
        {"level": 2, "kind": "analysis", "content": "Walk the tree with os.walk, skip anything for which os.path.islink is true or os.path.isfile is false, and hash each file's bytes with hashlib.sha256 inside try/except OSError. Group paths by digest, keep groups of two or more, sort each group, then sort the groups by their first path."},
    ],
    "model_connections": [
        "Deduplication tools such as fdupes and rmlint filter by size, then by a partial hash, then compare full content, which is this exercise's second part.",
        "Backup systems like restic and borg split files into content-addressed chunks, so identical data is stored once across files and snapshots.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Grouping by a cryptographic digest needs no pairwise comparison and handles empty files with no special case.",
            "Filtering by size, then by a short sample, rules out nearly every unique file without reading it.",
            "A thread pool overlaps the time files spend waiting on storage, because reads release the GIL.",
        ],
        "cons": [
            "Every filter tier is another pass over the candidates, which costs more than it saves when most files are duplicates.",
            "Trusting a matching digest accepts a tiny risk of collision; a byte-for-byte check removes it at the cost of reading again.",
            "Threads do not help with CPU-bound hashing in pure Python; that needs processes and their start-up cost.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {"a.txt": b"hello world", "docs/b.txt": b"hello world", "docs/c.bin": b"hello_world",
                     "docs/old/d.txt": b"bye", "docs/old/e.txt": b"bye", "z.txt": b"", "docs/y.txt": b""})
    assert {fn}(root) == [
        [p(root, "a.txt"), p(root, "docs/b.txt")],
        [p(root, "docs/old/d.txt"), p(root, "docs/old/e.txt")],
        [p(root, "docs/y.txt"), p(root, "z.txt")],
    ]
"""},
        {"name": "Part 1: no duplicates and empty trees", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A tree without duplicates, an empty directory, or a lone empty file must give [], never None or single-file groups.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    assert {fn}(root) == []
    make_tree(root, {"only.txt": b""})
    os.makedirs(os.path.join(root, "nothing", "here"))
    assert {fn}(root) == []
    make_tree(root, {"a": b"1", "b": b"2", "deep/c": b"3"})
    assert {fn}(root) == []
"""},
        {"name": "Part 1: symbolic links are ignored", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A symbolic link, to a file, to a directory or to nothing, must never be followed or reported.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as root:
    make_tree(outside, {"x.txt": b"same", "y.txt": b"same"})
    make_tree(root, {"a.txt": b"same", "b.txt": b"lonely"})
    os.symlink(os.path.join(root, "a.txt"), os.path.join(root, "link_to_a"))
    os.symlink(outside, os.path.join(root, "linked_dir"))
    os.symlink(os.path.join(root, "missing"), os.path.join(root, "dangling"))
    assert {fn}(root) == []
    make_tree(root, {"sub/c.txt": b"same"})
    assert {fn}(root) == [[p(root, "a.txt"), p(root, "sub/c.txt")]]
"""},
        {"name": "Part 1: unreadable files are skipped", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A file that cannot be opened, or fails while being read, must be left out of every group without stopping the walk or raising.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {"a": b"dup", "b": b"dup", "c": b"dup", "d": b"pair", "e": b"pair"})
    with ReadMonitor(root, deny=[p(root, "b"), p(root, "e")]):
        result = {fn}(root)
    assert result == [[p(root, "a"), p(root, "c")]], result
    with ReadMonitor(root, broken=[p(root, "a"), p(root, "d")]):
        result = {fn}(root)
    assert result == [[p(root, "b"), p(root, "c")]], f"a file that fails while being read must be skipped: {result}"
"""},
        {"name": "Part 1: random trees match a byte-for-byte comparison", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random tree, the groups or their order differed from comparing every pair of files byte for byte.",
         "code": _HELPERS + r"""
for seed in range(25):
    rng = random.Random(seed)
    contents = [b"", b"a", b"b", b"ab", b"ba", bytes(5000), bytes(4999) + b"x", b"\xff\xfe"]
    with tempfile.TemporaryDirectory() as root:
        files = {}
        for i in range(rng.randint(1, 14)):
            folder = rng.choice(["", "x/", "x/y/", "z/"])
            files[f"{folder}f{i}"] = rng.choice(contents)
        make_tree(root, files)
        assert {fn}(root) == brute_force(root), (seed, files)
"""},
        {"name": "Part 2: unique sizes are never read", "part": 2, "behavior": "performance.complexity", "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {f"d{i % 3}/f{i}": bytes([i]) * (i + 1) for i in range(40)})
    make_tree(root, {"big.bin": bytes(1 << 20)})
    with ReadMonitor(root) as monitor:
        assert {fn}(root) == []
    assert monitor.bytes == 0, f"read {monitor.bytes} bytes though every size is unique"
"""},
        {"name": "Part 2: a different sample stops the read", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Two 1 MB files that differ in their first bytes must be told apart by sampling at most 4,096 bytes of each.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {"a.bin": b"A" + bytes((1 << 20) - 1), "b.bin": b"B" + bytes((1 << 20) - 1)})
    with ReadMonitor(root) as monitor:
        assert {fn}(root) == []
    assert monitor.bytes <= 8192, f"read {monitor.bytes} bytes"
"""},
        {"name": "Part 2: same size and sample still compares everything", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Files that share their size and first 4,096 bytes but differ later must not be grouped, and full reads must come in chunks of at most 1 MiB.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    head = bytes(range(256)) * 32
    make_tree(root, {"a": head + b"tail-1", "b": head + b"tail-2", "c/d": head + b"tail-1",
                     "e": bytes(3 << 20), "f": bytes(3 << 20)})
    with ReadMonitor(root) as monitor:
        result = {fn}(root)
    assert result == [[p(root, "a"), p(root, "c/d")], [p(root, "e"), p(root, "f")]], result
    assert monitor.bytes <= 2 * (3 << 20) + 3 * len(head) + 64 * 1024, f"read {monitor.bytes} bytes"
    assert monitor.largest <= 1 << 20, f"one read returned {monitor.largest} bytes; read in chunks of at most 1 MiB"
"""},
        {"name": "Part 2: random trees with shared prefixes match", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random tree built to share sizes and prefixes, the groups differed from a byte-for-byte comparison.",
         "code": _HELPERS + r"""
for seed in range(25):
    rng = random.Random(seed)
    head = bytes(rng.randrange(256) for _ in range(4096))
    contents = [head + b"1", head + b"2", head + b"12", b"1" + head, head[:100], head[:100]]
    with tempfile.TemporaryDirectory() as root:
        files = {f"{rng.choice(['', 'q/', 'q/r/'])}g{i}": rng.choice(contents) for i in range(rng.randint(2, 10))}
        make_tree(root, files)
        assert {fn}(root) == brute_force(root), seed
"""},
        {"name": "Part 3: reads overlap", "part": 3, "behavior": "concurrency.thread_safety", "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {f"f{i}": b"same-size-" + bytes([i % 2]) for i in range(8)})
    expected = [[p(root, f"f{i}") for i in range(0, 8, 2)], [p(root, f"f{i}") for i in range(1, 8, 2)]]
    with ReadMonitor(root, delay=0.1) as monitor:
        assert {fn}(root, max_workers=4) == expected
    assert monitor.peak >= 3, f"at most {monitor.peak} reads were in flight at once"
"""},
        {"name": "Part 3: max_workers is respected", "part": 3, "visibility": "unshown", "behavior": "concurrency.thread_safety",
         "failure_message": "With max_workers=1 no two reads may overlap, and the default call must still return the right groups.",
         "code": _HELPERS + r"""
with tempfile.TemporaryDirectory() as root:
    make_tree(root, {f"f{i}": b"abc" for i in range(5)})
    with ReadMonitor(root, delay=0.02) as monitor:
        assert {fn}(root, max_workers=1) == [[p(root, f"f{i}") for i in range(5)]]
    assert monitor.peak == 1, f"{monitor.peak} reads overlapped with one worker"
    assert {fn}(root) == [[p(root, f"f{i}") for i in range(5)]]
"""},
        {"name": "Part 3: the answer does not depend on timing", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With several workers, a random tree's groups differed from the one-worker answer or from a byte-for-byte comparison.",
         "code": _HELPERS + r"""
for seed in range(12):
    rng = random.Random(seed)
    contents = [b"", b"aa", b"ab", b"ba", bytes(4097), bytes(4096) + b"\x01"]
    with tempfile.TemporaryDirectory() as root:
        files = {f"{rng.choice(['', 'k/', 'k/l/'])}h{i}": rng.choice(contents) for i in range(rng.randint(2, 12))}
        make_tree(root, files)
        deny = [p(root, rng.choice(list(files)))]
        with ReadMonitor(root, delay=0.001, deny=deny):
            many = {fn}(root, max_workers=6)
            one = {fn}(root, max_workers=1)
        for path in deny:
            os.remove(path)
        assert many == one == brute_force(root), seed
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SAMPLE_SIZE = 4096
CHUNK_SIZE = 1 << 20


def _regular_files(root_path):
    for dirpath, _dirnames, filenames in os.walk(root_path):  # never follows directory links
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path) and os.path.isfile(path):
                yield path


def _sample(path):
    try:
        with open(path, "rb") as f:
            return f.read(SAMPLE_SIZE)
    except OSError:
        return None


def _digest(path):
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.digest()


def _regroup(groups, key_of, pool):
    """Splits every group by key_of(path), reading in parallel, and drops what ends up alone."""
    paths = [path for group in groups for path in group]
    keys = dict(zip(paths, pool.map(key_of, paths)))
    out = []
    for group in groups:
        by_key = defaultdict(list)
        for path in group:
            if keys[path] is not None:
                by_key[keys[path]].append(path)
        out.extend(g for g in by_key.values() if len(g) >= 2)
    return out


def find_duplicate_files(root_path, max_workers=None):
    by_size = defaultdict(list)
    for path in _regular_files(root_path):
        try:
            by_size[os.path.getsize(path)].append(path)
        except OSError:
            continue
    groups = [paths for paths in by_size.values() if len(paths) >= 2]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        groups = _regroup(groups, _sample, pool)
        groups = _regroup(groups, _digest, pool)
    return sorted((sorted(group) for group in groups), key=lambda group: group[0])
''',
    "interview_questions": interview(
        concept=[
            "What do you use as the grouping key for a file's content, and why not the content itself?",
            "How does your walk treat symbolic links to files and to directories?",
        ],
        deep_dive=[
            "Where can reading a file fail, and how do you keep one bad file from breaking the whole result?",
        ],
        tradeoffs=[
            "Why compare sizes before reading anything, and why is a matching sample still not proof of a duplicate?",
            "Why do threads speed up reading files despite the GIL, and when would processes be the better pool?",
            "When would you add a byte-for-byte comparison after the hashes match?",
        ],
    ),
}
