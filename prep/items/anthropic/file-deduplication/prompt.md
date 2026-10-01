Write `find_duplicate_files`, which walks a directory tree and reports every group of files whose *content* is identical.

`root_path` is the path to a directory. Walk it recursively — files can be nested arbitrarily deep — and partition every file under it into groups by content: two files belong to the same group exactly when they hold the same sequence of bytes, compared byte for byte rather than as text, so no encoding ever enters into the comparison. Each returned group is a list of at least two paths; a file with no duplicate anywhere in the tree (a *singleton*) is excluded from the result entirely, and a tree with no duplicate files at all produces an empty list, not `None` and not a list of one-element groups.

The order of the result is fixed, so that two correct implementations produce exactly the same output: within a group, paths are sorted lexicographically ascending, and the groups themselves are sorted by their own (already-sorted) first path, also lexicographically ascending. A path may be returned exactly as `os.walk` produces it (joined with `os.path.join` under `root_path`); nothing needs to be made absolute or otherwise normalised.

Three situations each have a precise rule:

- *Empty files.* A file with zero bytes is, by the byte-for-byte definition above, identical to every other empty file: if the tree contains two or more empty files, they form one duplicate group together, regardless of their names or locations.
- *Symbolic links.* A symbolic link is never followed and never reported: it contributes no path to any group, whether it points at a file already in the tree, at a directory, or at nothing (a *dangling* link). This holds even when the link's target is itself part of a duplicate group — the link still never appears in the output. A directory reached only through a symbolic link is, for the same reason, not walked into: only the tree of real directories under `root_path` is visited.
- *Unreadable files.* Opening a file can fail — most commonly a permissions error, but a file can also be removed between being listed and being read. Such a file is skipped: it never appears in any group, and a permissions error or similar on one file never raises out of `find_duplicate_files` or stops the rest of the walk from grouping correctly.

```text
root/
  a.txt             "hello world"
  notes/
    b.txt           "hello world"
    c.bin           11 bytes, not "hello world" but the same length
  notes/log/
    d.txt           "goodbye"
    e.txt           "goodbye"
  empty1.txt        ""
  notes/empty2.txt  ""

find_duplicate_files("root") == [
    ["root/a.txt", "root/notes/b.txt"],                    # both 11 bytes, "hello world"
    ["root/empty1.txt", "root/notes/empty2.txt"],          # both empty
    ["root/notes/log/d.txt", "root/notes/log/e.txt"],      # both 7 bytes, "goodbye"
]
# root/notes/c.bin is a singleton: same size as a.txt and b.txt, but different content -- excluded
```

### Part 1 — Correct grouping

```py
def find_duplicate_files(root_path: str) -> list[list[str]]: ...
```

This part has no efficiency requirement: any implementation that returns the groups defined above, in the order defined above, is acceptable, including one that reads every file in full regardless of size.

### Part 2 — Large trees

`root_path` may now hold many files, some of them large, and most sharing their size with no one else. Implement `find_duplicate_files_fast`, with the exact contract of `find_duplicate_files` above, subject to one added requirement: on a tree where every file has a distinct size, the function must produce its answer without reading the content of a single file — size alone already proves every file unique, and that fact (`os.path.getsize`, not the file's bytes) is all the answer may cost. More generally, a file may be read in full only once some other file has been shown to share both its size and a cheap sample of its content — its first $k$ bytes, for a fixed, small $k$ you choose (4,096 is a reasonable default).

```py
def find_duplicate_files_fast(root_path: str) -> list[list[str]]: ...   # same contract as Part 1
```

### Part 3 — Parallel hashing, same answer

Implement `find_duplicate_files_concurrent`, with the same contract again, that computes file hashes concurrently instead of one file at a time.

```py
def find_duplicate_files_concurrent(root_path: str, max_workers: int | None = None,
                                     use_processes: bool = False) -> list[list[str]]: ...
```

`max_workers` is passed straight through to the underlying worker pool (`None` uses its default); `use_processes` selects a process pool instead of the default thread pool. Whichever pool is used and however many workers it has, the returned groups — including their order — must be exactly what `find_duplicate_files_fast` returns for the same tree: concurrency is allowed to change only how fast the answer arrives, never the answer itself.
