Implement `cd`, a function that computes the absolute path `pwd` would print after running `cd destination` in a shell (from Part 3 on, the physical path that `cd -P destination` followed by `pwd` prints). The function grows a parameter at a time across three parts: `cd_relative` takes a relative destination, `cd_absolute` adds absolute destinations and `~`, and `cd` adds symbolic links. A final part asks about the real shell command instead of code.

### Part 1 — Normalize a Relative Destination

`cwd` is given as an absolute, already-normalized path such as `"/srv/www/docs"`: it starts with `/`, contains no `.` or `..` segment, no empty segment from a repeated `/`, and no trailing `/` (except `cwd == "/"` itself). `destination` is relative (it never starts with `/`) and may contain `.` (stay), `..` (go up one level), repeated slashes, and a trailing slash. Compute the absolute, normalized result of starting at `cwd` and following `destination` component by component. Going above `/` leaves you at `/` instead of erroring.

```py
def cd_relative(cwd: str, destination: str) -> str:
    ...
```

```text
cd_relative("/srv/www/docs", "guides")               -> "/srv/www/docs/guides"
cd_relative("/srv/www/docs", "../assets")             -> "/srv/www/assets"
cd_relative("/srv/www/docs", "../guides/v2/")         -> "/srv/www/guides/v2"
cd_relative("/srv/www/docs", "./guides/")             -> "/srv/www/docs/guides"
cd_relative("/srv/www/docs", "../guides/v2/../")      -> "/srv/www/guides"
cd_relative("/srv/www", "../../../static")            -> "/static"        (clamped at root, then descends)
cd_relative("/", "../..")                             -> "/"
cd_relative("/srv/www/docs", "../../logs/./today//")  -> "/srv/logs/today"
```

### Part 2 — Absolute Paths and `~`

Extend the function so `destination` may also be an absolute path (starts with `/`) or start with `~`, which stands for the caller's home directory, given explicitly as `home` (also an absolute, normalized path). `~` expands to exactly `home`; `~/rest` expands to `home` joined with `rest`, and `rest` is then subject to the same `.`/`..`/repeated-slash handling as Part 1. A destination that starts with `~` but is neither `~` nor `~/...` (for example `~bob`, another user's home directory) is out of scope: raise `ValueError`. `~` is only special as the very first character of `destination` — `a/~b` is an ordinary two-component relative path, not an expansion.

```py
def cd_absolute(cwd: str, destination: str, home: str) -> str:
    ...
```

```text
cd_absolute("/srv/www/docs", "/opt/tools/bin", "/home/priya")  -> "/opt/tools/bin"
cd_absolute("/srv/www/docs", "~", "/home/priya")               -> "/home/priya"
cd_absolute("/srv/www/docs", "~/scripts", "/home/priya")       -> "/home/priya/scripts"
cd_absolute("/tmp/scratch", "../backups", "/home/priya")       -> "/tmp/backups"
cd_absolute("/home/priya", "a/~b", "/home/priya")              -> "/home/priya/a/~b"
cd_absolute("/home/priya", "~bob", "/home/priya")              -> raises ValueError
```

### Part 3 — Symbolic Links

You are additionally given `symlinks`, a mapping from the absolute, normalized path of a symbolic link to its target, which may itself be an absolute path or a path relative to the directory that contains the link:

```python
symlinks = {
    "/srv/www/current": "/srv/archive/2025-09",
}
```

Extend the function to resolve symbolic links the way a real filesystem does when it follows a path *physically* (as `cd -P` or `realpath` would), matching these rules exactly:

- `destination` is followed one component at a time, left to right. Whenever the path reached so far is a key of `symlinks`, you are in fact at that link's target, and the remaining components continue from there.
- `..` goes to the parent of the physical directory reached so far: right after a symbolic link, that is the parent of the directory the link leads to, not the directory that contains the link. At `/`, `..` stays at `/` as in Part 1.
- A relative target is resolved against the directory that *contains the link*, not against `cwd`.
- A target may contain `.`, `..` and other symbolic links, at its end or in its middle; it is followed by these same rules.
- Keys match whole path components, not substrings: `/srv/www/current-backup` is unrelated to a key `/srv/www/current`. Keys, `cwd` and `home` are physical paths: none of them lies under a key, since a symbolic link has no entries. Every path that is not a key is an ordinary, existing directory — the function does not check the real filesystem.
- Each replacement of a link by its target counts as one expansion, including another expansion of a link already expanded in the same call. If one call needs more than `MAX_SYMLINK_HOPS = 20` expansions, raise `SymlinkLoopError`.

```py
class SymlinkLoopError(Exception): ...

MAX_SYMLINK_HOPS = 20

def cd(cwd: str, destination: str, home: str, symlinks: dict[str, str]) -> str:
    ...
```

With `cwd = "/srv/www"`, `symlinks = {"/srv/www/current": "/srv/archive/2025-09"}`, and `destination = "current/gallery/../notes"`, the link is entered first, then `gallery/..` cancels inside the target, so the result is `"/srv/archive/2025-09/notes"`.

### Part 4 — How Does a Shell Actually Run `cd`?

No code for this part. Explain why `cd` has to be a built-in command of the shell rather than a separate program the shell launches, and describe, briefly, the steps a shell goes through to execute a `cd` command.
