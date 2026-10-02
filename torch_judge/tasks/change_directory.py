"""The directory `pwd` prints after `cd`: relative paths, then absolute paths and `~`, then symbolic links followed physically."""

from ._interview import interview

_HELPERS = r"""
import posixpath, random

def raises(kind, call):
    try:
        call()
    except Exception as e:
        assert type(e).__name__ == kind, f"expected {kind}, got {type(e).__name__}: {e}"
        return e
    raise AssertionError(f"expected {kind}, nothing was raised")

WORDS = ["etc", "opt", "var", "lib"]

def random_dir(rng):
    return "/" + "/".join(rng.choice(WORDS) for _ in range(rng.randint(0, 3)))

def random_tail(rng):
    first = rng.choice(WORDS + [".", ".."])
    rest = [rng.choice(WORDS + [".", "..", ""]) for _ in range(rng.randint(0, 6))]
    return "/".join([first] + rest)

def spec(cwd, destination, home, links, cap=16):
    # Follows the rules literally and recursively: (path, hops), or (None, hops) past the cap.
    if destination == "~" or destination.startswith("~/"):
        start, rest = home, destination[2:]
    elif destination.startswith("/"):
        start, rest = "/", destination
    else:
        start, rest = cwd, destination
    hops = [0]
    class Over(Exception):
        pass
    def follow(parts, path):
        for piece in path.split("/"):
            if piece in ("", "."):
                continue
            if piece == "..":
                parts = parts[:-1]
                continue
            here = "/" + "/".join(parts + [piece])
            if here not in links:
                parts = parts + [piece]
                continue
            hops[0] += 1
            if hops[0] > cap:
                raise Over()
            target = links[here]
            parts = follow([] if target.startswith("/") else parts, target)
        return parts
    try:
        return "/" + "/".join(follow([p for p in start.split("/") if p], rest)), hops[0]
    except Over:
        return None, hops[0]
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": r"""
assert {fn}("/data/runs/exp3", "ckpt") == "/data/runs/exp3/ckpt"
assert {fn}("/data/runs/exp3", "..//exp4/./logs/") == "/data/runs/exp4/logs"
assert {fn}("/a", "../../b/../c") == "/c"
"""},
    {"name": "Part 1: dots, slashes and the root", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "A '.' and an empty component change nothing, '..' drops the last component and stays at '/' when there is none, a trailing slash is ignored, and the result is '/' alone at the root, with no trailing slash elsewhere.",
     "code": _HELPERS + r"""
assert {fn}("/", "a") == "/a"
assert {fn}("/", ".") == "/"
assert {fn}("/", "./a/..//") == "/"
assert {fn}("/q", "..") == "/"
assert {fn}("/q/r", "./././") == "/q/r"
assert {fn}("/q/r", "s///t") == "/q/r/s/t"
assert {fn}("/q/r", "../../../../q2/../r2") == "/r2"
assert {fn}("/q/r", "...") == "/q/r/...", "three dots is an ordinary name"
assert {fn}("/q/r", ".hidden/x.y") == "/q/r/.hidden/x.y"
assert {fn}("/Q", "q") == "/Q/q", "names are case-sensitive"
"""},
    {"name": "Part 1: random relative paths", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random relative destinations, the result differed from normalising cwd + '/' + destination with '..' stopping at '/'.",
     "code": _HELPERS + r"""
rng = random.Random(11)
for _ in range(3000):
    cwd, tail = random_dir(rng), random_tail(rng)
    want = posixpath.normpath(cwd.rstrip("/") + "/" + tail)
    assert {fn}(cwd, tail) == want, (cwd, tail, want)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "contract.signature", "code": r"""
try:
    {fn}("/data/runs", "~kai", "/users/kai")
except ValueError:
    pass
else:
    raise AssertionError("~kai names another user's home and must raise ValueError")
assert {fn}("/data/runs", "notes/~old", "/users/kai") == "/data/runs/notes/~old"
assert {fn}("/srv", "~/../etc", "/") == "/etc"
assert {fn}("/data/runs", "/etc//ssl/", "/users/kai") == "/etc/ssl"
"""},
    {"name": "Part 2: where ~ and / count", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "An absolute destination ignores cwd; '~' and '~/rest' start from home and only as the first character; '~name' raises ValueError; '/', '//x', '~/' and '~/..' all normalise like Part 1.",
     "code": _HELPERS + r"""
home = "/h/me"
assert {fn}("/a/b", "/", home) == "/"
assert {fn}("/a/b", "//x//", home) == "/x"
assert {fn}("/a/b", "/../..", home) == "/"
assert {fn}("/a/b", "~/", home) == "/h/me"
assert {fn}("/a/b", "~/..", home) == "/h"
assert {fn}("/a/b", "~//./c/", home) == "/h/me/c"
assert {fn}("/a/b", "./~", home) == "/a/b/~"
assert {fn}("/a/b", "c/~/d", home) == "/a/b/c/~/d"
assert {fn}("/", "~", "/") == "/"
for bad in ["~x", "~x/y", "~.", "~~"]:
    raises("ValueError", lambda: {fn}("/a/b", bad, home))
assert {fn}("/a/b", "c", None) == "/a/b/c", "home may be left out when the destination does not use it"
"""},
    {"name": "Part 2: random destinations", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random relative, absolute and ~/ destinations, the result differed from normalising the expanded path with '..' stopping at '/'.",
     "code": _HELPERS + r"""
rng = random.Random(12)
for _ in range(3000):
    cwd, home, tail = random_dir(rng), random_dir(rng), random_tail(rng)
    kind = rng.choice(["relative", "absolute", "home"])
    if kind == "relative":
        destination, full = tail, cwd.rstrip("/") + "/" + tail
    elif kind == "absolute":
        destination, full = "/" + tail, "/" + tail
    else:
        destination, full = "~/" + tail, home.rstrip("/") + "/" + tail
    want = posixpath.normpath(full)
    assert {fn}(cwd, destination, home) == want, (cwd, destination, home, want)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "contract.signature", "code": r"""
links = {"/data/latest": "runs/exp9", "/data/runs/exp9/out": "/scratch/out9"}
assert {fn}("/data", "latest/../exp2", None, links) == "/data/runs/exp2"
assert {fn}("/data", "latest/out/..", None, links) == "/scratch"
loop = {"/l/a": "b", "/l/b": "a"}
try:
    {fn}("/l", "a", None, loop)
except Exception as e:
    assert type(e).__name__ == "LinkLoopError", f"expected LinkLoopError, got {type(e).__name__}"
else:
    raise AssertionError("a two-link loop must raise LinkLoopError")
"""},
    {"name": "Part 3: link rules", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "A link is replaced by its target as soon as it is reached; a relative target starts from the link's own directory; '..' after a link leaves the target, not the link's directory; keys match whole components; targets may hold '.', '..' and other links; up to 16 expansions are fine and the 17th raises LinkLoopError.",
     "code": _HELPERS + r"""
assert {fn}("/w", "ln/x", None, {"/w/ln": "../t/u"}) == "/t/u/x", "a relative target starts from /w"
assert {fn}("/w", "ln/..", None, {"/w/ln": "/p/q/r"}) == "/p/q"
assert {fn}("/w", "ln2/z", None, {"/w/ln": "/p"}) == "/w/ln2/z", "keys match whole components"
assert {fn}("/w", "a/b", None, {"/w/a": "/m/n/..", "/m/b": "/k"}) == "/k", "a target ending in .. then another link"
assert {fn}("/w", "a", None, {"/w/a": "b/./c", "/w/b": "/v", "/v/c": "../done"}) == "/done"
assert {fn}("/", "up/..", None, {"/up": "/"}) == "/", ".. at the root after a link stays at /"
assert {fn}("/w", "ln/x", "/h", {}) == "/w/ln/x"
assert {fn}("/w", "~/s", "/h", {"/h/s": "/w"}) == "/w"
assert {fn}("/w", "/w/ln/x", None, {"/w/ln": "y"}) == "/w/y/x"
me = {"/s/me": "."}
assert {fn}("/s", "/".join(["me"] * 16) + "/f", None, me) == "/s/f", "16 expansions are allowed"
raises("LinkLoopError", lambda: {fn}("/s", "/".join(["me"] * 17), None, me))
chain = {f"/c/l{k}": f"l{k + 1}" for k in range(1, 16)}
chain["/c/l16"] = "end"
assert {fn}("/c", "l1", None, chain) == "/c/end"
chain["/c/l16"] = "l17"
chain["/c/l17"] = "end"
raises("LinkLoopError", lambda: {fn}("/c", "l1", None, chain))
raises("LinkLoopError", lambda: {fn}("/z", "self", None, {"/z/self": "/z/self"}))
raises("LinkLoopError", lambda: {fn}("/l", "a", None, {"/l/a": "b", "/l/b": "a"}))
"""},
    {"name": "Part 3: random links", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random link tables, including loops, targets that climb past '/', and links reached through other links, the result differed from following each component in order and expanding each link where it is reached.",
     "code": _HELPERS + r"""
rng = random.Random(13)
dirs = ["/"] + ["/" + a for a in "pq"] + ["/" + a + "/" + b for a in "pq" for b in "pq"]
targets = ["/q/p", "/p/x", "..", "../..", "../../../p", ".", "y", "../x/q", "x/..", "./y/p", "/"]
seen_links = seen_loops = 0
for trial in range(400):
    links = {}
    for d in dirs:
        for name in ("x", "y"):
            if rng.random() < 0.5:
                links[d.rstrip("/") + "/" + name] = rng.choice(targets)
    for _ in range(20):
        cwd, home = rng.choice(dirs), rng.choice(dirs)
        tail = "/".join(rng.choice(["p", "q", "x", "y", "x", ".", "..", ""]) for _ in range(rng.randint(1, 6)))
        destination = rng.choice([tail, "/" + tail, "~/" + tail])
        want, hops = spec(cwd, destination, home, links)
        seen_links += hops > 0
        if want is None:
            seen_loops += 1
            raises("LinkLoopError", lambda: {fn}(cwd, destination, home, links))
        else:
            assert {fn}(cwd, destination, home, links) == want, (cwd, destination, home, links, want)
assert seen_links > 2000 and seen_loops > 100, (seen_links, seen_loops)
"""},
]

TASK = {
    "title": "Resolve a cd Destination",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "change_directory",
    "description_en": r"""Write `change_directory`, which returns the absolute path `pwd` would print after `cd destination`, first for relative destinations, then for absolute ones and `~`, then with symbolic links followed physically.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `change_directory` function passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `cwd` and `home` are already in the form this function returns.
- Split `destination` on `/` and handle the pieces left to right. An empty piece, from a repeated or trailing slash, and `.` change nothing. `..` drops the last component, and at `/` it stays at `/`. Any other piece is a name and is added; names are case-sensitive.
- Return the result as `/` followed by the components joined with `/`, so the root is `"/"`.
- Nothing is looked up on disk: unless Part 3 says a path is a link, it is a plain directory.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it starts as a stack exercise that most candidates finish quickly. Each later part adds one requirement, and the link part tests whether `..` is applied to the real directory reached so far rather than to the text typed.

**Where it is used:** shells (`cd -P`, `pwd -P`), `realpath`, path handling in build tools and container runtimes, and sandbox checks that must not be fooled by a link that leads outside.

Adapted from the cd command question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as one function, `change_directory`, in place of `cd_relative`, `cd_absolute` and `cd`; its `home` and `symlinks` arguments are optional and arrive in Parts 2 and 3. The expansion cap is 16 instead of 20, and `SymlinkLoopError` is renamed `LinkLoopError`. The source's last part, how a shell runs `cd`, has no code and becomes an interview question.""",
    "parts": [
        {
            "title": "Relative destinations",
            "description_en": r"""**Signature:** `change_directory(cwd, destination) -> str`

- `destination` is relative: it never starts with `/` or `~`. Start at `cwd` and apply its pieces as the rules say.

**Example:**
- `change_directory("/data/runs/exp3", "ckpt")` is `"/data/runs/exp3/ckpt"`
- `change_directory("/data/runs/exp3", "..//exp4/./logs/")` is `"/data/runs/exp4/logs"`
- `change_directory("/a", "../../b/../c")` is `"/c"`: the second `..` has nothing left to drop""",
        },
        {
            "title": "Absolute destinations and ~",
            "description_en": r"""Keep Part 1 and add a third argument.

**Signature:** `change_directory(cwd, destination, home=None) -> str`

- A `destination` starting with `/` is absolute: start at `/` instead of `cwd`.
- `destination == "~"` is `home`, and `"~/rest"` starts at `home` and applies `rest` as in Part 1.
- Any other `destination` starting with `~`, such as `"~kai"`, raises `ValueError`.
- A `~` anywhere except at position 0 is an ordinary character.
- `home` is always given when `destination` starts with `~`.

**Example:**
- `change_directory("/data/runs", "~kai", "/users/kai")` raises `ValueError`
- `change_directory("/data/runs", "notes/~old", "/users/kai")` is `"/data/runs/notes/~old"`
- `change_directory("/srv", "~/../etc", "/")` is `"/etc"`: home is the root, so `..` stays there
- `change_directory("/data/runs", "/etc//ssl/", "/users/kai")` is `"/etc/ssl"`""",
        },
        {
            "title": "Symbolic links",
            "description_en": r"""Keep Parts 1–2 and add a fourth argument.

**Signature:** `change_directory(cwd, destination, home=None, symlinks=None) -> str`, plus an exception class `LinkLoopError`

- `symlinks` maps a link's full path, in the form this function returns, to its target; `None` means no links. Lookups compare whole paths, so `/a/bc` is not affected by a link at `/a/b`.
- After each name is added, if the path so far is a key, replace that last component by the link's target: an absolute target starts again from `/`, and a relative target starts from the directory that holds the link. The target's pieces follow the same rules, including `.`, `..` and further links, and then the rest of `destination` continues from where they end.
- `..` always removes the last component of the resolved path, so right after a link it climbs out of the link's target.
- Count every replacement, whether or not the same link came up before. More than 16 in one call raises `LinkLoopError`.
- Assume `cwd`, `home` and the keys pass through no link.

**Example:** with `symlinks = {"/data/latest": "runs/exp9", "/data/runs/exp9/out": "/scratch/out9"}`:
- `change_directory("/data", "latest/../exp2", None, symlinks)` is `"/data/runs/exp2"`, not `"/data/exp2"`
- `change_directory("/data", "latest/out/..", None, symlinks)` is `"/scratch"`
- `change_directory("/l", "a", None, {"/l/a": "b", "/l/b": "a"})` raises `LinkLoopError`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If the current directory is a list of names, what does each kind of piece in the destination do to that list? What should happen when `..` meets an empty list, and how do you turn an empty list back into a path?"},
        {"level": 2, "kind": "analysis", "content": "Split cwd on \"/\" and keep the non-empty pieces as a stack. For each piece of destination.split(\"/\"): skip \"\" and \".\", pop on \"..\" only if the stack is non-empty, otherwise push. Return \"/\" + \"/\".join(stack), which is \"/\" for an empty stack. Every step is O(1), so the whole call is linear in the input length."},
    ],
    "model_connections": [
        "Sandboxes for code-running agents resolve every path physically before checking it is inside the workspace, since a link can point outside.",
        "Dataset and checkpoint loaders follow links such as `latest` to a dated run directory, where `..` must mean the real parent.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A stack of components handles `.`, `..`, repeated slashes and the root clamp in one linear pass.",
            "Putting a link's target pieces in front of the remaining pieces reuses the same loop for chains of links.",
            "Counting expansions bounds the work even when links form a loop.",
        ],
        "cons": [
            "A fixed expansion cap rejects very long but valid chains of links.",
            "Building the path string for each link lookup costs time proportional to the depth, so deep paths with many links are quadratic.",
            "Without a real file system the function cannot tell a missing directory from an existing one.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
from collections import deque

MAX_EXPANSIONS = 16


class LinkLoopError(Exception):
    pass


def _components(path):
    return [piece for piece in path.split("/") if piece]


def _start(cwd, destination, home):
    """The directory to start from and the pieces still to apply."""
    if destination == "~" or destination.startswith("~/"):
        return home, destination[2:]  # "~"[2:] is "", so a bare "~" stays at home
    if destination.startswith("~"):
        raise ValueError(f"~user is not supported: {destination!r}")
    if destination.startswith("/"):
        return "/", destination
    return cwd, destination


def change_directory(cwd, destination, home=None, symlinks=None):
    symlinks = symlinks or {}
    base, rest = _start(cwd, destination, home)
    stack = _components(base)  # cwd and home never lie inside a link
    pending = deque(rest.split("/"))
    expansions = 0
    while pending:
        piece = pending.popleft()
        if piece in ("", "."):
            continue
        if piece == "..":
            if stack:  # at the root, .. stays at the root
                stack.pop()
            continue
        target = symlinks.get("/" + "/".join(stack + [piece]))
        if target is None:
            stack.append(piece)
            continue
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            raise LinkLoopError(f"more than {MAX_EXPANSIONS} link expansions resolving {destination!r}")
        if target.startswith("/"):
            stack = []
        # a relative target starts from the link's own directory, which is what the stack holds now
        pending.extendleft(reversed(target.split("/")))
    return "/" + "/".join(stack)
''',
    "interview_questions": interview(
        concept=[
            "Why does a stack of path components make `.` and `..` easy to handle?",
            "Why should `..` at the root stay at the root instead of raising an error?",
        ],
        deep_dive=[
            "Which inputs make splitting on `/` produce empty pieces, and why is skipping them correct in each case?",
        ],
        tradeoffs=[
            "Why does `~` expand only at the start of the destination, and why reject `~name` here?",
            "Why must a link be expanded as soon as its component is added, before a later `..` can remove it?",
            "Why cap the number of expansions instead of failing the first time a link repeats?",
            "Why must `cd` be built into the shell instead of being a separate program, and what does the shell do when it runs it?",
        ],
    ),
}
