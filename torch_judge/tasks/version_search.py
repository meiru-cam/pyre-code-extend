"""Find the earliest version that supports a feature with few probes, then resolve a dependency set."""

from ._interview import interview

# Probes that count calls and catch repeats, and generators for the support patterns of each part.
_PROBES = r"""
import itertools, math, random

def key(v):
    return tuple(int(x) for x in v.split("."))

class Probe:
    def __init__(self, support):
        self.support, self.calls, self.seen = dict(support), 0, set()
    def __call__(self, version):
        assert version in self.support, f"probed {version!r}, which is not in the list"
        assert version not in self.seen, f"probed {version!r} twice"
        self.seen.add(version)
        self.calls += 1
        return self.support[version]

def budget(n):
    return 0 if n == 0 else math.ceil(math.log2(n)) + 1

def random_versions(rng, n, zeros=False):
    out = set()
    while len(out) < n:
        parts = [rng.randint(0, 12) for _ in range(3)]
        out.add(".".join(("0" + str(p)) if zeros and rng.random() < 0.3 else str(p) for p in parts))
    versions = list(out)
    rng.shuffle(versions)
    return versions

def first_true(versions, support):
    return next((v for v in sorted(versions, key=key) if support[v]), None)

def grouped_support(rng, majors, minors, patches):
    # Random versions and a support map that satisfies the three grouping rules of Part 3.
    tree = {}
    for M in rng.sample(range(20), majors):
        tree[M] = {m: sorted(rng.sample(range(30), rng.randint(1, patches))) for m in rng.sample(range(20), rng.randint(1, minors))}
    support, order = {}, sorted(tree)
    first_major = rng.randint(0, len(order))
    for i, M in enumerate(order):
        minor_order = sorted(tree[M])
        first_minor = len(minor_order) if i < first_major else rng.randint(0, len(minor_order) - 1)
        for j, m in enumerate(minor_order):
            ps = tree[M][m]
            first_patch = len(ps) if j < first_minor else rng.randint(0, len(ps) - 1)
            for k, p in enumerate(ps):
                support[f"{M}.{m}.{p}"] = k >= first_patch
    versions = list(support)
    rng.shuffle(versions)
    return versions, support, tree
"""

# A registry like the one in the statement, plus an independent checker and a brute force.
_REGISTRY = r"""
import itertools, operator, random
from collections import deque

class Registry:
    def __init__(self):
        self._versions = {}
    def add_version(self, name, version, requires=None):
        self._versions.setdefault(name, {})[version] = dict(requires or {})
    def versions(self, name):
        return list(self._versions.get(name, {}))
    def requires(self, name, version):
        return dict(self._versions[name][version])

def key(v):
    return tuple(int(x) for x in v.split("."))

OPS = [("==", operator.eq), (">=", operator.ge), ("<=", operator.le), (">", operator.gt), ("<", operator.lt)]

def holds(version, constraint):
    for op, fn in OPS:
        if constraint.startswith(op):
            return fn(key(version), key(constraint[len(op):]))
    raise ValueError(constraint)

def check_order(registry, root, order):
    assert isinstance(order, list), f"expected a list, got {order!r}"
    chosen, position = {}, {}
    for i, entry in enumerate(order):
        name, version = entry.rsplit("@", 1)
        assert name not in chosen, f"{name} listed twice"
        assert version in registry.versions(name), f"{entry} is not in the registry"
        chosen[name], position[name] = version, i
    assert root in chosen, "the root package is missing"
    reached, stack = set(), [root]
    while stack:
        name = stack.pop()
        if name in reached:
            continue
        reached.add(name)
        for dep, constraint in registry.requires(name, chosen[name]).items():
            assert dep in chosen, f"{name}@{chosen[name]} needs {dep}, which is missing"
            assert holds(chosen[dep], constraint), f"{dep}@{chosen[dep]} breaks {name}'s {constraint}"
            assert position[dep] < position[name], f"{dep} must come before {name}"
            stack.append(dep)
    assert reached == set(chosen), f"not needed: {sorted(set(chosen) - reached)}"

def solvable(registry, root):
    names = list(registry._versions)
    for combo in itertools.product(*[registry.versions(n) for n in names]):
        pick = dict(zip(names, combo))
        reached, stack, ok = set(), [root], root in pick
        while stack and ok:
            name = stack.pop()
            if name in reached:
                continue
            reached.add(name)
            for dep, constraint in registry.requires(name, pick[name]).items():
                if dep not in pick or not holds(pick[dep], constraint):
                    ok = False
                    break
                stack.append(dep)
        if not ok:
            continue
        indegree = {n: len(registry.requires(n, pick[n])) for n in reached}
        users = {n: [] for n in reached}
        for n in reached:
            for dep in registry.requires(n, pick[n]):
                users[dep].append(n)
        queue, seen = deque(n for n in reached if indegree[n] == 0), 0
        while queue:
            n = queue.popleft()
            seen += 1
            for user in users[n]:
                indegree[user] -= 1
                if indegree[user] == 0:
                    queue.append(user)
        if seen == len(reached):
            return True
    return False

def random_registry(rng, names):
    registry = Registry()
    for name in names:
        others = [n for n in names if n != name] + ["ghost"]
        for v in {f"{rng.randint(1, 2)}.{rng.randint(0, 2)}.{rng.randint(0, 2)}" for _ in range(rng.randint(1, 3))}:
            requires = {}
            for dep in rng.sample(others, k=min(2, len(others))):
                if rng.random() < (0.55 if dep != "ghost" else 0.05):
                    op = rng.choice(["==", ">=", "<=", ">", "<"])
                    requires[dep] = f"{op}{rng.randint(1, 2)}.{rng.randint(0, 2)}.{rng.randint(0, 2)}"
            registry.add_version(name, v, requires)
    return registry
"""

TASK = {
    "title": "Package Version Search",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "PackageManager",
    "description_en": r"""Build `PackageManager`, which answers questions about the published versions of packages.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `PackageManager` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A version is a string `"major.minor.patch"` of three base-10 integers, which may have leading zeros: `"4.010.2"` is major 4, minor 10, patch 2.
- Versions compare as `(major, minor, patch)` number tuples, never as strings: `"4.10.0"` is later than `"4.9.3"`.
- A list of versions holds distinct strings in any order. Methods return versions exactly as written in the input.
- `is_supported(version)` is a slow, rate-limited call to a registry. Never call it twice for the same version, and never for a version that is not in the list.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it starts as a binary search and checks whether you notice when the conditions for one stop holding. Each later part adds one requirement.

**Where it is used:** `git bisect`, package managers that find when a feature or bug appeared, and dependency resolvers such as pip and npm.

Adapted from the version dependency question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Support that never goes away",
            "description_en": r"""**Signature:** `PackageManager().earliest_supported(versions, is_supported) -> str | None`

- In version order, `is_supported` is `False` for some first stretch and `True` for the rest. Either stretch may be empty.
- Return the earliest version for which `is_supported` is `True`, or `None` if there is none.
- With `n` versions, call `is_supported` at most `⌈log₂ n⌉ + 1` times, and not at all when the list is empty.

**Example:** `versions = ["4.2.0", "4.10.1", "4.9.3", "4.10.0"]`, supported from `"4.10.0"` on:
- in version order the list is `4.2.0`, `4.9.3`, `4.10.0`, `4.10.1`
- the answer is `"4.10.0"`, although as a string `"4.10.0"` sorts before `"4.2.0"`""",
        },
        {
            "title": "Support that can regress",
            "description_en": r"""Keep Part 1 and add a search that does not assume support stays.

**Signature:** `earliest_supported_any(versions, is_supported) -> str | None`

- `is_supported` may now turn `False` again in a later version, and back to `True` after that.
- Return the earliest version for which it is `True`, or `None`. `earliest_supported` keeps Part 1's assumption and limit.

**Example:** `versions = ["7.0.3", "7.0.0", "7.0.4", "7.0.1", "7.0.2"]`, where patches `0` to `4` answer `False`, `False`, `True`, `False`, `True`:
- the answer is `"7.0.2"`; `"7.0.4"` is supported too, but later
- a binary search can land on `7.0.4`, because the answers are not one `False` stretch then one `True` stretch""",
        },
        {
            "title": "Few probes over grouped versions",
            "description_en": r"""Keep Parts 1–2 and add a search that uses the structure of version numbers.

**Signature:** `earliest_supported_grouped(versions, is_supported) -> str | None`

A group's representative is its latest version. Support satisfies three rules:
- Within each `(major, minor)` group, in patch order, support is a `False` stretch then a `True` stretch.
- Within each major, the representatives of its minor groups, in minor order, are a `False` stretch then a `True` stretch.
- The representatives of the majors, in major order, are a `False` stretch then a `True` stretch.

The full list in version order need not follow that pattern.

- Return the earliest supported version, or `None`, exactly as `earliest_supported_any` would.
- Call `is_supported` at most `⌈log₂ M⌉ + ⌈log₂ N⌉ + ⌈log₂ P⌉ + 3` times. `M` is the number of majors, `N` the most minor groups in one major, and `P` the most versions in one `(major, minor)` group.

**Example:** `2.5.1` no; `3.0.0` no, `3.0.1` yes; `3.1.0` no, `3.1.1` no, `3.1.2` yes; `3.2.0` yes:
- in version order the answers go no, no, yes, no, no, yes, yes, so a plain binary search is unsafe
- major representatives `2.5.1` no and `3.2.0` yes lead to major 3; minor representatives `3.0.1`, `3.1.2` and `3.2.0` are all yes, so the answer is in `3.0`; in `3.0`, `3.0.0` is no
- the answer is `"3.0.1"`, found with 5 calls""",
        },
        {
            "title": "Resolve dependencies",
            "description_en": r"""Keep Parts 1–3 and add dependency resolution.

**Signature:** `resolve(registry, root) -> list[str] | None`

- `registry.versions(name)` lists a package's versions (empty if unknown). `registry.requires(name, version)` maps each direct dependency's name to one constraint: `==`, `>=`, `<=`, `>` or `<` followed by a version, such as `">=2.0.0"`.
- Choose one version of `root` and of every package that the chosen versions need, directly or through others, so that every constraint of every chosen version holds.
- Return the choice as `"name@version"` strings in install order: each package after all of its dependencies. List only packages that are needed.
- Dependencies that form a cycle have no install order, so such a choice does not count.
- Return `None` if no choice works. Any working choice is accepted.

**Example:** `app 1.0.0` needs `ui>=2.0.0` and `net>=1.0.0`. `ui 2.1.0` needs `net>=2.0.0` and `crypto>=3.0.0`; `ui 2.0.0` needs `net<1.5.0`. `net 2.0.0` needs `crypto<3.0.0`, `net 1.5.0` needs nothing, `net 1.0.0` needs `crypto==2.0.0`. `crypto` has `2.0.0` and `3.0.0`.
- `ui 2.1.0` cannot work: it needs `crypto>=3.0.0`, while the only `net` it allows needs `crypto<3.0.0`
- so `ui 2.0.0`, which allows only `net 1.0.0`, which needs `crypto 2.0.0`
- result: `["crypto@2.0.0", "net@1.0.0", "ui@2.0.0", "app@1.0.0"]`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What order does sorted() give for \"4.10.0\" and \"4.9.3\", and what key fixes it? Once the list is in version order and the answers are all False then all True, which classic search finds the first True, and how many probes does it need?"},
        {"level": 2, "kind": "analysis", "content": "Sort by tuple(int(x) for x in v.split(\".\")). Then binary search: keep lo, hi and the best index found; probe mid; on True record it and search left of mid, on False search right of it. Stop when lo > hi and return the recorded version, or None. Each step halves the range."},
    ],
    "model_connections": [
        "Finding the first checkpoint or library version where an evaluation regressed is the same bisection, run with expensive training or eval jobs as the probe.",
        "ML environments pin torch, CUDA and driver versions together, and resolvers such as pip and conda search for a compatible set.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Sorting by number tuples and binary search needs only O(log n) probes when support is monotone.",
            "Searching majors, then minors, then patches keeps a logarithmic probe count when only the groups are monotone.",
            "Backtracking over versions finds a valid set whenever one exists.",
        ],
        "cons": [
            "When support can regress anywhere, no search beats probing versions one by one in order.",
            "The grouped search silently returns a wrong answer if a group breaks its monotone rule.",
            "Dependency resolution is NP-complete, so backtracking can take exponential time on adversarial registries.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
support = {"4.2.0": False, "4.9.3": False, "4.10.0": True, "4.10.1": True}
calls = []
def is_supported(v):
    calls.append(v)
    return support[v]
assert {fn}().earliest_supported(["4.2.0", "4.10.1", "4.9.3", "4.10.0"], is_supported) == "4.10.0"
assert len(calls) <= 3 and len(set(calls)) == len(calls), calls
"""},
        {"name": "Part 1: random lists within the probe limit", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "On a version list where support never goes away, the result was not the earliest supported version in numeric order, or is_supported was called more than ceil(log2 n) + 1 times or twice for one version.",
         "code": _PROBES + r"""
for seed in range(200):
    rng = random.Random(seed)
    versions = random_versions(rng, rng.randint(1, 200), zeros=seed % 2 == 1)
    ordered = sorted(versions, key=key)
    cut = rng.randint(0, len(ordered))
    support = {v: i >= cut for i, v in enumerate(ordered)}
    probe = Probe(support)
    assert {fn}().earliest_supported(versions, probe) == first_true(versions, support), seed
    assert probe.calls <= budget(len(versions)), (seed, probe.calls, len(versions))
"""},
        {"name": "Part 1: leading zeros, empty lists and no support", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Leading zeros must compare as numbers and the version must come back exactly as written; an empty list returns None without a call, and no support returns None.",
         "code": _PROBES + r"""
manager = {fn}()
probe = Probe({})
assert manager.earliest_supported([], probe) is None and probe.calls == 0
support = {"1.02.3": False, "1.2.10": True, "01.2.9": False, "1.10.0": True}
assert manager.earliest_supported(list(support), Probe(support)) == "1.2.10"
none = {f"0.0.{i}": False for i in range(9)}
assert manager.earliest_supported(list(none), Probe(none)) is None
one = {"9.9.9": True}
assert manager.earliest_supported(["9.9.9"], Probe(one)) == "9.9.9"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
answers = {"7.0.0": False, "7.0.1": False, "7.0.2": True, "7.0.3": False, "7.0.4": True}
assert {fn}().earliest_supported_any(["7.0.3", "7.0.0", "7.0.4", "7.0.1", "7.0.2"], lambda v: answers[v]) == "7.0.2"
"""},
        {"name": "Part 2: support that comes and goes", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "When support can turn off and on again, the result must be the numerically earliest supported version, or None, with no version probed twice.",
         "code": _PROBES + r"""
manager = {fn}()
for seed in range(400):
    rng = random.Random(seed)
    versions = random_versions(rng, rng.randint(0, 40), zeros=seed % 3 == 0)
    support = {v: rng.random() < rng.choice([0.05, 0.3, 0.7]) for v in versions}
    assert manager.earliest_supported_any(versions, Probe(support)) == first_true(versions, support), seed
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "performance.complexity", "code": _PROBES + r"""
support = {"2.5.1": False, "3.0.0": False, "3.0.1": True, "3.1.0": False, "3.1.1": False, "3.1.2": True, "3.2.0": True}
probe = Probe(support)
assert {fn}().earliest_supported_grouped(list(support)[::-1], probe) == "3.0.1"
assert probe.calls <= 5, probe.calls
"""},
        {"name": "Part 3: grouped support within the probe limit", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "On versions whose support follows the three grouping rules, the result differed from a full scan, or the probes exceeded ceil(log2 M) + ceil(log2 N) + ceil(log2 P) + 3.",
         "code": _PROBES + r"""
manager = {fn}()
for seed in range(200):
    rng = random.Random(seed)
    versions, support, tree = grouped_support(rng, rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 20))
    probe = Probe(support)
    assert manager.earliest_supported_grouped(versions, probe) == first_true(versions, support), seed
    M = len(tree)
    N = max(len(minors) for minors in tree.values())
    P = max(len(ps) for minors in tree.values() for ps in minors.values())
    limit = math.ceil(math.log2(M)) + math.ceil(math.log2(N)) + math.ceil(math.log2(P)) + 3
    assert probe.calls <= limit, (seed, probe.calls, limit)
probe = Probe({})
assert manager.earliest_supported_grouped([], probe) is None and probe.calls == 0
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": _REGISTRY + r"""
registry = Registry()
registry.add_version("app", "1.0.0", {"ui": ">=2.0.0", "net": ">=1.0.0"})
registry.add_version("ui", "2.1.0", {"net": ">=2.0.0", "crypto": ">=3.0.0"})
registry.add_version("ui", "2.0.0", {"net": "<1.5.0"})
registry.add_version("net", "2.0.0", {"crypto": "<3.0.0"})
registry.add_version("net", "1.5.0")
registry.add_version("net", "1.0.0", {"crypto": "==2.0.0"})
registry.add_version("crypto", "2.0.0")
registry.add_version("crypto", "3.0.0")
assert {fn}().resolve(registry, "app") == ["crypto@2.0.0", "net@1.0.0", "ui@2.0.0", "app@1.0.0"]
"""},
        {"name": "Part 4: cycles and missing packages", "part": 4, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A choice whose dependencies form a cycle does not count, but another choice that avoids the cycle does; an unknown root or dependency leaves nothing to choose.",
         "code": _REGISTRY + r"""
manager = {fn}()
assert manager.resolve(Registry(), "missing") is None
loop = Registry()
loop.add_version("app", "1.0.0", {"x": ">=1.0.0"})
loop.add_version("x", "1.0.0", {"y": ">=1.0.0"})
loop.add_version("y", "1.0.0", {"x": ">=1.0.0"})
assert manager.resolve(loop, "app") is None, "x and y need each other, so there is no install order"
escape = Registry()
escape.add_version("app", "1.0.0", {"x": ">=1.0.0"})
escape.add_version("x", "2.0.0", {"y": ">=1.0.0"})
escape.add_version("x", "1.0.0")
escape.add_version("y", "1.0.0", {"x": ">=1.0.0"})
order = manager.resolve(escape, "app")
assert order is not None
check_order(escape, "app", order)
assert "x@1.0.0" in order
gone = Registry()
gone.add_version("app", "1.0.0", {"lib": ">=0.0.0"})
assert manager.resolve(gone, "app") is None
alone = Registry()
alone.add_version("solo", "0.1.0")
alone.add_version("unused", "1.0.0")
assert manager.resolve(alone, "solo") == ["solo@0.1.0"]
zeros = Registry()
zeros.add_version("app", "1.0.0", {"lib": ">=1.10.0"})
zeros.add_version("lib", "1.9.0")
zeros.add_version("lib", "1.010.0")
assert manager.resolve(zeros, "app") == ["lib@1.010.0", "app@1.0.0"]
"""},
        {"name": "Part 4: random registries", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random registry, resolve returned None although a valid choice exists, returned a choice when none exists, or returned a list that breaks a constraint, the install order, or lists an unneeded package.",
         "code": _REGISTRY + r"""
manager = {fn}()
for seed in range(600):
    rng = random.Random(seed)
    names = list("ABCDE")[:rng.randint(2, 5)]
    registry = random_registry(rng, names)
    got = manager.resolve(registry, names[0])
    if solvable(registry, names[0]):
        assert got is not None, (seed, "a valid choice exists")
        check_order(registry, names[0], got)
    else:
        assert got is None, (seed, got)
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import operator
from collections import deque

_OPS = [("==", operator.eq), (">=", operator.ge), ("<=", operator.le), (">", operator.gt), ("<", operator.lt)]


def _key(version):
    return tuple(int(part) for part in version.split("."))  # numbers, not strings: "4.10.0" > "4.9.3"


def _first_true(n, test):
    """Leftmost index in range(n) where test is True, assuming False...False True...True; else None."""
    lo, hi, found = 0, n - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if test(mid):
            found, hi = mid, mid - 1
        else:
            lo = mid + 1
    return found


def _satisfies(version, constraint):
    for op, compare in _OPS:  # two-character operators first, or ">=" would be read as ">"
        if constraint.startswith(op):
            return compare(_key(version), _key(constraint[len(op):]))
    raise ValueError(f"bad constraint {constraint!r}")


class PackageManager:
    def earliest_supported(self, versions, is_supported):
        ordered = sorted(versions, key=_key)
        i = _first_true(len(ordered), lambda j: is_supported(ordered[j]))
        return None if i is None else ordered[i]

    def earliest_supported_any(self, versions, is_supported):
        for version in sorted(versions, key=_key):  # the first True in order is the earliest
            if is_supported(version):
                return version
        return None

    def earliest_supported_grouped(self, versions, is_supported):
        cache = {}

        def probe(version):
            if version not in cache:
                cache[version] = is_supported(version)
            return cache[version]

        tree = {}  # major -> minor -> versions in patch order
        for version in sorted(versions, key=_key):
            major, minor, _ = _key(version)
            tree.setdefault(major, {}).setdefault(minor, []).append(version)
        # A False representative means the whole group is unsupported, so the first True one holds the answer.
        majors = sorted(tree)
        i = _first_true(len(majors), lambda j: probe(tree[majors[j]][max(tree[majors[j]])][-1]))
        if i is None:
            return None
        groups = tree[majors[i]]
        minors = sorted(groups)
        j = _first_true(len(minors), lambda k: probe(groups[minors[k]][-1]))
        patches = groups[minors[j]]
        return patches[_first_true(len(patches), lambda k: probe(patches[k]))]

    def resolve(self, registry, root):
        def order_of(chosen):
            """Kahn's algorithm; None if the chosen versions' dependencies form a cycle."""
            waiting = {name: len(registry.requires(name, v)) for name, v in chosen.items()}
            users = {name: [] for name in chosen}
            for name, version in chosen.items():
                for dep in registry.requires(name, version):
                    users[dep].append(name)
            ready = deque(sorted(name for name, count in waiting.items() if count == 0))
            order = []
            while ready:
                name = ready.popleft()
                order.append(f"{name}@{chosen[name]}")
                for user in sorted(users[name]):
                    waiting[user] -= 1
                    if waiting[user] == 0:
                        ready.append(user)
            return order if len(order) == len(chosen) else None

        def search(chosen, pending):
            if not pending:
                return order_of(chosen)  # a cyclic choice fails here and the search goes on
            (name, constraint), rest = pending[0], pending[1:]
            if name in chosen:
                return search(chosen, rest) if _satisfies(chosen[name], constraint) else None
            for version in sorted(registry.versions(name), key=_key, reverse=True):  # newest first
                if _satisfies(version, constraint):
                    found = search({**chosen, name: version}, rest + list(registry.requires(name, version).items()))
                    if found is not None:
                        return found
            return None

        return search({}, [(root, ">=0.0.0")])
''',
    "interview_questions": interview(
        concept=[
            "Why must versions be compared as number tuples, and what goes wrong with plain string comparison?",
            "What property of the answers makes binary search valid here, and how many probes does it need?",
        ],
        deep_dive=[
            "How do you make sure a binary search returns the first True and not just any True?",
        ],
        tradeoffs=[
            "If support can regress, why can no strategy guarantee fewer probes than checking versions in order?",
            "Why does a False representative rule out its whole group, and how does that give a logarithmic probe count?",
            "Why is dependency resolution hard in general, and what do real resolvers do to prune the search?",
            "How do you detect that a chosen set of versions has a dependency cycle?",
        ],
    ),
}
