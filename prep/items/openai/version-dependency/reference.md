Worth confirming before coding: `is_supported` is deterministic, and Part 4's constraints are always one operator plus one version.

### Part 1

Sorting turns `is_supported` into a monotone predicate over indices, so the leftmost `True` is found by binary search instead of a linear scan -- the same routine Part 3 reuses at every level.

```python
def parse_version(version):
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)   # NOTE: int() drops leading zeros; never compare the raw strings


def first_true_index(n, predicate):
    """Leftmost index in range(n) with predicate(i) True, assuming predicate is monotone; else None."""
    left, right, result = 0, n - 1, None
    while left <= right:
        mid = (left + right) // 2
        if predicate(mid):
            result = mid
            right = mid - 1          # an earlier True may still exist -- keep searching left
        else:
            left = mid + 1
    return result


def find_earliest_supported_monotone(versions, is_supported):
    ordered = sorted(versions, key=parse_version)   # NOTE: numeric order; "1.10.0" < "1.9.0" as plain strings
    idx = first_true_index(len(ordered), lambda i: is_supported(ordered[i]))
    return ordered[idx] if idx is not None else None
```

### Part 2

Once regressions are allowed, `is_supported(ordered[i])` is no longer monotone in `i`: on Part 2's pattern (`False, True, False, True, False`), `first_true_index` lands on the `True` at index 3 and returns `"3.1.3"` instead of `"3.1.1"` (checked below). Sorting is still worth keeping: it turns the search into a left-to-right scan that returns on the first hit, since nothing later can be smaller.

```python
def find_earliest_supported(versions, is_supported):
    for v in sorted(versions, key=parse_version):
        if is_supported(v):        # sorted, so this first True is already the earliest
            return v
    return None
```

### Part 3

A group's representative tells whether the group contains any supported version. Inside a `(major, minor)` group this is A1: if any patch is supported, so is the latest one. For a major, its representative is also the representative of its latest minor group, because tuples compare minor before patch; if some version of the major is supported, A1 makes the representative of its minor group `True`, and A2 then makes every later minor representative `True`, the major's own included. So a `False` representative means the whole group is unsupported. By A3 the majors with a `True` representative form a suffix; bisect for the first of them, $M$. Every earlier major is entirely unsupported, so the answer lies in $M$. The same argument with A2 finds the first minor group of $M$ whose representative is `True`, and A1 allows a final bisection over its patches.

Enumerating every support pattern that satisfies A1-A3 on small version sets (up to 12 versions), the result always equals that of the full scan. If A1 is violated (`True, False, True` inside one group), the patch bisection returns the later `True`.

```python
def find_earliest_supported_hierarchical(versions, is_supported):
    cache = {}
    def probe(v):
        if v not in cache:                     # NOTE: never call is_supported twice for one version
            cache[v] = is_supported(v)
        return cache[v]

    tree = {}                                   # major -> minor -> patch-sorted versions
    for v in versions:
        major, minor, _ = parse_version(v)
        tree.setdefault(major, {}).setdefault(minor, []).append(v)
    for minors in tree.values():
        for patches in minors.values():
            patches.sort(key=parse_version)

    majors = sorted(tree)
    rep = lambda i: max((p[-1] for p in tree[majors[i]].values()), key=parse_version)   # NOTE: latest = rep
    mi = first_true_index(len(majors), lambda i: probe(rep(i)))
    if mi is None:
        return None

    minors = sorted(tree[majors[mi]])
    ni = first_true_index(len(minors), lambda i: probe(tree[majors[mi]][minors[i]][-1]))
    patches = tree[majors[mi]][minors[ni]]
    pi = first_true_index(len(patches), lambda i: probe(patches[i]))
    return patches[pi]
```

This probes `"0.9.0", "1.3.3"`, then `"1.1.0", "1.2.1"`, then `"1.2.0"` (`"1.2.1"` cached) on the example above: 5 calls for 9 versions, matching $O(\log M + \log N_{\text{minor}} + \log P)$.

### Part 4

Choosing the versions is a constraint-satisfaction search. `pending` holds the `(name, constraint)` pairs declared by the versions chosen so far and not yet enforced. Take the first pair: if its package already has a version, the constraint can only be checked; otherwise try every version that satisfies it, newest first, and continue with the rest of the list plus the requirements of the version just chosen. Each trial carries the whole remaining list, so a failure anywhere later returns to the most recent choice and tries its next version. The search is therefore exhaustive and returns `None` only when no valid choice exists. When nothing is pending, every constraint holds, and Kahn's algorithm builds the install order by repeatedly emitting a package whose dependencies have all been emitted. Packages that are never emitted lie on a cycle; that choice is rejected and the search goes on. On 2000 random registries the result agrees with a brute force over all combinations of versions.

```python
import operator
from collections import deque

_OPS = {"==": operator.eq, ">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def satisfies(version, constraint):
    for op in ("==", ">=", "<=", ">", "<"):        # NOTE: two-character operators first, or ">=" is read as ">"
        if constraint.startswith(op):
            return _OPS[op](parse_version(version), parse_version(constraint[len(op):]))
    raise ValueError(f"bad constraint: {constraint!r}")


def install_order(registry, chosen):
    """Kahn's algorithm over the chosen versions; None if their dependencies form a cycle."""
    indegree = {name: 0 for name in chosen}
    dependents = {name: [] for name in chosen}
    for name, version in chosen.items():
        for dep in registry.requires(name, version):
            dependents[dep].append(name)
            indegree[name] += 1
    queue = deque(sorted(name for name in chosen if indegree[name] == 0))
    order = []
    while queue:
        name = queue.popleft()
        order.append(f"{name}@{chosen[name]}")
        for other in sorted(dependents[name]):
            indegree[other] -= 1
            if indegree[other] == 0:
                queue.append(other)
    return order if len(order) == len(chosen) else None      # NOTE: a shorter order means a cycle


def resolve(registry, root):
    def search(chosen, pending):
        """chosen: name -> version so far; pending: declared (name, constraint) pairs not yet enforced."""
        if not pending:
            return install_order(registry, chosen)            # a cyclic choice fails here and the search goes on
        (name, constraint), rest = pending[0], pending[1:]
        if name in chosen:                                     # version already fixed: the constraint is only checked
            return search(chosen, rest) if satisfies(chosen[name], constraint) else None
        for version in sorted(registry.versions(name), key=parse_version, reverse=True):   # newest first
            if satisfies(version, constraint):
                found = search({**chosen, name: version}, rest + list(registry.requires(name, version).items()))
                if found is not None:
                    return found
        return None                                            # NOTE: the caller then tries its own next version

    return search({}, [(root, ">=0.0.0")])
```

### Follow-ups

- Resolution with one version per package is NP-complete (3-SAT reduces to it), so the worst case is exponential. Practical resolvers keep a candidate set per package, shrink it on every new constraint so that an empty set fails at once, and record the cause of each conflict to skip the same dead end later.
- A cycle between some versions does not make a registry unsolvable: if `x@2.0.0` and `y@1.0.0` require each other but `x@1.0.0` requires nothing, the search rejects the cyclic choice and returns `x@1.0.0`.
- If the registry accepts concurrent calls, probe all representatives of one level at once: three rounds of latency in total, paid for with $O(M + N_{\text{minor}} + P)$ calls instead of a logarithmic number.
- A pre-release suffix such as `2.1.0-beta` sorts before its release, so parse it into an extra tuple field: `(2, 1, 0, 0, "beta")` for the pre-release and `(2, 1, 0, 1, "")` for `2.1.0`.
- If `is_supported` can time out, retry with backoff and never cache a failure as `False`: one wrong `False` on a representative sends the bisection into the wrong group.

```python
# --- Parts 1-3 ---
versions1 = ["2.0.0", "1.10.0", "1.9.1", "1.9.0"]
support1 = {"1.9.0": False, "1.9.1": True, "1.10.0": True, "2.0.0": True}
assert find_earliest_supported_monotone(versions1, lambda v: support1[v]) == "1.9.1"

versions2 = ["3.1.4", "3.1.0", "3.1.2", "3.1.1", "3.1.3"]
support2 = dict(zip(sorted(versions2, key=parse_version), [False, True, False, True, False]))
assert find_earliest_supported(versions2, lambda v: support2[v]) == "3.1.1"
# the naive bisection of Part 1 is unsound on this same regression pattern:
assert find_earliest_supported_monotone(versions2, lambda v: support2[v]) == "3.1.3"

support3 = {
    "0.9.0": False,
    "1.0.0": False, "1.1.0": False,
    "1.2.0": False, "1.2.1": True,
    "1.3.0": False, "1.3.1": False, "1.3.2": False, "1.3.3": True,
}
probe3 = CountingProbe(lambda v: support3[v])
result3 = find_earliest_supported_hierarchical(list(support3), probe3)
assert result3 == "1.2.1" and find_earliest_supported(list(support3), lambda v: support3[v]) == result3
assert probe3.calls == 5   # against 9 versions total

# violating A1 (True, False, True within one (major, minor) group) breaks the hierarchical search
support_bad = {"1.0.0": True, "1.0.1": False, "1.0.2": True}
assert find_earliest_supported(list(support_bad), lambda v: support_bad[v]) == "1.0.0"
assert find_earliest_supported_hierarchical(list(support_bad), lambda v: support_bad[v]) == "1.0.2"

# exhaustive enumeration: every A1-A3-respecting pattern on small version sets agrees with a full scan
import itertools


def enumerate_a1_a3(majors, minors, patches):
    versions = sorted((f"{M}.{m}.{p}" for M in range(majors) for m in range(minors) for p in range(patches)),
                       key=parse_version)
    checked = 0
    for bits in itertools.product([False, True], repeat=len(versions)):
        support = dict(zip(versions, bits))
        ok = True
        for M in range(majors):
            for m in range(minors):
                seq = [support[f"{M}.{m}.{p}"] for p in range(patches)]
                ok &= not any(seq[i] and not seq[i + 1] for i in range(len(seq) - 1))
            reps = [support[f"{M}.{m}.{patches - 1}"] for m in range(minors)]
            ok &= not any(reps[i] and not reps[i + 1] for i in range(len(reps) - 1))
        major_reps = [support[f"{M}.{minors - 1}.{patches - 1}"] for M in range(majors)]
        ok &= not any(major_reps[i] and not major_reps[i + 1] for i in range(len(major_reps) - 1))
        if not ok:
            continue
        checked += 1
        expected = find_earliest_supported(versions, lambda v: support[v])
        got = find_earliest_supported_hierarchical(versions, lambda v: support[v])
        assert got == expected, (support, expected, got)
    return checked


for shape in [(2, 2, 2), (3, 2, 2), (2, 3, 2), (2, 2, 3)]:
    assert enumerate_a1_a3(*shape) > 0

# --- Part 4 ---
def build_example():
    reg = Registry()
    reg.add_version("toolkit", "1.0.0", {"netlib": ">=2.0.0", "parser": ">=1.0.0"})
    reg.add_version("netlib", "1.0.0", {})
    reg.add_version("netlib", "2.0.0", {"coreutil": "==1.0.0"})
    reg.add_version("netlib", "2.1.0", {"coreutil": ">=1.2.0"})
    reg.add_version("parser", "1.0.0", {"coreutil": "<1.2.0"})
    reg.add_version("parser", "1.5.0", {"coreutil": "<1.0.0"})
    reg.add_version("coreutil", "1.0.0", {})
    reg.add_version("coreutil", "1.2.0", {})
    reg.add_version("coreutil", "1.3.0", {})
    return reg


order = resolve(build_example(), "toolkit")
assert order == ["coreutil@1.0.0", "netlib@2.0.0", "parser@1.0.0", "toolkit@1.0.0"]

reg_cycle = Registry()
reg_cycle.add_version("app", "1.0.0", {"x": ">=1.0.0"})
reg_cycle.add_version("x", "1.0.0", {"y": ">=1.0.0"})
reg_cycle.add_version("y", "1.0.0", {"x": ">=1.0.0"})     # x <-> y: no version of either escapes it
assert resolve(reg_cycle, "app") is None

reg_avoid = Registry()
reg_avoid.add_version("app", "1.0.0", {"x": ">=1.0.0"})
reg_avoid.add_version("x", "1.0.0", {})                   # no dependency at all: breaks the potential cycle
reg_avoid.add_version("x", "2.0.0", {"y": ">=1.0.0"})
reg_avoid.add_version("y", "1.0.0", {"x": ">=1.0.0"})     # only cycles back if x picks 2.0.0
order_avoid = resolve(reg_avoid, "app")
assert order_avoid is not None and dict(e.split("@") for e in order_avoid)["x"] == "1.0.0"


def is_valid_order(registry, root, order):
    """Independently re-derives validity: every listed package's constraints hold, dependencies precede
    dependents, and the closure from root is exactly the set listed."""
    chosen = {}
    for entry in order:
        name, version = entry.rsplit("@", 1)
        if name in chosen:
            return False
        chosen[name] = version
    if root not in chosen:
        return False
    position = {e.rsplit("@", 1)[0]: i for i, e in enumerate(order)}
    reachable, stack = set(), [root]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for dep_name, constraint in registry.requires(name, chosen[name]).items():
            if dep_name not in chosen or not satisfies(chosen[dep_name], constraint):
                return False
            if position[dep_name] >= position[name]:
                return False
            stack.append(dep_name)
    return reachable == set(chosen)


assert is_valid_order(build_example(), "toolkit", order) is True


def brute_force_satisfiable(registry, root):
    names = list(registry._versions)
    for combo in itertools.product(*[registry.versions(n) for n in names]):
        assignment = dict(zip(names, combo))
        reachable, stack, ok = set(), [root], True
        while stack and ok:
            name = stack.pop()
            if name in reachable:
                continue
            reachable.add(name)
            for dep_name, constraint in registry.requires(name, assignment[name]).items():
                if dep_name not in assignment or not satisfies(assignment[dep_name], constraint):
                    ok = False
                    break
                stack.append(dep_name)
        if not ok:
            continue
        indegree = {n: 0 for n in reachable}
        adj = {n: [] for n in reachable}
        for n in reachable:
            for dep_name in registry.requires(n, assignment[n]):
                adj[dep_name].append(n)
                indegree[n] += 1
        q = deque(n for n in reachable if indegree[n] == 0)
        seen = 0
        while q:
            n = q.popleft()
            seen += 1
            for nxt in adj[n]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.append(nxt)
        if seen == len(reachable):
            return True
    return False


assert brute_force_satisfiable(build_example(), "toolkit") is True
assert brute_force_satisfiable(reg_cycle, "app") is False


def random_registry(rng, names, max_versions=3, max_deps=2):
    reg = Registry()
    for name in names:
        vers = sorted({f"{rng.randint(1, 2)}.{rng.randint(0, 2)}.{rng.randint(0, 2)}"
                       for _ in range(rng.randint(1, max_versions))}) or ["1.0.0"]
        others = [n for n in names if n != name]
        for v in vers:
            requires = {}
            for dep in rng.sample(others, k=min(max_deps, len(others))) if others else []:
                if rng.random() < 0.6:
                    op = rng.choice([">=", "<=", ">", "<", "=="])
                    bound = f"{rng.randint(1, 2)}.{rng.randint(0, 2)}.{rng.randint(0, 2)}"
                    requires[dep] = f"{op}{bound}"
            reg.add_version(name, v, requires)
    return reg


import random

rng = random.Random(0)
mismatches = 0
for _ in range(2000):
    names = list("ABCDE")[:rng.randint(2, 5)]
    registry = random_registry(rng, names)
    root = names[0]
    got = resolve(registry, root)
    expected_sat = brute_force_satisfiable(registry, root)
    if expected_sat:
        mismatches += got is None or is_valid_order(registry, root, got) is not True
    else:
        mismatches += got is not None
assert mismatches == 0
```
