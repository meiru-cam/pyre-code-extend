A package manager needs to answer two kinds of questions about the packages it manages. Parts 1-3 find the earliest published version of one package that supports a given feature, using a slow probe. Part 4 resolves a full, mutually compatible set of dependency versions for an installation. Every version is a string `"{major}.{minor}.{patch}"`, each component a base-10 integer that may carry leading zeros (`"103.003.02"` means major 103, minor 3, patch 2); versions are compared numerically as `(major, minor, patch)` tuples, never as raw strings.

```py
def parse_version(version: str) -> tuple[int, int, int]:
    """Parses "{major}.{minor}.{patch}" into (major, minor, patch)."""
```

### Part 1 — Monotone support

`versions` is a list of distinct version strings of one package, in arbitrary order. `is_supported(version)` calls a slow endpoint of the registry and returns whether exactly that version supports a given feature. Once `versions` is ordered numerically, assume `is_supported` is monotone over that order: a (possibly empty) run of `False` followed by a (possibly empty) run of `True`. Implement `find_earliest_supported_monotone`, returning the numerically smallest version for which `is_supported` is `True`, or `None` if no version supports it.

```py
def is_supported(version: str) -> bool:
    """Provided by the package registry: a real network call, slow and rate-limited."""

def find_earliest_supported_monotone(versions: list[str], is_supported) -> str | None:
    """versions: distinct "{major}.{minor}.{patch}" strings, in arbitrary order. Assumes is_supported is
    monotone once versions are ordered numerically. Returns the earliest supported version, or None."""
```

For example, from the input `["2.0.0", "1.10.0", "1.9.1", "1.9.0"]` (already out of numeric order), with support starting at `"1.9.1"`, the answer is `"1.9.1"`: numerically `(1, 9, 1) < (1, 10, 0) < (2, 0, 0)`, even though `"1.10.0"` sorts before `"1.9.1"` as a plain string.

### Part 2 — Regressions

Support can regress: a later version does not necessarily still support the feature, so `is_supported` may return `True`, then `False`, then `True` again as the version increases. Implement `find_earliest_supported`, which drops Part 1's assumption and still returns the numerically earliest version for which `is_supported` is `True`, or `None` if none is.

```py
def find_earliest_supported(versions: list[str], is_supported) -> str | None:
    """Same contract as find_earliest_supported_monotone, but is_supported need not be monotone."""
```

For example, from `["3.1.4", "3.1.0", "3.1.2", "3.1.1", "3.1.3"]`, with `is_supported` returning `False, True, False, True, False` for patches 0 through 4 respectively, the answer is `"3.1.1"`, even though `"3.1.3"` supports the feature too.

### Part 3 — Rate-limited probes over a hierarchy

Every call to `is_supported` now counts against a quota, so the number of calls must be as small as possible. Support is not monotone over the whole list, but it has the following structure ("ordered" always means ordered numerically):

- **(A1)** Within one `(major, minor)` group, support over its patches, ordered, is monotone (Part 1's pattern, applied only inside the group).
- **(A2)** Within one major, take each of its minor groups' *representative* — the numerically latest version in that group. Support over these representatives, ordered by minor, is monotone.
- **(A3)** Take each major's *representative* — the numerically latest version in that major. Support over these representatives, ordered by major, is monotone.

These three properties constrain only the representatives and the inside of each individual group; they do not force the full, flat, numerically ordered list of every version to be monotone. For example, majors 0 and 1 with:

```text
0.9.0 -> False
1.0.0 -> False   1.1.0 -> False
1.2.0 -> False   1.2.1 -> True
1.3.0 -> False   1.3.1 -> False   1.3.2 -> False   1.3.3 -> True
```

satisfy A1-A3: every `(major, minor)` group is internally monotone; within major 1 the minor representatives `1.0.0, 1.1.0, 1.2.1, 1.3.3` read `False, False, True, True`; the major representatives `0.9.0, 1.3.3` read `False, True`. Yet the flat ordered list of all nine versions reads `False, False, False, False, True, False, False, False, True` — support appears at `1.2.1`, disappears again for the whole of `1.3.0`-`1.3.2`, and returns at `1.3.3`. Bisecting that flat list directly is therefore unsound. Implement `find_earliest_supported_hierarchical`, which must instead bisect major groups, then minor groups within the chosen major, then patches within the chosen minor.

```python
class CountingProbe:
    """Wraps is_supported and counts how many times it is actually called."""
    def __init__(self, is_supported):
        self._is_supported = is_supported
        self.calls = 0

    def __call__(self, version: str) -> bool:
        self.calls += 1
        return self._is_supported(version)
```

```py
def find_earliest_supported_hierarchical(versions: list[str], is_supported) -> str | None:
    """Same contract as find_earliest_supported, but assumes A1-A3 above and must call is_supported
    (which may be a CountingProbe) O(log M + log N_minor + log P) times, where M is the number of
    majors, N_minor the largest number of minors within one major, and P the largest number of patches
    within one (major, minor) group -- never probing the same version string twice."""
```

### Part 4 — Dependency resolution

To install a package, the package manager must choose one version for it and for every package it needs, directly or transitively, so that every declared constraint holds. Each `(package, version)` declares its direct dependencies in a `requires` mapping from the dependency's package name to one *constraint*: a comparison operator (`==`, `>=`, `<=`, `>`, `<`) followed by one version, e.g. `">=2.0.0"`. Several packages may constrain the same dependency; all of their constraints must hold for the one version chosen for it. The available packages are held in a `Registry`:

```python
class Registry:
    """The available packages: name -> version -> requires."""
    def __init__(self):
        self._versions = {}

    def add_version(self, name: str, version: str, requires: dict[str, str] | None = None) -> None:
        """Registers one available (name, version) and its direct dependencies."""
        self._versions.setdefault(name, {})[version] = dict(requires or {})

    def versions(self, name: str) -> list[str]:
        """All versions registered for name (empty if the name is unknown)."""
        return list(self._versions.get(name, {}))

    def requires(self, name: str, version: str) -> dict[str, str]:
        """The constraints declared by exactly this (name, version)."""
        return self._versions[name][version]
```

```py
def resolve(registry: Registry, root: str) -> list[str] | None:
    """Chooses one version for root and for every package the chosen versions require, so that every
    declared constraint holds, and returns them as "name@version" strings in an install order: every
    package appears after all of its dependencies. Packages the chosen versions do not need are not
    listed. A choice whose dependencies form a cycle has no install order and is therefore not valid.
    Returns None if no valid choice exists."""
```

For example, a registry with:

```text
toolkit  1.0.0             requires netlib>=2.0.0, parser>=1.0.0
netlib   1.0.0
netlib   2.0.0             requires coreutil==1.0.0
netlib   2.1.0             requires coreutil>=1.2.0
parser   1.0.0             requires coreutil<1.2.0
parser   1.5.0             requires coreutil<1.0.0
coreutil 1.0.0, 1.2.0, 1.3.0   (no dependencies)
```

resolving `root = "toolkit"` gives `["coreutil@1.0.0", "netlib@2.0.0", "parser@1.0.0", "toolkit@1.0.0"]` (`netlib` and `parser` may swap places). `netlib@2.1.0` cannot be used: it needs `coreutil >= 1.2.0`, while `parser@1.0.0` needs `coreutil < 1.2.0` and `parser@1.5.0` needs `coreutil < 1.0.0`.
