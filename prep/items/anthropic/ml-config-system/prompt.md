A *config* is a value built from `dict`s with `str` keys, `list`s, and scalars — `int`, `float`, `bool`, `str`, or `None` — nested to any depth. A *config store* is a `dict[str, dict]` mapping a name to a config that is itself a `dict` at its top level; every function below takes a store as its first argument, already loaded in memory (nothing here parses YAML or any other text format), and returns new configs without mutating the store or any config it was given.

### Part 1 — Composition

A config may declare `_base_: [n1, n2, ...]` — a list of other names in the store — as a key at its own top level; a config without a `_base_` key has none. `DELETE` is a sentinel value exported by this module, usable anywhere a config may hold a scalar.

`deep_merge(base, overlay)` returns a new dict, computed key by key over the union of `base`'s and `overlay`'s keys:

- a key whose `overlay` value is `DELETE` is absent from the result, whether or not `base` had it;
- a key whose `overlay` value is a `dict` maps, in the result, to `deep_merge(b, overlay[key])`, where `b` is `base[key]` when that is itself a `dict`, and `{}` otherwise — so a `DELETE` nested inside `overlay[key]` still removes a key of that inner dict even where `base` had no dict there to begin with;
- a key present only in `base` keeps `base`'s value;
- every other key — present in `overlay`, and neither `DELETE` nor a `dict` — takes `overlay`'s value, replacing whatever `base` held there entirely, dict or not.

No value in the result is shared with `base` or `overlay`.

`resolve_bases(store, name)` returns `name`'s fully composed config: an empty dict, with the resolution of every name in `store[name]`'s own `_base_` list deep-merged in, left to right, then `store[name]`'s own keys (every key except `_base_`) deep-merged in on top. A name in `_base_` may itself have a `_base_`, resolved the same way, recursively; the result never contains a `_base_` key. `resolve_bases` raises `ConfigNotFoundError` when `name`, or some name reachable through a `_base_` list, is not a key of `store`, and raises `ConfigCycleError` when a config is its own base, directly or through others — with the cycle spelled out in the message as the sequence of names it passes through, e.g. `"a -> b -> a"`.

```py
DELETE = ...  # sentinel; usable as a value anywhere in a config

class ConfigError(Exception):
    """Base class for every error raised below."""

class ConfigNotFoundError(ConfigError):
    """name, or a name in some _base_ list, is not in the store."""

class ConfigCycleError(ConfigError):
    """The _base_ graph has a cycle reachable from name."""

def deep_merge(base: dict, overlay: dict) -> dict:
    """Returns a new dict combining base with overlay on top; see the rule above."""

def resolve_bases(store: dict[str, dict], name: str) -> dict:
    """Returns name's fully composed config. The result has no _base_ key."""
```

Example, over a store of four named configs:

```text
store = {
    "base": {"trainer": {"epochs": 10, "batch_size": 32, "lr": 0.1},
             "optimizer": {"name": "sgd", "momentum": 0.9}},
    "wide": {"_base_": ["base"], "trainer": {"batch_size": 128}},
    "regularized": {"trainer": {"weight_decay": 0.01}},
    "experiment": {"_base_": ["wide", "regularized"], "trainer": {"epochs": 20},
                   "optimizer": {"name": "adam", "momentum": DELETE}},
}

resolve_bases(store, "experiment")
# trainer:   base gives epochs=10, batch_size=32, lr=0.1
#          + wide overlays batch_size=128                 -> epochs=10, batch_size=128, lr=0.1
#          + regularized overlays weight_decay=0.01        -> ...,  weight_decay=0.01
#          + experiment's own epochs=20                    -> epochs=20, batch_size=128, lr=0.1, weight_decay=0.01
# optimizer: base gives name="sgd", momentum=0.9
#          + experiment's own name="adam", momentum=DELETE -> name="adam" (momentum removed)
-> {"trainer": {"epochs": 20, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01},
    "optimizer": {"name": "adam"}}
```

### Part 2 — Overrides and interpolation

An *override* is a string `path=value` or `+path=value`, where `path` is one or more identifiers (`[A-Za-z_]\w*`) joined by `.`, addressing a key inside nested `dict`s — never a list index. Without the `+` prefix, every segment of `path`, including the last, must already exist as a key of the config being overridden, or the override raises `OverrideKeyError`; a segment that exists but is not itself a `dict` also raises `OverrideKeyError`, since there is nothing to descend into further. With the `+` prefix, a missing segment — including the last — is created as an empty `dict` (the last one is then set, not left empty); `+` never fails because a key is missing, only because an existing, non-final segment is not a `dict`.

`value` is exactly one of: `null`, `true`, `false`, an integer (`-?[0-9]+`), a float (a `-?[0-9]+` with a `.` and at least one following digit, an exponent, or both), a string in matching single or double quotes with no escape sequences, or a list `[v1, v2, ...]` (or `[]`) of any of the former — never of another list. Whitespace may appear only immediately inside a list's brackets and around its commas; nowhere else in an override. Anything else raises `OverrideSyntaxError`. `apply_overrides(config, overrides)` applies every string in `overrides` to a copy of `config`, in order — a later override to the same path replaces an earlier one — and returns the copy; `config` itself is left untouched.

Once bases and overrides are both resolved, a string leaf may contain one or more `${a.b}` references, each a `path` of the same shape as above, addressing another key of the same config (never a list index). A leaf whose value is *exactly* one reference, with nothing else in the string, takes on the referenced value verbatim, whatever type it is. A reference that is only part of a larger string requires the value it addresses to be an `int`, `float`, `bool`, or `str` — never a `list`, `dict`, or `None` — and is replaced by its text: `str(x)` for an `int`, `float`, or `str`, and `"true"`/`"false"` for a `bool`. A referenced value may itself contain references, resolved first. `resolve_interpolations(config)` returns a copy of `config` with every reference replaced this way; it raises `InterpolationError` when a reference addresses a path absent from `config`, or embeds a `list`, `dict`, or `None` inside a larger string, and raises `InterpolationCycleError` when resolving some path needs, directly or indirectly, its own value — with the cycle spelled out as the sequence of paths it passes through.

```py
class OverrideSyntaxError(ConfigError):
    """An override string does not match path=value or +path=value."""

class OverrideKeyError(ConfigError):
    """A path segment is missing without '+', or an existing segment is not a dict."""

def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Applies every override in order to a copy of config; returns the copy."""

class InterpolationError(ConfigError):
    """A ${...} reference addresses a missing path, or embeds a non-scalar in a string."""

class InterpolationCycleError(ConfigError):
    """Resolving some path's ${...} references needs that path's own value."""

def resolve_interpolations(config: dict) -> dict:
    """Returns a copy of config with every ${a.b} reference replaced by its value."""
```

Example, continuing Part 1's `experiment`, with two more keys added to it:

```text
store["experiment"]["run_name"] = "exp-${optimizer.name}-bs${trainer.batch_size}"
store["experiment"]["eval_batch_size"] = "${trainer.batch_size}"
resolved = resolve_bases(store, "experiment")   # as in Part 1, plus the two keys above verbatim

overrides = ["trainer.epochs=30", "+trainer.warmup_steps=100", 'optimizer.name="adamw"']
overridden = apply_overrides(resolved, overrides)
# trainer.epochs:      20 -> 30 (already existed)
# trainer.warmup_steps: created at 100 (needed '+': did not exist yet)
# optimizer.name:      "adam" -> "adamw" (a quoted string -- not the bare word adamw)

final = resolve_interpolations(overridden)
# run_name:        "exp-" + optimizer.name ("adamw") + "-bs" + trainer.batch_size (128, stringified)
# eval_batch_size: the whole string is one reference -> trainer.batch_size's own value, the int 128
-> {"trainer": {"epochs": 30, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01, "warmup_steps": 100},
    "optimizer": {"name": "adamw"},
    "run_name": "exp-adamw-bs128",
    "eval_batch_size": 128}
```

### Part 3 — Validation

A *schema* is a class decorated with `@dataclasses.dataclass` whose fields are typed `int`, `float`, `bool`, `str`, a list of one of these or of another schema class (`list[X]`), or another schema class directly, nested to any depth. A field is *required* when it has neither a `default` nor a `default_factory`; a field with either is optional, and its absence from a config is not an error.

`validate_config(config, schema)` checks a fully resolved config (Part 2's output — no `_base_`, no `${...}` left) against `schema` and returns a `list[FieldError]`, one entry per mismatch, each carrying the *full dotted path* from the config's root, collecting every mismatch rather than stopping at the first:

- a required field absent from `config` — `"required field is missing"`;
- a key of `config`, at any level, not declared by the matching schema class — `"unexpected field, not declared in the schema"`;
- a present field whose value does not match its declared type: an `int` field needs an `int` that is not a `bool` (Python's `bool` is a subclass of `int`, but a `bool` value never satisfies an `int` or `float` field); a `float` field accepts an `int` or a `float`, again never a `bool`; a `list[X]` field needs a `list`, each of whose elements is checked against `X` at its own indexed path (`....3`); a nested-schema field needs a `dict`, checked recursively against that schema.

An empty list means `config` satisfies `schema`. `validate_config` does not raise on a mismatch; it only raises if `config` itself is not a `dict`.

```py
@dataclasses.dataclass
class FieldError:
    path: str      # full dotted path from the config's root, e.g. "trainer.lr"
    message: str

def validate_config(config: dict, schema: type) -> list[FieldError]:
    """Returns every mismatch between config and schema, collected rather than raised."""
```

Example, against a schema for the config Part 2 produced:

```text
@dataclass
class OptimizerConfig:
    name: str
    momentum: float = 0.0

@dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.0
    warmup_steps: int = 0

@dataclass
class ExperimentConfig:
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    run_name: str
    eval_batch_size: int
    tags: list[str] = field(default_factory=list)

validate_config(final, ExperimentConfig)   # final, from Part 2's example -> [] (valid)

broken = {
    "trainer": {"epochs": 30, "batch_size": 128, "lr": "0.1", "weight_decay": 0.01,
                "warmup_steps": 100, "debug": True},
    "optimizer": {"name": "adamw"},
    "run_name": "exp-adamw-bs128",
}   # trainer.lr is a string; trainer has an undeclared "debug" key; eval_batch_size is missing

validate_config(broken, ExperimentConfig) -> [
    FieldError("trainer.lr", "expected float, got str"),
    FieldError("trainer.debug", "unexpected field, not declared in the schema"),
    FieldError("eval_batch_size", "required field is missing"),
]   # all three, from one call -- not just the first one found
```

### Part 4 — Reproducibility

`freeze_config(config)` returns an immutable snapshot of a fully resolved config: every `dict` becomes a read-only mapping (a `types.MappingProxyType`, recursively, over already-frozen values) and every `list` becomes a `tuple` of already-frozen elements; scalars are returned as is. Assigning into any level of the result raises `TypeError`.

`config_hash(config)` returns a stable content hash of a fully resolved config: the same hash for two configs equal as nested dicts/lists/scalars regardless of the order their keys were inserted in or which order composition and overrides produced them in, and a different hash whenever some value differs — including `2` versus `2.0`, and `0.0` versus `-0.0`, which compare equal in Python (`2 == 2.0`, `-0.0 == 0.0`) but are not the same value. It raises `ValueError` if `config` contains a `NaN` or an infinite `float`, neither of which this hash can represent.

State, in your solution, what else — beyond `config` and `config_hash(config)` — must be recorded when a run starts, for that exact run to be reproducible weeks later, possibly on different hardware.

```py
def freeze_config(config: dict):
    """Returns an immutable snapshot: nested dicts -> MappingProxyType, nested lists -> tuple."""

def config_hash(config: dict) -> str:
    """Returns a stable content hash of config, independent of key order."""
```

Example:

```text
frozen = freeze_config(final)
frozen["trainer"]["epochs"]          # 30 -- reads normally
frozen["trainer"]["epochs"] = 999    # raises TypeError

c1 = {"trainer": {"epochs": 30, "batch_size": 128}, "optimizer": {"name": "adamw"}}
c2 = {"optimizer": {"name": "adamw"}, "trainer": {"batch_size": 128, "epochs": 30}}  # same content, different key order
config_hash(c1) == config_hash(c2)                                    # True

c3 = {**c1, "trainer": {**c1["trainer"], "epochs": 31}}
config_hash(c3) != config_hash(c1)                                    # True
```
