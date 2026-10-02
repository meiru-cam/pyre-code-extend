"""Compose ML configs from bases, apply overrides and references, validate against dataclasses, then freeze and hash them."""

from ._interview import interview

# Error checks and an independent deep-merge model.
_HELPERS = r"""
import copy, dataclasses, math, random, types

def raises(kind, call, text=None):
    try:
        call()
    except Exception as e:
        names = [c.__name__ for c in type(e).__mro__]
        assert names[0] == kind, f"expected {kind}, got {names[0]}: {e}"
        assert kind == "TypeError" or kind == "ValueError" or "ConfigError" in names, f"{kind} must subclass ConfigError"
        if text is not None:
            assert text in str(e), f"{kind} message {str(e)!r} should contain {text!r}"
        return e
    raise AssertionError(f"expected {kind}, nothing was raised")

def merge(base, overlay, DELETE):
    out = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in overlay.items():
        if v is DELETE:
            out.pop(k, None)
        elif isinstance(v, dict):
            out[k] = merge(base[k] if isinstance(base.get(k), dict) else {}, v, DELETE)
        else:
            out[k] = copy.deepcopy(v)
    return out
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
D = {fn}.DELETE
store = {
    "defaults": {"model": {"layers": 6, "width": 512, "dropout": 0.1}, "data": {"path": "/data/tiny", "shuffle": True}},
    "big": {"_base_": ["defaults"], "model": {"layers": 24, "width": 1024}},
    "noisy": {"data": {"augment": "flip"}},
    "run7": {"_base_": ["big", "noisy"], "model": {"dropout": D}, "data": {"shuffle": False}},
}
assert {fn}(store).resolve("run7") == {"model": {"layers": 24, "width": 1024},
                                       "data": {"path": "/data/tiny", "shuffle": False, "augment": "flip"}}
"""},
    {"name": "Part 1: merge rules and copies", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "deep_merge must drop DELETE keys (also inside a dict whose base was not a dict), let a non-dict overlay replace a dict, merge dicts key by key, and return values shared with neither input; resolve must leave the store unchanged and never return _base_.",
     "code": _HELPERS + r"""
cs = {fn}({})
D = {fn}.DELETE
base = {"a": {"x": 1, "y": [1, 2]}, "b": 2, "c": {"k": 1}, "d": 5}
over = {"a": {"y": D, "z": {"deep": D, "keep": 3}}, "b": {"n": D, "m": 1}, "c": 7, "d": D, "e": [{"q": 1}]}
out = cs.deep_merge(base, over)
assert out == {"a": {"x": 1, "z": {"keep": 3}}, "b": {"m": 1}, "c": 7, "e": [{"q": 1}]}, out
out["a"]["x"] = 99
out["e"][0]["q"] = 99
assert base["a"]["x"] == 1 and over["e"][0]["q"] == 1, "results must not share values with the inputs"
assert cs.deep_merge({}, {}) == {} and cs.deep_merge({"a": 1}, {"a": D, "b": D}) == {}
plain = cs.deep_merge(base, {})
plain["a"]["y"].append(3)
assert base["a"]["y"] == [1, 2]
store = {"p": {"_base_": [], "v": {"w": 1}}, "q": {"_base_": ["p", "p"], "v": {"u": 2}}}
before = copy.deepcopy(store)
got = {fn}(store).resolve("q")
assert got == {"v": {"w": 1, "u": 2}} and "_base_" not in got
got["v"]["w"] = 5
assert store == before
"""},
    {"name": "Part 1: missing bases and cycles", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "A missing name anywhere in the _base_ graph raises ConfigNotFoundError; a config that is its own base, directly or through others, raises ConfigCycleError whose message spells the loop as 'a -> b -> a'; both subclass ConfigError; a diamond is not a cycle.",
     "code": _HELPERS + r"""
raises("ConfigNotFoundError", lambda: {fn}({}).resolve("nope"))
raises("ConfigNotFoundError", lambda: {fn}({"a": {"_base_": ["b"]}, "b": {"_base_": ["gone"]}}).resolve("a"))
raises("ConfigCycleError", lambda: {fn}({"s": {"_base_": ["s"]}}).resolve("s"), "s -> s")
loop = {"top": {"_base_": ["x"]}, "x": {"_base_": ["y"]}, "y": {"_base_": ["z"]}, "z": {"_base_": ["x"]}}
e = raises("ConfigCycleError", lambda: {fn}(loop).resolve("top"), "x -> y -> z -> x")
assert "top" not in str(e), f"the loop starts at x, so 'top' is not part of it: {str(e)!r}"
diamond = {"d": {"_base_": ["l", "r"]}, "l": {"_base_": ["root"], "v": 1}, "r": {"_base_": ["root"], "v": 2}, "root": {"v": 0, "w": 0}}
assert {fn}(diamond).resolve("d") == {"v": 2, "w": 0}
"""},
    {"name": "Part 1: random stores", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random acyclic stores, resolve differed from merging each base's resolution left to right and then the config's own keys.",
     "code": _HELPERS + r"""
D = {fn}.DELETE
def value(rng, depth):
    r = rng.random()
    if depth < 2 and r < 0.35:
        return {rng.choice("abcd"): value(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    if r < 0.5:
        return D
    return rng.choice([0, 1, 2.5, "s", None, True, [1, 2]])
for seed in range(200):
    rng = random.Random(seed)
    names = [f"c{i}" for i in range(6)]
    store = {}
    for i, name in enumerate(names):
        cfg = {rng.choice("abcd"): value(rng, 0) for _ in range(rng.randint(0, 3))}
        if i and rng.random() < 0.8:
            cfg["_base_"] = rng.sample(names[:i], rng.randint(1, min(2, i)))
        store[name] = cfg
    def resolved(n):
        out = {}
        for b in store[n].get("_base_", []):
            out = merge(out, resolved(b), D)
        return merge(out, {k: v for k, v in store[n].items() if k != "_base_"}, D)
    cs = {fn}(store)
    for name in names:
        assert cs.resolve(name) == resolved(name), (seed, name)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "protocol.validation", "code": r"""
cs = {fn}({})
config = {"model": {"layers": 24, "width": 1024}, "data": {"path": "/data/tiny", "shuffle": False, "augment": "flip"},
          "tag": "L${model.layers}-w${model.width}", "eval_width": "${model.width}"}
over = cs.apply_overrides(config, ["model.layers=12", "+data.workers=4", "data.path='/data/full'"])
assert config["model"]["layers"] == 24, "the input config must not change"
final = cs.resolve_interpolations(over)
assert final == {"model": {"layers": 12, "width": 1024},
                 "data": {"path": "/data/full", "shuffle": False, "augment": "flip", "workers": 4},
                 "tag": "L12-w1024", "eval_width": 1024}, final
"""},
    {"name": "Part 2: override grammar", "part": 2, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "Values are null, true, false, integers, floats (with a dot and digits, or an exponent), quoted strings without escapes, or flat lists of those with spaces only inside brackets and around commas; anything else raises OverrideSyntaxError. Paths need existing dict keys unless prefixed with +, which creates missing dicts; descending into a non-dict raises OverrideKeyError. Later overrides win.",
     "code": _HELPERS + r"""
cs = {fn}({})
base = {"a": {"b": 1}, "s": "x", "n": 0}
ok = {"n=null": None, "n=true": True, "n=false": False, "n=-12": -12, "n=3.50": 3.5, "n=1e3": 1000.0, "n=-2.5E-1": -0.25,
      "n='a b'": "a b", 'n="it\'s"': "it's", "n=[]": [], "n=[ 1 ,2.0,'x' , null,true ]": [1, 2.0, "x", None, True], "n=''": ""}
for text, want in ok.items():
    got = cs.apply_overrides(base, [text])["n"]
    assert got == want and type(got) is type(want), (text, got)
for bad in ["n=", "n=abc", "n=1.", "n=.5", "n= 1", "n =1", "n=[[1]]", "n=[1,]", "n='x", "n=\"a'", "n=TRUE", "n=01x", "=1", "a..b=1", "1a=2", "n:1", "n=1 "]:
    raises("OverrideSyntaxError", lambda: cs.apply_overrides(base, [bad]), None)
raises("OverrideKeyError", lambda: cs.apply_overrides(base, ["a.c=1"]))
raises("OverrideKeyError", lambda: cs.apply_overrides(base, ["z.c=1"]))
raises("OverrideKeyError", lambda: cs.apply_overrides(base, ["s.c=1"]))
raises("OverrideKeyError", lambda: cs.apply_overrides(base, ["+s.c=1"]))
raises("OverrideKeyError", lambda: cs.apply_overrides(base, ["+a.b.c=1"]))
assert cs.apply_overrides(base, ["+x.y.z=1", "+a.c=2", "a.b=[]", "a.b=3"]) == {"a": {"b": 3, "c": 2}, "s": "x", "n": 0, "x": {"y": {"z": 1}}}
assert cs.apply_overrides(base, ["a=null"]) == {"a": None, "s": "x", "n": 0}
assert base == {"a": {"b": 1}, "s": "x", "n": 0}
"""},
    {"name": "Part 2: interpolation rules", "part": 2, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "A string that is exactly one ${path} takes the referenced value of any type, resolved first; a reference inside a longer string needs an int, float, bool (true/false) or str; references work inside lists and chain through other references; a missing path raises InterpolationError and a loop raises InterpolationCycleError naming the paths.",
     "code": _HELPERS + r"""
cs = {fn}({})
cfg = {"a": {"n": 3, "f": 0.5, "b": True, "s": "hi", "l": [1, 2], "none": None, "d": {"k": "${a.n}"}},
       "whole_list": "${a.l}", "whole_none": "${a.none}", "whole_dict": "${a.d}",
       "mix": "${a.n}/${a.f}/${a.b}/${a.s}", "chain": "${mix}!", "inlist": ["${a.n}", "x${a.s}"], "plain": "$a.n {a.n}"}
out = cs.resolve_interpolations(cfg)
assert out["whole_list"] == [1, 2] and out["whole_none"] is None and out["whole_dict"] == {"k": 3}
assert out["mix"] == "3/0.5/true/hi" and out["chain"] == "3/0.5/true/hi!"
assert out["inlist"] == [3, "xhi"] and out["plain"] == "$a.n {a.n}" and out["a"]["d"]["k"] == 3
out["whole_list"].append(9)
assert cfg["a"]["l"] == [1, 2] and cfg["a"]["d"]["k"] == "${a.n}"
raises("InterpolationError", lambda: cs.resolve_interpolations({"x": "${y}"}))
raises("InterpolationError", lambda: cs.resolve_interpolations({"x": "${y.z}", "y": 1}))
raises("InterpolationError", lambda: cs.resolve_interpolations({"x": "v${y}", "y": [1]}))
raises("InterpolationError", lambda: cs.resolve_interpolations({"x": "v${y}", "y": None}))
e = raises("InterpolationCycleError", lambda: cs.resolve_interpolations({"p": {"q": "${r}"}, "r": "x${p.q}"}))
assert "p.q -> r -> p.q" in str(e) or "r -> p.q -> r" in str(e), str(e)
raises("InterpolationCycleError", lambda: cs.resolve_interpolations({"s": "${s}"}), "s -> s")
e = raises("InterpolationCycleError", lambda: cs.resolve_interpolations({"entry": "${p.q}", "p": {"q": "${r}"}, "r": "x${p.q}"}), "p.q -> r -> p.q")
assert "entry" not in str(e), f"the loop starts at p.q, so 'entry' is not part of it: {str(e)!r}"
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "contract.signature", "code": r"""
import dataclasses
@dataclasses.dataclass
class ModelCfg:
    layers: int
    width: int
    dropout: float = 0.0
@dataclasses.dataclass
class DataCfg:
    path: str
    shuffle: bool
    workers: int = 0
@dataclasses.dataclass
class RunCfg:
    model: ModelCfg
    data: DataCfg
    tag: str
    seeds: list[int] = dataclasses.field(default_factory=list)
cs = {fn}({})
good = {"model": {"layers": 12, "width": 1024}, "data": {"path": "/data/full", "shuffle": False, "workers": 4}, "tag": "L12"}
assert cs.validate(good, RunCfg) == []
bad = {"model": {"layers": 12, "width": "1024", "depth": 3}, "data": {"path": "/d", "shuffle": False}, "seeds": [1, "2"]}
assert sorted(cs.validate(bad, RunCfg)) == sorted([("model.width", "expected int, got str"), ("model.depth", "not in schema"),
                                                    ("tag", "missing"), ("seeds.1", "expected int, got str")])
"""},
    {"name": "Part 3: types, bools and nesting", "part": 3, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "A bool never satisfies int or float, an int satisfies float, a list[X] checks each element at path.index, a nested schema needs a dict and is checked inside, None fails every type, optional fields may be absent, every mismatch is collected, and a non-dict config raises TypeError.",
     "code": _HELPERS + r"""
@dataclasses.dataclass
class Leaf:
    v: float
@dataclasses.dataclass
class Root:
    i: int
    f: float
    b: bool
    s: str
    leaves: list[Leaf]
    names: list[str]
    one: Leaf
    opt: int = 7
cs = {fn}({})
ok = {"i": 1, "f": 2, "b": False, "s": "", "leaves": [{"v": 1.5}, {"v": 0}], "names": [], "one": {"v": -1.0}}
assert cs.validate(ok, Root) == []
bad = {"i": True, "f": False, "b": 1, "s": None, "leaves": [{"v": "x"}, 3, {"w": 1}], "names": "ab", "one": [1], "opt": 1.0, "extra": {}}
want = [("i", "expected int, got bool"), ("f", "expected float, got bool"), ("b", "expected bool, got int"),
        ("s", "expected str, got None"), ("leaves.0.v", "expected float, got str"), ("leaves.1", "expected Leaf, got int"),
        ("leaves.2.v", "missing"), ("leaves.2.w", "not in schema"), ("names", "expected list, got str"),
        ("one", "expected Leaf, got list"), ("opt", "expected int, got float"), ("extra", "not in schema")]
assert sorted(cs.validate(bad, Root)) == sorted(want), sorted(cs.validate(bad, Root))
assert sorted(cs.validate({}, Root)) == sorted((n, "missing") for n in ["i", "f", "b", "s", "leaves", "names", "one"])
raises("TypeError", lambda: cs.validate([1], Root))
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": r"""
cs = {fn}({})
final = {"model": {"layers": 12, "width": 1024}, "data": {"path": "/data/full", "workers": 4}, "seeds": [1, 2]}
frozen = cs.freeze(final)
assert frozen["model"]["layers"] == 12 and frozen["seeds"] == (1, 2)
try:
    frozen["model"]["layers"] = 99
    raise AssertionError("assigning into a frozen config must raise TypeError")
except TypeError:
    pass
same = {"seeds": [1, 2], "data": {"workers": 4, "path": "/data/full"}, "model": {"width": 1024, "layers": 12}}
assert cs.config_hash(final) == cs.config_hash(same)
assert cs.config_hash(final) != cs.config_hash({**final, "seeds": [2, 1]})
"""},
    {"name": "Part 4: freezing and hashing edge cases", "part": 4, "visibility": "unshown", "behavior": "numerics.stability",
     "failure_message": "freeze must turn every dict into a read-only mapping and every list into a tuple at all depths without changing the input; config_hash returns the same str for equal content regardless of key order or object identity, a different one for 2 vs 2.0, 0.0 vs -0.0, True vs 1, 'a' vs ['a'] and changed nesting, and raises ValueError for NaN or infinity.",
     "code": _HELPERS + r"""
cs = {fn}({})
cfg = {"a": [{"b": [1, {"c": 2}]}], "d": {}}
fz = cs.freeze(cfg)
assert isinstance(fz["a"], tuple) and isinstance(fz["a"][0]["b"], tuple) and fz["a"][0]["b"][1]["c"] == 2
def put(target, key, value):
    target[key] = value
for target, key in [(fz, "x"), (fz["a"][0], "b"), (fz["a"][0]["b"][1], "c"), (fz["d"], "e"), (fz["a"], 0)]:
    try:
        put(target, key, 1)
        raise AssertionError("frozen configs must reject assignment at every depth")
    except TypeError:
        pass
cfg["a"][0]["b"].append(5)
assert len(fz["a"][0]["b"]) == 2 and isinstance(cfg["a"], list), "freeze must not share or change the input"
h = cs.config_hash
assert isinstance(h({}), str) and h({"x": 1}) == h(copy.deepcopy({"x": 1})) == {fn}({}).config_hash({"x": 1})
pairs = [({"x": 2}, {"x": 2.0}), ({"x": 0.0}, {"x": -0.0}), ({"x": True}, {"x": 1}), ({"x": "a"}, {"x": ["a"]}),
         ({"x": {"y": 1}}, {"x.y": 1}), ({"x": None}, {"x": "None"}), ({"x": [1, [2]]}, {"x": [[1], 2]}), ({"x": "1"}, {"x": 1}),
         ({"a": "b,c"}, {"a": "b", "c": ""}), ({"x": 0.1 + 0.2}, {"x": 0.3})]
for left, right in pairs:
    assert h(left) != h(right), (left, right)
for bad in [float("nan"), float("inf"), -float("inf")]:
    raises("ValueError", lambda: h({"deep": [1, {"v": bad}]}))
rng = random.Random(4)
for _ in range(50):
    keys = [f"k{i}" for i in range(6)]
    a = {k: rng.choice([1, 2.0, "s", None, [1]]) for k in keys}
    shuffled = dict(sorted(a.items(), key=lambda kv: rng.random()))
    assert h(a) == h(shuffled)
"""},
]

TASK = {
    "title": "ML Config System",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ConfigStore",
    "description_en": r"""Build `ConfigStore`, which composes training configs from named bases, applies command-line overrides and `${...}` references, validates the result against dataclasses, and freezes and hashes it.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ConfigStore` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A config is a `dict` with `str` keys whose values are dicts, lists, or scalars: `int`, `float`, `bool`, `str` or `None`, nested to any depth.
- `ConfigStore(store)` takes a `dict` from config name to config. Every method returns new objects and never changes `store` or any argument.
- Errors are your own classes, recognised by class name. Every one of them subclasses a class named `ConfigError`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** every ML team has a config system, and the question checks whether you can write a precise recursive merge, a tiny parser, and a reference resolver with cycle detection. Each later part adds one requirement.

**Where it is used:** Hydra and OmegaConf (composition, overrides like `+key=value`, `${...}` interpolation), mmcv's `_base_` files, and experiment trackers that hash configs to identify runs.

Adapted from the ML config system question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The functions become methods, validation errors are `(path, message)` tuples with short fixed messages, and the question about what else to record for reproducibility is left to the interview questions.""",
    "parts": [
        {
            "title": "Compose from bases",
            "description_en": r"""**Signatures:** `ConfigStore.DELETE`, a sentinel value; `deep_merge(base, overlay) -> dict`; `resolve(name) -> dict`

**`deep_merge`**, key by key:
- a key whose overlay value is `DELETE` is left out
- a key whose overlay value is a dict becomes `deep_merge(b, overlay[key])`, where `b` is `base[key]` if that is a dict and `{}` otherwise
- any other overlay value replaces the base value; keys only in `base` keep their value
- the result shares no object with either input

**`resolve`:**
- A config may have a top-level key `_base_`, a list of other names. Start from `{}`, merge in the resolution of each base from left to right, then merge in the config's own keys except `_base_`.
- Raise `ConfigNotFoundError` if `name` or any name reached through `_base_` is missing.
- Raise `ConfigCycleError` if a config is its own base, directly or through others. Its message spells the loop starting and ending at the repeated name, such as `"x -> y -> z -> x"`.

**Example:** `defaults` holds `model` `{"layers": 6, "width": 512, "dropout": 0.1}` and `data` `{"path": "/data/tiny", "shuffle": True}`; `big` has `_base_: ["defaults"]` and `model` `{"layers": 24, "width": 1024}`; `noisy` holds `data` `{"augment": "flip"}`; `run7` has `_base_: ["big", "noisy"]`, `model` `{"dropout": DELETE}` and `data` `{"shuffle": False}`:
- `resolve("run7")` is `{"model": {"layers": 24, "width": 1024}, "data": {"path": "/data/tiny", "shuffle": False, "augment": "flip"}}`""",
        },
        {
            "title": "Overrides and references",
            "description_en": r"""Keep Part 1 and add two methods.

**`apply_overrides(config, overrides) -> dict`** applies each string in order to a copy:
- An override is `path=value` or `+path=value`. `path` is names matching `[A-Za-z_]\w*` joined by `.`, each a dict key.
- Without `+`, every key on the path must exist, or raise `OverrideKeyError`. With `+`, missing keys are created as dicts and the last one is set. Either way, a key before the last that exists but is not a dict raises `OverrideKeyError`.
- `value` is `null`, `true`, `false`, an integer like `-12`, a float with a dot and digits or an exponent like `3.50` or `1e3`, a string in single or double quotes with no escapes, or a list of those in `[...]`. Spaces are allowed only just inside the brackets and around commas. Anything else raises `OverrideSyntaxError`.

**`resolve_interpolations(config) -> dict`** replaces `${path}` references in string values, also inside lists:
- A string that is exactly one reference becomes the referenced value, of any type, with its own references resolved first.
- A reference inside a longer string needs an `int`, `float`, `bool` or `str` and is replaced by its text; bools write `true` and `false`.
- Raise `InterpolationError` for a missing path or a list, dict or `None` inside a longer string. Raise `InterpolationCycleError` if a value needs itself; its message spells the loop of paths, such as `"p.q -> r -> p.q"`.

**Example:** with `tag` `"L${model.layers}-w${model.width}"` and `eval_width` `"${model.width}"` added to `run7`'s result, the overrides `model.layers=12`, `+data.workers=4` and `data.path='/data/full'`, then `resolve_interpolations`, give `layers` `12`, `workers` `4`, `path` `"/data/full"`, `tag` `"L12-w1024"` and `eval_width` the int `1024`.""",
        },
        {
            "title": "Validate against a schema",
            "description_en": r"""Keep Parts 1–2 and add validation.

**Signature:** `validate(config, schema) -> list[tuple[str, str]]`

- `schema` is a `dataclasses.dataclass` class. Its fields are typed `int`, `float`, `bool`, `str`, another schema class, or `list[X]` of one of these. A field with a default or `default_factory` is optional.
- Return one `(path, message)` per problem, in any order, collecting all of them. `path` is the dotted path from the root, with list positions as numbers, such as `"seeds.1"`.
- Messages: `"missing"` for an absent required field; `"not in schema"` for a key the schema class does not declare; `"expected {type}, got {type}"` for a wrong type, using class names and `None`, such as `"expected int, got str"`.
- A `bool` never satisfies `int` or `float`. An `int` satisfies `float`. A nested schema needs a dict and is checked inside; a `list[X]` needs a list and each element is checked.
- Raise `TypeError` if `config` is not a dict.

**Example:** for a `RunCfg` with `model: ModelCfg` (`layers: int`, `width: int`, `dropout: float = 0.0`), `data: DataCfg`, `tag: str` and `seeds: list[int]` with a default:
- a config with `width` `"1024"`, an extra `model.depth`, no `tag`, and `seeds` `[1, "2"]` gives `("model.width", "expected int, got str")`, `("model.depth", "not in schema")`, `("tag", "missing")` and `("seeds.1", "expected int, got str")`""",
        },
        {
            "title": "Freeze and hash",
            "description_en": r"""Keep Parts 1–3 and add two methods.

- `freeze(config)` returns a read-only copy: every dict becomes a `types.MappingProxyType` and every list a `tuple`, at every depth. Assigning into any level raises `TypeError`.
- `config_hash(config) -> str` returns the same string for configs with equal content, whatever the key order. It differs whenever any value differs, including `2` and `2.0`, `0.0` and `-0.0`, and `True` and `1`. It raises `ValueError` if the config holds a `NaN` or an infinite float.

**Example:**
- `freeze(final)["model"]["layers"]` reads `12`, and `freeze(final)["seeds"]` is `(1, 2)`; assigning to `["model"]["layers"]` raises `TypeError`
- the same config written with its keys in another order has the same hash; with `seeds` `[2, 1]` the hash changes""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "deep_merge has four cases per key. Which one must you check first so a DELETE inside a nested dict still works when the base had no dict there? For resolve, what do you need to remember while recursing to notice that a name leads back to itself?"},
        {"level": 2, "kind": "analysis", "content": "deep_merge: start from deep copies of base's keys; for each overlay key, skip and drop it if the value is DELETE, recurse with base[key] or {} if it is a dict, else store a deep copy. resolve(name, path=()): if name in path, raise with the loop; if missing, raise not found; merge each base's resolve(base, path + (name,)), then the own keys."},
    ],
    "model_connections": [
        "Hydra and OmegaConf drive most large training stacks: defaults lists compose configs, +key=value adds keys, and ${...} ties values together.",
        "Experiment trackers identify a run by a hash of its resolved config, so 2 versus 2.0 or a reordered dict must not change it by accident.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Recursive deep merge with a DELETE sentinel lets an overlay remove keys as well as change them.",
            "Strict override syntax catches typos such as a bare word where a string was meant.",
            "A canonical encoding with type tags gives a stable hash that tells 2 and 2.0 apart.",
        ],
        "cons": [
            "Deep copies on every merge cost time and memory proportional to the config size.",
            "References resolved after merging can point at keys an override removed, which only shows up late.",
            "Validation against dataclasses covers only the types it knows; unions and optionals need more rules.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import copy
import dataclasses
import hashlib
import math
import re
import types
import typing


class ConfigError(Exception):
    pass


class ConfigNotFoundError(ConfigError):
    pass


class ConfigCycleError(ConfigError):
    pass


class OverrideSyntaxError(ConfigError):
    pass


class OverrideKeyError(ConfigError):
    pass


class InterpolationError(ConfigError):
    pass


class InterpolationCycleError(ConfigError):
    pass


class _Delete:
    def __repr__(self):
        return "DELETE"


_PATH = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
_SCALAR = r"""null|true|false|-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|'[^']*'|"[^"]*\""""
_OVERRIDE = re.compile(rf"(\+?)({_PATH})=(.*)", re.S)
_LIST = re.compile(rf"\[\s*(?:(?:{_SCALAR})(?:\s*,\s*(?:{_SCALAR}))*)?\s*\]")
_REF = re.compile(rf"\$\{{({_PATH})\}}")


def _scalar(text):
    if text == "null":
        return None
    if text in ("true", "false"):
        return text == "true"
    if text[0] in "'\"":
        return text[1:-1]
    return float(text) if any(c in text for c in ".eE") else int(text)


def _parse_value(text):
    if re.fullmatch(_SCALAR, text):
        return _scalar(text)
    if _LIST.fullmatch(text):
        return [_scalar(m.group(0)) for m in re.finditer(_SCALAR, text[1:-1])]
    raise OverrideSyntaxError(f"bad value {text!r}")


def _type_name(value):
    return "None" if value is None else type(value).__name__


def _encode(value):
    """A canonical text form that keeps 2 apart from 2.0 and 0.0 apart from -0.0."""
    if value is None:
        return "n"
    if isinstance(value, bool):
        return "b1" if value else "b0"
    if isinstance(value, int):
        return f"i{value}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN and infinity cannot be hashed")
        return f"f{value.hex()}"
    if isinstance(value, str):
        return f"s{len(value)}:{value}"
    if isinstance(value, (list, tuple)):
        return "l[" + ",".join(_encode(v) for v in value) + "]"
    if isinstance(value, (dict, types.MappingProxyType)):
        return "d{" + ",".join(f"{_encode(k)}={_encode(value[k])}" for k in sorted(value)) + "}"
    raise TypeError(f"cannot hash {type(value).__name__}")


class ConfigStore:
    DELETE = _Delete()

    def __init__(self, store):
        self._store = store

    # -- part 1 ----------------------------------------------------------
    def deep_merge(self, base, overlay):
        result = {key: copy.deepcopy(value) for key, value in base.items() if key not in overlay}
        for key, value in overlay.items():
            if value is self.DELETE:
                continue
            if isinstance(value, dict):
                inner = base.get(key)
                result[key] = self.deep_merge(inner if isinstance(inner, dict) else {}, value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def resolve(self, name, _path=()):
        if name in _path:
            loop = list(_path[_path.index(name):]) + [name]
            raise ConfigCycleError(" -> ".join(loop))
        if name not in self._store:
            raise ConfigNotFoundError(name)
        own = self._store[name]
        result = {}
        for base in own.get("_base_", []):
            result = self.deep_merge(result, self.resolve(base, _path + (name,)))
        return self.deep_merge(result, {k: v for k, v in own.items() if k != "_base_"})

    # -- part 2 ----------------------------------------------------------
    def apply_overrides(self, config, overrides):
        result = copy.deepcopy(config)
        for text in overrides:
            match = _OVERRIDE.fullmatch(text)
            if match is None:
                raise OverrideSyntaxError(f"bad override {text!r}")
            plus, path, raw = match.groups()
            value = _parse_value(raw)
            keys = path.split(".")
            node = result
            for key in keys[:-1]:
                if key not in node:
                    if not plus:
                        raise OverrideKeyError(f"{path}: {key} is missing")
                    node[key] = {}
                if not isinstance(node[key], dict):
                    raise OverrideKeyError(f"{path}: {key} is not a dict")
                node = node[key]
            if keys[-1] not in node and not plus:
                raise OverrideKeyError(f"{path}: {keys[-1]} is missing")
            node[keys[-1]] = value
        return result

    def resolve_interpolations(self, config):
        done = {}

        def lookup(path):
            node = config
            for key in path.split("."):
                if not isinstance(node, dict) or key not in node:
                    raise InterpolationError(f"{path} does not exist")
                node = node[key]
            return node

        def value_at(path, stack):
            if path in stack:
                loop = list(stack[stack.index(path):]) + [path]
                raise InterpolationCycleError(" -> ".join(loop))
            if path not in done:
                done[path] = walk(lookup(path), path, stack + [path])
            return done[path]

        def text(value, path):
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float, str)):
                return str(value)
            raise InterpolationError(f"{path} is {_type_name(value)}, which cannot sit inside a string")

        def walk(node, here, stack):
            if isinstance(node, dict):  # every keyed value goes through value_at, so loops are seen from where they start
                return {k: value_at(f"{here}.{k}" if here else k, stack) for k in node}
            if isinstance(node, list):
                return [walk(v, here, stack) for v in node]
            if not isinstance(node, str):
                return node
            whole = _REF.fullmatch(node)
            if whole:
                return copy.deepcopy(value_at(whole.group(1), stack))
            return _REF.sub(lambda m: text(value_at(m.group(1), stack), m.group(1)), node)

        return walk(config, "", [])

    # -- part 3 ----------------------------------------------------------
    def validate(self, config, schema):
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        errors = []

        def check(value, kind, path):
            if dataclasses.is_dataclass(kind):
                if not isinstance(value, dict):
                    errors.append((path, f"expected {kind.__name__}, got {_type_name(value)}"))
                else:
                    record(value, kind, path + ".")
            elif typing.get_origin(kind) is list:
                if not isinstance(value, list):
                    errors.append((path, f"expected list, got {_type_name(value)}"))
                else:
                    for i, item in enumerate(value):
                        check(item, typing.get_args(kind)[0], f"{path}.{i}")
            else:
                ok = (isinstance(value, kind) and not (isinstance(value, bool) and kind is not bool)) or (
                    kind is float and isinstance(value, int) and not isinstance(value, bool))
                if not ok:
                    errors.append((path, f"expected {kind.__name__}, got {_type_name(value)}"))

        def record(node, kind, prefix):
            fields = {f.name: f for f in dataclasses.fields(kind)}
            for name, f in fields.items():
                if name in node:
                    check(node[name], f.type, prefix + name)
                elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    errors.append((prefix + name, "missing"))
            for name in node:
                if name not in fields:
                    errors.append((prefix + name, "not in schema"))

        record(config, schema, "")
        return errors

    # -- part 4 ----------------------------------------------------------
    def freeze(self, config):
        if isinstance(config, dict):
            return types.MappingProxyType({k: self.freeze(v) for k, v in config.items()})
        if isinstance(config, list):
            return tuple(self.freeze(v) for v in config)
        return config

    def config_hash(self, config):
        return hashlib.sha256(_encode(config).encode()).hexdigest()
''',
    "interview_questions": interview(
        concept=[
            "What should deep_merge do when the overlay holds a dict but the base holds a scalar, and why?",
            "Why must merge results share no objects with their inputs?",
        ],
        deep_dive=[
            "How do you detect a cycle in the _base_ graph and report the loop, and why is a diamond not a cycle?",
        ],
        tradeoffs=[
            "Why require + to create a key in an override instead of creating missing keys silently?",
            "Why can a whole-string reference keep the referenced type while an embedded one cannot?",
            "Why is a bool rejected for an int field, given that bool subclasses int in Python?",
            "Besides the config and its hash, what must a run record to be reproducible weeks later on other hardware?",
        ],
    ),
}
