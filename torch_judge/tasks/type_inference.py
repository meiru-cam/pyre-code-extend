"""Type strings and generic return-type inference over nested tuple types."""

from ._interview import interview

# Random types and a slow model written straight from the statement.
_HELPERS = r"""
import random

PRIMS = ["int", "float", "str", "bool", "char"]

def rand_type(rng, depth=0, names=PRIMS):
    if depth >= 3 or rng.random() < 0.55:
        return rng.choice(names)
    return [rand_type(rng, depth + 1, names) for _ in range(rng.randint(0, 3))]

def fmt(t):
    return t if isinstance(t, str) else "[" + ",".join(fmt(c) for c in t) + "]"

def generalise(rng, t, gens):
    # Replaces random subtrees of a concrete type by generics, reusing a name for an equal subtree.
    if rng.random() < 0.3:
        key = fmt(t)
        if key not in gens:
            gens[key] = f"T{len(gens) + 1}"
        return gens[key]
    if isinstance(t, list):
        return [generalise(rng, c, gens) for c in t]
    return t

def model_return(params, ret, args):
    if len(params) != len(args):
        return "ArityError"
    bound = {}
    def bind(e, a):
        if isinstance(e, str) and e not in PRIMS:
            if e in bound and bound[e] != a:
                raise LookupError("GenericConflictError")
            bound.setdefault(e, a)
        elif isinstance(e, list) and isinstance(a, list) and len(e) == len(a):
            for x, y in zip(e, a):
                bind(x, y)
        elif e != a:
            raise LookupError("TypeMismatchError")
    try:
        for e, a in zip(params, args):
            bind(e, a)
    except LookupError as err:
        return err.args[0]
    def sub(t):
        if isinstance(t, list):
            return [sub(c) for c in t]
        return t if t in PRIMS else bound[t]
    return sub(ret)

def outcome(checker, params, ret, args):
    try:
        return checker.return_type(params, ret, args)
    except Exception as e:
        return type(e).__name__

def raises_value_error(f, text):
    try:
        f(text)
    except ValueError:
        return True
    return False
"""

TASK = {
    "title": "Generic Type Inference",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TypeChecker",
    "description_en": r"""Build `TypeChecker`, which prints and parses function types and works out what a generic function returns for given arguments.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `TypeChecker` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A type is either a name (`str`) or a tuple of zero or more types (`list`), nested to any depth.
- The primitive names are `int`, `float`, `str`, `bool` and `char`. A generic is `T` followed by one or more digits, such as `T1` or `T12`.
- A function type is a list of parameter types and one return type.
- Never change a type you are given, and return new lists from every call: no two results share a list.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** printing a type is a warm-up; the work is matching a pattern against a value with consistent bindings, and each later part adds one requirement.

**Where it is used:** compilers and type checkers such as mypy infer the type of a generic call this way, and the same matching drives pattern matching and template engines.

Adapted from the type inference question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class. Types are plain strings and lists instead of node objects, and Part 1 adds parsing, the inverse of printing.""",
    "parts": [
        {
            "title": "Print and parse",
            "description_en": r"""**Signature:** `TypeChecker()`, `format_type(t) -> str`, `format_function(params, ret) -> str`, `parse_type(text)`, `parse_function(text) -> tuple[list, type]`

- `format_type` prints a name as itself and a tuple as `[` + its elements separated by `,` with no spaces + `]`. The empty tuple is `[]`.
- `format_function` prints `(` + the parameters, printed the same way + `) -> ` + the return type. There is exactly one space on each side of `->`.
- `parse_type` and `parse_function` are the exact inverses: they accept exactly the strings the format methods produce. Any other text raises `ValueError`, including extra spaces, a missing bracket, an empty element or an unknown name.
- `parse_function` returns `(params, ret)`, with `params` a `list`.

**Example:**
- `format_function(["int", ["T2", []], "T1"], ["T1", "str"])` is `"(int,[T2,[]],T1) -> [T1,str]"`
- `parse_function` of that string gives back `(["int", ["T2", []], "T1"], ["T1", "str"])`
- `parse_type("[int,]")`, `parse_type("[int, str]")` and `parse_type("string")` raise `ValueError`""",
        },
        {
            "title": "Return-type inference",
            "description_en": r"""Keep Part 1 and add inference.

**Signature:** `return_type(params, ret, args)`

- `args` holds one concrete type per parameter: a type with no generics in it.
- Match each parameter with its argument. Where the parameter is a generic, that generic is bound to the whole argument at that position, even a tuple. Where both are tuples of the same length, match element by element. Otherwise they must be equal.
- Every occurrence of one generic in one call must bind to equal types.
- Return `ret` with every generic replaced by its binding. Every generic in `ret` appears in some parameter. The result shares no `list` with the inputs, and no two parts of it share a `list`.
- Errors, as exception classes you define with these names: `ArityError` when `len(args) != len(params)`; `TypeMismatchError` when two types must be equal and are not; `GenericConflictError` when a generic would bind to two different types.
- Arity is checked first. Then parameters are matched from left to right, each one depth first from left to right, and the first problem found is raised.

**Example**, written with the Part 1 strings:
- `(T1,[int,T2]) -> [T2,T1]` called with `str`, `[int,bool]` returns `[bool,str]`
- `(T1) -> [T1,T1]` called with `[]` returns `[[],[]]`
- `(T1) -> T1` called with `int`, `int` raises `ArityError`
- `(int,T1) -> T1` called with `float`, `str` raises `TypeMismatchError`
- `(T1,[T1]) -> T1` called with `int`, `[str]` raises `GenericConflictError`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "A type is either a str or a list. Which of the two cases calls itself? When parsing, after reading one type, what single character tells you whether the tuple goes on or ends, and what should you do with anything else?"},
        {"level": 2, "kind": "analysis", "content": "format_type: return t for a str, else '[' + ','.join(format each element) + ']'. For parsing, write _parse(text, i) that returns (type, next index): on '[' read types separated by ',' until ']', handling '[]' first; otherwise match a primitive or T followed by digits. parse_type must also check that the whole string was used."},
    ],
    "model_connections": [
        "Tensor libraries type-check shapes and dtypes of generic operations, and tools such as jaxtyping bind a named dimension once per call and check every later use.",
        "Tool-calling models fill typed parameters; a schema with generics must bind each placeholder to one consistent type across arguments.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Separating matching from substitution keeps an unbound or partly bound generic from leaking into the result.",
            "A recursive parser that returns the next index makes every malformed string fail at a precise position.",
            "Checking whether the parameter itself is a generic, before comparing, lets a generic bind to a whole tuple.",
        ],
        "cons": [
            "One-way matching only works when the arguments are concrete; generics on both sides need full unification.",
            "Comparing types by recursion costs time proportional to their size on every conflict check.",
            "Exact error order is a contract callers can come to depend on, which makes later changes harder.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": _HELPERS + r"""
c = {fn}()
assert c.format_function(["int", ["T2", []], "T1"], ["T1", "str"]) == "(int,[T2,[]],T1) -> [T1,str]"
assert c.parse_function("(int,[T2,[]],T1) -> [T1,str]") == (["int", ["T2", []], "T1"], ["T1", "str"])
for bad in ["[int,]", "[int, str]", "string"]:
    assert raises_value_error(c.parse_type, bad), bad
"""},
        {"name": "Part 1: round trips and malformed text", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "A random type or function did not print as stated or did not parse back to the same value, or malformed text did not raise ValueError.",
         "code": _HELPERS + r"""
c = {fn}()
rng = random.Random(1)
names = PRIMS + ["T1", "T2", "T37"]
for _ in range(400):
    t = rand_type(rng, names=names)
    assert c.format_type(t) == fmt(t), t
    assert c.parse_type(fmt(t)) == t, fmt(t)
    params = [rand_type(rng, names=names) for _ in range(rng.randint(0, 3))]
    text = "(" + ",".join(fmt(p) for p in params) + ") -> " + fmt(t)
    assert c.format_function(params, t) == text, text
    got = c.parse_function(text)
    assert tuple(got) == (params, t) and isinstance(got[0], list), text
assert c.format_type([]) == "[]" and c.parse_type("[[]]") == [[]]
assert c.format_function([], "int") == "() -> int" and c.parse_function("() -> int") == ([], "int")
for bad in ["", "[", "]", "[int", "int]", "[,]", "[int,,str]", "Int", "T", "Tx", "intx", " int", "int ", "[int] ", "[[int]"]:
    assert raises_value_error(c.parse_type, bad), f"parse_type({bad!r}) should raise ValueError"
for bad in ["(int)->int", "(int) ->int", "(int,) -> int", "int -> int", "(int) -> ", "(int -> int", "(int) -> int,str", "() ->  int"]:
    assert raises_value_error(c.parse_function, bad), f"parse_function({bad!r}) should raise ValueError"
"""},
        {"name": "Part 1: inputs are not changed", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Printing changed the type it was given, or two parses of the same text returned the same list object.",
         "code": _HELPERS + r"""
c = {fn}()
t = [["T1", []], "int", ["bool", ["char"]]]
c.format_type(t)
c.format_function([t, t], t)
assert t == [["T1", []], "int", ["bool", ["char"]]]
a, b = c.parse_type("[[int]]"), c.parse_type("[[int]]")
a[0].append("str")
assert b == [["int"]], "each parse must build new lists"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "contract.signature", "code": _HELPERS + r"""
c = {fn}()
assert c.return_type(["T1", ["int", "T2"]], ["T2", "T1"], ["str", ["int", "bool"]]) == ["bool", "str"]
assert c.return_type(["T1"], ["T1", "T1"], [[]]) == [[], []]
assert outcome(c, ["T1"], "T1", ["int", "int"]) == "ArityError"
assert outcome(c, ["int", "T1"], "T1", ["float", "str"]) == "TypeMismatchError"
assert outcome(c, ["T1", ["T1"]], "T1", ["int", ["str"]]) == "GenericConflictError"
"""},
        {"name": "Part 2: random calls and error order", "part": 2, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "On a random generic function and call, the result or the error raised differed from the rules: arity first, then parameters left to right, depth first, and a generic may bind to a whole tuple.",
         "code": _HELPERS + r"""
c = {fn}()
rng = random.Random(2)
for trial in range(1500):
    args = [rand_type(rng) for _ in range(rng.randint(0, 4))]
    gens = {}
    params = [generalise(rng, a, gens) for a in args]
    names = sorted(set(gens.values())) or ["int"]
    ret = rand_type(rng, names=names + ["bool"])
    call = [list(a) if isinstance(a, list) else a for a in args]
    roll = rng.random()
    if roll < 0.15 and call:
        call.pop()
    elif roll < 0.6 and call:
        i = rng.randrange(len(call))
        call[i] = rand_type(rng)
    assert outcome(c, params, ret, call) == model_return(params, ret, call), (params, ret, call)
assert outcome(c, ["T1", "int"], "T1", [["str"], "float"]) == "TypeMismatchError"
assert outcome(c, ["T1", ["T1", "int"]], "T1", ["str", ["bool", "float"]]) == "GenericConflictError", "the conflict comes first, depth first"
assert outcome(c, [["int", "T1"]], "T1", [["int", "str", "bool"]]) == "TypeMismatchError", "tuples of different length"
assert outcome(c, [["T1"]], "T1", ["int"]) == "TypeMismatchError"
assert c.return_type(["T5", "T5"], ["T5", "char"], [["int", []], ["int", []]]) == [["int", []], "char"]
assert c.return_type([], ["float"], []) == ["float"]
"""},
        {"name": "Part 2: the result shares nothing", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "The returned type shared a list with the parameters, return type or arguments, or the call changed them.",
         "code": _HELPERS + r"""
c = {fn}()
params, ret, args = ["T1", ["bool"]], ["T1", "T1", ["bool"]], [["int", ["str"]], ["bool"]]
out = c.return_type(params, ret, args)
assert out == [["int", ["str"]], ["int", ["str"]], ["bool"]]
out[0][1].append("char")
out[2].append("float")
assert out[1] == ["int", ["str"]], "two uses of one generic must be separate lists"
assert params == ["T1", ["bool"]] and ret == ["T1", "T1", ["bool"]] and args == [["int", ["str"]], ["bool"]]
assert c.return_type(params, ret, args) == [["int", ["str"]], ["int", ["str"]], ["bool"]], "the same call must give the same answer again"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import re

PRIMITIVES = {"int", "float", "str", "bool", "char"}
_NAME = re.compile(r"int|float|str|bool|char|T[0-9]+")


class ArityError(Exception):
    pass


class TypeMismatchError(Exception):
    pass


class GenericConflictError(Exception):
    pass


class TypeChecker:
    def format_type(self, t):
        if isinstance(t, str):
            return t
        return "[" + ",".join(self.format_type(c) for c in t) + "]"

    def format_function(self, params, ret):
        return "(" + ",".join(self.format_type(p) for p in params) + ") -> " + self.format_type(ret)

    def parse_type(self, text):
        t, end = self._parse(text, 0)
        if end != len(text):
            raise ValueError(f"unexpected text at {end}: {text!r}")
        return t

    def parse_function(self, text):
        if not text.startswith("(") or ") -> " not in text:
            raise ValueError(f"not a function: {text!r}")
        params, i = [], 1
        if text[i:i + 1] != ")":
            while True:
                t, i = self._parse(text, i)
                params.append(t)
                if text[i:i + 1] != ",":
                    break
                i += 1
        if text[i:i + 5] != ") -> ":
            raise ValueError(f"expected ') -> ' at {i}: {text!r}")
        return params, self.parse_type(text[i + 5:])

    def _parse(self, text, i):
        """Parses one type starting at text[i]; returns (type, index just after it)."""
        if text[i:i + 1] == "[":
            items, i = [], i + 1
            if text[i:i + 1] == "]":
                return items, i + 1
            while True:
                t, i = self._parse(text, i)
                items.append(t)
                if text[i:i + 1] == "]":
                    return items, i + 1
                if text[i:i + 1] != ",":
                    raise ValueError(f"expected ',' or ']' at {i}: {text!r}")
                i += 1
        m = _NAME.match(text, i)
        if not m:
            raise ValueError(f"expected a type at {i}: {text!r}")
        return m.group(), m.end()

    def return_type(self, params, ret, args):
        if len(args) != len(params):
            raise ArityError(f"expected {len(params)} arguments, got {len(args)}")
        bindings = {}
        for expected, actual in zip(params, args):
            self._bind(expected, actual, bindings)
        return self._substitute(ret, bindings)

    def _bind(self, expected, actual, bindings):
        if isinstance(expected, str) and expected not in PRIMITIVES:  # the node itself is a generic
            if expected in bindings and bindings[expected] != actual:
                raise GenericConflictError(f"{expected} is bound to two different types: "
                                           f"{self.format_type(bindings[expected])} and {self.format_type(actual)}")
            bindings.setdefault(expected, actual)
            return
        if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
            for e, a in zip(expected, actual):
                self._bind(e, a, bindings)
            return
        if expected != actual:
            raise TypeMismatchError(f"expected {self.format_type(expected)} but received {self.format_type(actual)}")

    def _substitute(self, t, bindings):
        # always builds new lists, so the result never shares a list with the inputs
        if isinstance(t, list):
            return [self._substitute(c, bindings) for c in t]
        if t in PRIMITIVES:
            return t
        return self._substitute(bindings[t], {})
''',
    "interview_questions": interview(
        concept=[
            "How do you tell a name from a tuple in this representation, and which case recurses?",
            "Why must parsing check that the whole string was consumed?",
        ],
        deep_dive=[
            "How does a parser that returns the next index handle nested tuples and report the first bad position?",
        ],
        tradeoffs=[
            "Why must the check for a generic look at the parameter itself rather than at generics inside it?",
            "Why keep matching and substitution as two passes instead of one?",
            "Why must the returned type share no list with the inputs?",
            "What changes if arguments may contain generics too, and why is an occurs check needed?",
        ],
    ),
}
