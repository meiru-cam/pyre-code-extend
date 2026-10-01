"""A spreadsheet of integer formulas: evaluate on demand, then keep every cell fresh."""

from ._interview import interview

# Helpers shared by the cases. Learner-defined exceptions are matched by class name,
# since a case can only see the Spreadsheet class itself.
_HELPERS = r"""
import random, re, time

def raises(name, fn, *args):
    try:
        fn(*args)
    except Exception as error:
        assert type(error).__name__ == name, f"expected {name}, got {type(error).__name__}: {error}"
        return True
    return False

class Oracle:
    # Straight from the statement: store raw values, evaluate recursively on every read,
    # and reject a set that would close a cycle.
    def __init__(self):
        self.raw = {}
    def refs(self, value):
        return re.findall(r"[A-Z]+[0-9]+", value) if isinstance(value, str) else []
    def reaches(self, start, goal, seen=None):
        seen = set() if seen is None else seen
        if start == goal:
            return True
        if start in seen:
            return False
        seen.add(start)
        return any(self.reaches(ref, goal, seen) for ref in self.refs(self.raw.get(start)))
    def set(self, name, value):
        if any(self.reaches(ref, name) for ref in self.refs(value)):
            return "cycle"
        self.raw[name] = value
    def get(self, name):
        value = self.raw.get(name, 0)
        if isinstance(value, int):
            return value
        total, sign = 0, 1
        for token in re.findall(r"[A-Z]+[0-9]+|[0-9]+|[+-]", value):
            if token in "+-":
                sign = 1 if token == "+" else -1
            else:
                total += sign * (self.get(token) if token[0].isalpha() else int(token))
        return total

def random_formula(rng, cells):
    terms = [rng.choice(cells) if rng.random() < 0.7 else str(rng.randint(0, 99))
             for _ in range(rng.randint(1, 3))]
    out = "=" + terms[0]
    for term in terms[1:]:
        out += rng.choice([" + ", "-", " - ", "+"]) + term
    return out
"""

TASK = {
    "title": "Spreadsheet with Formula Dependencies",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "Spreadsheet",
    "description_en": r"""Build `Spreadsheet`, whose cells hold integers or formulas that add and subtract other cells.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Spreadsheet` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A cell name is one or more uppercase letters `A`–`Z` followed by one or more digits, such as `A1` or `AB12`. `a1`, `A` and `A1B` are not cell names.
- A cell holds an `int` or a formula: `=` followed by terms joined by exactly one `+` or `-` each, like `=R1 + 20 - S3` or `=5`.
- A term is a cell name or a non-negative integer of at most 18 digits. Spaces may surround terms and operators, but not sit inside a term.
- Nothing else is allowed: no `*`, `/`, parentheses, or operator before the first term or after the last.
- A cell that was never set reads as `0`.
- Define two exception classes, `FormulaError` and `CycleError`. `set` and `get` raise `FormulaError` for a bad cell name, and `set` raises it for a bad value.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first version is a recursive evaluator. Later requirements turn it into a dependency graph with cycle detection and ordered recomputation, which is where most bugs hide.

**Where it is used:** spreadsheet engines, build systems and reactive UI frameworks all keep a graph of what reads what and recompute only what a change reaches.

Adapted from the spreadsheet dependencies question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class, with part 1's cycle rule relaxed so that the parts build on each other.""",
    "parts": [
        {
            "title": "Evaluate formulas",
            "description_en": r"""**Signature:** `set(name, value) -> None`, `get(name) -> int`

- `set` stores `value` for `name`, replacing what it held.
- `get` returns the cell's value: the integer itself, or the formula evaluated with every referenced cell's current value.
- Cells form a cycle when a formula reads its own cell, directly or through other cells. A cycle must be reported as `CycleError`, either by the `set` that closes it or by a `get` that needs a cell on it.
- Assume a chain of references never passes through more than 200 cells.

**Example:**
- `set("P4", 6)`, `set("Q2", 17)`, `set("M3", "=Q2 - P4")`, `set("M8", "=M3 - P4 + 30")`
- `get("M8")` is `35`
- `set("P4", 14)`, then `get("M8")` is `19`""",
        },
        {
            "title": "Update eagerly",
            "description_en": r"""Reads must now be instant. Keep Part 1 and add:

- `set` recomputes every cell its change affects before returning, so `get` only returns a stored value and runs in O(1).
- A `set` that would create a cycle raises `CycleError` and changes nothing: the cell's previous content, every dependency and every stored value stay as they were.

**Example:**
- `set("U1", "=U2")` succeeds and `get("U1")` is `0`
- `set("U2", "=U1")` raises `CycleError`, and `U1` and `U2` are unchanged""",
        },
        {
            "title": "Edge cases",
            "description_en": r"""Keep Parts 1–2 and make sure these hold:

- Overwriting a formula, with a new formula or an integer, drops its old dependencies: a cell it used to read no longer affects it, and can now read it back without a cycle.
- A formula that reads its own cell, `set("H2", "=H2")`, raises `CycleError` like any other cycle.
- A formula may read a cell that is not set yet. Setting that cell later updates every cell that depends on it, directly or not.
- When two cells read a common cell and a fourth reads both of them, changing the common cell gives the fourth the correct value.
- Overwriting an integer with a formula updates the cell and every cell that reads it in that same `set`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which pattern accepts exactly the valid names and formulas, and nothing else? How does get evaluate a term that is itself a formula cell? What would get need to remember while evaluating to notice it came back to a cell it is still evaluating?"},
        {"level": 2, "kind": "analysis", "content": "Validate with two full-match regular expressions: one for a cell name and one for a whole formula, then split the formula into signed terms. Evaluate recursively, keeping the set of cells on the current evaluation path; meeting one of them again is a cycle, so raise CycleError there."},
    ],
    "model_connections": [
        "Spreadsheet engines such as LibreOffice Calc keep a dependency graph and recompute only the cells a change reaches, in dependency order.",
        "Build systems like Bazel and make reject dependency cycles and rebuild targets in topological order, which is part 2's recomputation.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Eager recomputation makes every read O(1), which suits read-heavy workloads.",
            "Rejecting a cycle at set time keeps the graph acyclic, so recomputation always terminates.",
            "Recomputing in topological order evaluates each affected cell exactly once per change.",
        ],
        "cons": [
            "A set on a widely read cell recomputes every dependent before returning, so writes get slow.",
            "Keeping both directions of the dependency graph doubles the bookkeeping on every formula change.",
            "Lazy evaluation stays cheaper when most cells are written far more often than read.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
sheet = {fn}()
sheet.set("P4", 6)
sheet.set("Q2", 17)
sheet.set("M3", "=Q2 - P4")
sheet.set("M8", "=M3 - P4 + 30")
assert sheet.get("M8") == 35
sheet.set("P4", 14)
assert sheet.get("M8") == 19
assert sheet.get("Z99") == 0
"""},
        {"name": "Part 1: names and formulas are validated", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "set and get must raise FormulaError for a bad cell name, and set for a value that is neither an int nor a formula of the stated shape.",
         "code": _HELPERS + r"""
sheet = {fn}()
for bad in ("a1", "A", "A1B", "1A", "", "A 1", "Ä1"):
    assert raises("FormulaError", sheet.set, bad, 1), f"set({bad!r}, 1) must raise FormulaError"
    assert raises("FormulaError", sheet.get, bad), f"get({bad!r}) must raise FormulaError"
for bad in ("=-R1", "=R1 - -S3", "=R1 2", "=", "=  ", "R1", "=R1*2", "=(1)", "=R1+", "=1234567890123456789", "=r1", 3.5, None, [1]):
    assert raises("FormulaError", sheet.set, "A1", bad), f"set('A1', {bad!r}) must raise FormulaError"
for good in ("=5", "= R1 + 20 - S3 ", "=AB12", "=123456789012345678", "=0-0+B2"):
    sheet.set("A1", good)
sheet.set("A1", -7)
assert sheet.get("A1") == -7
"""},
        {"name": "Part 1: cycles are reported", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A cycle must raise CycleError, from the set that closes it or from a get that needs a cell on it.",
         "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("U1", "=U2")
sheet.set("W1", "=U1 + 1")
if not raises("CycleError", sheet.set, "U2", "=U1"):
    assert raises("CycleError", sheet.get, "U1"), "get on a cycle must raise CycleError"
    assert raises("CycleError", sheet.get, "W1"), "get on a cell that reads a cycle must raise CycleError"
sheet.set("C1", "=C3 + 1")
sheet.set("C2", "=C1")
if not raises("CycleError", sheet.set, "C3", "=C2"):
    assert raises("CycleError", sheet.get, "C3")
"""},
        {"name": "Part 1: random acyclic sheets match a reference", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random sheet without cycles evaluated to a different value than the rules give.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    sheet, oracle = {fn}(), Oracle()
    cells = [f"C{i}" for i in range(15)]
    for i, cell in enumerate(cells):
        value = rng.randint(-50, 50) if i < 3 or rng.random() < 0.3 else random_formula(rng, cells[:i])
        sheet.set(cell, value)
        oracle.set(cell, value)
    for _ in range(10):
        cell = rng.choice(cells[:3])
        value = rng.randint(-50, 50)
        sheet.set(cell, value)
        oracle.set(cell, value)
        for c in cells:
            assert sheet.get(c) == oracle.get(c), (seed, c)
"""},
        {"name": "Part 2: a cycle is rejected at set", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("U1", "=U2")
assert sheet.get("U1") == 0
assert raises("CycleError", sheet.set, "U2", "=U1"), "set must reject a cycle"
assert sheet.get("U1") == 0 and sheet.get("U2") == 0
sheet.set("U2", 5)
assert sheet.get("U1") == 5
"""},
        {"name": "Part 2: a rejected set changes nothing", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After set raises CycleError, the cell's old content, its dependencies and every stored value must be exactly as before.",
         "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("A1", 1)
sheet.set("B1", "=A1 + 1")
sheet.set("C1", "=B1 + 1")
assert raises("CycleError", sheet.set, "A1", "=C1 + D1"), "set must reject a cycle"
assert (sheet.get("A1"), sheet.get("B1"), sheet.get("C1")) == (1, 2, 3)
sheet.set("D1", 50)
assert sheet.get("A1") == 1, "the rejected formula left a dependency on D1 behind"
sheet.set("A1", 10)
assert (sheet.get("B1"), sheet.get("C1")) == (11, 12)
"""},
        {"name": "Part 2: get does not evaluate", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "get on the end of a long chain was much slower than on a single cell; recompute in set and let get return the stored value.",
         "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("A0", 1)
for i in range(1, 200):
    sheet.set(f"A{i}", f"=A{i - 1} + 1")
sheet.set("B0", 1)
def timed(name):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(3000):
            sheet.get(name)
        best = min(best, time.perf_counter() - start)
    return best
assert sheet.get("A199") == 200
ratio = timed("A199") / max(timed("B0"), 1e-9)
assert ratio < 15, f"get on a 200-deep chain took {ratio:.0f}x as long as on a plain cell"
"""},
        {"name": "Part 2: random sheets with cycles match a reference", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "In a random sheet, set accepted or rejected a different formula than the rules give, or a value differed afterwards.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    sheet, oracle = {fn}(), Oracle()
    cells = [f"D{i}" for i in range(10)]
    for cell in rng.sample(cells, len(cells)):
        value = rng.randint(-9, 9) if rng.random() < 0.3 else random_formula(rng, cells)
        rejected = raises("CycleError", sheet.set, cell, value)
        assert rejected == (oracle.set(cell, value) == "cycle"), (seed, cell, value)
        for c in cells:
            assert sheet.get(c) == oracle.get(c), (seed, c)
"""},
        {"name": "Part 3: overwriting drops old dependencies", "part": 3, "behavior": "state.invariant", "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("A1", "=B1 + 1")
sheet.set("A1", 5)
sheet.set("B1", "=A1")
assert sheet.get("B1") == 5, "A1 no longer reads B1, so B1 may read A1"
sheet.set("C1", "=D1")
sheet.set("C1", "=E1")
sheet.set("D1", 100)
assert sheet.get("C1") == 0
sheet.set("D1", "=C1")
assert sheet.get("D1") == 0
"""},
        {"name": "Part 3: a cell reading itself", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A formula that reads its own cell is a cycle; set must raise CycleError and leave the cell as it was.",
         "code": _HELPERS + r"""
sheet = {fn}()
assert raises("CycleError", sheet.set, "H2", "=H2")
assert sheet.get("H2") == 0
sheet.set("H3", 4)
assert raises("CycleError", sheet.set, "H3", "=1 + H3 - 1")
assert sheet.get("H3") == 4
"""},
        {"name": "Part 3: later cells, diamonds, and literals turned formulas", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A set must update every cell that depends on the changed one, directly or not, including cells set before their inputs existed.",
         "code": _HELPERS + r"""
sheet = {fn}()
sheet.set("X1", "=X2 + X3")
sheet.set("Y1", "=X1")
sheet.set("X3", 4)
assert (sheet.get("X1"), sheet.get("Y1")) == (4, 4)
sheet.set("T1", 1)
sheet.set("L1", "=T1 + 1")
sheet.set("R1", "=T1 + 10")
sheet.set("B1", "=L1 + R1")
sheet.set("T1", 5)
assert sheet.get("B1") == 21
sheet.set("Q1", 3)
sheet.set("Q2", "=Q1")
sheet.set("Q1", "=Q3 + 1")
assert (sheet.get("Q1"), sheet.get("Q2")) == (1, 1)
sheet.set("Q3", 4)
assert (sheet.get("Q1"), sheet.get("Q2")) == (5, 5)
"""},
        {"name": "Part 3: random edits match a reference", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "In a random sequence of overwrites, a value or a cycle decision differed from what the rules give.",
         "code": _HELPERS + r"""
for seed in range(80):
    rng = random.Random(seed)
    sheet, oracle = {fn}(), Oracle()
    cells = [f"E{i}" for i in range(8)]
    for _ in range(40):
        cell = rng.choice(cells)
        value = rng.randint(-9, 9) if rng.random() < 0.3 else random_formula(rng, cells)
        rejected = raises("CycleError", sheet.set, cell, value)
        assert rejected == (oracle.set(cell, value) == "cycle"), (seed, cell, value)
        for c in cells:
            assert sheet.get(c) == oracle.get(c), (seed, c)
"""},
    ],
    "solution": r'''import re

_NAME = re.compile(r"[A-Z]+[0-9]+")
_TERM = r"(?:[A-Z]+[0-9]+|[0-9]{1,18})"
_FORMULA = re.compile(rf"=\s*{_TERM}(?:\s*[+-]\s*{_TERM})*\s*")
_TOKEN = re.compile(r"[A-Z]+[0-9]+|[0-9]+|[+-]")


class FormulaError(ValueError):
    pass


class CycleError(ValueError):
    pass


class Spreadsheet:
    def __init__(self):
        self._terms = {}    # name -> list of (sign, cell name or int); only formula cells
        self._literal = {}  # name -> int; only literal cells
        self._reads = {}    # name -> set of cells its formula reads
        self._readers = {}  # name -> set of cells whose formula reads it
        self._values = {}

    def _check_name(self, name):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise FormulaError(f"bad cell name: {name!r}")

    def _parse(self, value):
        if type(value) is int:
            return None
        if not isinstance(value, str) or not _FORMULA.fullmatch(value):
            raise FormulaError(f"bad value: {value!r}")
        terms, sign = [], 1
        for token in _TOKEN.findall(value[1:]):
            if token in "+-":
                sign = 1 if token == "+" else -1
            else:
                terms.append((sign, token if token[0].isalpha() else int(token)))
        return terms

    def _reaches(self, start, goal):
        stack, seen = [start], set()
        while stack:
            cell = stack.pop()
            if cell == goal:
                return True
            if cell not in seen:
                seen.add(cell)
                stack.extend(self._reads.get(cell, ()))
        return False

    def set(self, name, value):
        self._check_name(name)
        terms = self._parse(value)
        reads = {t for _, t in terms if isinstance(t, str)} if terms is not None else set()
        if any(self._reaches(cell, name) for cell in reads):
            raise CycleError(f"{name} would depend on itself")
        for cell in self._reads.pop(name, ()):
            self._readers[cell].discard(name)
        for cell in reads:
            self._readers.setdefault(cell, set()).add(name)
        self._terms.pop(name, None)
        self._literal.pop(name, None)
        if terms is None:
            self._literal[name] = value
        else:
            self._terms[name] = terms
            self._reads[name] = reads
        self._recompute(name)

    def get(self, name):
        self._check_name(name)
        return self._values.get(name, 0)

    def _recompute(self, start):
        # Every cell the change reaches, evaluated once each in dependency order.
        order, seen = [], set()

        def visit(cell):
            if cell in seen:
                return
            seen.add(cell)
            for reader in self._readers.get(cell, ()):
                visit(reader)
            order.append(cell)

        visit(start)
        for cell in reversed(order):
            if cell in self._literal:
                self._values[cell] = self._literal[cell]
            elif cell in self._terms:
                self._values[cell] = sum(
                    sign * (self._values.get(t, 0) if isinstance(t, str) else t)
                    for sign, t in self._terms[cell]
                )
''',
    "interview_questions": interview(
        concept=[
            "How do you validate a cell name and a formula so that exactly the stated shapes are accepted?",
            "How does get evaluate a formula whose terms are themselves formula cells?",
        ],
        deep_dive=[
            "How does a recursive get notice a cycle, and why is a set of visited cells not enough on its own?",
        ],
        tradeoffs=[
            "Compare evaluating on get with recomputing on set: which reads and writes does each make cheap?",
            "Why must set check for a cycle before changing any state, and how do you check it?",
            "In a diamond, how do you make sure the bottom cell is recomputed once, after both of its inputs?",
        ],
    ),
}
