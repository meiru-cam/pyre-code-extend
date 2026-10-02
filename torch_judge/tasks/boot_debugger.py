"""Run a tiny boot program, find the one corrupted line that makes it loop, then parse it from text."""

from ._interview import interview

# A brute-force model and generators for programs with exactly one repairing flip.
_HELPERS = r"""
import random, time

class Bad(Exception):
    pass

def model_run(program):
    n, pc, acc, seen = len(program), 0, 0, set()
    while True:
        if pc == n:
            return True, acc, None
        if pc < 0 or pc > n:
            raise Bad()
        if pc in seen:
            return False, acc, pc
        seen.add(pc)
        op, v = program[pc]
        acc += v if op == "add" else 0
        pc += v if op == "goto" else 1

SWAP = {"skip": "goto", "goto": "skip"}

def fixes(program):
    out = []
    for i, (op, v) in enumerate(program):
        if op in SWAP:
            p = list(program)
            p[i] = (SWAP[op], v)
            try:
                done, acc, _ = model_run(p)
            except Bad:
                continue
            if done:
                out.append((i, acc))
    return out

def random_program(rng, n):
    prog = []
    for i in range(n):
        op = rng.choice(["add", "skip", "goto", "goto"])
        v = rng.randint(-i, n - i) if op != "add" else rng.randint(-9, 9)
        if op == "skip" and rng.random() < 0.5:
            v = rng.randint(-20, 20)
        prog.append((op, v))
    return prog

def one_fix_programs(rng, count, lo, hi):
    out = []
    while len(out) < count:
        prog = random_program(rng, rng.randint(lo, hi))
        try:
            looped = not model_run(prog)[0]
        except Bad:
            continue
        found = fixes(prog) if looped else []
        if len(found) == 1:
            out.append((prog, found[0]))
    return out

def raises(name, call):
    try:
        call()
    except Exception as e:
        assert type(e).__name__ == name, f"expected {name}, got {type(e).__name__}: {e}"
        return e
    raise AssertionError(f"expected {name}, nothing was raised")
"""

TASK = {
    "title": "Boot Program Debugger",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "BootDebugger",
    "description_en": r"""Build `BootDebugger`, which runs a tiny boot program, finds the one corrupted line that traps it in a loop, and reads programs from text.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `BootDebugger` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A program is a list of `(op, value)` pairs, one per line, numbered from `0`. `op` is `"add"`, `"skip"` or `"goto"`, and `value` is an `int`.
- Running starts with `pc = 0` and `acc = 0`, where `pc` is the line about to run:
- `add v`: `acc += v`, then `pc += 1`
- `skip v`: `pc += 1`; the value is ignored
- `goto v`: `pc += v`; `v` may be negative or `0`
- With `n` lines, the run ends normally when `pc` becomes exactly `n`. If `pc` becomes anything else outside `0..n`, the program is out of range.
- If `pc` is about to run a line it already ran in this run, the program is in a loop.
- Define `class ProgramError(Exception)` yourself. Tests recognise it by its class name.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the simulation is short, so the interview is about the second step: the obvious repair tries every line and reruns the program, and the question is whether you see the linear version. Each later part adds one requirement.

**Where it is used:** emulators and bytecode interpreters detect loops and bad jumps the same way, and control-flow graphs with one successor per node are how compilers and static analysers reason about reachability.

Adapted from the boot loader loop question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class, with renamed operations and methods (`run`, `find_fix`, `parse`). Values use ASCII digits only. The linear repair is checked by speed on large programs.""",
    "parts": [
        {
            "title": "Run and detect the loop",
            "description_en": r"""**Signature:** `BootDebugger().run(program) -> tuple[bool, int, int | None]`

- On a normal end, return `(True, acc, None)`.
- On a loop, return `(False, acc, line)`: `line` is the line about to run a second time, and `acc` is its value before that line runs again.
- If a step sends `pc` out of range, raise `ProgramError`.
- Run in `O(n)` time: a program of `100000` lines must finish well under a second.

**Example:**
- lines `0` to `7`: `skip +2`, `add +5`, `goto +4`, `add -1`, `goto -3`, `add +20`, `goto -3`, `add +2`
- the run visits lines `0`, `1`, `2`, `6`, `3`, `4`, and line `4` jumps back to line `1`
- the result is `(False, 4, 1)`: `5 - 1`""",
        },
        {
            "title": "Find the corrupted line",
            "description_en": r"""Keep Part 1 and add a repair.

**Signature:** `find_fix(program) -> tuple[int, int]`

- Exactly one line was corrupted: a `goto` became a `skip`, or a `skip` became a `goto`, with its value unchanged. `add` lines are never corrupted.
- So `run` loops on the given program, and switching exactly one line between `skip` and `goto` makes it end normally. No other single switch does.
- Return `(line, acc)`: that line, and `acc` at the normal end of the repaired program.
- Run in `O(n)` time. Trying every line and rerunning the program costs `O(n²)`, which is too slow for the tests: a program of `40000` lines must finish in about a second.

**Example**, the program from Part 1:
- switching line `6` to `skip -3` gives the path `0`, `1`, `2`, `6`, `7`, then `pc = 8`, the end
- switching line `4` instead runs `add +20` and comes back to line `6`, a loop, so it is not the repair
- the result is `(6, 7)`: `5 + 2`""",
        },
        {
            "title": "Parse the program text",
            "description_en": r"""Keep Parts 1–2 and read programs from text.

**Signature:** `parse(text) -> list[tuple[str, int]]`

- Each line of `text` holds at most one instruction. Everything from a `#` to the end of its line is a comment.
- A line that is empty or only whitespace after removing the comment adds nothing.
- Every other line must split on whitespace into exactly two tokens: an op from the rules, then a sign `+` or `-` followed by one or more digits `0` to `9`. `+0` and `-0` are both zero; `12`, `+1e3`, `--4` and `+ 7` are not valid.
- Return the instructions in order, numbered from `0` no matter how many lines were skipped.
- Define `class ParseError(ValueError)` with an attribute `line_number`. On a malformed line, raise it with the 1-based number of that line in `text`, counting blank and comment lines.

**Example:** these eight lines parse to `[("goto", 3), ("add", 1), ("goto", 3), ("add", 2), ("goto", -3)]`, and `run` on it gives `(True, 3, None)`:
- line 1: `goto +3   # over the next two`
- line 2: `add +1`
- line 3: blank
- line 4: `goto +3`
- line 5: `# back edge below`
- line 6: `add +2`
- line 7: only spaces
- line 8: `goto -3`

In a text whose first three lines are a comment, `add +1` and a blank line, a fourth line `goto 4` raises `ParseError` with `line_number == 4`; so do `nop +1`, `add +2 +3` and a bare `add`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the only thing that distinguishes a loop from progress, given that acc never changes control flow? What should you check first when pc changes: the end, out of range, or already visited, and does the order matter?"},
        {"level": 2, "kind": "analysis", "content": "Keep a set of visited lines. In a while loop: if pc == n return (True, acc, None); if pc < 0 or pc > n raise ProgramError; if pc in seen return (False, acc, pc); add pc, then apply the op. Every line runs at most once, so it is O(n)."},
    ],
    "model_connections": [
        "Agent loops detect repeated states the same way: a set of seen states stops a tool-calling loop that has started cycling.",
        "Graph compilers such as XLA and TorchInductor reason about reachability on control-flow graphs, the structure used by the linear repair.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A visited set detects a loop on the first repeat, in O(n) time and memory.",
            "Reverse reachability from the end finds the repair in O(n) instead of rerunning the program for every line.",
            "Strict parsing with line numbers turns a bad program into a precise error.",
        ],
        "cons": [
            "The visited set costs O(n) memory; tortoise-and-hare would use O(1) but needs more steps.",
            "The linear repair relies on there being exactly one corrupted line.",
            "A hand-written parser must be kept in sync with every new op or syntax.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
program = [("skip", 2), ("add", 5), ("goto", 4), ("add", -1), ("goto", -3), ("add", 20), ("goto", -3), ("add", 2)]
assert {fn}().run(program) == (False, 4, 1)
"""},
        {"name": "Part 1: ends, bounds and self-loops", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "An empty program ends at once with acc 0; landing exactly on n ends normally; any other pc outside 0..n raises ProgramError; goto 0 is a loop on its own line; the loop's acc is the value before the repeat.",
         "code": _HELPERS + r"""
d = {fn}()
assert d.run([]) == (True, 0, None)
assert d.run([("add", -4), ("goto", 1)]) == (True, -4, None)
assert d.run([("add", 2), ("goto", 0)]) == (False, 2, 1)
assert d.run([("add", 5), ("add", 1), ("goto", -2)]) == (False, 6, 0)
raises("ProgramError", lambda: d.run([("goto", 2)]))
raises("ProgramError", lambda: d.run([("add", 1), ("goto", -2)]))
assert d.run([("skip", -7), ("skip", 99)]) == (True, 0, None), "skip ignores its value"
"""},
        {"name": "Part 1: random programs and a long one", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "On random programs the result or the ProgramError differed from a plain simulation, or a 100000-line program took more than about a second.",
         "code": _HELPERS + r"""
d = {fn}()
rng = random.Random(3)
for trial in range(400):
    prog = random_program(rng, rng.randint(1, 30))
    try:
        want = model_run(prog)
    except Bad:
        raises("ProgramError", lambda: d.run(prog))
        continue
    assert d.run(prog) == want, (trial, prog)
n = 100000
prog = [("add", 1)] * (n - 1) + [("goto", -(n - 1))]
start = time.perf_counter()
assert d.run(prog) == (False, n - 1, 0)
assert time.perf_counter() - start < 2.0, "run must be linear in the program length"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
program = [("skip", 2), ("add", 5), ("goto", 4), ("add", -1), ("goto", -3), ("add", 20), ("goto", -3), ("add", 2)]
assert {fn}().find_fix(program) == (6, 7)
"""},
        {"name": "Part 2: random corrupted programs", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random program with exactly one repairing switch, the line or the final acc differed from trying every switch; a switch that sends pc out of range is never the repair.",
         "code": _HELPERS + r"""
d = {fn}()
for prog, want in one_fix_programs(random.Random(11), 300, 2, 25):
    assert d.find_fix(prog) == want, (prog, want)
assert d.find_fix([("skip", 9), ("add", 1), ("goto", -2)]) == (2, 1), "switching line 0 sends pc out of range"
assert d.find_fix([("add", 4), ("skip", -1), ("goto", -2)]) == (2, 4)
"""},
        {"name": "Part 2: linear time on a long program", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "find_fix took more than about a second on a 40000-line program whose repair is in the middle: rerunning the program for every candidate line is O(n^2).",
         "code": _HELPERS + r"""
d = {fn}()
n = 40000
a = n // 2
prog = [("skip", 0)] * (n - 2) + [("goto", -(n - 2)), ("goto", 0)]
prog[a] = ("skip", n - a)
prog[a - 1] = ("add", 7)
start = time.perf_counter()
assert d.find_fix(prog) == (a, 7)
assert time.perf_counter() - start < 2.0, "find_fix must be linear, not one rerun per line"
chain = [("add", 7)] + [("goto", 1)] * (n - 2) + [("goto", -(n - 1))]
start = time.perf_counter()
assert d.find_fix(chain) == (n - 1, 7)
assert time.perf_counter() - start < 2.0, "find_fix must be linear even when every wrong switch runs far before looping"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "protocol.validation", "code": _HELPERS + r"""
text = "goto +3   # over the next two\nadd +1\n\ngoto +3\n# back edge below\nadd +2\n   \ngoto -3"
d = {fn}()
program = d.parse(text)
assert program == [("goto", 3), ("add", 1), ("goto", 3), ("add", 2), ("goto", -3)]
assert d.run(program) == (True, 3, None)
for head, bad, line in [("# boot\nadd +1\n\n", "goto 4", 4), ("# boot\nadd +1\n\n", "nop +1", 4),
                        ("add +1\n", "add +2 +3", 2), ("\n\n\n\n# x\n", "add", 6)]:
    e = raises("ParseError", lambda: d.parse(head + bad + "\nadd +1"))
    assert e.line_number == line, (bad, e.line_number, line)
"""},
        {"name": "Part 3: strict tokens and line numbers", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "A value needs exactly one sign and only ASCII digits; ParseError subclasses ValueError and carries the 1-based line number counting blank and comment lines; whitespace-only and comment-only lines add nothing; text after # is ignored.",
         "code": _HELPERS + r"""
d = {fn}()
assert d.parse("") == [] and d.parse("\n  \n# only\n\t#x\n") == []
assert d.parse("  goto   -0\t\nskip +0#c\nadd +007 # x # y\r\n") == [("goto", 0), ("skip", 0), ("add", 7)]
for bad in ["add 12", "add +1e3", "add --4", "add + 7", "add +", "add -x", "ADD +1", "add +٣", "add +5 # ok\nadd 5", "+5 add"]:
    text = "# header\n\nadd +1\n" + bad
    e = raises("ParseError", lambda: d.parse(text))
    assert isinstance(e, ValueError), "ParseError must subclass ValueError"
    want = 4 + bad.count("\n")
    assert e.line_number == want, (bad, e.line_number, want)
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import re
from collections import defaultdict

_VALUE = re.compile(r"[+-][0-9]+")
_SWAP = {"skip": "goto", "goto": "skip"}


class ProgramError(Exception):
    """pc left 0..n without landing exactly on n."""


class ParseError(ValueError):
    def __init__(self, line_number, message):
        super().__init__(f"line {line_number}: {message}")
        self.line_number = line_number  # 1-based, counting blank and comment lines


def _next(program, i):
    op, value = program[i]
    return i + value if op == "goto" else i + 1


class BootDebugger:
    def run(self, program):
        n, pc, acc, seen = len(program), 0, 0, set()
        while True:
            if pc == n:
                return True, acc, None
            if pc < 0 or pc > n:
                raise ProgramError(f"pc {pc} is outside 0..{n}")
            if pc in seen:
                return False, acc, pc
            seen.add(pc)
            op, value = program[pc]
            if op == "add":
                acc += value
            pc = _next(program, pc)

    def find_fix(self, program):
        n = len(program)
        # Every line has one successor, so a line ends the run exactly when the end is reachable from it.
        into = defaultdict(list)
        for i in range(n):
            j = _next(program, i)
            if 0 <= j <= n:
                into[j].append(i)
        ends = {n}
        stack = [n]
        while stack:
            for i in into[stack.pop()]:
                if i not in ends:
                    ends.add(i)
                    stack.append(i)
        # Walk the looping path; the first switch whose other successor reaches the end is the repair.
        pc, seen = 0, set()
        while pc not in seen and 0 <= pc < n:
            seen.add(pc)
            op, value = program[pc]
            if op in _SWAP:
                other = pc + 1 if op == "goto" else pc + value
                if other in ends:
                    fixed = list(program)
                    fixed[pc] = (_SWAP[op], value)
                    return pc, self.run(fixed)[1]
            pc = _next(program, pc)
        raise ValueError("no single switch repairs this program")

    def parse(self, text):
        program = []
        for number, line in enumerate(text.split("\n"), start=1):
            tokens = line.split("#", 1)[0].split()
            if not tokens:
                continue
            if len(tokens) != 2 or tokens[0] not in ("add", "skip", "goto") or not _VALUE.fullmatch(tokens[1]):
                raise ParseError(number, f"expected '<add|skip|goto> <+n|-n>', got {line.strip()!r}")
            program.append((tokens[0], int(tokens[1])))
        return program
''',
    "interview_questions": interview(
        concept=[
            "Why is a set of visited lines enough to detect a loop here, and what makes the state just pc?",
            "Why must landing past the end be an error rather than another way to finish?",
        ],
        deep_dive=[
            "Why must an out-of-range pc be caught before indexing program[pc], and what does Python do with a negative index?",
        ],
        tradeoffs=[
            "Why is trying every switch and rerunning O(n^2), and how does reachability from the end make the repair O(n)?",
            "Why can only lines on the looping path be the corrupted one?",
            "How would you report every malformed line at once instead of stopping at the first?",
            "If several lines could be corrupted, how would the repair search change?",
        ],
    ),
}
