Two points are worth confirming before coding. First, what "repair" means: Part 2 below finds the single line whose flip makes the *whole* run terminate. A narrower reading some interviewers use instead is a purely local patch — the moment a `jump` is about to land on an already-visited line, treat that one `jump` as a `next` for just that step and carry on, without searching for a global fix. The two are not the same task: the local rule only ever patches a revisit caused by a `jump` landing on a visited line, so it has nothing to say when the revisit is instead reached by an ordinary `pc += 1` falling onto a line a much earlier jump already visited, which can happen once jumps move both forward and backward through the file. Second, that landing outside `0 .. len(program)` is always invalid, never a third way of finishing: a jump that overshoots the end by more than one line is not normal termination, even though it, too, leaves no next line to run.

### Part 1

A single pass suffices: keep a set of visited line indices; the instant `pc` would repeat one already in it, stop and report the loop at that line; the instant `pc` equals `n`, stop and report the accumulator; any other `pc` outside `0 .. n` is invalid. Because a program has only `n` distinct lines, the visited set can never grow past `n` entries before one of these three outcomes is forced, so the loop always finishes within `n + 1` steps, however the instructions are arranged.

```python
class ProgramError(Exception):
    """Raised when the program counter would move outside 0 .. len(program)."""


def run_program(program: list[tuple[str, int]]) -> tuple[bool, int, int | None]:
    n = len(program)
    pc, acc, visited = 0, 0, set()
    while True:
        if pc == n:
            return True, acc, None
        if pc < 0 or pc > n:
            raise ProgramError(f"line {pc} is out of range for a {n}-line program")
        if pc in visited:
            return False, acc, pc
        visited.add(pc)
        op, value = program[pc]
        if op == "plus":
            acc += value
            pc += 1
        elif op == "next":
            pc += 1
        elif op == "jump":
            pc += value
        else:
            raise ValueError(f"unknown op {op!r}")
```

### Part 2

The immediate approach tries every candidate: for each `jump`/`next` line, flip it and rerun the whole program from the top with `run_program`, stopping at the first flip that terminates. With up to `n` such candidates and an `O(n)` rerun each, this costs `O(n^2)` in the worst case.

```python
FLIPPED = {"jump": "next", "next": "jump"}


def find_fix_brute_force(program: list[tuple[str, int]]) -> tuple[int, int]:
    for line, (op, value) in enumerate(program):
        if op not in FLIPPED:
            continue                                    # NOTE: plus is never the corrupted line
        candidate = list(program)
        candidate[line] = (FLIPPED[op], value)
        try:
            terminated, acc, _ = run_program(candidate)
        except ProgramError:
            continue                                     # this flip sends pc out of range -- not the fix
        if terminated:
            return line, acc
    raise ValueError("no single flip terminates this program")
```

A single pass finds the fix in linear time instead. Run the given, corrupted program once and record the path of distinct lines $p_0 = 0, p_1, \dots, p_{k-1}$ it visits before it is about to repeat one of them — exactly Part 1's simulation, which costs $O(n)$ since no line appears in it twice. The corrupted line must be one of $p_0, \dots, p_{k-1}$: a line the corrupted run never reaches behaves identically whether or not it is later flipped, so flipping it could never turn a looping run into a terminating one.

Separately, define the *successor* of every line exactly as the program is given (corrupted, unflipped): `succ(i) = i + 1` for `plus`/`next`, `i + value` for `jump`, undefined when it falls outside `0 .. n`. This makes a graph in which every line has at most one outgoing edge and line `n` (termination) has none, so from any starting line, repeatedly following `succ` has only two possible fates: it reaches `n`, or it re-enters some line already seen and cycles forever — nothing else is possible once every node has at most one way out. Compute the set $R$ of lines that reach `n` this way with one traversal from `n` over the *reversed* edges (line `j`'s predecessors are the lines whose `succ` is `j`): a line is in $R$ exactly when it is reachable from `n` backwards, because following its forward edge, then the forward edge of the line after it, and so on, is precisely how that backward walk got there. This costs $O(n)$: `n + 1` nodes, at most `n` edges.

Now walk $p_0, \dots, p_{k-1}$ in order. For each `jump`/`next` line $p_t$, compute the *other* successor it would have if only this one line were flipped — $p_t + 1$ for a `jump` flipped to `next`, or $p_t + \text{value}$ for a `next` flipped to `jump` — and check whether that lands in $R$. The first $p_t$ for which it does is the corrupted line, and no other line needs to be tried, for two reasons taken together:

- If the alternative successor's chain, followed through the *unflipped* graph, ever passed back through any of $p_0, \dots, p_t$, it would from there on retrace the tail of the original run and close the very same loop, so it could never reach $n$ — it could never land in $R$ in the first place. So landing in $R$ already guarantees that chain avoids the whole prefix, which makes it exactly what the flipped program does next, all the way to `n`, with no line repeated anywhere: a genuine fix.
- The line that truly is the fix cannot, once flipped, revisit $p_0, \dots, p_t$ either — doing so would itself be a second visit, still a loop, not the termination the problem guarantees this flip produces. So its continuation is identical whether or not that one line is flipped, and it does reach `n`, which puts its alternative successor in $R$.

Either direction alone would leave room for $R$ to answer the wrong question about the *true* fix's own successor; together they pin down that membership in $R$ is exactly the test that finds it, without rerunning the program for every candidate.

```python
def find_fix(program: list[tuple[str, int]]) -> tuple[int, int]:
    n = len(program)

    def succ(i: int) -> int:
        op, value = program[i]
        return i + value if op == "jump" else i + 1

    predecessors = {j: [] for j in range(n + 1)}
    for i in range(n):
        s = succ(i)
        if 0 <= s <= n:
            predecessors[s].append(i)

    reaches_end = [False] * (n + 1)
    reaches_end[n] = True
    stack = [n]
    while stack:
        j = stack.pop()
        for i in predecessors[j]:
            if not reaches_end[i]:
                reaches_end[i] = True
                stack.append(i)

    pc, acc, seen = 0, 0, set()
    while pc not in seen and 0 <= pc < n:
        seen.add(pc)
        op, value = program[pc]
        if op in FLIPPED:
            alternative = pc + 1 if op == "jump" else pc + value
            if 0 <= alternative <= n and reaches_end[alternative]:
                return pc, acc + _run_from(program, alternative)
        if op == "plus":
            acc += value
        pc = succ(pc)
    raise ValueError("no single flip terminates this program")


def _run_from(program: list[tuple[str, int]], start: int) -> int:
    """Sums the plus values from start to termination, following the UNFLIPPED program -- valid
    because membership in reaches_end already guarantees this chain runs to n without a repeat."""
    n, acc, pc = len(program), 0, start
    while pc != n:
        op, value = program[pc]
        if op == "plus":
            acc += value
        pc = pc + value if op == "jump" else pc + 1
    return acc
```

Building `predecessors` and `reaches_end` is $O(n)$, the walk over $p_0, \dots, p_{k-1}$ is $O(n)$, and `_run_from` is called once, on the winning candidate only, for another $O(n)$ — $O(n)$ total, with no line ever simulated twice from the same starting point.

### Part 3

Comments are stripped with a single `split("#", 1)` before anything else is checked, so a comment can sit on its own line or trail a real instruction. `str.split()` with no separator already collapses any run of whitespace and drops leading and trailing whitespace, so extra spacing between the op and the value needs no special case.

```python
import re

_SIGNED_INT = re.compile(r"^[+-]\d+$")
_OPS = {"plus", "next", "jump"}


class ParseError(ValueError):
    def __init__(self, line_number: int, raw_line: str):
        super().__init__(f"line {line_number}: cannot parse {raw_line.strip()!r} as '<op> <signed integer>'")
        self.line_number = line_number


def parse_program(text: str) -> list[tuple[str, int]]:
    program = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        code = raw_line.split("#", 1)[0].strip()   # NOTE: strip the comment BEFORE checking for blank
        if not code:
            continue
        tokens = code.split()
        if len(tokens) != 2 or tokens[0] not in _OPS or not _SIGNED_INT.match(tokens[1]):
            raise ParseError(line_number, raw_line)
        program.append((tokens[0], int(tokens[1])))
    return program
```

`text.splitlines()` runs once, over the whole input, and every later step per line is `O(1)` beyond the length of that line, so parsing is `O(len(text))`.

### Follow-ups

- **Parallelising the brute force.** Each candidate flip in `find_fix_brute_force` is independent of every other, so submitting one task per `jump`/`next` line to a thread or process pool parallelises the `O(n^2)` approach with no shared state to coordinate — worth doing only when `n` is large enough that `O(n^2)` is still the intended approach at all. `find_fix`'s single linear pass has no per-candidate work left to hand out this way.
- **More than one corrupted line.** With two corrupted lines instead of one, `find_fix`'s argument breaks: fixing the first line $p_t$ found this way no longer guarantees termination, because the run from its alternative successor can still hit the *other* corrupted line and loop again. Finding all minimal repairs then means re-simulating after each accepted fix rather than reading everything off one reverse traversal computed up front.
- **Real hardware's version of this.** Firmware guards against exactly this class of bug with a watchdog timer: free-running hardware that resets the chip unless software explicitly "pets" it on a schedule. A boot routine that hangs or loops never gets back around to petting it, the timer reaches zero, and the chip restarts — a coarser, hardware-level relative of Part 1's visited-set loop detector, with no notion of which line was at fault.
- **Labels instead of relative offsets.** A real assembler lets a jump target be a symbolic label rather than a raw line offset. Extending `parse_program` to accept `jump some_label` needs two passes: one to record every label's line number, a second to resolve each label reference to the relative offset `target_line - current_line` once every label's position is known.
- **Confirming uniqueness is worth checking, not assuming.** Both `find_fix` implementations return the first candidate that works and never check whether a second one also would; asked to also detect zero or more than one working flip, `find_fix_brute_force` would need to keep scanning after its first hit instead of returning immediately, and `find_fix` would need to keep walking past $p_t$ as well, since a later line on the path could turn out to be a second, equally valid fix.

```python
import random
import time

# --- Part 1: the worked example ---
demo = [
    ("next", 0), ("plus", 6), ("next", 5), ("plus", 100), ("jump", 2),
    ("plus", 50), ("jump", -3), ("plus", 7), ("next", 0),
]
assert run_program(demo) == (False, 106, 3)

# --- Part 2: the same program, fixed ---
assert find_fix_brute_force(demo) == (2, 13)
assert find_fix(demo) == (2, 13)
fixed_demo = list(demo)
fixed_demo[2] = ("jump", 5)
assert run_program(fixed_demo) == (True, 13, None)

# --- Part 3: the worked example, and the malformed-line examples ---
text = """\
# module 7 boot sequence
next +0

plus +4      # initial credit
jump +2
plus +999    # skipped trap
plus +3
"""
parsed = parse_program(text)
assert parsed == [("next", 0), ("plus", 4), ("jump", 2), ("plus", 999), ("plus", 3)]
assert run_program(parsed) == (True, 7, None)

for bad_line, why in [("plus 4", "no sign"), ("boot +1", "unknown op"),
                       ("jump +1 +2", "three tokens"), ("jump", "missing value")]:
    try:
        parse_program(f"next +0\n{bad_line}\nplus +1\n")
        raise AssertionError(f"expected ParseError for {bad_line!r} ({why})")
    except ParseError as e:
        assert e.line_number == 2, (bad_line, e.line_number)
assert parse_program("# only comments\n\n   \n# another\n") == []

# --- edge cases ---
assert run_program([]) == (True, 0, None)                         # empty program
assert run_program([("jump", 0)]) == (False, 0, 0)                # jump 0: immediate self-loop
try:
    run_program([("plus", 1), ("jump", 5)])                       # overshoots the end by 3, not exactly onto it
    raise AssertionError("expected ProgramError")
except ProgramError:
    pass
try:
    run_program([("jump", -1)])                                   # negative pc
    raise AssertionError("expected ProgramError")
except ProgramError:
    pass
assert run_program([("jump", 2), ("plus", 999), ("next", 0)]) == (True, 0, None)  # jump over a mid-range trap

print("worked examples and edge cases passed")


# --- an independent brute force, written from the statement, with a defensive step cap ---
def spec_run(program, step_cap):
    n = len(program)
    pc, acc, seen = 0, 0, []
    for _ in range(step_cap + 1):
        if pc == n:
            return True, acc, None
        if pc < 0 or pc > n:
            raise ProgramError(f"out of range: {pc}")
        if pc in seen:
            return False, acc, pc
        seen.append(pc)
        op, value = program[pc]
        if op == "plus":
            acc, pc = acc + value, pc + 1
        elif op == "next":
            pc = pc + 1
        else:
            pc = pc + value
    raise AssertionError("exceeded the step cap -- a well-formed run never needs more than n + 1 steps")


def random_terminating_program(rng, n):
    """A random n-line program that terminates normally from line 0, built by laying out a random
    path of distinct lines from 0 to the virtual exit n, then filling every unused line with junk
    that the correct run never reaches."""
    unused = set(range(n))
    unused.discard(0)
    path, program, pc = [0], [None] * n, 0
    while pc != n:
        options = []
        if pc + 1 == n or (pc + 1) in unused:
            options += [pc + 1, pc + 1]                  # weighted up, so next/plus steps are common too
        if unused:
            options.append(rng.choice(list(unused)))
        options.append(n)                                # always allowed to jump straight to the exit
        target = rng.choice(options)
        if target == pc + 1:
            program[pc] = (rng.choice(["next", "plus"]), rng.randint(-50, 50))
        else:
            program[pc] = ("jump", target - pc)
        if target != n:
            unused.discard(target)
            path.append(target)
        pc = target
    path.append(n)
    span = max(3, min(30, n))     # local filler jumps: an excursion off the path tends to loop, not fly out
    for i in range(n):
        if program[i] is None:
            op = rng.choice(["plus", "next", "jump"])
            value = rng.randint(-span, span) if op == "jump" else rng.randint(-50, 50)
            program[i] = (op, value)
    return program, path


def single_fix_instance(rng, n):
    """A random n-line program with exactly one corrupted jump/next line, verified (independently
    of run_program/find_fix) to loop as given and to have a unique terminating flip."""
    for _ in range(20):
        program, path = random_terminating_program(rng, n)
        corruptible = [i for i in path[:-1] if program[i][0] in FLIPPED]
        if not corruptible:
            continue
        line = rng.choice(corruptible)
        op, value = program[line]
        corrupted = list(program)
        corrupted[line] = (FLIPPED[op], value)
        try:
            if spec_run(corrupted, n + 5)[0]:
                continue                                  # coincidence: the flip changed nothing observable
        except ProgramError:
            continue

        fixes = []
        for i, (iop, ival) in enumerate(corrupted):
            if iop not in FLIPPED:
                continue
            trial = list(corrupted)
            trial[i] = (FLIPPED[iop], ival)
            try:
                ok, acc, _ = spec_run(trial, n + 5)
            except ProgramError:
                continue
            if ok:
                fixes.append((i, acc))
        if len(fixes) == 1:
            return corrupted, fixes[0], op
    return None


trials, by_direction = 0, {"jump": 0, "next": 0}
for seed in range(800):
    rng = random.Random(seed)
    result = single_fix_instance(rng, rng.randint(4, 30))
    if result is None:
        continue
    corrupted, (expected_line, expected_acc), original_op = result
    by_direction[original_op] += 1

    assert run_program(corrupted) == spec_run(corrupted, len(corrupted) + 5)
    assert find_fix_brute_force(corrupted) == (expected_line, expected_acc), seed
    assert find_fix(corrupted) == (expected_line, expected_acc), seed
    trials += 1

assert trials > 600 and by_direction["jump"] > 50 and by_direction["next"] > 50, (trials, by_direction)
print(f"brute-force cross-check passed on {trials} random single-fix programs "
      f"({by_direction['jump']} corrupted jumps, {by_direction['next']} corrupted nexts)")

# --- find_fix is O(n): a deliberately adversarial program where every wrong flip loops only after
# touching every line, so find_fix_brute_force pays for it and find_fix does not ---
n = 3000
worst_case = [("jump", 1)] * (n - 1) + [("jump", -(n - 1))]      # a single n-line cycle if never fixed
assert run_program(worst_case) == (False, 0, 0)
t0 = time.perf_counter()
slow = find_fix_brute_force(worst_case)
brute_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
fast = find_fix(worst_case)
fast_seconds = time.perf_counter() - t0
assert slow == fast == (n - 1, 0)
assert brute_seconds > 20 * fast_seconds, (brute_seconds, fast_seconds)   # measured: roughly 500x on n=3,000

big = [("jump", 1)] * 199_999 + [("jump", -199_999)]
t0 = time.perf_counter()
assert find_fix(big) == (199_999, 0)
big_seconds = time.perf_counter() - t0
assert big_seconds < 5.0        # measured: well under a second; generous for a loaded 2-core machine

print(f"find_fix stayed linear: {brute_seconds:.3f}s (brute) vs {fast_seconds:.3f}s (fast) at n={n}, "
      f"{big_seconds:.3f}s at n=200,000")

print("all checks passed")
```
