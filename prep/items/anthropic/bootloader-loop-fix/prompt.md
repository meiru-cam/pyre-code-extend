A *boot program* is a sequence of instructions, one per line, each written `<op> <value>` where `op` is one of `plus`, `next`, `jump` and `value` is a signed integer. Execution keeps a *program counter* `pc` — the zero-based index of the line about to run — and a global *accumulator* `acc`; both start at `pc = 0` and `acc = 0`. Each instruction updates them, in this order, and execution then continues at the new `pc`:

- `plus v` — adds `v` to `acc`, then moves to the next line (`pc += 1`).
- `next v` — leaves `acc` unchanged and moves to the next line (`pc += 1`); `v` plays no role in execution at all and is never inspected.
- `jump v` — leaves `acc` unchanged and moves by `v` lines relative to the current one (`pc += v`); `v` may be positive, negative or zero.

A program of `n` lines is indexed `0` to `n - 1`. Execution *terminates normally* the instant `pc` becomes exactly `n` — one line past the end — at which point the current `acc` is the answer. If instead an instruction sets `pc` to any other value outside `0 .. n` (negative, or more than one past the end), the program is *out of range*: there is no line to run next and no rule for what happens, and a simulator must always treat this as distinct from normal termination, never as though the program had finished. Finally, if `pc` is about to take a value it has already held earlier in the very same run — about to execute one of its own lines for the second time — the program is in an *infinite loop*: nothing about its future can differ from the first time it reached that `pc`, so a simulator may stop right there.

### Part 1 — Detect the Loop

Write `run_program`, which executes a boot program from `pc = 0`, `acc = 0`, following the rules above.

```py
def run_program(program: list[tuple[str, int]]) -> tuple[bool, int, int | None]:
    ...
```

`program[i]` is the pair `(op, value)` for line `i`. The return value is `(terminated, acc, loop_line)`:

- on normal termination, `terminated` is `True`, `acc` is the accumulator at that point, and `loop_line` is `None`;
- on an infinite loop, `terminated` is `False`, `loop_line` is the index of the line about to run for the second time, and `acc` is the accumulator exactly as it stood after the previous instruction — before this repeat, not after it;
- if a jump would send `pc` out of range, raise `ProgramError` instead of returning.

```py
class ProgramError(Exception): ...
```

```text
line 0: next +0
line 1: plus +6
line 2: next +5
line 3: plus +100
line 4: jump +2
line 5: plus +50
line 6: jump -3
line 7: plus +7
line 8: next +0

pc=0  next +0    -> ignore the value, pc=1                     # acc=0
pc=1  plus +6    -> acc=0+6=6, pc=2                             # acc=6
pc=2  next +5    -> ignore the value, pc=3                      # acc=6
pc=3  plus +100  -> acc=6+100=106, pc=4                         # acc=106
pc=4  jump +2    -> pc=4+2=6                                    # acc=106
pc=6  jump -3    -> pc=6-3=3                                    # acc=106
pc=3  — already ran two steps ago: about to run a second time, report the loop here

run_program(program) == (False, 106, 3)
```

### Part 2 — Repair the Corrupted Line

In the programs of this part, exactly one `jump` or `next` line has been corrupted: its op was swapped for the other one — a `jump` written as `next`, or a `next` written as `jump` — while its numeric value is exactly what it should be; a `plus` line is never corrupted. Because of this, `run_program` never terminates normally on one of these programs: it always reaches an infinite loop. Restoring exactly one line to its correct op (its value unchanged) makes the whole program terminate normally; no other single flip does. Write `find_fix`, which returns that line and the accumulator at termination.

```py
def find_fix(program: list[tuple[str, int]]) -> tuple[int, int]:
    ...
```

Continuing the program from Part 1, line 2 (`next +5`) is the corrupted line — it should read `jump +5`. Restoring it changes nothing about lines 0 and 1, then skips straight past the trap at lines 3 to 6:

```text
pc=0  next +0          -> pc=1                    # acc=0
pc=1  plus +6          -> acc=6, pc=2              # acc=6
pc=2  jump +5 (fixed)  -> pc=7                     # acc=6
pc=7  plus +7          -> acc=13, pc=8             # acc=13
pc=8  next +0          -> pc=9 = len(program): terminate

find_fix(program) == (2, 13)
```

### Part 3 — Parsing the Program File

So far `program` was already a parsed list. In practice it starts as text, one instruction per line, interspersed with blank lines and comments. A *comment* is introduced by `#`: everything from the `#` to the end of that line is not part of the program, whether the line is nothing but a comment or the `#` trails a real instruction on the same line. A line that is empty, all whitespace, or empty once its comment is stripped, contributes no instruction. Every other line must reduce, once its comment and surrounding whitespace are removed, to exactly two whitespace-separated tokens: one of `plus`, `next`, `jump`, then a signed integer written with an explicit leading `+` or `-` (`+0` and `-0` are both valid and both mean zero; `3`, `3.0` and `++3` are not signed integers and are rejected). A line that does not fit this shape — the wrong number of tokens, an unrecognised op, or a malformed value — is malformed and must be reported with the 1-based number of the line it is on, counting every line of the input text, including blank and comment lines.

Write `parse_program`, turning the full text of a program into the `list[tuple[str, int]]` used by `run_program` and `find_fix` above, in the order the instructions appear, indexed from `0` regardless of how many comment or blank lines preceded them.

```py
class ParseError(ValueError):
    """line_number is 1-based and counts every line of the input text -- not the 0-based index
    the matching instruction gets in the returned program."""
    line_number: int


def parse_program(text: str) -> list[tuple[str, int]]:
    ...
```

```text
# module 7 boot sequence
next +0

plus +4      # initial credit
jump +2
plus +999    # skipped trap
plus +3

parse_program(text) == [
    ("next", 0), ("plus", 4), ("jump", 2), ("plus", 999), ("plus", 3),
]
run_program(parse_program(text)) == (True, 7, None)   # "jump +2" skips the trap at parsed line 3
```

```text
"plus 4"       -> malformed: 4 has no explicit sign
"boot +1"      -> malformed: "boot" is not plus/next/jump
"jump +1 +2"   -> malformed: three tokens, not two
"jump"         -> malformed: missing value
```
