"""Schedule a kernel for a simulated in-order vector machine: pack bundles, vectorize with a scalar tail, then keep the load unit busy every cycle."""

from ._interview import interview

# The machine, adapted from the source's simulator, and helpers that run a learner's program.
_MODEL = r"""
import math, random
VLEN = 8
NUM_SREG = 8
NUM_VREG = 8
LOAD_LATENCY = 4
UNIT_WIDTH = {"scalar": 2, "vector": 1, "load": 1, "store": 1}

# Every opcode belongs to exactly one issue unit.
OP_UNIT = {
    "li": "scalar", "mov": "scalar", "add": "scalar", "addi": "scalar", "mul": "scalar",
    "sload": "load", "vload": "load",
    "sstore": "store", "vstore": "store",
    "vadd": "vector", "vmul": "vector", "vmadd": "vector", "vbcast": "vector", "vrsum": "vector",
}


def _operand_regs(instr):
    '''The (reads, writes) of one instruction, each a list of ("s" | "v", index).'''
    op, *args = instr
    if op == "li": return [], [("s", args[0])]
    if op == "mov": return [("s", args[1])], [("s", args[0])]
    if op in ("add", "mul"): return [("s", args[1]), ("s", args[2])], [("s", args[0])]
    if op == "addi": return [("s", args[1])], [("s", args[0])]
    if op == "sload": return [("s", args[1])], [("s", args[0])]
    if op == "vload": return [("s", args[1])], [("v", args[0])]
    if op == "sstore": return [("s", args[0]), ("s", args[1])], []
    if op == "vstore": return [("s", args[0]), ("v", args[1])], []
    if op in ("vadd", "vmul"): return [("v", args[1]), ("v", args[2])], [("v", args[0])]
    if op == "vmadd": return [("v", args[1]), ("v", args[2]), ("v", args[3])], [("v", args[0])]
    if op == "vbcast": return [("s", args[1])], [("v", args[0])]
    if op == "vrsum": return [("v", args[1])], [("s", args[0])]
    raise ValueError(f"unknown opcode {op!r}")


class VectorMachine:
    '''An in-order vector processor. A program is a list of bundles; a bundle is a list of
    instructions issued in the same cycle, at most UNIT_WIDTH[unit] per unit. See the machine
    description above for the read/write and stall rules.'''

    def __init__(self, mem):
        self.s = [0] * NUM_SREG
        self.v = [[0.0] * VLEN for _ in range(NUM_VREG)]
        self.mem = mem
        self.cycles = 0

    def _eval(self, instr):
        '''The writes this instruction produces, computed from the register file as it stood
        before the enclosing bundle. Returns a list of (file, index, value); a store's effect on
        memory is applied immediately, since nothing here re-reads a location before its own
        prior write to it has retired.'''
        op, *args = instr
        s, v, mem = self.s, self.v, self.mem
        if op == "li": return [("s", args[0], args[1])]
        if op == "mov": return [("s", args[0], s[args[1]])]
        if op == "add": return [("s", args[0], s[args[1]] + s[args[2]])]
        if op == "mul": return [("s", args[0], s[args[1]] * s[args[2]])]
        if op == "addi": return [("s", args[0], s[args[1]] + args[2])]
        if op == "sload":
            addr = s[args[1]]
            if not 0 <= addr < len(mem):
                raise IndexError(f"sload: address {addr} out of range")
            return [("s", args[0], mem[addr])]
        if op == "vload":
            base = s[args[1]]
            if base < 0 or base + VLEN > len(mem):
                raise IndexError(f"vload: range [{base}, {base + VLEN}) out of range")
            return [("v", args[0], mem[base:base + VLEN])]
        if op == "sstore":
            addr = s[args[0]]
            if not 0 <= addr < len(mem):
                raise IndexError(f"sstore: address {addr} out of range")
            mem[addr] = s[args[1]]
            return []
        if op == "vstore":
            base = s[args[0]]
            if base < 0 or base + VLEN > len(mem):
                raise IndexError(f"vstore: range [{base}, {base + VLEN}) out of range")
            mem[base:base + VLEN] = v[args[1]]
            return []
        if op == "vadd": return [("v", args[0], [p + q for p, q in zip(v[args[1]], v[args[2]])])]
        if op == "vmul": return [("v", args[0], [p * q for p, q in zip(v[args[1]], v[args[2]])])]
        if op == "vmadd":
            return [("v", args[0], [p * q + r for p, q, r in zip(v[args[1]], v[args[2]], v[args[3]])])]
        if op == "vbcast": return [("v", args[0], [s[args[1]]] * VLEN)]
        if op == "vrsum": return [("s", args[0], sum(v[args[1]]))]
        raise ValueError(f"unknown opcode {op!r}")

    def run(self, program):
        '''Executes `program` (a list of bundles) and returns its cycle count.'''
        ready = {("s", i): 0 for i in range(NUM_SREG)}
        ready.update({("v", i): 0 for i in range(NUM_VREG)})
        cycle = 0
        for bnum, bundle in enumerate(program):
            used = {"scalar": 0, "vector": 0, "load": 0, "store": 0}
            reads = []
            for instr in bundle:
                unit = OP_UNIT[instr[0]]
                used[unit] += 1
                if used[unit] > UNIT_WIDTH[unit]:
                    raise ValueError(f"bundle {bnum}: more than {UNIT_WIDTH[unit]} {unit} op(s): {bundle}")
                r, _w = _operand_regs(instr)
                reads += r
            issue = max([cycle] + [ready[reg] for reg in reads])   # stall until every source is ready
            pending, seen = [], set()
            for instr in bundle:
                unit = OP_UNIT[instr[0]]
                for file, idx, val in self._eval(instr):
                    if (file, idx) in seen:
                        raise ValueError(f"bundle {bnum}: two writes to {file}{idx}")
                    seen.add((file, idx))
                    latency = LOAD_LATENCY if unit == "load" else 1
                    pending.append((file, idx, val, issue + latency))
            for file, idx, val, ready_at in pending:
                (self.s if file == "s" else self.v)[idx] = val
                ready[(file, idx)] = ready_at
            cycle = issue + 1
        self.cycles = cycle
        return cycle


def run(build, n, x, y, a, b):
    mem = list(x) + list(y) + [float(a), float(b)] + [0.0] * n
    before = list(mem[:2 * n + 2])
    program = build(n)
    cycles = VectorMachine(mem).run(program)
    assert mem[:2 * n + 2] == before, "the program changed x, y, a or b"
    return mem[2 * n + 2:], cycles

def check(build, n, seed, label):
    rng = random.Random(seed)
    x = [rng.uniform(-4, 4) for _ in range(n)]
    y = [rng.uniform(-4, 4) for _ in range(n)]
    a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
    z, cycles = run(build, n, x, y, a, b)
    want = [a * p + b * q for p, q in zip(x, y)]
    bad = [i for i, (g, w) in enumerate(zip(z, want)) if not math.isclose(g, w, rel_tol=1e-12, abs_tol=1e-12)]
    assert len(z) == n and not bad, (label, n, "wrong z at", bad[:5])
    return cycles
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "performance.complexity", "code": _MODEL + r"""
z, _ = run({fn}, 3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 2.0, -1.0)
assert z == [-2.0, -1.0, 0.0], z
cycles = check({fn}, 40, 0, "example")
assert cycles <= 340, f"{cycles} cycles at n = 40; at most 340 allowed"
"""},
    {"name": "Part 1: every small n, packed", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "For some n from 0 to 40, the program wrote a wrong z, changed the inputs, broke a bundle rule, or took more than 8n + 20 cycles.",
     "code": _MODEL + r"""
for n in range(41):
    cycles = check({fn}, n, 100 + n, "small")
    assert cycles <= 8 * n + 20, (n, cycles, 8 * n + 20)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "performance.complexity", "code": _MODEL + r"""
cycles = check({fn}, 256, 1, "example")
assert cycles <= 360, f"{cycles} cycles at n = 256; at most 360 allowed"
"""},
    {"name": "Part 2: vectorized sizes and tails", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "The program wrote a wrong z for an n that is not a multiple of 8, or took more than 1.25n + 40 cycles at n = 128, 256 or 512.",
     "code": _MODEL + r"""
for n in (1, 7, 8, 9, 15, 16, 17, 63, 100, 129, 263):
    check({fn}, n, 200 + n, "tail")
for n in (128, 256, 512):
    cycles = check({fn}, n, 300 + n, "vector")
    assert cycles <= 1.25 * n + 40, (n, cycles, 1.25 * n + 40)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "performance.complexity", "code": _MODEL + r"""
cycles = check({fn}, 1024, 2, "example")
assert cycles <= 280, f"{cycles} cycles at n = 1024; at most 280 allowed"
"""},
    {"name": "Part 3: near the load bound", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "At some n from 512 to 2053 the program took more than 2 * (n // 8) + 8 * (n % 8) + 24 cycles, or wrote a wrong z.",
     "code": _MODEL + r"""
for n in (512, 1024, 1029, 1031, 2048, 2053):
    cycles = check({fn}, n, 400 + n, "bound")
    limit = 2 * (n // 8) + 8 * (n % 8) + 24
    assert cycles <= limit, (n, cycles, limit)
"""},
]

TASK = {
    "title": "Vector Machine Kernel Scheduling",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "build_kernel",
    "description_en": r"""Write `build_kernel(n)`, which returns a program for a simulated in-order vector machine that computes `z[i] = a * x[i] + b * y[i]` for arrays of length `n`. The starter code holds the machine, `VectorMachine`, which is the exact specification, and a correct but slow program. Each part asks for fewer simulated cycles.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `build_kernel` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Memory is a list of floats: `x` at `0..n-1`, `y` at `n..2n-1`, `a` at `2n`, `b` at `2n+1`, and `z` must be written to `2n+2..3n+1`. `x`, `y`, `a` and `b` must stay unchanged.
- A program is a list of bundles, each a list of instruction tuples such as `("vload", 2, 0)`. A bundle issues in one cycle and holds at most 2 scalar, 1 vector, 1 load and 1 store instruction.
- Instructions in a bundle read registers as they were before the bundle, and two of them may not write the same register. There are 8 scalar registers, and 8 vector registers of 8 lanes each.
- Loads become readable 4 cycles after they issue, and every other result 1 cycle after. A bundle whose inputs are not ready waits. `VectorMachine.run` returns the cycle count.
- `z` must equal `a * x[i] + b * y[i]` for every `n >= 0`, including sizes that are not a multiple of 8. The program may depend on `n` only.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** performance take-homes hand you a small machine model and a slow kernel, and judge whether you find the bottleneck unit at each step and know how close you are to the bound. Each later part adds one requirement: vector instructions with a scalar tail, then enough work in flight to hide load latency.

**Where it is used:** compilers and kernel authors for accelerators do exactly this by hand or by scheduler: bundle independent instructions, vectorize, unroll, and software-pipeline until one unit is busy every cycle.

Adapted from the kernel optimisation take-home in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The simulator is kept, with `vrsum` and the other unused opcodes still available. The kernel is new: an element-wise `a * x + b * y` written back to memory replaces the source's reduction to one scalar. The speedup score becomes three cycle limits, and the written report on bottlenecks is left out; the interview questions ask for it instead.""",
    "parts": [
        {
            "title": "Correct and packed",
            "description_en": r"""**Signature:** `build_kernel(n) -> list[list[tuple]]`

- The program is correct for every `n`, and takes at most `8n + 20` cycles for `n` from `0` to `40`.
- Load `a` and `b` once, and put independent instructions in the same bundle.

**Example:**
- with `x = [1, 2, 3]`, `y = [4, 5, 6]`, `a = 2` and `b = -1`, the program writes `z = [-2, -1, 0]`
- at `n = 40` it takes at most `340` cycles, where the starter takes `685`""",
        },
        {
            "title": "Vectorize",
            "description_en": r"""Keep Part 1. Use the vector unit: `vload`, `vmul`, `vmadd`, `vbcast` and `vstore` handle 8 elements at once.

- At `n = 128`, `256` and `512`, take at most `1.25n + 40` cycles.
- Sizes that are not a multiple of 8 stay correct, so the last `n % 8` elements need scalar code.

**Example:** at `n = 256` the program takes at most `360` cycles; a scalar program cannot, since it loads one value per cycle.""",
        },
        {
            "title": "Near the load bound",
            "description_en": r"""Keep Parts 1–2. Each 8 elements need two vector loads, and the machine issues one load per cycle, so the load unit bounds the program at about `n / 4` cycles.

- Take at most `2 * (n // 8) + 8 * (n % 8) + 24` cycles for `n` from `512` to `2053`.

**Example:** at `n = 1024` the program takes at most `280` cycles, against a bound of `256`.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "In the starter, which instructions wait for a load, and which do not depend on each other at all? Which values never change between elements, and how often does the starter read them from memory?"},
        {"level": 2, "kind": "analysis", "content": "Load a and b into registers once at the start. Per element, the two loads go to the load unit on consecutive cycles, and their pointer increments can ride in the same bundles as scalar instructions. The two multiplies can share a bundle, then the add, then the store with its pointer increment: about 8 cycles per element without overlapping elements."},
    ],
    "model_connections": [
        "Accelerator kernels for attention and matrix multiply are scheduled the same way: keep the limiting unit, usually memory loads, busy every cycle.",
        "Compilers for VLIW and in-order machines use list scheduling and software pipelining to hide load latency, which is what hand-tuned kernels imitate.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Bundling and hoisting constants is cheap and safe, and removes most of the baseline's wasted cycles.",
            "Vector instructions do 8 elements per load, which changes the bound by a factor of 8.",
            "Rotating registers across several chunks lets later loads issue while earlier results wait, approaching the load bound.",
        ],
        "cons": [
            "Hand-scheduled straight-line code grows with n; a real kernel would loop and pay for it in instruction memory.",
            "Register rotation is limited by the 8 vector registers, which caps how many chunks can be in flight.",
            "A scalar tail costs several cycles per element, which matters for small n.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
from collections import defaultdict

VLEN = 8
LOAD_LATENCY = 4
UNIT_WIDTH = {"scalar": 2, "vector": 1, "load": 1, "store": 1}
OP_UNIT = {
    "li": "scalar", "mov": "scalar", "add": "scalar", "addi": "scalar", "mul": "scalar",
    "sload": "load", "vload": "load", "sstore": "store", "vstore": "store",
    "vadd": "vector", "vmul": "vector", "vmadd": "vector", "vbcast": "vector", "vrsum": "vector",
}


def regs(instr):
    """(reads, writes) of one instruction, as ("s" | "v", index) pairs."""
    op, *a = instr
    if op == "li":
        return [], [("s", a[0])]
    if op in ("addi", "sload"):
        return [("s", a[1])], [("s", a[0])]
    if op in ("add", "mul"):
        return [("s", a[1]), ("s", a[2])], [("s", a[0])]
    if op == "vload":
        return [("s", a[1])], [("v", a[0])]
    if op == "sstore":
        return [("s", a[0]), ("s", a[1])], []
    if op == "vstore":
        return [("s", a[0]), ("v", a[1])], []
    if op == "vmul":
        return [("v", a[1]), ("v", a[2])], [("v", a[0])]
    if op == "vmadd":
        return [("v", a[1]), ("v", a[2]), ("v", a[3])], [("v", a[0])]
    if op == "vbcast":
        return [("s", a[1])], [("v", a[0])]
    raise ValueError(f"unexpected opcode {op!r}")


def schedule(ops):
    """Pack a straight-line op list into bundles: each op goes to the earliest cycle that keeps
    every read after its producer is ready, every write after earlier reads and writes of that
    register, and its unit under capacity."""
    ready = defaultdict(int)  # register -> cycle its newest value can be read
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(int)
    used = defaultdict(int)  # (cycle, unit) -> ops placed
    bundles = defaultdict(list)
    for op in ops:
        reads, writes = regs(op)
        unit = OP_UNIT[op[0]]
        t = max([ready[r] for r in reads] + [last_write[w] + 1 for w in writes] + [last_read[w] for w in writes] + [0])
        while used[t, unit] == UNIT_WIDTH[unit]:
            t += 1
        used[t, unit] += 1
        bundles[t].append(op)
        for r in reads:
            last_read[r] = max(last_read[r], t)
        for w in writes:
            last_write[w] = t
            ready[w] = t + (LOAD_LATENCY if unit == "load" else 1)
    return [bundles[t] for t in sorted(bundles)]  # empty cycles dropped: the machine stalls instead


def build_kernel(n):
    """z[i] = a * x[i] + b * y[i], with x at 0, y at n, a at 2n, b at 2n + 1 and z at 2n + 2."""
    A, B, PX, PY, PZ, T, U, W = range(8)
    ops = [("li", A, 2 * n), ("li", B, 2 * n + 1), ("sload", A, A), ("sload", B, B),
           ("li", PX, 0), ("li", PY, n), ("li", PZ, 2 * n + 2)]
    chunks, tail = divmod(n, VLEN)
    if chunks:
        ops += [("vbcast", 0, A), ("vbcast", 1, B)]
    for k in range(chunks):
        vx, vy = 2 + 2 * (k % 3), 3 + 2 * (k % 3)  # three chunks in flight hide the load latency
        ops += [("vload", vx, PX), ("vload", vy, PY), ("addi", PX, PX, VLEN), ("addi", PY, PY, VLEN),
                ("vmul", vy, 1, vy), ("vmadd", vx, 0, vx, vy),  # a * x + (b * y), the reference's order
                ("vstore", PZ, vx), ("addi", PZ, PZ, VLEN)]
    for _ in range(tail):  # the last n % VLEN elements, one at a time
        ops += [("sload", T, PX), ("sload", U, PY), ("addi", PX, PX, 1), ("addi", PY, PY, 1),
                ("mul", T, A, T), ("mul", U, B, U), ("add", W, T, U), ("sstore", PZ, W), ("addi", PZ, PZ, 1)]
    return schedule(ops)
''',
    "interview_questions": interview(
        concept=[
            "Why does the starter take far more cycles per element than it has instructions, and which waits cause it?",
            "Which instructions can share a bundle, and why can an instruction write a register another one in the same bundle reads?",
        ],
        deep_dive=[
            "Which unit limits a well-packed scalar program, and how many cycles per element does that imply?",
        ],
        tradeoffs=[
            "How do you handle n that is not a multiple of 8, and what does the tail cost?",
            "Why does the load unit bound the vector program at about n / 4 cycles, and how close can you get?",
            "How many chunks must be in flight to hide a 4-cycle load latency, and what limits it on this machine?",
            "Would you write the schedule by hand or write a small scheduler, and what would you check to trust it?",
        ],
    ),
}
