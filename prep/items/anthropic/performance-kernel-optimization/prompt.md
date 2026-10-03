You are given a simulator for a small vector processor and a kernel that runs correctly but slowly on it. The task is to rewrite the kernel so that it computes the same result in fewer simulated cycles, without changing the processor itself.

### The machine

The processor has 8 scalar registers `s0`–`s7`, each holding one number, and 8 vector registers `v0`–`v7`, each holding `VLEN = 8` lanes of floats. Memory is a flat, word-addressed list of floats: address `i` names one float, not one byte. There is no branch instruction. A *program* is a fixed list of *bundles*, built in advance — the way a compiler unrolls a loop whose trip count it already knows, rather than a stream that a hardware program counter walks with a decision at the end of every iteration.

A *bundle* is a list of instructions issued together, in a single cycle. Every instruction belongs to one *issue unit*, and a bundle may carry at most this many instructions of each unit, simultaneously: 2 scalar-ALU, 1 vector-ALU, 1 load (scalar or vector) and 1 store (scalar or vector). The instruction set:

- mnemonic: `li` · operands: `rd, imm` · effect: `s[rd] = imm` · unit: scalar
- mnemonic: `mov` · operands: `rd, rs` · effect: `s[rd] = s[rs]` · unit: scalar
- mnemonic: `add` · operands: `rd, rs1, rs2` · effect: `s[rd] = s[rs1] + s[rs2]` · unit: scalar
- mnemonic: `addi` · operands: `rd, rs, imm` · effect: `s[rd] = s[rs] + imm` · unit: scalar
- mnemonic: `mul` · operands: `rd, rs1, rs2` · effect: `s[rd] = s[rs1] * s[rs2]` · unit: scalar
- mnemonic: `sload` · operands: `rd, raddr` · effect: `s[rd] = mem[s[raddr]]` · unit: load
- mnemonic: `vload` · operands: `vd, raddr` · effect: `v[vd] = mem[s[raddr] : s[raddr] + VLEN]` · unit: load
- mnemonic: `sstore` · operands: `raddr, rs` · effect: `mem[s[raddr]] = s[rs]` · unit: store
- mnemonic: `vstore` · operands: `raddr, vs` · effect: `mem[s[raddr] : s[raddr] + VLEN] = v[vs]` · unit: store
- mnemonic: `vadd` · operands: `vd, vs1, vs2` · effect: `v[vd] = v[vs1] + v[vs2]`, lane by lane · unit: vector
- mnemonic: `vmul` · operands: `vd, vs1, vs2` · effect: `v[vd] = v[vs1] * v[vs2]`, lane by lane · unit: vector
- mnemonic: `vmadd` · operands: `vd, vs1, vs2, vs3` · effect: `v[vd] = v[vs1] * v[vs2] + v[vs3]`, lane by lane · unit: vector
- mnemonic: `vbcast` · operands: `vd, rs` · effect: every lane of `v[vd]` is set to `s[rs]` · unit: vector
- mnemonic: `vrsum` · operands: `rd, vs` · effect: `s[rd]` = the sum of the `VLEN` lanes of `v[vs]` · unit: vector

Every instruction in a bundle reads the register file as it stood *before* the bundle, and only once every instruction has been read from does any of the bundle's writes commit. Two independent instructions may therefore share a bundle even when one reads a register that another writes in the same bundle: the reader sees the pre-bundle value. Two instructions may never write the same register in one bundle.

A written register is not necessarily readable by the very next bundle. `li`, `mov`, `add`, `addi`, `mul`, `vadd`, `vmul`, `vmadd`, `vbcast` and `vrsum` make their result readable one cycle after the bundle that produced it issues. `sload` and `vload` take `LOAD_LATENCY = 4` cycles instead. A bundle whose inputs are not all ready *stalls*: cycles pass with nothing issued until every register the bundle reads is ready, and only then does it issue. The *cycle count* of a program is one more than the cycle at which its last bundle issues (an empty program costs 0 cycles, and every bundle — issued or stalled — advances the cycle counter by at least 1). Memory is not separately hazard-checked: no program below reads an address before its own prior write to that address has already completed.

For example, this sequence of bundles:

```text
li(s0, 0)         # no inputs; issues at cycle 0; s0 is readable from cycle 1
vload(v0, s0)     # reads s0 (ready at 1, and cycle 1 is this bundle's earliest slot anyway);
                  #   issues at cycle 1; v0 is a load result, so it is not readable until 1 + 4 = 5
vadd(v1, v0, v0)  # reads v0; its earliest slot would be cycle 2, but v0 is not ready until 5 --
                  #   stalls through cycles 2, 3 and 4, then issues at cycle 5
```

costs 6 cycles: bundles issue at cycles 0, 1 and 5, and the count is `5 + 1`.

The simulator below implements exactly this machine.

```python
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
    """The (reads, writes) of one instruction, each a list of ("s" | "v", index)."""
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
    """An in-order vector processor. A program is a list of bundles; a bundle is a list of
    instructions issued in the same cycle, at most UNIT_WIDTH[unit] per unit. See the machine
    description above for the read/write and stall rules."""

    def __init__(self, mem):
        self.s = [0] * NUM_SREG
        self.v = [[0.0] * VLEN for _ in range(NUM_VREG)]
        self.mem = mem
        self.cycles = 0

    def _eval(self, instr):
        """The writes this instruction produces, computed from the register file as it stood
        before the enclosing bundle. Returns a list of (file, index, value); a store's effect on
        memory is applied immediately, since nothing here re-reads a location before its own
        prior write to it has retired."""
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
        """Executes `program` (a list of bundles) and returns its cycle count."""
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
```

### The kernel to optimise

Given arrays `x` and `y` of the same length `n` and scalars `a` and `b`, the kernel computes

```math
S = \sum_{i=0}^{n-1} (a \cdot x_i + b) \cdot y_i .
```

```python
def reference(x, y, a, b):
    """The exact quantity the kernel must produce."""
    return sum((a * xi + b) * yi for xi, yi in zip(x, y))
```

A program receives its input through memory and must leave its output there too, laid out as follows: `mem[0:n]` holds `x`, `mem[n:2n]` holds `y`, `mem[2n]` holds `a`, `mem[2n + 1]` holds `b`, and the program must write `S` to `mem[2n + 2]`. `n` can be any non-negative integer, including one that is not a multiple of `VLEN`.

```python
def mem_layout(n):
    """Where each input and the output live, for arrays of length n."""
    return dict(base_x=0, base_y=n, addr_a=2 * n, addr_b=2 * n + 1, addr_out=2 * n + 2)


def make_memory(x, y, a, b):
    n = len(x)
    assert len(y) == n
    return list(x) + list(y) + [float(a), float(b), 0.0]


def run_program(build, n, x, y, a, b):
    """Builds a program for this n, runs it on a fresh VectorMachine, and returns (S, cycles)."""
    machine = VectorMachine(make_memory(x, y, a, b))
    cycles = machine.run(build(n))
    return machine.mem[mem_layout(n)["addr_out"]], cycles
```

The baseline kernel below is correct but naive: one instruction per bundle, one element of `x` and `y` at a time, and it re-reads `a` and `b` from memory on every iteration instead of keeping them in a register.

```python
def build_baseline(n):
    S_ADDR_A, S_ADDR_B, S_PTR_X, S_PTR_Y, S_ACC, S_TMP_AB, S_TMP_XY, S_PROD = range(8)
    L = mem_layout(n)
    prog = [
        [("li", S_ADDR_A, L["addr_a"])],
        [("li", S_ADDR_B, L["addr_b"])],
        [("li", S_PTR_X, L["base_x"])],
        [("li", S_PTR_Y, L["base_y"])],
        [("li", S_ACC, 0.0)],
    ]
    for _ in range(n):
        prog += [
            [("sload", S_TMP_AB, S_ADDR_A)],        # re-read a
            [("sload", S_TMP_XY, S_PTR_X)],          # x[i]
            [("mul", S_PROD, S_TMP_AB, S_TMP_XY)],    # a * x[i]
            [("sload", S_TMP_AB, S_ADDR_B)],         # re-read b
            [("add", S_PROD, S_PROD, S_TMP_AB)],      # a * x[i] + b
            [("sload", S_TMP_XY, S_PTR_Y)],           # y[i]
            [("mul", S_PROD, S_PROD, S_TMP_XY)],       # (a * x[i] + b) * y[i]
            [("add", S_ACC, S_ACC, S_PROD)],          # accumulate
            [("addi", S_PTR_X, S_PTR_X, 1)],
            [("addi", S_PTR_Y, S_PTR_Y, 1)],
        ]
    prog += [[("li", S_TMP_AB, L["addr_out"])], [("sstore", S_TMP_AB, S_ACC)]]
    return prog
```

Running `run_program(build_baseline, n, x, y, a, b)` reproduces `reference(x, y, a, b)` for every `n`, at a cycle cost that grows with `n`.

### Deliverables and scoring

You have 2 hours. AI assistants — Claude, ChatGPT, Copilot or any other — may be used throughout.

Write a function `build_optimised(n)` with the same contract as `build_baseline` above: given `n`, it returns a program that, run through `run_program`, writes `reference(x, y, a, b)` to `mem[2n + 2]` for every `n`, including sizes that are not a multiple of `VLEN`. Its score is the *speedup*

```math
\text{speedup} = \frac{\text{cycles of build\_baseline}(n)}{\text{cycles of build\_optimised}(n)} ,
```

both measured by `run_program` on the same `n`. Submit `build_optimised` together with a short write-up (under one page) that states, for each change made: the cycle count measured before and after it, at a size tested; which issue unit was believed to be the bottleneck at that point and why; and, once the kernel is vectorised, how the final cycle count compares to a lower bound derived from the machine's own per-unit throughput.
