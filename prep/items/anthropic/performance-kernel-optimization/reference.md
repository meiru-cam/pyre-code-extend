The kernel keeps a running *vector* accumulator and reduces it to a single scalar only once, at the very end — reducing after every chunk would itself spend a vector-ALU instruction, competing with the arithmetic for the machine's one vector issue slot. Steps 1 through 3 also assume `n` is a multiple of `VLEN`, so that each technique can be introduced without the leftover elements complicating it; Step 4 removes that assumption and is the version to submit as `build_optimised`. All five versions below are measured on the same worked example: 64 elements, drawn once with a fixed seed.

### Step 1 — Pack independent operations and hoist the constants

The baseline spends one whole bundle each on two things it does not have to: incrementing the two pointers, one at a time, though they are independent; and — since `a` and `b` never change — reading them back from memory on every iteration. Packing `addi(S_PTR_X, ...)` and `addi(S_PTR_Y, ...)` into a single bundle removes one bundle per element. Loading `a` and `b` once, before the loop, rather than once per element, removes two more per element — and, being loads, those were the more expensive kind to repeat, since only one may issue per cycle regardless of how many scalar-ALU slots are free.

```python
def build_step1(n):
    S_A, S_B, S_PTR_X, S_PTR_Y, S_ACC, S_TMP, S_PROD, S_SPARE = range(8)
    L = mem_layout(n)
    prog = [
        [("li", S_PTR_X, L["base_x"]), ("li", S_PTR_Y, L["base_y"])],
        [("li", S_PROD, L["addr_a"]), ("li", S_TMP, L["addr_b"])],   # S_PROD, S_TMP as scratch addresses
        [("sload", S_A, S_PROD)],
        [("sload", S_B, S_TMP), ("li", S_ACC, 0.0)],
    ]
    for _ in range(n):
        prog += [
            [("sload", S_TMP, S_PTR_X)],
            [("mul", S_PROD, S_A, S_TMP)],
            [("add", S_PROD, S_PROD, S_B)],
            [("sload", S_TMP, S_PTR_Y)],
            [("mul", S_PROD, S_PROD, S_TMP)],
            [("add", S_ACC, S_ACC, S_PROD)],
            [("addi", S_PTR_X, S_PTR_X, 1), ("addi", S_PTR_Y, S_PTR_Y, 1)],
        ]
    prog += [[("li", S_SPARE, L["addr_out"])], [("sstore", S_SPARE, S_ACC)]]
    return prog
```

`run_program(build_step1, 64, x, y, a, b)` costs 838 cycles against the baseline's 1223 — a 1.46× speedup. Every remaining bundle but one sits on a single dependency chain: load `x[i]`, multiply, add, multiply, add, each step needing the previous one's result. The exception is the load of `y[i]`, which only needs `S_PTR_Y` — but it cannot move any earlier in this schedule either, because it reuses `S_TMP`, the very register `x[i]`'s load used, so moving it up would overwrite `x[i]` before the first multiply has read it. Giving independent operands independent registers, so that an independent load really can run ahead of the arithmetic waiting on a different one, is the deeper version of this idea that Steps 3 and 4 use. Here, though, the chain is serial regardless of scheduling, and vectorising — replacing one lane of work per instruction with eight — is the far larger win, and is what the next step does.

### Step 2 — Vectorise

Switch from one element to one `VLEN`-wide chunk at a time. `a` and `b` are broadcast into vector registers once, before the loop — the same hoisting as Step 1, now applied to the broadcast rather than the scalar value. The running sum is kept as a vector, `V_ACC`, and reduced to a scalar only once, with `vrsum`, right before the store.

Each chunk needs exactly 2 loads (one for `x`'s chunk, one for `y`'s) and, because `vmadd` fuses a multiply and an add into one instruction, exactly 2 vector-ALU instructions: `V_X <- a * x + b`, then `V_ACC <- (a * x + b) * y + V_ACC`, both written with `vmadd`. Writing the same arithmetic with unfused `vmul`/`vadd` would need 4 vector-ALU instructions per chunk against only 2 loads, making the vector unit the bottleneck on its own; fusing brings the two units back to the same rate — 2 instructions each per chunk, the fewest either one can be made to do, since the kernel is one multiply-add feeding a second multiply-add, by construction.

```python
V_A, V_B, V_ACC = 0, 1, 2      # persistent vector registers: broadcast a, broadcast b, running sum
P_X, P_Y = 0, 1                # persistent scalar registers: pointers into x and y


def _setup(L):
    a_val, b_val, zero = 4, 5, 6     # scratch, free again once broadcast
    return [
        [("li", P_X, L["base_x"]), ("li", P_Y, L["base_y"])],
        [("li", 2, L["addr_a"]), ("li", 3, L["addr_b"])],
        [("sload", a_val, 2), ("li", zero, 0.0)],
        [("sload", b_val, 3)],
        [("vbcast", V_A, a_val)],
        [("vbcast", V_B, b_val)],
        [("vbcast", V_ACC, zero)],
    ]


def _finish(L, scalar_acc_reg):
    out_addr = 6
    return [[("li", out_addr, L["addr_out"])], [("sstore", out_addr, scalar_acc_reg)]]


def build_step2(n):
    assert n % VLEN == 0
    L = mem_layout(n)
    V_X, V_Y = 3, 4
    prog = _setup(L)
    for _ in range(n // VLEN):
        prog += [
            [("vload", V_X, P_X)],
            [("vload", V_Y, P_Y)],
            [("vmadd", V_X, V_A, V_X, V_B)],          # V_X <- a * x + b
            [("vmadd", V_ACC, V_X, V_Y, V_ACC)],       # V_ACC <- (a * x + b) * y + V_ACC
            [("addi", P_X, P_X, VLEN), ("addi", P_Y, P_Y, VLEN)],
        ]
    prog.append([("vrsum", 4, V_ACC)])
    prog += _finish(L, 4)
    return prog
```

For 64 elements (8 chunks), this costs 68 cycles: a fixed 12 for broadcasting the constants, reducing the sum and storing it, plus 7 per chunk — above the 2 loads and 2 vector-ALU instructions each chunk actually needs. The reason is that every chunk's first `vmadd` waits out the `vload` that feeds it: the load issued only 2 bundles earlier, 2 cycles short of its 4-cycle latency, so the chunk stalls for those 2 cycles before it can even start computing. 68 cycles is a 17.99× speedup over the baseline. Filling those stalled cycles with another chunk's independent work is what the next step does.

### Step 3 — Unroll to fill both units

Process two chunks per pass instead of one, each in its own pair of vector registers (`V_X0, V_Y0` and `V_X1, V_Y1`): reusing one register for both chunks would let the second chunk's load overwrite the first chunk's data before the first chunk's `vmadd` has read it. Issue all four loads for the pair first, then both chunks' pair of `vmadd`s:

```python
def build_step3(n):
    assert n % (2 * VLEN) == 0
    L = mem_layout(n)
    V_X0, V_Y0, V_X1, V_Y1 = 3, 4, 5, 6
    prog = _setup(L)
    for _ in range(n // (2 * VLEN)):
        prog += [
            [("vload", V_X0, P_X), ("addi", P_X, P_X, VLEN)],
            [("vload", V_Y0, P_Y), ("addi", P_Y, P_Y, VLEN)],
            [("vload", V_X1, P_X), ("addi", P_X, P_X, VLEN)],
            [("vload", V_Y1, P_Y), ("addi", P_Y, P_Y, VLEN)],
            [("vmadd", V_X0, V_A, V_X0, V_B)],
            [("vmadd", V_ACC, V_X0, V_Y0, V_ACC)],
            [("vmadd", V_X1, V_A, V_X1, V_B)],
            [("vmadd", V_ACC, V_X1, V_Y1, V_ACC)],
        ]
    prog.append([("vrsum", 4, V_ACC)])
    prog += _finish(L, 4)
    return prog
```

By the time the first `vmadd` needs `V_X0`, four load bundles have already issued — this one's own load, plus three more — so `V_X0`'s 4-cycle latency has fully elapsed: no stall. The same holds for every `vmadd` in the block. But the four loads and the four `vmadd`s are still two separate phases: the vector unit is idle for all four load bundles, and the load unit is idle for all four `vmadd` bundles, so each unit is busy only half the time. 44 cycles for 64 elements is `4 * 8 + 12` — 4 cycles per chunk, down from Step 2's 7: the stalls are gone, but neither unit is kept continuously busy, which is exactly what the next step fixes. 44 cycles is a 27.80× speedup over the baseline.

### Step 4 — Software-pipeline, and handle the tail

Keep both units busy on *every* cycle, instead of alternating between them, by working on two chunks at once: while chunk `i`'s `vmadd` consumes the data sitting in one of two rotating register slots, pack a `vload` for chunk `i + 2` into the *same* bundle, overwriting that same slot. The read (of chunk `i`'s value, already there) and the write (of chunk `i + 2`'s value, not yet needed) are two different instructions in the bundle, so — as stated in the machine description — the read sees the pre-bundle value: this is safe, and it is what lets two vector registers do the work that Step 3 spent four on. A short *prologue* fills both slots with chunks 0 and 1 before the steady state starts; a short *epilogue* drains the last two chunks, already loaded, once there is nothing left to prefetch.

```python
def build_step4(n):
    L = mem_layout(n)
    num_chunks, tail = n // VLEN, n % VLEN
    SLOTS = [(3, 4), (5, 6)]        # two rotating (v_x, v_y) register pairs
    V_TMP = 7
    prog = _setup(L)

    if num_chunks == 1:
        vx, vy = SLOTS[0]
        prog.append([("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
        prog.append([("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        prog.append([("vmadd", V_TMP, V_A, vx, V_B)])
        prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC)])
    elif num_chunks >= 2:
        for k in (0, 1):                                    # prologue: fill both slots
            vx, vy = SLOTS[k]
            prog.append([("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
            prog.append([("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        for i in range(num_chunks - 2):                     # steady state: compute i, prefetch i + 2
            vx, vy = SLOTS[i % 2]
            prog.append([("vmadd", V_TMP, V_A, vx, V_B), ("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
            prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC), ("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        for i in range(max(num_chunks - 2, 0), num_chunks):  # epilogue: drain the last two chunks
            vx, vy = SLOTS[i % 2]
            prog.append([("vmadd", V_TMP, V_A, vx, V_B)])
            prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC)])

    scalar_acc = 2
    prog.append([("vrsum", scalar_acc, V_ACC)])
    if tail:
        a_val, b_val, addr_a, addr_b = 3, 4, 5, 6
        prog.append([("li", addr_a, L["addr_a"]), ("li", addr_b, L["addr_b"])])
        prog.append([("sload", a_val, addr_a)])
        prog.append([("sload", b_val, addr_b)])
        prod, tmp = 5, 6         # reuse addr_a's and addr_b's registers once a_val/b_val are loaded
        for _ in range(tail):
            prog += [
                [("sload", prod, P_X)],
                [("mul", prod, a_val, prod)],
                [("add", prod, prod, b_val)],
                [("sload", tmp, P_Y)],
                [("mul", prod, prod, tmp)],
                [("add", scalar_acc, scalar_acc, prod)],
                [("addi", P_X, P_X, 1), ("addi", P_Y, P_Y, 1)],
            ]
    prog += _finish(L, scalar_acc)
    return prog
```

The steady state now issues exactly one `vmadd`, one `vload` and one `addi` in every bundle: zero stalls, with the load unit and the vector unit both at their one-per-cycle limit simultaneously. For 64 elements (8 chunks, no tail), this is `2 * 8 + 16 = 32` cycles — a 38.22× speedup over the baseline, and the version to return as `build_optimised`. The `n % VLEN` leftover elements, once the vectorised part is done, fall back to a version of Step 1's per-element loop, with `a` and `b` already loaded once and reused for the whole tail: each leftover element costs the same 13 cycles Step 1's steady state costs, after 3 more cycles to fetch `a` and `b` for the tail. `build_step4` produces a correct program for every `n`, including `n` not a multiple of `VLEN` and `n` too small to fill even one pipeline slot.

### The lower bound

Every chunk needs exactly 2 vector loads and, once the two multiply-adds are fused, exactly 2 vector-ALU instructions. Since the load unit and the vector unit each admit at most 1 instruction per cycle, no program built from whole `VLEN`-wide chunks can process `n // VLEN` of them in under `2 * (n // VLEN)` cycles, no matter how the bundles are packed. Step 4 reaches this bound exactly in its steady state — Step 3 does not, sitting at exactly twice it, one unit idle half the time — and pays a further fixed cost on top, measured at 16 cycles, for the one-off broadcast of `a` and `b`, filling and draining the two-deep pipeline, the final reduction and the store. That fixed cost does not grow with `n`, so it matters less as `n` grows:

- `n`: 64 · Step 4 cycles: 32 · `2 * (n // VLEN)`: 16 · ratio: 2.00
- `n`: 128 · Step 4 cycles: 48 · `2 * (n // VLEN)`: 32 · ratio: 1.50
- `n`: 256 · Step 4 cycles: 80 · `2 * (n // VLEN)`: 64 · ratio: 1.25
- `n`: 512 · Step 4 cycles: 144 · `2 * (n // VLEN)`: 128 · ratio: 1.125
- `n`: 1024 · Step 4 cycles: 272 · `2 * (n // VLEN)`: 256 · ratio: 1.0625

Deciding what to try next means finding which unit is idle: either read it off a bundle trace, or compare, for each unit, its total instruction count against `cycles * UNIT_WIDTH[unit]`. A unit at that product is saturated, and only reducing its own instruction count helps further — fewer loads, or fusing arithmetic, as Step 2's move from `vmul`/`vadd` to `vmadd` does. A unit below it still has slack that better scheduling can recover — packing, unrolling, pipelining — without changing what the kernel computes at all. This comparison, cycles measured at each step against the bound derived from the machine's own throughput, is what the write-up should end with.

### Follow-ups

- **A larger `LOAD_LATENCY`.** The steady state above hides exactly 4 cycles of latency with a 2-deep pipeline (2 loads per chunk × 2 chunks of lookahead). `LOAD_LATENCY = 8` would need a 4-deep rotation — 8 vector registers just for in-flight chunk data, on top of the 4 already spent on the two broadcasts, the accumulator and the temporary product: more than the machine's 8 vector registers provide, forcing a smaller, less latency-hiding pipeline instead.
- **Store bandwidth.** This kernel stores once, at the very end, so the store unit is never a candidate bottleneck. A kernel that wrote a full elementwise result back to memory, rather than only a reduction, would need to weigh vector stores against vector loads the same way Step 2 weighs loads against vector-ALU instructions.
- **Summation order.** `vrsum` sums a vector register's lanes in lane order, and the running sum itself accumulates chunk by chunk, so the total is not summed in the same order as `reference`'s left-to-right sum over `x` and `y`; floating-point addition is not associative, so the two can differ in the last few bits, which is why the checks below compare with a tolerance rather than for exact equality.
- **A tighter setup.** The broadcast of the running sum's initial zero could be issued while `a` and `b`'s loads are still in flight, rather than after their broadcasts, shaving a cycle or two off the fixed part of every version above — a small further win once the per-chunk rate is already optimal.
- **A shallower pipeline.** With only one vector register free per operand instead of two, a 1-deep pipeline could still hide latency by prefetching one chunk ahead, provided a chunk's own processing took at least `LOAD_LATENCY` cycles on its own; at 2 cycles per chunk here, it does not, which is why Step 4 needs two rotating slots rather than one.

```python
demo = VectorMachine([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
demo_cycles = demo.run([[("li", 0, 0)], [("vload", 0, 0)], [("vadd", 1, 0, 0)]])
assert demo_cycles == 6 and demo.v[1] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
print("machine-description trace: OK (6 cycles)")

import random

rng = random.Random(0)
n_demo = 64
x_demo = [rng.uniform(-3, 3) for _ in range(n_demo)]
y_demo = [rng.uniform(-3, 3) for _ in range(n_demo)]
a_demo, b_demo = 1.7, -0.4
want_demo = reference(x_demo, y_demo, a_demo, b_demo)

exact_cycles = [(build_baseline, 1223), (build_step1, 838), (build_step2, 68),
                (build_step3, 44), (build_step4, 32)]
for build, expected in exact_cycles:
    got, cycles = run_program(build, n_demo, x_demo, y_demo, a_demo, b_demo)
    assert abs(got - want_demo) < 1e-6, (build.__name__, got, want_demo)
    assert cycles == expected, (build.__name__, cycles, expected)
print("worked example (n = 64): every version matches the reference; cycle counts as quoted")

baseline_cycles = dict(exact_cycles)[build_baseline]
speedups = {build.__name__: baseline_cycles / cycles for build, cycles in exact_cycles}
for name, quoted in [("build_step1", 1.46), ("build_step2", 17.99), ("build_step3", 27.80),
                      ("build_step4", 38.22)]:
    assert abs(speedups[name] - quoted) < 0.01, (name, speedups[name], quoted)
print("speedups match the quoted figures")

# closed-form cycle counts (also checked for correctness, not just cycles)
for chunks in range(1, 13):
    n = chunks * VLEN
    got2, c2 = run_program(build_step2, n, [1.0] * n, [1.0] * n, 1.0, 0.5)
    assert c2 == 7 * chunks + 12 and abs(got2 - reference([1.0] * n, [1.0] * n, 1.0, 0.5)) < 1e-9
    if chunks % 2 == 0:
        got3, c3 = run_program(build_step3, n, [1.0] * n, [1.0] * n, 1.0, 0.5)
        assert c3 == 4 * chunks + 12 and abs(got3 - reference([1.0] * n, [1.0] * n, 1.0, 0.5)) < 1e-9
    got4, c4 = run_program(build_step4, n, [1.0] * n, [1.0] * n, 1.0, 0.5)
    assert c4 == 2 * chunks + 16 and abs(got4 - reference([1.0] * n, [1.0] * n, 1.0, 0.5)) < 1e-9
for n in range(0, 30):
    want_n = reference([1.0] * n, [1.0] * n, 1.0, 0.5)
    gotb, cb = run_program(build_baseline, n, [1.0] * n, [1.0] * n, 1.0, 0.5)
    got1, c1 = run_program(build_step1, n, [1.0] * n, [1.0] * n, 1.0, 0.5)
    assert cb == 19 * n + 7 and abs(gotb - want_n) < 1e-9
    assert c1 == 13 * n + 6 and abs(got1 - want_n) < 1e-9
print("closed-form cycle counts verified for many sizes, against the reference too")

# Step 4 cycles against the derived lower bound, for growing n
for n, expected_cycles in [(64, 32), (128, 48), (256, 80), (512, 144), (1024, 272)]:
    _, cycles = run_program(build_step4, n, [0.3] * n, [0.2] * n, 1.1, -0.2)
    assert cycles == expected_cycles
    assert cycles >= 2 * (n // VLEN)
print("Step 4 cycle counts against the derived lower bound: as quoted")

# random correctness sweep for build_optimised = build_step4: many sizes, including tails and n = 0
rng2 = random.Random(1)
for _ in range(300):
    n = rng2.choice([0, 1, 2, 3, 7, 8, 9, 15, 16, 17]) if rng2.random() < 0.6 else rng2.randint(0, 90)
    x = [rng2.uniform(-5, 5) for _ in range(n)]
    y = [rng2.uniform(-5, 5) for _ in range(n)]
    a, b = rng2.uniform(-4, 4), rng2.uniform(-4, 4)
    got, cycles = run_program(build_step4, n, x, y, a, b)
    want = reference(x, y, a, b)
    assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (n, got, want)
    assert cycles >= 0
print("random correctness sweep (many sizes, including tails and n = 0): OK")


def expect_error(machine, program, exc):
    try:
        machine.run(program)
        raise AssertionError(f"expected {exc.__name__}")
    except exc:
        pass


expect_error(VectorMachine([0.0] * 32), [[("vload", 0, 0), ("vload", 1, 0)]], ValueError)   # 2 loads
expect_error(VectorMachine([0.0] * 32), [[("li", 0, 1), ("li", 0, 2)]], ValueError)          # dup write
oob_machine = VectorMachine([0.0] * 4)
oob_machine.s[0] = 2
expect_error(oob_machine, [[("vload", 0, 0)]], IndexError)                                   # 2 + VLEN > 4
print("the simulator rejects over-width bundles, duplicate writes and out-of-range addresses")

print("all checks passed")
```
