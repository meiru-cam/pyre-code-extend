"""Model matmul cost with a roofline: one matmul, a layer stack on one device, pipeline against tensor parallel on two, and sharding a feed-forward block."""

from ._interview import interview

# An independent model of every formula, plus a tolerant comparison.
_MODEL = r"""
import math, random

HW = dict(peak_flops=400e12, hbm_bandwidth=2e12, hbm_capacity=24e9, link_bandwidth=200e9)

def near(got, want, label):
    assert isinstance(got, (int, float)) and not isinstance(got, bool), (label, "not a number", got)
    assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-30), (label, got, want)

def m_mm(hw, s, m, k, n):
    fl, by = 2 * m * k * n, s * (m * k + k * n + m * n)
    c, mem = fl / hw["peak_flops"], by / hw["hbm_bandwidth"]
    return dict(flops=fl, bytes=by, intensity=fl / by, compute_time=c, memory_time=mem, time=max(c, mem),
                bound="compute" if c >= mem else "memory")

def shapes(L, T, d, f):
    return [(T, d, f) if i % 2 else (T, f, d) for i in range(1, L + 1)]

def m_stack(hw, s, L, T, d, f):
    mms = [m_mm(hw, s, *x) for x in shapes(L, T, d, f)]
    res = s * d * f * L + s * T * (d + f)
    return dict(compute_time=sum(x["flops"] for x in mms) / hw["peak_flops"],
                transfer_time=sum(x["bytes"] for x in mms) / hw["hbm_bandwidth"],
                time=sum(x["time"] for x in mms), resident_bytes=res,
                fits=res <= hw["hbm_capacity"], headroom=hw["hbm_capacity"] - res)

def m_two(hw, s, L, T, d, f):
    sh = shapes(L, T, d, f)
    h = L // 2
    act = s * T * sh[h - 1][2]
    t = lambda xs: sum(m_mm(hw, s, *x)["time"] for x in xs)
    pipe = dict(latency=t(sh[:h]) + act / hw["link_bandwidth"] + t(sh[h:]),
                peak_bytes=max(h, L - h) * s * d * f + s * T * (d + f), sent_bytes=act)
    lat = sent = 0
    for (m, k, n) in sh:
        lat += m_mm(hw, s, m, k, n // 2)["time"] + s * m * (n // 2) / hw["link_bandwidth"]
        sent += s * m * (n // 2)
    ten = dict(latency=lat, peak_bytes=L * s * d * f / 2 + s * T * (d + f), sent_bytes=sent)
    return dict(pipeline=pipe, tensor=ten)

def m_ffn(hw, s, T, d, f, p, scheme, link=None):
    link = link or hw["link_bandwidth"]
    if scheme == "tensor":
        mms, wb, cb = [m_mm(hw, s, T, d, f // p), m_mm(hw, s, T, f // p, d)], 2 * d * f * s / p, 2 * (p - 1) / p * T * d * s
    else:
        mms, wb, cb = [m_mm(hw, s, T // p, d, f), m_mm(hw, s, T // p, f, d)], 2 * d * f * s, 0
    c, mem, cm = sum(x["compute_time"] for x in mms), sum(x["memory_time"] for x in mms), cb / link
    order = [("compute", c), ("memory", mem), ("communication", cm)]
    bound = max(order, key=lambda kv: (kv[1], -order.index(kv)))[0]
    return dict(flops=sum(x["flops"] for x in mms), weight_bytes=wb, compute_time=c, memory_time=mem,
                comm_bytes=cb, comm_time=cm, latency=sum(x["time"] for x in mms) + cm, bound=bound)

def check(got, want, label):
    assert isinstance(got, dict), (label, "not a dict", type(got))
    assert set(want) <= set(got), (label, "missing keys", sorted(set(want) - set(got)))
    for key, w in want.items():
        if isinstance(w, (str, bool)):
            assert got[key] == w, (label, key, got[key], w)
        else:
            near(got[key], w, (label, key))

def rand_hw(rng):
    return dict(peak_flops=rng.choice([100e12, 312e12, 400e12, 989e12]), hbm_bandwidth=rng.choice([0.9e12, 1.5e12, 2e12, 3.35e12]),
                hbm_capacity=rng.choice([16e9, 24e9, 40e9, 80e9]), link_bandwidth=rng.choice([25e9, 50e9, 200e9, 450e9]))
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "numerics.stability", "code": _MODEL + r"""
r = {fn}(**HW)
near(r.ridge(), 200.0, "ridge")
a = r.matmul(16, 8192, 8192)
assert a["flops"] == 2147483648 and a["bytes"] == 134742016, a
assert a["bound"] == "memory" and abs(a["time"] - 6.7371008e-05) < 1e-12, a
b = r.matmul(4096, 4096, 4096)
assert b["bound"] == "compute" and abs(b["intensity"] - 1365.333) < 1e-3, b
"""},
    {"name": "Part 1: random shapes and machines", "part": 1, "visibility": "unshown", "behavior": "numerics.stability",
     "failure_message": "matmul disagreed with 2mkn FLOPs, s(mk + kn + mn) bytes, their ratio, the two times, their maximum or the bound label on random shapes, machines and element sizes, or an exact tie between compute and memory time was not labelled compute.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(seed)
    hw, s = rand_hw(rng), rng.choice([1, 2, 4])
    m, k, n = (rng.choice([1, 3, 8, 64, 512, 4096, 16384]) for _ in range(3))
    check({fn}(**hw, bytes_per_element=s).matmul(m, k, n), m_mm(hw, s, m, k, n), (seed, m, k, n))
tie = {fn}(peak_flops=2.0 ** 40, hbm_bandwidth=2.0 ** 33, hbm_capacity=1e9, link_bandwidth=1e9)
x = tie.matmul(384, 384, 384)
assert x["compute_time"] == x["memory_time"] and x["bound"] == "compute", x
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "budget.enforcement", "code": _MODEL + r"""
st = {fn}(**HW).stack(40, 8192, 6144, 24576)
assert st["resident_bytes"] == 12582912000 and st["fits"] is True, st
assert abs(st["headroom"] - 11417088000) < 1, st
assert abs(st["compute_time"] - 0.2473901162496) < 1e-9, st
"""},
    {"name": "Part 2: random stacks", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "stack disagreed with the summed compute and transfer times, the summed per-layer kernel times, the resident bytes (every weight plus one layer's input and output), the fit or the headroom, for odd and even layer counts.",
     "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(1000 + seed)
    hw, s = rand_hw(rng), rng.choice([1, 2, 4])
    L, T, d, f = rng.randint(1, 96), rng.choice([1, 16, 2048, 20000]), rng.choice([512, 4096, 8192]), rng.choice([2048, 16384, 32768])
    check({fn}(**hw, bytes_per_element=s).stack(L, T, d, f), m_stack(hw, s, L, T, d, f), (seed, L, T, d, f))
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "scheduler.concurrency", "code": _MODEL + r"""
two = {fn}(**HW).two_devices(40, 8192, 6144, 24576)
pipe, ten = two["pipeline"], two["tensor"]
assert pipe["sent_bytes"] == 100663296 and ten["sent_bytes"] == 5033164800, two
assert pipe["peak_bytes"] == ten["peak_bytes"] == 6543114240, two
assert abs(pipe["latency"] - 0.2478934327296) < 1e-9 and abs(ten["latency"] - 0.1488608821248) < 1e-9, two
"""},
    {"name": "Part 3: random two-device splits", "part": 3, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "two_devices disagreed with the model for pipeline (first half of the layers on one device, one activation handed over) or tensor parallel (each layer's output columns halved, the halves exchanged after every layer), in latency, peak bytes or bytes sent, for odd and even layer counts.",
     "code": _MODEL + r"""
for seed in range(200):
    rng = random.Random(2000 + seed)
    hw, s = rand_hw(rng), rng.choice([1, 2])
    L, T, d, f = rng.randint(2, 96), rng.choice([16, 2048, 20000]), rng.choice([512, 4096, 8192]), rng.choice([2048, 16384, 32768])
    got = {fn}(**hw, bytes_per_element=s).two_devices(L, T, d, f)
    want = m_two(hw, s, L, T, d, f)
    check(got["pipeline"], want["pipeline"], (seed, "pipeline", L))
    check(got["tensor"], want["tensor"], (seed, "tensor", L))
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "routing.selection", "code": _MODEL + r"""
r = {fn}(**HW)
ten = r.ffn(2048, 6144, 24576, 8, "tensor")
tok = r.ffn(2048, 6144, 24576, 8, "token")
slow = r.ffn(2048, 6144, 24576, 8, "tensor", link_bandwidth=12.5e9)
assert ten["weight_bytes"] == 75497472 and tok["weight_bytes"] == 603979776, (ten, tok)
assert abs(ten["comm_bytes"] - 44040192) < 1 and tok["comm_bytes"] == 0, (ten, tok)
assert ten["bound"] == tok["bound"] == "compute" and slow["bound"] == "communication", (ten["bound"], tok["bound"], slow["bound"])
assert abs(ten["latency"] - 0.00060674801664) < 1e-12 and abs(tok["latency"] - 0.00038654705664) < 1e-12, (ten, tok)
"""},
    {"name": "Part 4: random sharding sweeps", "part": 4, "visibility": "unshown", "behavior": "routing.selection",
     "failure_message": "ffn disagreed with the model on FLOPs, weight bytes, compute, memory or communication time, latency or the bound for tensor or token sharding over p from 1 to 64 with or without a link override, or an unknown scheme did not raise ValueError.",
     "code": _MODEL + r"""
for seed in range(300):
    rng = random.Random(3000 + seed)
    hw, s = rand_hw(rng), rng.choice([1, 2])
    p = rng.choice([1, 2, 4, 8, 16, 32, 64])
    T, d, f = rng.choice([64, 2048, 16384]), rng.choice([1024, 4096, 8192]), rng.choice([4096, 16384, 32768])
    scheme = rng.choice(["tensor", "token"])
    link = rng.choice([None, 12.5e9, 25e9, 900e9])
    r = {fn}(**hw, bytes_per_element=s)
    got = r.ffn(T, d, f, p, scheme) if link is None else r.ffn(T, d, f, p, scheme, link_bandwidth=link)
    check(got, m_ffn(hw, s, T, d, f, p, scheme, link), (seed, scheme, p, link))
try:
    {fn}(**HW).ffn(2048, 1024, 4096, 2, "expert")
except Exception as e:
    assert type(e).__name__ == "ValueError", type(e).__name__
else:
    raise AssertionError("an unknown scheme must raise ValueError")
"""},
]

TASK = {
    "title": "Roofline Performance Model",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "Roofline",
    "description_en": r"""Write `Roofline`, a cost model that predicts how long matrix multiplies take on an accelerator, how much memory a layer stack needs, and what splitting work across devices costs. Every answer follows from the formulas below, so the tests compare numbers.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Roofline` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `Roofline(peak_flops, hbm_bandwidth, hbm_capacity, link_bandwidth, bytes_per_element=2)`: FLOP/s, bytes/s, bytes, and bytes/s that one device can send to another while also receiving as much. `s = bytes_per_element` is the size of every stored or sent value.
- A matmul of an `m×k` matrix by a `k×n` matrix costs `2*m*k*n` FLOPs. It moves `s*(m*k + k*n + m*n)` bytes: each input is read once and the output written once.
- Its compute time is FLOPs over `peak_flops`, its memory time is bytes over `hbm_bandwidth`, and its time is the larger of the two, as compute and memory traffic overlap inside one kernel.
- Return plain numbers in dicts with exactly the keys named. The tests compare with a relative tolerance of `1e-9`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** performance roles ask you to estimate before you profile: is this kernel limited by compute or by memory, does the model fit, and what does adding a device cost in communication. Each later part adds one requirement: a layer stack, two devices, then sharding sweeps.

**Where it is used:** choosing batch sizes for inference, sizing a model to an accelerator, and picking tensor, pipeline or data parallelism all start from these back-of-envelope numbers.

Adapted from the matmul performance modelling question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The hand estimates become graded functions on a `Roofline` class with the hardware passed in, and the machines, shapes and layer counts are new. The written explanations are left out. The interview questions ask for them instead.""",
    "parts": [
        {
            "title": "One matmul",
            "description_en": r"""**Signature:** `Roofline(...)` with `ridge() -> float` and `matmul(m, k, n) -> dict`

- `ridge()` is `peak_flops / hbm_bandwidth`, the intensity at which both times are equal.
- `matmul` returns `flops`, `bytes`, `intensity` (FLOPs over bytes), `compute_time`, `memory_time`, `time`, and `bound`.
- `bound` is `"compute"` when compute time is at least memory time, else `"memory"`.

**Example:** `Roofline(400e12, 2e12, 24e9, 200e9)` has `ridge()` `200.0`:
- `matmul(16, 8192, 8192)` has `flops` `2147483648`, `bytes` `134742016`, intensity about `15.9`, `bound` `"memory"` and `time` `6.7371008e-05`
- `matmul(4096, 4096, 4096)` has intensity about `1365.3` and `bound` `"compute"`""",
        },
        {
            "title": "A layer stack on one device",
            "description_en": r"""Keep Part 1. Add `stack(layers, T, d, f) -> dict` for `layers` matmul layers run one after another, each its own kernel.

- Layer `i`, counting from 1, multiplies a `T×d` input by a `d×f` weight when `i` is odd, and a `T×f` input by an `f×d` weight when `i` is even. Each layer's output is the next one's input.
- `compute_time` sums every layer's FLOPs over `peak_flops`, `transfer_time` sums every layer's bytes over `hbm_bandwidth`, and `time` sums every layer's kernel time.
- `resident_bytes` holds every weight plus one layer's input and output: `s*d*f*layers + s*T*(d + f)`. `fits` is `resident_bytes <= hbm_capacity`, and `headroom` is `hbm_capacity - resident_bytes`, negative when it does not fit.

**Example:** on the Part 1 machine, `stack(40, 8192, 6144, 24576)`:
- `resident_bytes` is `12582912000`, `fits` is `True` and `headroom` is `11417088000`
- `compute_time` is about `0.2474` seconds""",
        },
        {
            "title": "Pipeline or tensor parallel",
            "description_en": r"""Keep Parts 1–2. Add `two_devices(layers, T, d, f) -> {"pipeline": {...}, "tensor": {...}}` for the Part 2 stack on two devices. Each scheme reports `latency`, `peak_bytes` (the larger of the two devices) and `sent_bytes` (the most either device sends).

- Pipeline: with `h = layers // 2`, the first device runs layers `1..h` and sends layer `h`'s output to the second device, which runs the rest. `latency` is the first device's summed kernel times, plus that output's bytes over `link_bandwidth`, plus the second device's summed kernel times.
- Pipeline peak memory on a device is its own layers' weights plus `s*T*(d + f)`.
- Tensor: each layer's weight is split by output columns, so each device computes a `T×k` by `k×(n/2)` matmul from the full input. Each device then sends its `T×(n/2)` half to the other while receiving the other half. `latency` sums, over layers, the half matmul's time plus the half's bytes over `link_bandwidth`.
- Tensor peak memory is half of every weight plus `s*T*(d + f)`.

**Example:** on the Part 1 machine, `two_devices(40, 8192, 6144, 24576)`:
- pipeline sends `100663296` bytes once; tensor sends `5033164800` bytes in total
- both peak at `6543114240` bytes
- pipeline latency is about `0.2479` seconds and tensor latency about `0.1489`""",
        },
        {
            "title": "Sharding a feed-forward block",
            "description_en": r"""Keep Parts 1–3. Add `ffn(T, d, f, p, scheme, link_bandwidth=None) -> dict` for the block `act(X W1) W2`, with `X` `T×d`, `W1` `d×f` and `W2` `f×d`, on `p` devices. The pointwise `act` costs nothing. `p` divides `T` and `f`.

- `"tensor"`: each device runs a `T×d` by `d×(f/p)` matmul, then a `T×(f/p)` by `(f/p)×d` one. It holds `2*s*d*f/p` weight bytes. An all-reduce then sends `comm_bytes = 2*(p-1)/p * T*d*s` from each device.
- `"token"`: each device runs a `(T/p)×d` by `d×f` matmul, then a `(T/p)×f` by `f×d` one. It holds `2*s*d*f` weight bytes and sends nothing.
- Any other scheme raises `ValueError`.
- Return `flops`, `weight_bytes`, `compute_time` and `memory_time` (each summed over the two matmuls), and `comm_bytes`. `comm_time` is `comm_bytes` over `link_bandwidth`, or over the argument when it is given.
- `latency` is the two kernel times plus `comm_time`, since the all-reduce cannot overlap. `bound` names the largest of compute, memory and communication time, as `"compute"`, `"memory"` or `"communication"`, with ties going to the earlier name.

**Example:** on the Part 1 machine, `ffn(2048, 6144, 24576, 8, ...)`:
- `"tensor"` holds `75497472` weight bytes, sends `44040192` bytes, is compute bound, and takes about `607` microseconds
- `"token"` holds `603979776` weight bytes, is compute bound, and takes about `387` microseconds
- `"tensor"` with `link_bandwidth=12.5e9` is communication bound""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "For a matmul with m = 16 rows, which term of the byte count dominates, and how does intensity change as m grows with k and n fixed? At what intensity are compute time and memory time equal on this machine?"},
        {"level": 2, "kind": "analysis", "content": "Intensity is 2mkn / s(mk + kn + mn). With m small, the kn weight term dominates the bytes, so intensity is about 2m/s and the kernel is memory bound. Compute both times and take the larger; the label follows from which one is larger, with a tie going to compute, which matches intensity >= ridge."},
    ],
    "model_connections": [
        "Decoding with a small batch is a memory-bound matmul, which is why batching requests raises throughput almost for free until the ridge point.",
        "Tensor parallelism inside a node and pipeline or data parallelism across nodes follow from comparing compute time with all-reduce time on each link.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Taking the larger of compute and memory time gives a quick lower bound that names the bottleneck.",
            "Counting resident tensors exactly tells you whether a model fits before you try it.",
            "Per-device FLOPs, bytes and communication make the trade-off between sharding schemes explicit.",
        ],
        "cons": [
            "Real kernels reach only part of peak FLOP/s and bandwidth, and small shapes pay launch overhead the model ignores.",
            "The model assumes perfect overlap inside a kernel and none between kernels or with communication.",
            "Caches, fusion and recomputation change the bytes moved in ways the formulas leave out.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).


class Roofline:
    def __init__(self, peak_flops, hbm_bandwidth, hbm_capacity, link_bandwidth, bytes_per_element=2):
        self.peak = peak_flops  # FLOP/s
        self.bw = hbm_bandwidth  # bytes/s
        self.capacity = hbm_capacity  # bytes
        self.link = link_bandwidth  # bytes/s sent by one device, full duplex
        self.s = bytes_per_element

    # -- part 1: one matmul
    def ridge(self):
        return self.peak / self.bw

    def matmul(self, m, k, n):
        flops = 2 * m * k * n
        moved = self.s * (m * k + k * n + m * n)  # read A and B once, write C once
        compute, memory = flops / self.peak, moved / self.bw
        return {"flops": flops, "bytes": moved, "intensity": flops / moved,
                "compute_time": compute, "memory_time": memory, "time": max(compute, memory),
                "bound": "compute" if compute >= memory else "memory"}

    # -- part 2: a stack of alternating layers on one device
    def _layer(self, i, T, d, f):
        return (T, d, f) if i % 2 == 1 else (T, f, d)  # odd layers widen d -> f, even ones narrow back

    def stack(self, layers, T, d, f):
        mms = [self.matmul(*self._layer(i, T, d, f)) for i in range(1, layers + 1)]
        weights = self.s * d * f * layers
        resident = weights + self.s * T * (d + f)  # every weight, plus one layer's input and output
        return {"compute_time": sum(x["flops"] for x in mms) / self.peak,
                "transfer_time": sum(x["bytes"] for x in mms) / self.bw,
                "time": sum(x["time"] for x in mms),
                "resident_bytes": resident, "fits": resident <= self.capacity,
                "headroom": self.capacity - resident}

    # -- part 3: two devices, pipeline against tensor parallel
    def two_devices(self, layers, T, d, f):
        half = layers // 2
        width = f if half % 2 == 1 else d  # the activation leaving layer `half`
        first, second = self._kernels(1, half, T, d, f), self._kernels(half + 1, layers, T, d, f)
        handoff = self.s * T * width
        pipeline = {"latency": first + handoff / self.link + second,
                    "peak_bytes": self.s * d * f * (layers - half) + self.s * T * (d + f),
                    "sent_bytes": handoff}
        latency = sent = 0
        for i in range(1, layers + 1):
            _, k, n = self._layer(i, T, d, f)
            shard = self.matmul(T, k, n // 2)  # this device's half of the output columns
            out = self.s * T * (n // 2)  # sent to the other device, received from it at the same time
            latency += shard["time"] + out / self.link
            sent += out
        tensor = {"latency": latency,
                  "peak_bytes": self.s * (d * f // 2) * layers + self.s * T * (d + f),
                  "sent_bytes": sent}
        return {"pipeline": pipeline, "tensor": tensor}

    def _kernels(self, start, stop, T, d, f):
        """Summed kernel time of layers start..stop on one device."""
        return sum(self.matmul(*self._layer(i, T, d, f))["time"] for i in range(start, stop + 1))

    # -- part 4: sharding one feed-forward block over p devices
    def ffn(self, T, d, f, p, scheme, link_bandwidth=None):
        link = self.link if link_bandwidth is None else link_bandwidth
        if scheme == "tensor":
            mms = [self.matmul(T, d, f // p), self.matmul(T, f // p, d)]
            weight_bytes = 2 * self.s * d * f // p
            comm_bytes = 2 * (p - 1) * T * d * self.s / p  # ring all-reduce of the T x d partial outputs
        elif scheme == "token":
            mms = [self.matmul(T // p, d, f), self.matmul(T // p, f, d)]
            weight_bytes = 2 * self.s * d * f
            comm_bytes = 0
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        compute = sum(x["compute_time"] for x in mms)
        memory = sum(x["memory_time"] for x in mms)
        comm = comm_bytes / link
        times = {"compute": compute, "memory": memory, "communication": comm}
        return {"flops": sum(x["flops"] for x in mms), "weight_bytes": weight_bytes,
                "compute_time": compute, "memory_time": memory,
                "comm_bytes": comm_bytes, "comm_time": comm,
                "latency": sum(x["time"] for x in mms) + comm,  # the all-reduce cannot overlap
                "bound": max(times, key=times.get)}  # ties go to the earlier name
''',
    "interview_questions": interview(
        concept=[
            "What is arithmetic intensity, and why does a matmul with few rows end up memory bound?",
            "Can growing m, k or n alone ever move a matmul from compute bound to memory bound? Argue for each of the three.",
        ],
        deep_dive=[
            "Why is the predicted time the larger of compute time and memory time rather than their sum, and when is that wrong?",
        ],
        tradeoffs=[
            "Which tensors must be resident to run a layer stack, and what can you drop to fit a model that does not?",
            "For a single request on two devices, why does tensor parallel beat a pipeline, and why are pipelines used anyway?",
            "As p grows, when does tensor sharding of a feed-forward block become communication bound, and how does a slower link change that?",
            "What would let the all-reduce overlap with compute in a real model, given that it cannot within one block?",
        ],
    ),
}
