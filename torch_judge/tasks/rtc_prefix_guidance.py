"""Real-Time Chunking: prefix weights over the frozen part of a chunk, then guided flow-matching velocity."""

from ._interview import interview

# An independent model of both parts, written from the paper's closed forms rather than the
# linspace construction the reference uses.
_MODEL = r"""
import math, random, torch

def want_weights(start, end, total, schedule):
    end = min(end, total)
    start = min(start, end)
    out = []
    for i in range(total):
        if schedule == "ones":
            w = 1.0 if i < end else 0.0
        elif schedule == "zeros":
            w = 1.0 if i < start else 0.0
        elif i < start:
            w = 1.0
        elif i >= end:
            w = 0.0
        else:
            w = (end - i) / (end - start + 1)
            if schedule == "exp":
                w = w * math.expm1(w) / (math.e - 1)
        out.append(w)
    return torch.tensor(out, dtype=torch.float64)

def want_gain(time, max_w):
    tau = 1.0 - time
    if tau == 0.0:
        return max_w
    if tau == 1.0:
        return 0.0
    inv_r2 = ((1 - tau) ** 2 + tau ** 2) / (1 - tau) ** 2
    return min((1 - tau) / tau * inv_r2, max_w)

def padded(prev, steps):
    out = torch.zeros(prev.shape[:-2] + (steps, prev.shape[-1]), dtype=prev.dtype)
    out[..., : prev.shape[-2], :] = prev
    return out

def guided():
    g = {fn}.__globals__.get("guided_velocity")
    assert callable(g), "define guided_velocity next to prefix_weights"
    return g
"""

TESTS = [
    {"name": "Part 1: the four schedules on start=2, end=6, total=10", "part": 1, "behavior": "attention.masking", "code": _MODEL + r"""
lin = {fn}(2, 6, 10, "linear")
assert lin.shape == (10,), lin.shape
assert torch.allclose(lin.double(), torch.tensor([1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], dtype=torch.float64), atol=1e-6), lin
assert {fn}(2, 6, 10, "zeros").tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
assert {fn}(2, 6, 10, "ones").tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
exp = {fn}(2, 6, 10, "exp")
assert torch.allclose(exp.double(), want_weights(2, 6, 10, "exp"), atol=1e-6), exp
"""},
    {"name": "Part 1: random schedules match the closed form", "part": 1, "visibility": "unshown", "behavior": "attention.masking",
     "failure_message": "The ramp over start <= i < end must fall from 1 toward 0 without reaching either end, (end - i) / (end - start + 1), and exp multiplies it by expm1(w) / (e - 1).",
     "code": _MODEL + r"""
rng = random.Random(7)
for _ in range(200):
    total = rng.randint(1, 16)
    start, end = rng.randint(0, total), rng.randint(0, total)
    schedule = rng.choice(["zeros", "ones", "linear", "exp"])
    got = {fn}(start, end, total, schedule)
    assert got.shape == (total,), (start, end, total, schedule, got.shape)
    assert torch.allclose(got.double(), want_weights(start, end, total, schedule), atol=1e-6), (start, end, total, schedule, got)
"""},
    {"name": "Part 1: start past end, end past total, empty ramps", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Clamp end to total first, then start to end, so a late start never outlives end and an end past the chunk ramps only inside it.",
     "code": _MODEL + r"""
cases = [(5, 3, 8), (2, 12, 10), (12, 15, 10), (0, 0, 6), (0, 6, 6), (3, 3, 6), (6, 9, 6), (4, 2, 1), (0, 4, 4)]
for start, end, total in cases:
    for schedule in ("zeros", "ones", "linear", "exp"):
        got = {fn}(start, end, total, schedule)
        assert got.shape == (total,), (start, end, total, schedule, got.shape)
        assert torch.allclose(got.double(), want_weights(start, end, total, schedule), atol=1e-6), (start, end, total, schedule, got)
assert {fn}(0, 3, 0, "linear").shape == (0,)
"""},
    {"name": "Part 1: float32 output and an unknown schedule", "part": 1, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "Return a float32 tensor of shape (total,), and raise ValueError for a schedule other than zeros, ones, linear or exp.",
     "code": _MODEL + r"""
for schedule in ("zeros", "ones", "linear", "exp"):
    got = {fn}(1, 4, 7, schedule)
    assert isinstance(got, torch.Tensor) and got.dtype == torch.float32 and not got.requires_grad, (schedule, got)
for bad in ("cosine", "LINEAR", ""):
    try:
        {fn}(1, 4, 7, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted schedule {bad!r}")
"""},
    {"name": "Part 2: guidance pulls the committed prefix toward the previous chunk", "part": 2, "behavior": "gradient.flow", "code": _MODEL + r"""
g = guided()
x_t = torch.zeros(5, 1)
v_fn = lambda x: torch.ones_like(x)
prev = torch.full((3, 1), 2.0)
out = g(x_t, v_fn, prev, 0.5, 1, 3, 10.0, "linear")
# x1 = 0 - 0.5 * 1 = -0.5; weights [1, 2/3, 1/3, 0, 0]; error 2.5 * w; gain 2 at time 0.5.
assert out.shape == (5, 1), out.shape
assert torch.allclose(out.flatten(), torch.tensor([-4.0, -7 / 3, -2 / 3, 1.0, 1.0]), atol=1e-5), out.flatten()
assert torch.equal(g(x_t, v_fn, None, 0.5, 1, 3, 10.0, "linear"), torch.ones(5, 1))
"""},
    {"name": "Part 2: the vector-Jacobian product goes through the velocity", "part": 2, "visibility": "unshown", "behavior": "gradient.flow",
     "failure_message": "Make x_t require grad before calling v_fn, so the correction is the error times the Jacobian of x1 = x_t - time * v, not the error alone.",
     "code": _MODEL + r"""
g = guided()
gen = torch.Generator().manual_seed(3)
for seed in range(5):
    B, T, A = 2, 6, 3
    M = torch.randn(A, A, generator=gen, dtype=torch.float64)
    b = torch.randn(A, generator=gen, dtype=torch.float64)
    x_t = torch.randn(B, T, A, generator=gen, dtype=torch.float64)
    prev = torch.randn(B, 4, A, generator=gen, dtype=torch.float64)
    time, max_w = 0.3 + 0.1 * seed, 50.0
    for kind in ("linear", "tanh"):
        v_fn = (lambda x: x @ M + b) if kind == "linear" else (lambda x: torch.tanh(x @ M) + b)
        out = g(x_t, v_fn, prev, time, 2, 5, max_w, "exp")
        v = v_fn(x_t)
        w = want_weights(2, 4, T, "exp")[:, None]
        err = (padded(prev, T) - (x_t - time * v)) * w
        if kind == "linear":
            jvp = err @ M.T
        else:
            jvp = (err * (1 - torch.tanh(x_t @ M) ** 2)) @ M.T
        want = v - want_gain(time, max_w) * (err - time * jvp)
        assert torch.allclose(out, want, rtol=1e-5, atol=1e-8), (seed, kind, (out - want).abs().max())
"""},
    {"name": "Part 2: guidance weight over time, with the clamp and both endpoints", "part": 2, "visibility": "unshown", "behavior": "numerics.stability",
     "failure_message": "Use tau = 1 - time, weight min((1 - tau) / tau * inv_r2, max_guidance_weight), max_guidance_weight at time 1 and 0 at time 0, never nan.",
     "code": _MODEL + r"""
g = guided()
x_t = torch.zeros(4, 2, dtype=torch.float64)
v_fn = lambda x: torch.zeros_like(x) + 0.25
prev = torch.ones(4, 2, dtype=torch.float64)
for max_w in (0.5, 3.0, 100.0):
    for time in (0.0, 0.05, 0.2, 0.5, 0.7, 0.9, 0.99, 1.0):
        out = g(x_t, v_fn, prev, time, 4, 4, max_w, "ones")
        assert torch.isfinite(out).all(), (time, max_w, out)
        err = 1.0 + time * 0.25
        gain = (0.25 - out[0, 0].item()) / err
        assert math.isclose(gain, want_gain(time, max_w), rel_tol=1e-5, abs_tol=1e-6), (time, max_w, gain, want_gain(time, max_w))
"""},
    {"name": "Part 2: a short previous chunk is zero-padded and caps the horizon", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "Zero-pad prev_chunk at the end up to the chunk length and use min(execution_horizon, prev steps) as the end of the weights.",
     "code": _MODEL + r"""
g = guided()
gen = torch.Generator().manual_seed(11)
x_t = torch.randn(8, 2, generator=gen, dtype=torch.float64)
prev = torch.randn(3, 2, generator=gen, dtype=torch.float64) + 3.0
v_fn = lambda x: torch.zeros_like(x) - 1.0
time, max_w = 0.6, 10.0
out = g(x_t, v_fn, prev, time, 1, 6, max_w, "linear")
x1 = x_t + time
err = (padded(prev, 8) - x1) * want_weights(1, 3, 8, "linear")[:, None]
want = -1.0 - want_gain(time, max_w) * err
assert torch.allclose(out, want, rtol=1e-5, atol=1e-8), (out, want)
assert torch.equal(out[3:], torch.full((5, 2), -1.0, dtype=torch.float64)), out
"""},
    {"name": "Part 2: batched seeded sweep against an independent model", "part": 2, "visibility": "unshown", "behavior": "tensor.shape",
     "failure_message": "Broadcast the (T,) weights over batch and action dims, keep the dtype and shape of x_t, and pass the schedule through to prefix_weights.",
     "code": _MODEL + r"""
g = guided()
rng = random.Random(5)
gen = torch.Generator().manual_seed(5)
for _ in range(12):
    B, T, A = rng.randint(1, 3), rng.randint(2, 9), rng.randint(1, 4)
    dtype = rng.choice([torch.float32, torch.float64])
    P = rng.randint(1, T)
    d, h = rng.randint(0, T), rng.randint(0, T + 2)
    schedule = rng.choice(["zeros", "ones", "linear", "exp"])
    time, max_w = rng.uniform(0.05, 0.95), rng.choice([1.0, 5.0, 10.0])
    scale = torch.randn(A, generator=gen, dtype=torch.float64).to(dtype)
    x_t = torch.randn(B, T, A, generator=gen, dtype=torch.float64).to(dtype)
    prev = torch.randn(B, P, A, generator=gen, dtype=torch.float64).to(dtype)
    v_fn = lambda x: x * scale
    out = g(x_t, v_fn, prev, time, d, h, max_w, schedule)
    assert out.shape == x_t.shape and out.dtype == dtype, (out.shape, out.dtype)
    x, pv = x_t.double(), padded(prev, T).double()
    s = scale.double()
    err = (pv - (x - time * x * s)) * want_weights(d, min(h, P), T, schedule)[:, None]
    want = x * s - want_gain(time, max_w) * err * (1 - time * s)
    tol = 1e-4 if dtype == torch.float32 else 1e-6
    assert torch.allclose(out.double(), want, rtol=tol, atol=tol), (B, T, A, P, d, h, schedule, (out.double() - want).abs().max())
"""},
    {"name": "Part 2: works under no_grad and leaves the caller's tensors alone", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Enable grad locally, differentiate a detached copy of x_t, call v_fn once, and return a detached result.",
     "code": _MODEL + r"""
g = guided()
gen = torch.Generator().manual_seed(2)
x_t = torch.randn(1, 5, 2, generator=gen)
before = x_t.clone()
prev = torch.randn(1, 5, 2, generator=gen)
calls = []
def v_fn(x):
    calls.append(1)
    return torch.sin(x)
with torch.no_grad():
    out = g(x_t, v_fn, prev, 0.4, 2, 4, 10.0, "linear")
assert len(calls) == 1, f"v_fn called {len(calls)} times"
assert not out.requires_grad and out.grad_fn is None, out
assert not x_t.requires_grad and torch.equal(x_t, before), x_t
calls.clear()
again = g(x_t, v_fn, prev, 0.4, 2, 4, 10.0, "linear")
assert not again.requires_grad and again.grad_fn is None, "return a detached tensor outside no_grad too"
assert torch.equal(again, out) and not x_t.requires_grad, again
calls.clear()
plain = g(x_t, v_fn, None, 0.4, 2, 4, 10.0, "linear")
assert len(calls) == 1 and torch.equal(plain, torch.sin(x_t)), plain
"""},
    {"name": "Part 2: rejects a previous chunk that does not fit", "part": 2, "visibility": "unshown", "behavior": "contract.signature",
     "failure_message": "Raise ValueError when prev_chunk has more steps than x_t, a different action dim, or a different number of dims.",
     "code": _MODEL + r"""
g = guided()
x_t = torch.zeros(2, 4, 3)
v_fn = lambda x: x
for prev in (torch.zeros(2, 5, 3), torch.zeros(2, 4, 2), torch.zeros(4, 3)):
    try:
        g(x_t, v_fn, prev, 0.5, 1, 3, 10.0, "linear")
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted prev_chunk of shape {tuple(prev.shape)}")
"""},
]

TASK = {
    "title": "Real-Time Chunking Prefix Guidance",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "prefix_weights",
    "description_en": r"""Implement Real-Time Chunking (RTC): while a robot executes the previous action chunk, a flow-matching policy denoises the next one, and guidance keeps the new chunk consistent with the actions that are already committed.

The requirement arrives in parts. Each part keeps every earlier behavior, so one solution passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A chunk has `T` steps. Step `i` counts from 0.
- Define both functions at module level in one solution. Part 2 calls `prefix_weights` from Part 1.
- Use only torch and the standard library.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The problem RTC solves.** A VLA such as pi0 predicts a chunk of H actions per inference call, but inference takes d control steps. By the time the new chunk is ready, the robot has already executed d more actions of the old chunk, so the first d actions of the new chunk are fixed. Naively switching chunks causes jerks at the boundary; RTC treats the switch as inpainting: the frozen prefix is known, the rest is generated to agree with it.

**Two pieces.** A per-step weight says how strongly each step of the new chunk must match the leftover of the previous chunk: fully for the committed steps, fading over the overlap, not at all after it. Pseudo-inverse guidance (PiGDM) then nudges the flow-matching velocity at every denoising step so the predicted clean chunk moves toward the previous one where the weights say so.

**Where it is used.** Physical Intelligence's RTC paper evaluates it on Kinetix and real pi0.5 robots. LeRobot ships it as `RTCProcessor` for pi0, pi0.5 and SmolVLA, with the same four schedules.""",
    "parts": [
        {
            "title": "Prefix weights",
            "description_en": r"""**Signature:** `prefix_weights(start, end, total, schedule) -> Tensor`

**Parameters:** non-negative ints `start`, `end`, `total`, and `schedule`, one of `"zeros"`, `"ones"`, `"linear"`, `"exp"`.

**Returns:** a float32 tensor of shape `(total,)`.

**Constraints:**
- Clamp first: `end = min(end, total)`, then `start = min(start, end)`.
- `"zeros"`: 1 for `i < start`, else 0.
- `"ones"`: 1 for `i < end`, else 0.
- `"linear"`: 1 for `i < start`, 0 for `i >= end`, and in between a ramp that falls toward 0 without touching 1 or 0: `w_i = (end - i) / (end - start + 1)`.
- `"exp"`: like `"linear"`, but each ramp value becomes `w * expm1(w) / (e - 1)`. The 1s and 0s stay as they are.
- Raise `ValueError` for any other schedule.

**Example:** `prefix_weights(2, 6, 10, "linear")` is

    [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0]

`"zeros"` gives `[1, 1, 0, ...]` and `"ones"` gives six 1s followed by four 0s.""",
        },
        {
            "title": "Guided velocity",
            "description_en": r"""Keep Part 1. Add:

**Signature:** `guided_velocity(x_t, v_fn, prev_chunk, time, inference_delay, execution_horizon, max_guidance_weight, schedule) -> Tensor`

**Parameters:**
- `x_t`: the noisy chunk, float tensor `(T, A)` or `(B, T, A)`.
- `v_fn`: a callable that maps a tensor shaped like `x_t` to the velocity, same shape.
- `prev_chunk`: the unexecuted leftover of the previous chunk, with the same number of dims as `x_t`, `P <= T` steps and the same action dim `A`; or `None`.
- `time`: a float in `[0, 1]`. Noise is at `time = 1` and data at `time = 0`, so `x_t - time * v` predicts the clean chunk.
- `inference_delay`, `execution_horizon`: ints, the `start` and `end` of the prefix weights.
- `max_guidance_weight`: a positive float. `schedule`: as in Part 1.

**Returns:** the guided velocity, with the shape and dtype of `x_t`, detached.

**Constraints:**
- If `prev_chunk` is `None`, return `v_fn(x_t)`.
- Raise `ValueError` if `prev_chunk` has more steps than `x_t`, a different action dim, or a different number of dims.
- Zero-pad `prev_chunk` at the end of the time axis to `T` steps.
- Weights: `prefix_weights(inference_delay, min(execution_horizon, P), T, schedule)`, broadcast over batch and action dims.
- Call `v_fn` exactly once, on a detached copy of `x_t` that requires grad, with grad enabled even if the caller is inside `torch.no_grad()`.
- Predict `x1 = x - time * v`. The error is `(prev - x1) * weights`, treated as a constant.
- The correction is the vector-Jacobian product of `x1` with respect to `x` against the error, from `torch.autograd.grad`. It includes the Jacobian of `v_fn`.
- With `tau = 1 - time`: `inv_r2 = ((1 - tau)^2 + tau^2) / (1 - tau)^2`, `c = (1 - tau) / tau`, and the guidance weight is `min(c * inv_r2, max_guidance_weight)`.
- At `time = 1` the guidance weight is `max_guidance_weight`. At `time = 0` it is 0. Never return nan.
- Return `v - weight * correction`. Do not modify `x_t`.

**Example:** with `x_t = zeros(5, 1)`, `v_fn = lambda x: torch.ones_like(x)`, `prev_chunk = full((3, 1), 2.0)`, `time = 0.5`, delay 1, horizon 3, max weight 10 and `"linear"`: the weights are `[1, 2/3, 1/3, 0, 0]`, `x1 = -0.5`, the guidance weight is 2, and the result is

    [-4, -7/3, -2/3, 1, 1]""",
        },
    ],
    "advisory_prerequisites": ["flow_matching", "ddim_step", "pi0_flow_matching_chunk", "act_temporal_ensemble"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which steps of the new chunk can no longer change, and which should merely lean toward the old chunk? If x_t does not require grad when v_fn runs, what does autograd.grad of x1 return? What happens to (1 - tau) / tau and inv_r2 at tau = 0 and tau = 1?"},
        {"level": 2, "kind": "analysis", "content": "Part 1: clamp end, then start, and concatenate three pieces: the committed ones, a ramp that excludes both of its endpoints, and the zeros. The exp schedule reshapes that same ramp. Part 2: the correction is a vector-Jacobian product, so the velocity must be computed from a copy of x_t that requires grad, with grad enabled even when the caller disabled it. Treat the error as a constant, and compute the guidance weight on tensors so that infinities and nan can be cleaned before the clamp."},
    ],
    "model_connections": [
        "LeRobot's RTCProcessor.denoise_step wraps the pi0, pi0.5 and SmolVLA denoisers with exactly this guidance, using the same four prefix schedules.",
        "Physical Intelligence's Kinetix reference computes the correction with jax.vjp through the whole denoiser, the same product this exercise takes with torch.autograd.grad.",
        "PiGDM (pseudo-inverse guided diffusion) supplies the r_tau^2 weighting; RTC adds the clamp at max_guidance_weight for few-step sampling.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Training-free: any flow-matching chunk policy gets smooth chunk transitions at inference time.",
            "The soft exp ramp keeps continuity past the frozen prefix without pinning the whole overlap.",
        ],
        "cons": [
            "Each denoising step needs a backward pass through the action expert, roughly doubling the sampling cost.",
            "The weight diverges near pure noise, so results depend on the max_guidance_weight clamp, a tuned hyperparameter.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot/blob/75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee/src/lerobot/policies/rtc/modeling_rtc.py",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/rtc/modeling_rtc.py",
            "symbol": "RTCProcessor.get_prefix_weights, RTCProcessor.denoise_step",
            "license": "Apache-2.0",
            "adapted": "Prefix weights and the guided velocity follow get_prefix_weights and denoise_step, with the config fields passed as arguments and lowercase schedule names.",
            "simplifications": "Upstream calls the denoiser before x_t requires grad, so its correction drops the Jacobian of the velocity and equals the weighted error; "
                               "the exercise differentiates through v_fn as the paper and Kinetix do. "
                               "Upstream also zero-pads a narrower action dim and pads in float32; the exercise rejects a different action dim and keeps the dtype of x_t. "
                               "An unknown schedule raises ValueError instead of UnboundLocalError. The debug tracker is dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py",
            "commit": "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b",
            "path": "src/model.py",
            "symbol": "get_prefix_weights, FlowPolicy.realtime_action",
            "license": "MIT",
            "adapted": "Cross-check for the closed-form ramp and the vjp-based correction, with time flipped to noise at 1 (tau = 1 - time, v negated).",
            "simplifications": "Kinetix keeps the ramp slope from the unclamped end, (end - i) / (end - start + 1), when end > total; "
                               "the exercise follows LeRobot and clamps end to total first, so the ramp reaches toward 0 inside the chunk. "
                               "The simulated-delay training path and bidirectional decoding are left out.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2506.07339",
            "section": "Black, Galliker and Levine 2025, Section 3.1 Eqs. 2-4 (guided velocity, clamp beta, r_tau^2) and Section 3.2 Eq. 5 (soft-masking weights); Algorithm 1",
            "note": "Real-Time Execution of Action Chunking Flow Policies: the RTC guidance and prefix weights.",
        },
    ],
    "tests": TESTS,
    "solution": '''import math
import torch


def prefix_weights(start, end, total, schedule):
    if schedule not in ("zeros", "ones", "linear", "exp"):
        raise ValueError(f"unknown schedule {schedule!r}")
    end = min(end, total)
    start = min(start, end)
    steps = torch.arange(total)
    if schedule == "zeros":
        return (steps < start).float()
    if schedule == "ones":
        return (steps < end).float()
    ramp = torch.linspace(1, 0, end - start + 2)[1:-1]
    if schedule == "exp":
        ramp = ramp * torch.expm1(ramp) / (math.e - 1)
    return torch.cat([torch.ones(start), ramp, torch.zeros(total - end)])


def guided_velocity(x_t, v_fn, prev_chunk, time, inference_delay, execution_horizon,
                    max_guidance_weight, schedule):
    if prev_chunk is None:
        return v_fn(x_t)
    steps, prev_steps = x_t.shape[-2], prev_chunk.shape[-2]
    if prev_chunk.dim() != x_t.dim() or prev_chunk.shape[-1] != x_t.shape[-1] or prev_steps > steps:
        raise ValueError("prev_chunk must match x_t except for fewer steps")
    horizon = min(execution_horizon, prev_steps)
    target = torch.nn.functional.pad(prev_chunk.to(x_t.dtype), (0, 0, 0, steps - prev_steps))
    weights = prefix_weights(inference_delay, horizon, steps, schedule).to(x_t)[:, None]

    with torch.enable_grad():
        x = x_t.detach().clone().requires_grad_(True)
        v = v_fn(x)
        x1 = x - time * v
        err = ((target - x1) * weights).detach()
        correction = torch.autograd.grad(x1, x, err)[0]

    tau = torch.tensor(1.0 - float(time), dtype=torch.float64)
    max_w = float(max_guidance_weight)
    inv_r2 = ((1 - tau) ** 2 + tau ** 2) / (1 - tau) ** 2
    c = torch.nan_to_num((1 - tau) / tau, posinf=max_w)
    gain = torch.nan_to_num(c * inv_r2, nan=0.0, posinf=max_w)
    gain = torch.clamp(gain, max=max_w)
    return (v - gain.to(x_t.dtype) * correction).detach()
''',
    "interview_questions": interview(
        concept=[
            "What problem does Real-Time Chunking solve when a policy predicts action chunks but inference takes several control steps?",
            "Why are the first inference_delay steps of the new chunk weighted 1, and why does the weight fade instead of dropping straight to 0 after them?",
        ],
        deep_dive=[
            "Walk through one guided denoising step: what is x1, what is the error, and what does the vector-Jacobian product compute?",
            "Why must x_t require grad before the velocity network runs, and what does the correction collapse to if it does not?",
            "How does the guidance weight behave as tau approaches 0 and 1, and how do nan_to_num and the clamp keep it finite?",
        ],
        tradeoffs=[
            "RTC guidance versus hard inpainting that overwrites the prefix after every denoising step: smoothness, cost and failure modes?",
            "RTC versus ACT-style temporal ensembling of overlapping chunks: latency, compute and action quality?",
        ],
    ),
}
