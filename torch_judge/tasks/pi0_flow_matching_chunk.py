"""pi0 flow matching on action chunks: training inputs and Euler sampling in both time conventions."""

from ._interview import interview

# A velocity field that couples horizon positions, so a missed clamp changes every output.
_MIXING_FIELD = r"""
import torch

def mixing_field(B, H, D, seed, log):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(H, H, generator=g, dtype=torch.float64) / H
    bias = torch.randn(1, H, D, generator=g, dtype=torch.float64) * 0.3
    def v(x, t):
        log.append((x.clone(), t.clone()))
        tt = t.reshape(t.shape[0], -1, 1)
        return torch.einsum("hk,bkd->bhd", W, torch.tanh(x)) * (1 + tt) + bias
    return v

def manual_sample(v, noise, n, convention, prefix=None, mask=None):
    dt = (-1.0 if convention == "noise_at_one" else 1.0) / n
    clean = 0.0 if convention == "noise_at_one" else 1.0
    B, H = noise.shape[:2]
    x = noise.clone()
    for k in range(n):
        t = torch.full((B,), (1.0 + k * dt) if convention == "noise_at_one" else k * dt, dtype=noise.dtype)
        if prefix is not None:
            x[mask] = prefix[mask]
            t = t[:, None].expand(B, H).clone()
            t[mask] = clean
        x = x + dt * v(x, t)
        if prefix is not None:
            x[mask] = prefix[mask]
    return x
"""

# The exact conditional velocity toward a known chunk: Euler integration of it is exact.
_EXACT_FIELD = r"""
import torch

def exact_field(actions, convention, log=None):
    def v(x, t):
        if log is not None:
            log.append(t.clone())
        tt = t.reshape(t.shape[0], -1, 1)
        if convention == "noise_at_one":
            return (x - actions) / tt
        return (actions - x) / (1 - tt)
    return v
"""

TASK = {
    "title": "pi0 Flow Matching on Action Chunks",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "Pi0FlowMatching",
    "description_en": r"""Build the two halves of a pi0-style flow-matching action head for a chunk of `H` future actions: the training inputs, and the Euler sampler that turns noise into a chunk. Both work in either time convention.

**Signature:** `Pi0FlowMatching()` with two methods:
- `make_inputs(actions, noise, time, convention, prefix_mask=None) -> (x_t, v_target, model_time)`
- `euler_sample(velocity_fn, noise, num_steps, convention, hard_prefix=None, hard_prefix_mask=None) -> Tensor`

**Conventions.** `convention` is the string `"noise_at_one"` or `"noise_at_zero"`. Any other value raises `ValueError`.
- `"noise_at_one"` (pi0, openpi): `x_t = t * noise + (1 - t) * actions` and `v_target = noise - actions`. Sampling runs from `t = 1` to `t = 0`.
- `"noise_at_zero"` (the existing `flow_matching` exercise): `x_t = (1 - t) * noise + t * actions` and `v_target = actions - noise`. Sampling runs from `t = 0` to `t = 1`.
- The clean time is the end that holds the actions: `0.0` for `"noise_at_one"`, `1.0` for `"noise_at_zero"`.

**make_inputs:**
- `actions` and `noise` are float tensors of shape `(B, H, D)`. `time` has shape `(B,)`. `prefix_mask` is an optional bool tensor of shape `(B, H)`.
- Without a prefix mask, `model_time` is `time` itself, shape `(B,)`.
- With a prefix mask, `model_time` has shape `(B, H)`: the clean time where the mask is true, `time[b]` elsewhere. `x_t` uses this per-position time, so prefix positions equal `actions` exactly.
- `v_target` is the full velocity everywhere, prefix positions included.
- Raise `ValueError` when `actions` is not 3-D, when `noise` does not have the shape of `actions`, when `time` is not `(B,)`, or when `prefix_mask` is not `(B, H)`.

**euler_sample:**
- Start from `x = noise`, shape `(B, H, D)`, and take `num_steps = N` Euler steps `x = x + dt * velocity_fn(x, time)`.
- `"noise_at_one"`: `dt = -1 / N` and step `k` (from 0) uses `t_k = 1 + k * dt`. `"noise_at_zero"`: `dt = 1 / N` and `t_k = k * dt`.
- `time` passed to `velocity_fn` is a tensor of shape `(B,)` filled with `t_k`, with the dtype and device of `noise`.
- With a hard prefix, `time` has shape `(B, H)` instead: the clean time where `hard_prefix_mask` is true, `t_k` elsewhere. Before each velocity call and after each update, set `x` to `hard_prefix` wherever the mask is true (all `D` values).
- `hard_prefix` has the shape of `noise`, and `hard_prefix_mask` is a bool tensor of shape `(B, H)`. Pass both or neither.
- Raise `ValueError` when `num_steps` is not a positive integer, when `noise` is not 3-D, when only one of `hard_prefix` and `hard_prefix_mask` is given, or when their shapes are wrong.
- Never modify `noise`, `actions` or `hard_prefix` in place.

**Example:**

    fm = Pi0FlowMatching()
    actions = torch.tensor([[[1.0], [2.0]]]); noise = torch.tensor([[[0.0], [4.0]]])
    fm.make_inputs(actions, noise, torch.tensor([0.25]), "noise_at_one")
    # x_t = [[[0.75], [2.5]]], v_target = [[[-1.0], [2.0]]], model_time = [0.25]
    fm.euler_sample(lambda x, t: noise - actions, noise, 4, "noise_at_one")
    # times seen: 1.0, 0.75, 0.5, 0.25; returns actions

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why two conventions.** Both describe the same straight path between noise and data and differ only in the direction of time. The pi0 paper writes `A_tau = tau * A + (1 - tau) * eps` (noise at 0), while openpi and every LeRobot port of it (pi0, pi0.5, SmolVLA) put the noise at `t = 1` and integrate down to 0, the convention common in the diffusion literature. GR00T-style heads keep the noise at 0. A checkpoint is tied to its convention: training with one and sampling with the other makes the network push along the wrong direction.

**Clean-sample identity.** In `"noise_at_one"` the clean chunk is `x_t - t * v`; in `"noise_at_zero"` it is `x_t + (1 - t) * v`. Euler sampling with the exact straight-line velocity therefore recovers the chunk in any number of steps; pi0 uses 10.

**Prefixes.** Real-time chunking executes the start of a chunk while the rest is still being generated, so the first actions are already committed. Training marks those positions as clean (time at the clean end, so they are never noised), and sampling clamps them to the committed values. Clamping only after a step lets the first velocity call see noise at the committed positions; clamping only before leaves the last update applied to them.

**Training times.** openpi draws `t = Beta(1.5, 1) * 0.999 + 0.001`, which favors noisy times and never hits the clean end exactly. That sampler is outside this exercise; `time` is given.""",
    "advisory_prerequisites": ["flow_matching", "ddim_step"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which end of the time interval holds the noise in each convention, and so which time is clean? Where must the per-position time replace the per-sample time, and at which two moments in each Euler step does the prefix need to be written back?"},
        {"level": 2, "kind": "analysis", "content": "Validate shapes first. In make_inputs, expand time to (B, H) with torch.where(prefix_mask, clean_time, time[:, None]) when a mask is given, then interpolate with time[..., None]. In euler_sample, compute dt = -1/N or +1/N and t_k from k, build the time tensor in noise's dtype, clamp with torch.where(mask[:, :, None], hard_prefix, x) before the call and after x = x + dt * v."},
    ],
    "model_connections": [
        "openpi's Pi0.compute_loss and Pi0.sample_actions use noise at t = 1, dt = -1/num_steps and 10 steps by default.",
        "LeRobot's make_flow_matching_inputs and euler_integrate serve pi0, pi0.5, SmolVLA and EO-1 (noise at one) and GR00T, EVO-1 and WALL-X (noise at zero) from one implementation.",
        "Real-time chunking (rtc_prefix_guidance) builds on the hard-prefix clamp and adds a guidance term toward the previous chunk.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A continuous chunk head models multimodal action distributions without discretization error, and a few Euler steps are cheap enough for 50 Hz control.",
            "Clamping a committed prefix lets sampling continue a chunk that is already executing, with no retraining.",
        ],
        "cons": [
            "The convention is implicit in the checkpoint; mixing them silently produces actions that move away from the data.",
            "Euler integration of a learned, curved field has discretization error that only more steps or a better solver remove.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/common/flow_matching.py",
            "symbol": "FlowConvention, make_flow_matching_inputs, euler_integrate",
            "license": "Apache-2.0",
            "adapted": "Training-input construction and the uniform-schedule Euler loop with the hard-prefix clamp, both conventions.",
            "simplifications": "Conventions are strings and an unknown one raises ValueError (upstream: a FlowConvention enum and TypeError); "
                               "noise must have the shape of actions rather than any broadcastable shape; "
                               "the sampler passes time in noise's dtype rather than always float32; "
                               "hard_prefix_mask is (B, H) rather than shaped like noise with the time taken from mask[..., 0]; "
                               "a mask without a prefix raises instead of being ignored; "
                               "RTC guidance, time_grid, velocity_scale and step_aware are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/Physical-Intelligence/openpi",
            "commit": "215abfb217dbac7d5f1273282331b9b1866c0479",
            "path": "src/openpi/models/pi0.py",
            "symbol": "Pi0.compute_loss, Pi0.sample_actions",
            "license": "Apache-2.0",
            "adapted": "The noise-at-one interpolation, velocity target and dt = -1/num_steps schedule, transcribed from JAX.",
            "simplifications": "openpi accumulates time as time + dt inside a while_loop; the exercise computes t_k = 1 + k * dt, which takes the same steps at the same times up to float rounding.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2410.24164",
            "section": "Black et al. 2024, Section IV (The pi0 Model) and Appendix A-B (Sampling the flow matching timestep)",
            "note": "Conditional flow-matching loss on action chunks, 10 forward Euler steps, and the Beta time distribution.",
        },
    ],
    "tests": [
        {"name": "Training inputs in both conventions", "behavior": "state.invariant", "code": r"""
import torch
fm = {fn}()
actions = torch.tensor([[[1.0], [2.0]]])
noise = torch.tensor([[[0.0], [4.0]]])
x_t, v, mt = fm.make_inputs(actions, noise, torch.tensor([0.25]), "noise_at_one")
assert torch.allclose(x_t, torch.tensor([[[0.75], [2.5]]])), x_t
assert torch.allclose(v, torch.tensor([[[-1.0], [2.0]]])), v
assert mt.shape == (1,) and torch.allclose(mt, torch.tensor([0.25])), mt
x_t, v, mt = fm.make_inputs(actions, noise, torch.tensor([0.25]), "noise_at_zero")
assert torch.allclose(x_t, torch.tensor([[[0.25], [3.5]]])), x_t
assert torch.allclose(v, torch.tensor([[[1.0], [-2.0]]])), v
"""},
        {"name": "Euler sampling from t = 1 recovers the chunk", "behavior": "state.invariant", "code": r"""
import torch
fm = {fn}()
actions = torch.tensor([[[1.0, -1.0], [2.0, 0.5]]])
noise = torch.tensor([[[0.3, 0.2], [-0.4, 1.0]]])
seen = []
def v(x, t):
    seen.append(t.tolist())
    return noise - actions
out = fm.euler_sample(v, noise, 4, "noise_at_one")
assert seen == [[1.0], [0.75], [0.5], [0.25]], seen
assert out.shape == (1, 2, 2) and torch.allclose(out, actions, atol=1e-6), out
"""},
        {"name": "A clean prefix gets the clean-end time", "behavior": "state.invariant", "code": r"""
import torch
fm = {fn}()
actions = torch.tensor([[[1.0], [2.0]]])
noise = torch.tensor([[[0.0], [4.0]]])
x_t, v, mt = fm.make_inputs(actions, noise, torch.tensor([0.25]), "noise_at_one", torch.tensor([[True, False]]))
assert mt.shape == (1, 2) and torch.allclose(mt, torch.tensor([[0.0, 0.25]])), mt
assert torch.allclose(x_t, torch.tensor([[[1.0], [2.5]]])), x_t
assert torch.allclose(v, torch.tensor([[[-1.0], [2.0]]])), v
"""},
        {"name": "Seeded training inputs match the formulas", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "noise_at_one uses x_t = t*noise + (1-t)*actions and v = noise - actions; noise_at_zero swaps which end holds the noise.",
         "code": r"""
import torch
fm = {fn}()
for seed in (0, 7, 21):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    n = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    t = torch.rand(3, generator=g, dtype=torch.float64)
    tt = t[:, None, None]
    x, v, mt = fm.make_inputs(a, n, t, "noise_at_one")
    assert torch.allclose(x, tt * n + (1 - tt) * a, atol=1e-12) and torch.allclose(v, n - a, atol=1e-12), seed
    assert torch.equal(mt, t), seed
    x, v, mt = fm.make_inputs(a, n, t, "noise_at_zero")
    assert torch.allclose(x, (1 - tt) * n + tt * a, atol=1e-12) and torch.allclose(v, a - n, atol=1e-12), seed
    assert torch.equal(mt, t), seed
"""},
        {"name": "Each convention's target points from x_t to the clean chunk", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Do not mix conventions: x_t - t*v must equal actions for noise_at_one and x_t + (1-t)*v for noise_at_zero.",
         "code": r"""
import torch
fm = {fn}()
g = torch.Generator().manual_seed(3)
a = torch.randn(4, 6, 3, generator=g, dtype=torch.float64)
n = torch.randn(4, 6, 3, generator=g, dtype=torch.float64)
t = torch.rand(4, generator=g, dtype=torch.float64) * 0.8 + 0.1
tt = t[:, None, None]
x1, v1, _ = fm.make_inputs(a, n, t, "noise_at_one")
assert torch.allclose(x1 - tt * v1, a, atol=1e-12), "noise_at_one target"
x0, v0, _ = fm.make_inputs(a, n, 1 - t, "noise_at_zero")
assert torch.allclose(x0 + tt * v0, a, atol=1e-12), "noise_at_zero target"
assert torch.allclose(x0, x1, atol=1e-12) and torch.allclose(v0, -v1, atol=1e-12), "mirror"
"""},
        {"name": "Training prefix positions are clean and keep the full target", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Prefix positions get the clean-end time (0 for noise_at_one, 1 for noise_at_zero) so x_t equals actions there; v_target is not masked.",
         "code": r"""
import torch
fm = {fn}()
for seed, conv in ((1, "noise_at_one"), (2, "noise_at_zero"), (5, "noise_at_one"), (9, "noise_at_zero")):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 6, 2, generator=g, dtype=torch.float64)
    n = torch.randn(3, 6, 2, generator=g, dtype=torch.float64)
    t = torch.rand(3, generator=g, dtype=torch.float64) * 0.8 + 0.1
    mask = torch.rand(3, 6, generator=g) < 0.5
    mask[0, 0], mask[0, 1] = True, False
    clean = 0.0 if conv == "noise_at_one" else 1.0
    x, v, mt = fm.make_inputs(a, n, t, conv, mask)
    assert mt.shape == (3, 6), mt.shape
    want_t = torch.where(mask, torch.full((3, 6), clean, dtype=torch.float64), t[:, None].expand(3, 6))
    assert torch.allclose(mt, want_t), (seed, mt)
    assert torch.allclose(x[mask], a[mask], atol=1e-12), seed
    ref, _, _ = fm.make_inputs(a, n, t, conv)
    assert torch.allclose(x[~mask], ref[~mask], atol=1e-12), seed
    assert torch.allclose(v, (n - a) if conv == "noise_at_one" else (a - n), atol=1e-12), seed
"""},
        {"name": "All-false and all-true training prefix masks", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "An all-false mask still returns a (B, H) model_time equal to time; an all-true mask returns the actions unchanged.",
         "code": r"""
import torch
fm = {fn}()
g = torch.Generator().manual_seed(11)
a = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
n = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
t = torch.tensor([0.3, 0.9], dtype=torch.float64)
for conv in ("noise_at_one", "noise_at_zero"):
    ref, _, _ = fm.make_inputs(a, n, t, conv)
    x, _, mt = fm.make_inputs(a, n, t, conv, torch.zeros(2, 4, dtype=torch.bool))
    assert mt.shape == (2, 4) and torch.allclose(mt, t[:, None].expand(2, 4)), mt
    assert torch.allclose(x, ref, atol=1e-12), conv
    x, _, mt = fm.make_inputs(a, n, t, conv, torch.ones(2, 4, dtype=torch.bool))
    assert torch.allclose(x, a, atol=1e-12), conv
    assert torch.all(mt == (0.0 if conv == "noise_at_one" else 1.0)), mt
"""},
        {"name": "Sampler times follow each convention's schedule", "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Step k evaluates the velocity at t = 1 + k*dt with dt = -1/N for noise_at_one, and at t = k/N for noise_at_zero; take exactly N steps.",
         "code": _EXACT_FIELD + r"""
fm = {fn}()
g = torch.Generator().manual_seed(4)
a = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
n = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
for N in (1, 3, 10):
    for conv in ("noise_at_one", "noise_at_zero"):
        log = []
        fm.euler_sample(exact_field(a, conv, log), n, N, conv)
        dt = (-1.0 if conv == "noise_at_one" else 1.0) / N
        want = [(1.0 + k * dt) if conv == "noise_at_one" else k * dt for k in range(N)]
        assert len(log) == N, (conv, N, len(log))
        for k, t in enumerate(log):
            assert t.shape == (2,), t.shape
            assert torch.allclose(t, torch.full((2,), want[k], dtype=torch.float64), atol=1e-12), (conv, N, k, t)
"""},
        {"name": "Exact conditional field recovers the chunk for any step count", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "x = x + dt * v with dt = -1/N for noise_at_one and +1/N for noise_at_zero lands exactly on the actions for the exact straight-line field.",
         "code": _EXACT_FIELD + r"""
fm = {fn}()
for seed in (0, 6, 13):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    n = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    for N in (1, 2, 7, 10):
        for conv in ("noise_at_one", "noise_at_zero"):
            out = fm.euler_sample(exact_field(a, conv), n, N, conv)
            assert out.shape == a.shape, out.shape
            assert torch.allclose(out, a, atol=1e-9), (seed, N, conv, (out - a).abs().max())
            const = (n - a) if conv == "noise_at_one" else (a - n)
            out = fm.euler_sample(lambda x, t: const, n, N, conv)
            assert torch.allclose(out, a, atol=1e-9), (seed, N, conv, "constant field")
"""},
        {"name": "Hard prefix is clamped before every velocity call", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Write hard_prefix into x where the mask is true before calling velocity_fn, including the first step.",
         "code": _MIXING_FIELD + r"""
fm = {fn}()
for seed, conv in ((2, "noise_at_one"), (8, "noise_at_zero")):
    g = torch.Generator().manual_seed(seed)
    n = torch.randn(2, 5, 3, generator=g, dtype=torch.float64)
    prefix = torch.randn(2, 5, 3, generator=g, dtype=torch.float64) + 3.0
    mask = torch.tensor([[True, True, False, False, False], [True, False, False, True, False]])
    log = []
    fm.euler_sample(mixing_field(2, 5, 3, seed, log), n, 4, conv, prefix, mask)
    assert len(log) == 4, len(log)
    for k, (x, _) in enumerate(log):
        assert torch.equal(x[mask], prefix[mask]), (conv, k)
"""},
        {"name": "Hard prefix positions see the clean time", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With a hard prefix, pass a (B, H) time: the clean-end time (0 for noise_at_one, 1 for noise_at_zero) at masked positions and t_k elsewhere.",
         "code": _MIXING_FIELD + r"""
fm = {fn}()
for conv in ("noise_at_one", "noise_at_zero"):
    n = torch.randn(2, 4, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    log = []
    fm.euler_sample(mixing_field(2, 4, 2, 5, log), n, 5, conv, n.clone(), mask)
    clean = 0.0 if conv == "noise_at_one" else 1.0
    dt = (-1.0 if conv == "noise_at_one" else 1.0) / 5
    for k, (_, t) in enumerate(log):
        tk = (1.0 + k * dt) if conv == "noise_at_one" else k * dt
        assert t.shape == (2, 4), t.shape
        assert torch.all(t[mask] == clean), (conv, k, t)
        assert torch.allclose(t[~mask], torch.full_like(t[~mask], tk), atol=1e-12), (conv, k, t)
"""},
        {"name": "Hard prefix sampling matches a step-by-step oracle", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Clamp the prefix again after the last update so the output equals hard_prefix there, and integrate the other positions with the full Euler loop.",
         "code": _MIXING_FIELD + r"""
fm = {fn}()
for seed, conv, N in ((3, "noise_at_one", 1), (4, "noise_at_one", 6), (10, "noise_at_zero", 3), (12, "noise_at_zero", 10)):
    g = torch.Generator().manual_seed(seed)
    n = torch.randn(3, 6, 2, generator=g, dtype=torch.float64)
    prefix = torch.randn(3, 6, 2, generator=g, dtype=torch.float64)
    mask = torch.rand(3, 6, generator=g) < 0.4
    mask[:, 0] = True
    out = fm.euler_sample(mixing_field(3, 6, 2, seed, []), n, N, conv, prefix, mask)
    want = manual_sample(mixing_field(3, 6, 2, seed, []), n, N, conv, prefix, mask)
    assert torch.equal(out[mask], prefix[mask]), (seed, "prefix not clamped at the end")
    assert torch.allclose(out, want, atol=1e-10), (seed, conv, N, (out - want).abs().max())
"""},
        {"name": "All-false and all-true hard prefix masks", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "An all-false mask samples exactly as without a prefix; an all-true mask returns hard_prefix.",
         "code": _MIXING_FIELD + r"""
fm = {fn}()
g = torch.Generator().manual_seed(17)
n = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
prefix = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
for conv in ("noise_at_one", "noise_at_zero"):
    plain = fm.euler_sample(mixing_field(2, 4, 3, 1, []), n, 5, conv)
    none = fm.euler_sample(mixing_field(2, 4, 3, 1, []), n, 5, conv, prefix, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.allclose(plain, none, atol=1e-12), conv
    full = fm.euler_sample(mixing_field(2, 4, 3, 1, []), n, 5, conv, prefix, torch.ones(2, 4, dtype=torch.bool))
    assert torch.equal(full, prefix), conv
"""},
        {"name": "Inputs are not modified and dtype is kept", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Build new tensors instead of updating noise in place, and give the time tensor noise's dtype.",
         "code": _MIXING_FIELD + r"""
fm = {fn}()
g = torch.Generator().manual_seed(23)
a = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
n = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
prefix = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
mask = torch.tensor([[True, False, False], [False, False, False]])
saved = (a.clone(), n.clone(), prefix.clone())
x, v, mt = fm.make_inputs(a, n, torch.tensor([0.2, 0.6], dtype=torch.float64), "noise_at_one", mask)
assert x.dtype == torch.float64 and v.dtype == torch.float64 and mt.dtype == torch.float64
for conv in ("noise_at_one", "noise_at_zero"):
    log = []
    out = fm.euler_sample(mixing_field(2, 3, 2, 0, log), n, 3, conv)
    assert out.dtype == torch.float64 and all(t.dtype == torch.float64 for _, t in log), "dtype"
    fm.euler_sample(mixing_field(2, 3, 2, 0, []), n, 3, conv, prefix, mask)
assert torch.equal(a, saved[0]) and torch.equal(n, saved[1]) and torch.equal(prefix, saved[2]), "an input was modified"
"""},
        {"name": "Rejects bad shapes, conventions and step counts", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a time that is not (B,), a mask that is not (B, H), an unknown convention, num_steps < 1, or a prefix without its mask.",
         "code": r"""
import torch
fm = {fn}()
a = torch.zeros(2, 3, 4); n = torch.ones(2, 3, 4)
bad_inputs = [
    lambda: fm.make_inputs(a, n, torch.zeros(2, 1), "noise_at_one"),
    lambda: fm.make_inputs(a, n, torch.zeros(3), "noise_at_one"),
    lambda: fm.make_inputs(a, n, torch.zeros(2), "noise_at_one", torch.zeros(2, 4, dtype=torch.bool)),
    lambda: fm.make_inputs(a, n, torch.zeros(2), "noise_at_half"),
    lambda: fm.make_inputs(a, torch.ones(2, 3, 5), torch.zeros(2), "noise_at_one"),
    lambda: fm.make_inputs(a[0], n[0], torch.zeros(3), "noise_at_one"),
    lambda: fm.euler_sample(lambda x, t: x, n, 0, "noise_at_one"),
    lambda: fm.euler_sample(lambda x, t: x, n, 3, "forward"),
    lambda: fm.euler_sample(lambda x, t: x, n, 3, "noise_at_one", hard_prefix=n),
    lambda: fm.euler_sample(lambda x, t: x, n, 3, "noise_at_one", n, torch.zeros(2, 4, dtype=torch.bool)),
]
for i, call in enumerate(bad_inputs):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch

_CONVENTIONS = ("noise_at_one", "noise_at_zero")


class Pi0FlowMatching:
    @staticmethod
    def _noise_at_one(convention):
        if convention not in _CONVENTIONS:
            raise ValueError(f"convention must be one of {_CONVENTIONS}, got {convention!r}")
        return convention == "noise_at_one"

    def make_inputs(self, actions, noise, time, convention, prefix_mask=None):
        noise_at_one = self._noise_at_one(convention)
        if actions.ndim != 3 or noise.shape != actions.shape:
            raise ValueError("actions and noise must both have shape (B, H, D)")
        if time.shape != actions.shape[:1]:
            raise ValueError("time must have shape (B,)")
        if prefix_mask is not None and prefix_mask.shape != actions.shape[:2]:
            raise ValueError("prefix_mask must have shape (B, H)")
        if prefix_mask is None:
            model_time = time
            t = time[:, None, None]
        else:
            # Clean actions sit at the far end of the schedule from the noise.
            clean_time = 0.0 if noise_at_one else 1.0
            per_position = time[:, None].expand(prefix_mask.shape)
            model_time = torch.where(prefix_mask, torch.full_like(per_position, clean_time), per_position)
            t = model_time[:, :, None]
        if noise_at_one:
            x_t = t * noise + (1 - t) * actions
            v_target = noise - actions
        else:
            x_t = (1 - t) * noise + t * actions
            v_target = actions - noise
        return x_t, v_target, model_time

    def euler_sample(self, velocity_fn, noise, num_steps, convention, hard_prefix=None, hard_prefix_mask=None):
        noise_at_one = self._noise_at_one(convention)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if noise.ndim != 3:
            raise ValueError("noise must have shape (B, H, D)")
        if (hard_prefix is None) != (hard_prefix_mask is None):
            raise ValueError("pass hard_prefix and hard_prefix_mask together")
        if hard_prefix is not None and (hard_prefix.shape != noise.shape or hard_prefix_mask.shape != noise.shape[:2]):
            raise ValueError("hard_prefix must match noise and hard_prefix_mask must be (B, H)")
        batch, horizon = noise.shape[:2]
        clean_time = 0.0 if noise_at_one else 1.0
        dt = (-1.0 if noise_at_one else 1.0) / num_steps
        x = noise
        for k in range(num_steps):
            t_k = (1.0 + k * dt) if noise_at_one else k * dt
            time = torch.full((batch,), t_k, dtype=noise.dtype, device=noise.device)
            if hard_prefix is not None:
                keep = hard_prefix_mask[:, :, None]
                x = torch.where(keep, hard_prefix, x)  # clamp before the velocity call
                clean = torch.full((batch, horizon), clean_time, dtype=noise.dtype, device=noise.device)
                time = torch.where(hard_prefix_mask, clean, time[:, None].expand(batch, horizon))
            x = x + dt * velocity_fn(x, time)
            if hard_prefix is not None:
                x = torch.where(keep, hard_prefix, x)  # and after the update
        return x
''',
    "interview_questions": interview(
        concept=[
            "In pi0's flow-matching action head, which end of the time interval holds the noise, and how does that change the velocity target compared with the noise-at-zero convention?",
            "Why does pi0 generate a whole chunk of future actions with flow matching instead of one discretized action token at a time?",
        ],
        deep_dive=[
            "Walk through one Euler step of pi0 sampling: what time does the network see, what is dt, and why does the exact straight-line field recover the chunk in any number of steps?",
            "How does a committed action prefix enter training and sampling, and why must the prefix be clamped both before and after each Euler step?",
            "openpi samples training times as Beta(1.5, 1) * 0.999 + 0.001. Which noise levels does this emphasize, and why keep away from the clean endpoint?",
        ],
        tradeoffs=[
            "Few Euler steps versus many at robot control rates: what do you gain and lose, and when would you reach for a higher-order solver or distillation?",
            "Flow-matching action chunks versus discrete action tokens such as FAST: compare training cost, inference latency and handling of multimodal actions.",
        ],
    ),
}
