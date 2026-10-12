"""Dreamer 4 shortcut forcing: the step-size and signal-level sampler and the x-space loss with the ramp weight."""

from ._interview import interview

# A deterministic x-prediction network that mixes frames causally, so feeding the
# wrong input at any frame of a half-step call changes later frames' outputs.
_FAKE = r"""
import math
import torch

def make_model(D, seed, log=None, theta=None):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, D, generator=g, dtype=torch.float64) / math.sqrt(D)
    a = torch.randn(D, generator=g, dtype=torch.float64)
    b = torch.randn(D, generator=g, dtype=torch.float64)
    c = torch.randn(D, generator=g, dtype=torch.float64) * 0.5
    def f(z, tau, d):
        if log is not None:
            log.append((z.detach().clone(), tau.detach().clone(), d.detach().clone(), torch.is_grad_enabled()))
        tt = tau[..., None].to(z.dtype)
        dd = d[..., None].to(z.dtype)
        h = torch.tanh(z @ W + tt * a + 3.0 * dd * b)
        steps = torch.arange(1, z.shape[1] + 1, dtype=z.dtype).reshape(1, -1, 1)
        out = h + 0.5 * torch.cumsum(h, dim=1) / steps + c * tt
        return out if theta is None else theta * out
    return f

def grid(B, T, k_max, seed, flow_frac=0.3):
    g = torch.Generator().manual_seed(seed + 1000)
    n = int(math.log2(k_max)) + 1
    steps = 2 ** torch.randint(0, n, (B, T), generator=g)
    steps[torch.rand(B, T, generator=g) < flow_frac] = k_max
    d = 1.0 / steps.double()
    tau = torch.floor(torch.rand(B, T, generator=g, dtype=torch.float64) * steps) * d
    return tau, d

def data(B, T, D, seed):
    g = torch.Generator().manual_seed(seed + 2000)
    z1 = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    z0 = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    return z1, z0

def parts(model, z1, z0, tau, d, d_min):
    # Unweighted per-frame flow and bootstrap terms, written out from the paper's Equation 7.
    t = tau[..., None]
    zt = (1 - t) * z0 + t * z1
    x_hat = model(zt, tau, d)
    flow = ((x_hat - z1) ** 2).sum(-1)
    with torch.no_grad():
        h = (d / 2)[..., None]
        b1 = (model(zt, tau, d / 2) - zt) / (1 - t)
        zm = zt + b1 * h
        b2 = (model(zm, tau + d / 2, d / 2) - zm) / (1 - t - h)
        target = (b1 + b2) / 2
    boot = (1 - tau) ** 2 * (((x_hat - zt) / (1 - t) - target) ** 2).sum(-1)
    return flow, boot

def expected(model, z1, z0, tau, d, d_min):
    flow, boot = parts(model, z1, z0, tau, d, d_min)
    return (0.9 * tau + 0.1) * torch.where(d == d_min, flow, boot)
"""

TASK = {
    "title": "Shortcut Forcing Loss (Dreamer 4)",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "shortcut_forcing_loss",
    "description_en": r"""Implement the training objective of the Dreamer 4 world model's dynamics: sample a step size and signal level per frame, then compute the shortcut forcing loss of an x-prediction network in data space, with the ramp weight.

**Signatures:**
- `sample_step_and_level(shape, k_max, generator=None) -> (tau, d)`
- `shortcut_forcing_loss(model, z1, z0, tau, d, d_min) -> Tensor`

**Conventions.** `tau` is the signal level: `tau = 0` is pure noise and `tau = 1` is clean data. The corrupted input is `z_tilde = (1 - tau) * z0 + tau * z1`. `d` is the step size, and `d_min = 1 / k_max` is the finest one.

**sample_step_and_level:**
- `k_max` is a positive power-of-two `int` (1, 2, 4, ...); a float such as `2.0` or a bool is not accepted. Otherwise raise `ValueError`.
- For every element of `shape`, draw `K` uniformly from `{1, 2, 4, ..., k_max}` and set `d = 1 / K`.
- Then draw `tau` uniformly from the grid that steps of size `d` reach from 0: `{0, d, 2d, ..., 1 - d}`. `tau = 1` is never drawn.
- Draw independently for each element. Use `generator` for all randomness. Return two float32 tensors of shape `shape`.

**shortcut_forcing_loss:**
- `z1` (clean) and `z0` (noise) have shape `(B, T, D)`. `tau` and `d` have shape `(B, T)`, one value per frame, on the grid above.
- `model(z, tau, d)` takes `(B, T, D)`, `(B, T)`, `(B, T)` tensors and returns an x-prediction, the clean frames, of shape `(B, T, D)`. It may mix information across frames.
- Prediction: `z1_hat = model(z_tilde, tau, d)`. Gradients flow through this call.
- A frame with `d == d_min` is a flow frame. Its loss is `||z1_hat - z1||^2`.
- Every other frame is a bootstrap frame. Take two half steps of size `h = d / 2`, each turning an x-prediction into a velocity:
    b1 = (model(z_tilde, tau, h) - z_tilde) / (1 - tau)
    z_mid = z_tilde + b1 * h
    b2 = (model(z_mid, tau + h, h) - z_mid) / (1 - (tau + h))
    target = stop_gradient((b1 + b2) / 2)
    loss = (1 - tau)^2 * ||(z1_hat - z_tilde) / (1 - tau) - target||^2
- Each half-step call gets the whole sequence, with `h` and `tau + h` (and `z_mid` in the second call) at every frame, flow frames included. The outputs at flow frames are not used.
- If every frame is a flow frame, call `model` only once. Otherwise call it exactly three times.
- `||x||^2` is the sum of squares over the last axis `D`.
- Multiply every frame's loss, flow or bootstrap, by the ramp weight `w = 0.9 * tau + 0.1`.
- Return the weighted loss per frame, shape `(B, T)`, in the dtype of `z1`. Do not reduce it to a scalar.
- No gradient may flow through `target`.
- Raise `ValueError` when `z1` is not 3-D, `z0` has a different shape, `tau` or `d` is not `(B, T)`, any `d < d_min`, any `tau < 0`, or any `tau + d > 1`.
- Do not modify any input in place.

**Example** (one bootstrap frame, `D = 1`):

    model = lambda z, tau, d: z * (1 + d[..., None])
    z1 = torch.tensor([[[2.0]]]); z0 = torch.zeros(1, 1, 1)
    tau = torch.tensor([[0.5]]); d = torch.tensor([[0.5]])
    shortcut_forcing_loss(model, z1, z0, tau, d, d_min=0.25)
    # z_tilde = 1.0, z1_hat = 1.5, b1 = 0.5, z_mid = 1.125, b2 = 1.125, target = 0.8125
    # loss = 0.25 * (1.0 - 0.8125)^2 = 0.0087890625; times w = 0.55 -> 0.004833984375

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Where this sits.** Dreamer 4 (Hafner, Yan and Lillicrap, 2025) learns a world model of Minecraft from video and trains its agent inside it. Its dynamics transformer predicts the next frame's tokenizer representation given actions, and it is trained with shortcut forcing, a combination of diffusion forcing and shortcut models, so that it needs only `K = 4` forward passes per generated frame.

**Diffusion forcing.** Every frame of a training sequence gets its own signal level, so each frame is both a denoising task and noisy context for later frames. At inference Dreamer 4 generates frame by frame and corrupts past frames slightly, to `tau_ctx = 0.1`, so the model tolerates small errors in its own generations.

**Shortcut models.** Frans et al. condition the network on the step size `d` as well as the level. The finest step is trained with plain flow matching (in their code; the paper frames it as the d -> 0 limit). Each larger step is trained to match two half steps of the model itself, so a step of size `d` learns to land where two steps of size `d / 2` would. At inference, conditioning on `d = 1 / K` gives `K` sampling steps. The Dreamer 4 paper prints the level grid as `{0, 1/d, ..., 1 - 1/d}`. It means the grid reached by steps of size `d`, `{0, d, ..., 1 - d}`, which is also what the shortcut-models code samples.

**Why x-prediction.** Shortcut models predict velocities. Dreamer 4 reports that velocity prediction trains the network to produce high-frequency outputs, which can cause small errors to accumulate over long frame-by-frame rollouts. Predicting the clean frame instead enabled high-quality rollouts of arbitrary length. The flow term is then a plain x-space error. The bootstrap target is defined in velocity space, so the prediction is converted with `v = (z1_hat - z_tilde) / (1 - tau)`. Since `||z1_hat - z1||^2 = (1 - tau)^2 * ||v_hat - v||^2`, the `(1 - tau)^2` factor brings the bootstrap term into the range of the x-space flow term. It also makes the bootstrap term exactly the x-space distance from `z1_hat` to the clean frame the target velocity implies, `z_tilde + (1 - tau) * target`.

**Why the ramp weight.** At low signal levels the flow term degenerates to predicting the dataset mean. The paper adds a weight that grows linearly with `tau`, from 0.1 at pure noise to 1.0 at clean data, to focus capacity where there is more learning signal. The paper defines `w(tau)` without writing out the product. This exercise applies it to both branches; the paper also uses it on a flow-only diffusion forcing baseline.

**What was measured.** In the paper's Minecraft design cascade (FVD, lower is better), adding the shortcut model at 4 sampling steps brings FVD from 875 back to 329. Then x-prediction gives 326, the x-space loss 151 and the ramp weight 102.""",
    "advisory_prerequisites": ["flow_matching", "pi0_flow_matching_chunk", "diffusion_forcing_schedule"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which grid of levels can a step of size d reach from pure noise without passing tau = 1? The network returns a clean frame, not a velocity: how do you get the velocity at level tau from it, and what happens to that velocity's scale near tau = 1? Which of the three model calls should gradients flow through?"},
        {"level": 2, "kind": "analysis", "content": "For the sampler, draw an exponent per element, set K = 2 to that power, then draw an integer index below K and set tau = index / K. For the loss, validate shapes and the grid, build z_tilde, and call the model once with gradients. If any frame has d above d_min, compute b1, z_mid and b2 under no_grad on the whole sequence, average them, convert z1_hat to a velocity, and form the scaled bootstrap term. Select per frame between the flow and bootstrap terms, then multiply by 0.9 * tau + 0.1."},
    ],
    "model_connections": [
        "Dreamer 4 trains its dynamics transformer with this objective and generates each frame with K = 4 shortcut steps of size d = 1/4.",
        "Shortcut models (Frans et al.) introduced the step-size-conditioned bootstrap in velocity space, generating images with one or a few steps.",
        "Diffusion forcing gives each frame of a sequence its own noise level; Dreamer 4 combines it with the shortcut bootstrap.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A single training run yields a model that samples with any power-of-two step count, with no separate distillation phase.",
            "Predicting clean frames avoids the high-frequency outputs that the paper found accumulate errors over long autoregressive rollouts.",
        ],
        "cons": [
            "Each bootstrap frame costs two extra forward passes, and its target is only as good as the model's own smaller steps.",
            "The ramp weight and the (1 - tau)^2 rescaling are heuristics that fix the relative scale of the two terms by hand.",
        ],
    },
    "sources": [
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2509.24527",
            "section": "Hafner, Yan, Lillicrap 2025, Training Agents Inside of Scalable World Models: Section 2 (Shortcut models), Section 3.2 (Shortcut forcing) and Section 4.4 (Model Design)",
            "equation": "Equation 4 (step-size and level sampling), Equation 6 (corruption and x-prediction), Equation 7 and footnote 1 (x-space shortcut forcing loss), Equation 8 (ramp weight)",
            "note": "The paper prints the level grid as {0, 1/d, ..., 1 - 1/d}; the exercise uses {0, d, ..., 1 - d}. Equation 7's z_tau and sg(b_1 + b_2) are read as z_tilde and sg(b' + b''). The paper does not write w(tau) times the loss; the exercise applies it to both branches. Dreamer 4 released no code; the oracle is transcribed from these equations.",
        },
        {
            "kind": "code",
            "url": "https://github.com/kvfrans/shortcut-models",
            "commit": "601004348667094e1b71f30942199759412d4432",
            "path": "targets_shortcut.py",
            "symbol": "get_targets",
            "license": "MIT",
            "adapted": "The two-half-step bootstrap target (v_b1, x_t2, v_b2, their average) and the level grid t = randint(0, 2**dt_base) / 2**dt_base, used to cross-check the x-space target.",
            "simplifications": "Upstream predicts velocities, computes the targets with EMA parameters, clips x_t2 and the target to [-4, 4], fills a fixed 1/8 of each batch with bootstrap samples and spreads their step sizes evenly, and adds classifier-free guidance options. The exercise predicts clean frames as Dreamer 4 does, uses the current model with a stop gradient, does not clip, and samples the step size uniformly per frame as in Dreamer 4's Equation 4.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2410.12557",
            "section": "Frans, Hafner, Levine, Abbeel 2024, One Step Diffusion via Shortcut Models: Section 3 (Shortcut Models for Few Step Generation) and Section 3.1 (Training Details)",
            "note": "The self-consistency bootstrap of a step of size 2d from two steps of size d, and flow matching at the finest step.",
        },
    ],
    "tests": [
        {"name": "Flow frames: x-space error times the ramp weight", "behavior": "state.invariant", "code": r"""
import torch
model = lambda z, tau, d: z
z1 = torch.tensor([[[1.0, 2.0], [3.0, 0.0]]], dtype=torch.float64)
z0 = torch.zeros(1, 2, 2, dtype=torch.float64)
tau = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
d = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
out = {fn}(model, z1, z0, tau, d, 0.5)
# frame 0: z_tilde = [0.5, 1.0], error 0.25 + 1.0 = 1.25, w = 0.55
# frame 1: z_tilde = 0, error 9.0, w = 0.1
assert out.shape == (1, 2), out.shape
assert torch.allclose(out, torch.tensor([[0.6875, 0.9]], dtype=torch.float64)), out
"""},
        {"name": "Bootstrap frame: two half steps converted from x-predictions", "behavior": "state.invariant", "code": r"""
import torch
model = lambda z, tau, d: z * (1 + d[..., None])
z1 = torch.tensor([[[2.0]]], dtype=torch.float64)
z0 = torch.zeros(1, 1, 1, dtype=torch.float64)
tau = torch.tensor([[0.5]], dtype=torch.float64)
d = torch.tensor([[0.5]], dtype=torch.float64)
out = {fn}(model, z1, z0, tau, d, 0.25)
# z1_hat = 1.5, b1 = 0.5, z_mid = 1.125, b2 = 1.125, target = 0.8125, v_hat = 1.0
assert out.shape == (1, 1), out.shape
assert torch.allclose(out, torch.tensor([[0.004833984375]], dtype=torch.float64)), out
"""},
        {"name": "Sampler draws d and tau on the grid", "behavior": "state.invariant", "code": r"""
import torch
sample = {fn}.__globals__["sample_step_and_level"]
tau, d = sample((4, 64), 8, torch.Generator().manual_seed(0))
assert tau.shape == (4, 64) and d.shape == (4, 64), (tau.shape, d.shape)
assert tau.dtype == torch.float32 and d.dtype == torch.float32, (tau.dtype, d.dtype)
assert set(d.unique().tolist()) <= {1.0, 0.5, 0.25, 0.125}, d.unique()
k = 1 / d
assert torch.all(tau * k == torch.round(tau * k)), "tau must be a multiple of d"
assert torch.all(tau >= 0) and torch.all(tau <= 1 - d), "tau must lie in {0, d, ..., 1 - d}"
"""},
        {"name": "Seeded sequences match the formulas", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Flow frames use ||z1_hat - z1||^2; bootstrap frames use (1-tau)^2 * ||(z1_hat - z_tilde)/(1-tau) - (b1+b2)/2||^2; both times 0.9*tau + 0.1.",
         "code": _FAKE + r"""
for seed, (B, T, D, k_max) in enumerate([(2, 5, 3, 8), (3, 4, 4, 4), (1, 7, 2, 16), (2, 6, 5, 2), (4, 3, 3, 8)]):
    model = make_model(D, seed)
    z1, z0 = data(B, T, D, seed)
    tau, d = grid(B, T, k_max, seed)
    out = {fn}(model, z1, z0, tau, d, 1.0 / k_max)
    want = expected(model, z1, z0, tau, d, 1.0 / k_max)
    assert out.shape == (B, T), (seed, out.shape)
    assert torch.allclose(out, want, atol=1e-10, rtol=1e-9), (seed, (out - want).abs().max())
"""},
        {"name": "A model that predicts the clean frames has zero loss", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "With z1_hat = z1 every half step lies on the straight path, so b1 = b2 = z1 - z0; check the denominators 1 - tau and 1 - (tau + d/2) and the input z_mid.",
         "code": _FAKE + r"""
for seed in (0, 1, 2):
    B, T, D = 3, 6, 4
    z1, z0 = data(B, T, D, seed)
    tau, d = grid(B, T, 16, seed, flow_frac=0.2)
    model = lambda z, t, s: z1.clone()
    out = {fn}(model, z1, z0, tau, d, 1.0 / 16)
    assert torch.allclose(out, torch.zeros(B, T, dtype=torch.float64), atol=1e-18), (seed, out.abs().max())
    # A model that adds a constant offset is penalized by the same squared offset in both branches.
    off = torch.randn(D, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    model = lambda z, t, s: z1 + off
    out = {fn}(model, z1, z0, tau, d, 1.0 / 16)
    boot = d != 1.0 / 16
    flow_want = (0.9 * tau + 0.1) * (off ** 2).sum()
    assert torch.allclose(out[~boot], flow_want[~boot], atol=1e-10), seed
"""},
        {"name": "The ramp weight multiplies both branches", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Multiply the flow term and the bootstrap term by w = 0.9 * tau + 0.1 (0.1 at pure noise, rising to 1 at clean data).",
         "code": _FAKE + r"""
B, T, D, k_max = 4, 8, 3, 8
model = make_model(D, 11)
z1, z0 = data(B, T, D, 11)
tau, d = grid(B, T, k_max, 11, flow_frac=0.5)
flow, boot = parts(model, z1, z0, tau, d, 1.0 / k_max)
out = {fn}(model, z1, z0, tau, d, 1.0 / k_max)
is_flow = d == 1.0 / k_max
assert is_flow.any() and (~is_flow).any()
w = 0.9 * tau + 0.1
assert torch.allclose(out[is_flow], (w * flow)[is_flow], atol=1e-10), "flow frames"
assert torch.allclose(out[~is_flow], (w * boot)[~is_flow], atol=1e-10), "bootstrap frames"
"""},
        {"name": "No gradient flows through the bootstrap target", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Compute b1 and b2 under torch.no_grad() (or detach the target); only the call with step d carries gradients.",
         "code": _FAKE + r"""
for seed in (3, 4):
    B, T, D, k_max = 2, 5, 3, 8
    z1, z0 = data(B, T, D, seed)
    tau, d = grid(B, T, k_max, seed, flow_frac=0.2)
    theta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    out = {fn}(make_model(D, seed, theta=theta), z1, z0, tau, d, 1.0 / k_max)
    (g,) = torch.autograd.grad(out.sum(), theta)
    theta2 = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    want = expected(make_model(D, seed, theta=theta2), z1, z0, tau, d, 1.0 / k_max)
    (g_want,) = torch.autograd.grad(want.sum(), theta2)
    assert abs(g_want.item()) > 1e-6
    assert torch.allclose(g, g_want, atol=1e-10), (seed, g.item(), g_want.item())
"""},
        {"name": "Half-step calls see the whole sequence with the right inputs", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Call model(z_tilde, tau, d), model(z_tilde, tau, d/2) and model(z_mid, tau + d/2, d/2) on all frames, with z_mid = z_tilde + b1 * d/2 everywhere.",
         "code": _FAKE + r"""
B, T, D, k_max = 2, 6, 3, 8
log = []
model = make_model(D, 5, log)
z1, z0 = data(B, T, D, 5)
tau, d = grid(B, T, k_max, 5, flow_frac=0.4)
assert (d == 1.0 / k_max).any() and (d != 1.0 / k_max).any()
{fn}(model, z1, z0, tau, d, 1.0 / k_max)
assert len(log) == 3, f"expected 3 model calls, got {len(log)}"
t = tau[..., None]
zt = (1 - t) * z0 + t * z1
by_kind = {}
for z, ta, dd, _ in log:
    if torch.allclose(dd, d):
        by_kind["full"] = (z, ta)
    elif torch.allclose(dd, d / 2) and torch.allclose(ta, tau):
        by_kind["first"] = (z, ta)
    elif torch.allclose(dd, d / 2) and torch.allclose(ta, tau + d / 2):
        by_kind["second"] = (z, ta)
assert set(by_kind) == {"full", "first", "second"}, f"calls seen: {sorted(by_kind)}"
assert torch.allclose(by_kind["full"][0], zt, atol=1e-12), "prediction input"
assert torch.allclose(by_kind["first"][0], zt, atol=1e-12), "first half-step input"
with torch.no_grad():
    b1 = (model(zt, tau, d / 2) - zt) / (1 - t)
zm = zt + b1 * (d / 2)[..., None]
assert torch.allclose(by_kind["second"][0], zm, atol=1e-10), "second half-step input must be z_mid at every frame"
"""},
        {"name": "All flow frames need a single model call", "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "When every frame has d == d_min, skip the two half-step calls and call the model exactly once.",
         "code": _FAKE + r"""
B, T, D = 2, 4, 3
log = []
model = make_model(D, 6, log)
z1, z0 = data(B, T, D, 6)
d = torch.full((B, T), 0.25, dtype=torch.float64)
tau = torch.tensor([[0.0, 0.25, 0.5, 0.75], [0.75, 0.5, 0.0, 0.25]], dtype=torch.float64)
out = {fn}(model, z1, z0, tau, d, 0.25)
assert len(log) == 1, f"expected 1 model call, got {len(log)}"
assert log[0][3], "the prediction call must keep gradients enabled"
assert torch.allclose(out, expected(make_model(D, 6), z1, z0, tau, d, 0.25), atol=1e-10)
"""},
        {"name": "Grid boundaries: pure noise, the last level and the one-step size", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Handle tau = 0, tau = 1 - d and d = 1 (a single step from pure noise); with k_max = 1 every frame is a flow frame at tau = 0.",
         "code": _FAKE + r"""
B, T, D = 1, 6, 3
model = make_model(D, 8)
z1, z0 = data(B, T, D, 8)
d = torch.tensor([[1.0, 0.5, 0.5, 0.125, 0.125, 0.25]], dtype=torch.float64)
tau = torch.tensor([[0.0, 0.5, 0.0, 0.875, 0.0, 0.75]], dtype=torch.float64)
out = {fn}(model, z1, z0, tau, d, 0.125)
want = expected(model, z1, z0, tau, d, 0.125)
assert torch.isfinite(out).all(), out
assert torch.allclose(out, want, atol=1e-10), (out, want)
# k_max = 1: d = d_min = 1 and tau = 0 everywhere.
d1 = torch.ones(2, 3, dtype=torch.float64)
t1 = torch.zeros(2, 3, dtype=torch.float64)
y1, y0 = data(2, 3, D, 9)
out = {fn}(model, y1, y0, t1, d1, 1.0)
assert torch.allclose(out, 0.1 * ((model(y0, t1, d1) - y1) ** 2).sum(-1), atol=1e-10), out
"""},
        {"name": "Keeps the dtype and leaves inputs untouched", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Return a (B, T) tensor in z1's dtype, and build new tensors instead of updating z1, z0, tau or d in place.",
         "code": _FAKE + r"""
B, T, D = 2, 5, 4
model = make_model(D, 10)
z1, z0 = data(B, T, D, 10)
tau, d = grid(B, T, 8, 10)
saved = [x.clone() for x in (z1, z0, tau, d)]
out = {fn}(model, z1, z0, tau, d, 0.125)
assert out.dtype == torch.float64 and out.shape == (B, T), (out.dtype, out.shape)
for x, s in zip((z1, z0, tau, d), saved):
    assert torch.equal(x, s), "an input was modified"
"""},
        {"name": "Rejects bad shapes and off-grid levels", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a non-3-D z1, mismatched z0, tau or d not (B, T), d < d_min, tau < 0, or tau + d > 1.",
         "code": r"""
import torch
model = lambda z, tau, d: z
z = torch.zeros(2, 3, 4, dtype=torch.float64)
tau = torch.zeros(2, 3, dtype=torch.float64)
d = torch.full((2, 3), 0.5, dtype=torch.float64)
bad = [
    lambda: {fn}(model, z[0], z[0], tau[0], d[0], 0.5),
    lambda: {fn}(model, z, torch.zeros(2, 3, 5, dtype=torch.float64), tau, d, 0.5),
    lambda: {fn}(model, z, z, tau[:, :2], d, 0.5),
    lambda: {fn}(model, z, z, tau, d[None], 0.5),
    lambda: {fn}(model, z, z, tau, torch.full((2, 3), 0.25, dtype=torch.float64), 0.5),
    lambda: {fn}(model, z, z, torch.full((2, 3), -0.5, dtype=torch.float64), d, 0.5),
    lambda: {fn}(model, z, z, torch.full((2, 3), 0.75, dtype=torch.float64), d, 0.25),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
        {"name": "Sampler is uniform over step counts and over each grid", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Draw K uniformly from {1, 2, 4, ..., k_max} (d = 1/K), then tau uniformly from {0, d, ..., 1 - d}.",
         "code": r"""
import torch
sample = {fn}.__globals__["sample_step_and_level"]
tau, d = sample((60000,), 8, torch.Generator().manual_seed(1))
tau, d = tau.double(), d.double()
n = tau.numel()
for k in (1, 2, 4, 8):
    sel = d == 1.0 / k
    frac = sel.double().mean().item()
    assert abs(frac - 0.25) < 0.02, f"P(d = 1/{k}) = {frac:.3f}, want 0.25"
    levels = torch.round(tau[sel] * k).long()
    assert levels.min().item() == 0 and levels.max().item() == k - 1, (k, levels.min(), levels.max())
    counts = torch.bincount(levels, minlength=k).double() / sel.sum()
    assert torch.all((counts - 1.0 / k).abs() < 0.025), (k, counts)
"""},
        {"name": "Sampler edge cases, validation and reproducibility", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "k_max = 1 gives d = 1 and tau = 0; a k_max that is not a positive power of two raises ValueError; the same generator seed gives the same draw.",
         "code": r"""
import torch
sample = {fn}.__globals__["sample_step_and_level"]
tau, d = sample((3, 5), 1, torch.Generator().manual_seed(2))
assert torch.all(d == 1.0) and torch.all(tau == 0.0), (tau, d)
for bad in (0, 3, 6, -4, 2.0):
    try:
        sample((2,), bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"k_max = {bad!r} was accepted")
a = sample((2, 50), 32, torch.Generator().manual_seed(7))
b = sample((2, 50), 32, torch.Generator().manual_seed(7))
assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "not reproducible from the generator"
tau, d = sample((20000,), 64, torch.Generator().manual_seed(3))
assert torch.all(tau <= 1 - d) and torch.all(tau >= 0), "tau left the grid"
assert (d == 1.0 / 64).any() and (tau == 1 - 1.0 / 64).any(), "the finest grid's last level was never drawn"
"""},
    ],
    "solution": '''import torch


def sample_step_and_level(shape, k_max, generator=None):
    if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 1 or k_max & (k_max - 1):
        raise ValueError("k_max must be a positive power of two")
    num_sizes = k_max.bit_length()  # step counts 1, 2, 4, ..., k_max
    exponent = torch.randint(0, num_sizes, shape, generator=generator)
    steps = 2 ** exponent  # K = 1 / d for each element
    # tau is uniform over the grid {0, d, ..., 1 - d} that steps of size d reach from 0.
    index = torch.floor(torch.rand(shape, generator=generator, dtype=torch.float64) * steps).long()
    d = 1.0 / steps.double()
    tau = index.double() * d
    return tau.float(), d.float()


def _sum_sq(x):
    return (x * x).sum(dim=-1)


def shortcut_forcing_loss(model, z1, z0, tau, d, d_min):
    if z1.ndim != 3 or z0.shape != z1.shape:
        raise ValueError("z1 and z0 must both have shape (B, T, D)")
    if tau.shape != z1.shape[:2] or d.shape != z1.shape[:2]:
        raise ValueError("tau and d must have shape (B, T)")
    if bool((d < d_min).any()) or bool((tau < 0).any()) or bool((tau + d > 1).any()):
        raise ValueError("need d >= d_min and 0 <= tau <= 1 - d")
    t = tau[..., None]
    z_tilde = (1 - t) * z0 + t * z1
    z1_hat = model(z_tilde, tau, d)

    flow = d == d_min
    loss = _sum_sq(z1_hat - z1)  # x-space flow loss at the finest step
    if not bool(flow.all()):
        half = d / 2
        h = half[..., None]
        with torch.no_grad():
            # Two half steps, each x-prediction turned into a velocity.
            b1 = (model(z_tilde, tau, half) - z_tilde) / (1 - t)
            z_mid = z_tilde + b1 * h
            b2 = (model(z_mid, tau + half, half) - z_mid) / (1 - (t + h))
            v_target = (b1 + b2) / 2
        v_hat = (z1_hat - z_tilde) / (1 - t)
        bootstrap = (1 - tau) ** 2 * _sum_sq(v_hat - v_target)
        loss = torch.where(flow, loss, bootstrap)
    return (0.9 * tau + 0.1) * loss  # ramp weight on both branches
''',
    "interview_questions": interview(
        concept=[
            "What does a shortcut model condition on beyond the noise level, and how is a large step trained when no ground-truth target for it exists?",
            "In Dreamer 4's shortcut forcing, what does the network predict, and why did the authors move away from velocity prediction for a world model that rolls out frame by frame?",
        ],
        deep_dive=[
            "Walk through the bootstrap target for one frame: the two model calls, their inputs and step sizes, and how each x-prediction becomes a velocity.",
            "Why is the bootstrap term multiplied by (1 - tau)^2, and what x-space quantity does the scaled term equal?",
            "Why does Dreamer 4 sample the signal level only on the grid {0, d, ..., 1 - d} for a step of size d, and what would go wrong with tau drawn from a continuous range near 1?",
        ],
        tradeoffs=[
            "The ramp weight down-weights low signal levels. What do you gain, and what could you lose at the high-noise end where generation starts?",
            "Shortcut forcing with 4 steps versus diffusion forcing with 64 steps or a separately distilled few-step model: compare training cost, inference speed and quality.",
        ],
    ),
}
