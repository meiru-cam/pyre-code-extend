"""Diffusion Forcing noise levels: independent per-frame training levels, scheduling matrices and context stabilization."""

from ._interview import interview

# An independent oracle for the pyramid family, written with plain loops and floor().
_ORACLE = r"""
import math, torch

def want_pyramid(horizon, S, u):
    height = S + math.floor((horizon - 1) * u) + 1
    rows = []
    for m in range(height):
        rows.append([min(S, max(0, S + math.floor(t * u) - m)) for t in range(horizon)])
    return torch.tensor(rows, dtype=torch.long)

def want_full(horizon, S):
    return torch.tensor([[S - m] * horizon for m in range(S + 1)], dtype=torch.long)

def fns(fn):
    g = fn.__globals__
    names = ("train_noise_levels", "step_noise_levels", "model_noise_levels")
    out = [g.get(n) for n in names]
    for n, f in zip(names, out):
        assert callable(f), f"define {n} next to scheduling_matrix"
    return out
"""

TASK = {
    "title": "Diffusion Forcing Noise Schedules",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "scheduling_matrix",
    "description_en": r"""Diffusion Forcing trains a video diffusion model with an independent noise level for every frame, then samples by walking down a **scheduling matrix**: row `m` gives the noise level of every frame after `m` denoising steps. Implement the four pieces of noise-level bookkeeping from the official `df_base.py`, on integer levels only. No model and no images are involved.

**Signatures:** define all four functions at the top level of your code.
- `train_noise_levels(num_frames, batch_size, timesteps, generator, frame_mask=None) -> Tensor`
- `scheduling_matrix(horizon, sampling_timesteps, kind, uncertainty_scale=1.0) -> Tensor`
- `step_noise_levels(matrix, m, n_context) -> (from_levels, to_levels, update)`
- `model_noise_levels(levels, timesteps, sampling_timesteps, stabilization_level) -> Tensor`

**Two level scales.** Training uses real diffusion steps `0 .. timesteps - 1`, where `0` is the least noise and `timesteps - 1` is pure noise. Sampling uses schedule levels `0 .. S` with `S = sampling_timesteps`, where `S` is pure noise and `0` is clean. Every returned level tensor is `torch.long`.

**train_noise_levels.** Draw the levels with exactly one call, `torch.randint(0, timesteps, (num_frames, batch_size), generator=generator)`, so frames are independent. `frame_mask` is `None` or a bool tensor of shape `(num_frames, batch_size)`; where it is `False` the frame is missing and its level becomes `timesteps - 1`. Return shape `(num_frames, batch_size)`.

**scheduling_matrix.** Returns shape `(height, horizon)`; column `t` is frame `t` of the chunk. With `u = uncertainty_scale` and `int` meaning Python's `int()` of a float:
- `"pyramid"`: `height = S + int((horizon - 1) * u) + 1` and `matrix[m, t] = clip(S + int(t * u) - m, 0, S)`.
- `"autoregressive"`: the pyramid with `u = S`; the `uncertainty_scale` argument is ignored.
- `"full_sequence"`: `height = S + 1` and `matrix[m, t] = S - m` for every `t`.

**step_noise_levels.** One sampling step moves from row `m` to row `m + 1`. `n_context` clean context frames come first, at schedule level 0. Return three 1-D tensors of length `n_context + horizon`: `from_levels = cat(zeros(n_context), matrix[m])`, `to_levels = cat(zeros(n_context), matrix[m + 1])`, and the bool `update = from_levels != to_levels` (only those frames are rewritten in this step).

**model_noise_levels.** Converts schedule levels to the real step given to the model, and stabilizes clean frames:
- `real = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1).long()[levels]`. Real step `-1` means clean. When `timesteps` is divisible by `S` this is `-1 + k * (timesteps // S)` for level `k`.
- Wherever `real` is `-1`, return `stabilization_level - 1` instead; otherwise return `real`. The output has the shape of `levels`.

**Constraints:**
- `train_noise_levels` uses only `generator`, never the global RNG, and calls `torch.randint` once even when `frame_mask` is given.
- Do not modify any input in place.
- `scheduling_matrix` raises `ValueError` unless `horizon` and `sampling_timesteps` are integers `>= 1`, `kind` is one of the three names, and `uncertainty_scale` is an int or float `>= 0`.
- `train_noise_levels` raises `ValueError` unless `num_frames`, `batch_size` and `timesteps` are integers `>= 1` and `frame_mask` is `None` or a bool tensor of shape `(num_frames, batch_size)`.
- `step_noise_levels` raises `ValueError` unless `matrix` is a 2-D `torch.long` tensor, `m` is an integer with `0 <= m <= height - 2`, and `n_context` is an integer `>= 0`.
- `model_noise_levels` raises `ValueError` unless `levels` is a `torch.long` tensor with values in `0 .. S`, `1 <= S <= timesteps`, and `1 <= stabilization_level <= timesteps`.
- Integers here are Python `int`; `bool` is rejected.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why independent levels.** Teacher forcing trains next-frame prediction from clean history; full-sequence diffusion noises every frame to one shared level. Diffusion Forcing (Chen et al. 2024, Section 3.2, Algorithm 1) samples a separate noise level for every token, so one network learns to denoise any mix of clean, partly noisy and fully noisy frames. That one network can then be sampled with many schedules without retraining. A missing frame gets the maximum level, which tells the model it carries no information.

**Reading the matrix.** In `df_base.py`, row 0 is all `S` (pure noise) and the last row is all `0` (clean); sampling applies one denoising step per pair of adjacent rows. The paper's Algorithm 2 writes the same grid with the row index running the other way, from `M - 1` down to `0`. With `"full_sequence"` every frame descends together. With `"pyramid"` and `u = 1`, frame `t` starts descending `t` rows after frame 0, so nearer frames are always cleaner: one instance of the paper's "zig-zag" schedules (any `u > 0`; Section 3.3's example offsets frames by `K / 2`), which keeps the far future more uncertain than the near future. `"autoregressive"` uses `u = S`: frame `t + 1` only starts once frame `t` is clean, which is next-frame generation inside one chunk. The upstream default config uses `"autoregressive"` with `chunk_size: 1`.

**Why stabilize.** Feeding the model its own generated frames as perfectly clean context lets small errors compound over a long rollout. Section 3.3 conditions on the past at a small noise level `0 < k << K` instead. Upstream does this in `ddim_sample_step`: a clean frame (real step `-1`) is shown to the model at step `stabilization_level - 1` (10 in the base config, 15 in the video config), and frames whose level does not change in a step are copied through unchanged. Oasis uses the same idea with `stabilization_level = 15` over 1000 steps.

**Two conventions to keep apart.** The paper samples training levels from `0 .. K` with `0` clean. The code draws real steps from `0 .. timesteps - 1`, so step `0` still carries one step of noise and no training frame is perfectly clean; clean only exists as `-1` at sampling time. The scheduling matrix is in sampling steps (`S` of them, fewer than `timesteps` for DDIM), and `model_noise_levels` maps it back to real steps.""",
    "advisory_prerequisites": ["st_transformer_attention", "noise_schedule", "ddim_step"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "In the pyramid formula, which row is the first where frame t starts to lose noise, and which row is the first where the last frame reaches 0? What does a scale of S do to the start rows of neighbouring frames? Which schedule level maps to real step -1, and what does the model see for it?"},
        {"level": 2, "kind": "analysis", "content": "Draw the training levels with the single randint call, then overwrite the missing frames with torch.where. Build the pyramid with two loops (or broadcasting) on Python ints, computing int(t * u) for each column before subtracting the row index, and clip to [0, S]. Route autoregressive to the pyramid with u = S and build full_sequence from a descending arange. For one step, prepend n_context zeros to rows m and m + 1 and compare them. For the model levels, index the long linspace with the levels and replace -1 by stabilization_level - 1. Validate every argument first."},
    ],
    "model_connections": [
        "Diffusion Forcing (Chen et al. 2024) trains a causal video or planning model with per-token noise levels and samples it with pyramid, autoregressive or full-sequence scheduling matrices.",
        "open-oasis (Etched and Decart), whose generate.py cites Diffusion Forcing, generates a Minecraft-like world one frame at a time and feeds past frames to the DiT at noise step stabilization_level - 1 = 14 of 1000.",
        "Dreamer 4 (Hafner et al. 2025) uses diffusion forcing in its dynamics model, assigning a signal level to each time step, and slightly corrupts past inputs at inference to make the model robust to its own errors.",
        "GameNGen (Valevski et al. 2024) adds a sampled amount of Gaussian noise to context frames during training and gives the model the discretized noise level, to avoid drift under autoregressive rollout.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One trained model supports full-sequence, autoregressive and zig-zag sampling; only the integer matrix changes.",
            "Missing frames are handled by setting their level to pure noise, with no separate masking path in the model.",
            "Conditioning on slightly noised context makes long autoregressive rollouts more stable.",
        ],
        "cons": [
            "The model must learn every combination of per-frame noise levels, a harder task than one shared level.",
            "A pyramid with u > 0 needs more rows, so more network calls, than full-sequence sampling of the same chunk.",
            "The stabilization level is a tuned hyperparameter: too high blurs the context, too low brings back drift.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/df_base.py",
            "commit": "475e0bcab87545e48b24b39fb46a81fe59d80594",
            "path": "algorithms/diffusion_forcing/df_base.py",
            "symbol": "DiffusionForcingBase._generate_noise_levels, _generate_scheduling_matrix, _generate_pyramid_scheduling_matrix, validation_step",
            "license": "MIT",
            "adapted": "The random_all training levels with missing frames set to timesteps - 1; the pyramid, autoregressive and full_sequence matrices; "
                       "the from/to levels of one validation step with zero-level context frames prepended.",
            "simplifications": "The generator is passed in instead of using the global RNG; frame_stack is fixed at 1, so the mask is (T, B); "
                               "levels are a torch.long tensor instead of numpy int64 and are not repeated over the batch; "
                               "an unknown kind raises ValueError (upstream returns None). "
                               "Divergence: the trapezoid kind is left out, because upstream never writes column horizon // 2 for an even horizon, "
                               "which stays clean in every row; the oracle test pins this quirk.",
        },
        {
            "kind": "code",
            "url": "https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/models/diffusion.py",
            "commit": "475e0bcab87545e48b24b39fb46a81fe59d80594",
            "path": "algorithms/diffusion_forcing/models/diffusion.py",
            "symbol": "Diffusion.sample_step, Diffusion.ddim_sample_step",
            "license": "MIT",
            "adapted": "The linspace(-1, timesteps - 1, sampling_timesteps + 1).long() conversion from schedule levels to real steps, "
                       "the substitution of stabilization_level - 1 for real step -1, and the rule that frames whose level does not change keep their input.",
            "simplifications": "Only the integer levels are computed; the context scaling by q_sample, the model call and the DDIM update are left out. "
                               "The update mask is returned by step_noise_levels on schedule levels, which is equivalent because the real-step map is strictly increasing when sampling_timesteps <= timesteps.",
        },
        {
            "kind": "code",
            "url": "https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/generate.py",
            "commit": "f59deef2c019c212bd0c5a3a5b986a51f3701847",
            "path": "generate.py",
            "symbol": "main (sampling loop, noise_range, stabilization_level, t_ctx)",
            "license": "MIT",
            "adapted": "Cross-check only: one-frame-at-a-time sampling with context frames at stabilization_level - 1 and the new frame at noise_range[noise_idx].",
            "simplifications": "The DiT, VAE, actions and sliding window are left out; only the level tensors are compared in the oracle test.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2407.01392",
            "section": "Chen et al. 2024, Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion, Section 3.2 (Algorithms 1 and 2) and Section 3.3",
            "note": "Independent per-token noise levels at training time, the M x T scheduling matrix, the zig-zag schedule that keeps the future uncertain, "
                    "and conditioning on slightly noisy past tokens to stabilize long rollouts.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2408.14837",
            "section": "Valevski et al. 2024, Diffusion Models Are Real-Time Game Engines, Section 3.2.1 (Noise Augmentation) and Section 4.2",
            "note": "Context-frame noise augmentation with the level given to the model (maximum 0.7, 10 buckets).",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2509.24527",
            "section": "Hafner et al. 2025, Training Agents Inside of Scalable World Models, Section 2 (Diffusion forcing) and Section 3.2 (inference paragraph)",
            "note": "Per-time-step signal levels in the dynamics model and slight corruption of past inputs at inference.",
        },
    ],
    "tests": [
        {"name": "Pyramid matrix for three frames", "behavior": "state.invariant", "code": r"""
import torch
out = {fn}(3, 3, "pyramid", 1.0)
assert out.dtype == torch.long, out.dtype
assert out.tolist() == [
    [3, 3, 3],
    [2, 3, 3],
    [1, 2, 3],
    [0, 1, 2],
    [0, 0, 1],
    [0, 0, 0],
], out
"""},
        {"name": "Autoregressive and full-sequence matrices", "behavior": "state.invariant", "code": r"""
import torch
ar = {fn}(2, 2, "autoregressive", 1.0)
assert ar.tolist() == [[2, 2], [1, 2], [0, 2], [0, 1], [0, 0]], ar
full = {fn}(2, 2, "full_sequence")
assert full.tolist() == [[2, 2], [1, 1], [0, 0]], full
"""},
        {"name": "Training levels, one sampling step and model levels", "behavior": "state.invariant", "code": r"""
import torch
g = {fn}.__globals__
mask = torch.ones(4, 2, dtype=torch.bool)
mask[3, 1] = False
levels = g["train_noise_levels"](4, 2, 10, torch.Generator().manual_seed(0), mask)
want = torch.randint(0, 10, (4, 2), generator=torch.Generator().manual_seed(0))
want[3, 1] = 9
assert levels.dtype == torch.long and torch.equal(levels, want), (levels, want)

matrix = {fn}(2, 2, "autoregressive")
frm, to, update = g["step_noise_levels"](matrix, 2, 1)
assert frm.tolist() == [0, 0, 2] and to.tolist() == [0, 0, 1], (frm, to)
assert update.tolist() == [False, False, True], update

real = g["model_noise_levels"](frm, 100, 2, 15)
assert real.tolist() == [14, 14, 99], real   # linspace(-1, 99, 3) = [-1, 49, 99]
"""},
        {"name": "Pyramid matches the formula for any horizon and scale", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Use height = S + int((horizon - 1) * u) + 1 and entry clip(S + int(t * u) - m, 0, S), taking int() of t * u before subtracting the row index.",
         "code": _ORACLE + r"""
cases = [(1, 1, 1.0), (4, 3, 0.5), (5, 4, 1.5), (3, 5, 2.0), (6, 2, 0.25), (4, 3, 2.5), (5, 3, 0.75), (2, 7, 1), (7, 1, 0.5), (3, 4, 0)]
for horizon, S, u in cases:
    got = {fn}(horizon, S, "pyramid", u)
    want = want_pyramid(horizon, S, u)
    assert got.dtype == torch.long, got.dtype
    assert got.shape == want.shape, (horizon, S, u, got.shape, want.shape)
    assert torch.equal(got, want), (horizon, S, u, got, want)
"""},
        {"name": "Autoregressive ignores the scale and finishes each frame before the next", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Autoregressive is the pyramid with uncertainty_scale replaced by sampling_timesteps, whatever scale is passed.",
         "code": _ORACLE + r"""
for horizon, S in ((1, 3), (3, 2), (4, 3), (2, 5)):
    for u in (0, 0.5, 1.0, 3.0):
        got = {fn}(horizon, S, "autoregressive", u)
        assert torch.equal(got, want_pyramid(horizon, S, S)), (horizon, S, u, got)
    got = {fn}(horizon, S, "autoregressive")
    for m in range(got.shape[0]):
        for t in range(1, horizon):
            if got[m, t] < S:
                assert got[m, t - 1] == 0, ("frame t started before frame t - 1 was clean", horizon, S, got)
"""},
        {"name": "Full-sequence matrix for any shape", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "full_sequence has S + 1 rows, from all S down to all 0, and ignores uncertainty_scale.",
         "code": _ORACLE + r"""
for horizon, S in ((1, 1), (3, 4), (5, 2), (2, 10)):
    for u in (0, 1.0, 2.5):
        got = {fn}(horizon, S, "full_sequence", u)
        assert got.dtype == torch.long and torch.equal(got, want_full(horizon, S)), (horizon, S, u, got)
assert torch.equal({fn}(4, 3, "pyramid", 0), want_full(4, 3)), "a pyramid with scale 0 is the full-sequence matrix"
"""},
        {"name": "Every schedule starts at pure noise, ends clean and never adds noise", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Row 0 must be all S, the last row all 0 and the only all-zero row, and each entry may drop by 0 or 1 from one row to the next.",
         "code": r"""
import torch
for kind in ("pyramid", "autoregressive", "full_sequence"):
    for horizon, S, u in ((1, 2, 1.0), (4, 3, 1.0), (5, 2, 0.5), (3, 4, 2.0), (6, 3, 1.5)):
        M = {fn}(horizon, S, kind, u)
        assert M.shape[1] == horizon, (kind, M.shape)
        assert (M[0] == S).all() and (M[-1] == 0).all(), (kind, horizon, S, u, M)
        assert int(M.min()) >= 0 and int(M.max()) <= S, (kind, M)
        drop = M[:-1] - M[1:]
        assert ((drop == 0) | (drop == 1)).all(), (kind, horizon, S, u, M)
        assert not (M[:-1] == 0).all(dim=1).any(), ("no all-zero row before the last", kind, horizon, S, u, M)
"""},
        {"name": "Training levels are independent, seeded and respect the frame mask", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Draw torch.randint(0, timesteps, (num_frames, batch_size), generator=generator) once, then set frames where frame_mask is False to timesteps - 1.",
         "code": _ORACLE + r"""
train, _, _ = fns({fn})
for seed, (T, B, steps) in enumerate(((5, 3, 1000), (8, 2, 7), (1, 1, 1), (6, 4, 50), (3, 5, 2))):
    gm = torch.Generator().manual_seed(100 + seed)
    mask = torch.rand(T, B, generator=gm) > 0.4
    mask_copy = mask.clone()
    for use_mask in (False, True):
        global_state = torch.get_rng_state()
        got = train(T, B, steps, torch.Generator().manual_seed(seed), mask if use_mask else None)
        assert torch.equal(torch.get_rng_state(), global_state), "train_noise_levels touched the global RNG"
        want = torch.randint(0, steps, (T, B), generator=torch.Generator().manual_seed(seed))
        if use_mask:
            want = torch.where(mask, want, torch.full_like(want, steps - 1))
        assert got.dtype == torch.long and got.shape == (T, B), (got.dtype, got.shape)
        assert torch.equal(got, want), (seed, use_mask, got, want)
    assert torch.equal(mask, mask_copy), "frame_mask was modified"
big = train(64, 64, 4, torch.Generator().manual_seed(9))
assert int(big.min()) == 0 and int(big.max()) == 3, "levels must cover 0 .. timesteps - 1"
"""},
        {"name": "Step levels prepend clean context and mark changed frames", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "from and to are zeros(n_context) followed by rows m and m + 1; update is from != to, so context frames are never updated.",
         "code": _ORACLE + r"""
_, step, _ = fns({fn})
M = want_pyramid(3, 4, 1.0)
M_copy = M.clone()
for m in range(M.shape[0] - 1):
    for n_ctx in (0, 1, 3):
        frm, to, update = step(M, m, n_ctx)
        z = [0] * n_ctx
        assert frm.dtype == torch.long and to.dtype == torch.long and update.dtype == torch.bool, (frm.dtype, to.dtype, update.dtype)
        assert frm.tolist() == z + M[m].tolist() and to.tolist() == z + M[m + 1].tolist(), (m, n_ctx, frm, to)
        assert update.tolist() == [a != b for a, b in zip(frm.tolist(), to.tolist())], (m, n_ctx, update)
        assert not update[:n_ctx].any()
assert torch.equal(M, M_copy), "matrix was modified"
frm, to, update = step(M, 0, 2)
assert update.tolist() == [False, False, True, False, False], update
"""},
        {"name": "Model levels map to real steps and stabilize clean frames", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Index torch.linspace(-1, timesteps - 1, S + 1).long() with the levels and replace real step -1 (clean) by stabilization_level - 1.",
         "code": _ORACLE + r"""
_, _, model = fns({fn})
for T, S, stab in ((1000, 10, 15), (1000, 50, 10), (100, 4, 1), (8, 8, 3), (12, 3, 12), (1, 1, 1)):
    levels = torch.arange(S + 1).repeat(2, 1)
    got = model(levels, T, S, stab)
    want = torch.tensor([[stab - 1] + [-1 + k * (T // S) for k in range(1, S + 1)]] * 2)
    assert got.dtype == torch.long and got.shape == levels.shape, (got.dtype, got.shape)
    assert torch.equal(got, want), (T, S, stab, got, want)
lv = torch.tensor([3, 0, 2, 0])
lv_copy = lv.clone()
assert model(lv, 30, 3, 5).tolist() == [29, 4, 19, 4]
assert torch.equal(lv, lv_copy), "levels were modified"
"""},
        {"name": "A full chunk rollout denoises each new frame exactly S times", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Walk every pair of adjacent rows with step_noise_levels; each new frame must be updated S times and each context frame never, and model levels must never show -1.",
         "code": _ORACLE + r"""
_, step, model = fns({fn})
for kind, horizon, S, u, n_ctx in (("pyramid", 4, 5, 1.0, 2), ("autoregressive", 3, 3, 1.0, 1), ("full_sequence", 5, 4, 1.0, 0), ("pyramid", 6, 2, 0.5, 3)):
    M = {fn}(horizon, S, kind, u)
    count = torch.zeros(n_ctx + horizon, dtype=torch.long)
    for m in range(M.shape[0] - 1):
        frm, to, update = step(M, m, n_ctx)
        count += update.long()
        real = model(frm, 1000, S, 15)
        assert (real >= 0).all(), real
        assert (real[:n_ctx] == 14).all(), real
    assert count.tolist() == [0] * n_ctx + [S] * horizon, (kind, count)
"""},
        {"name": "Rejects bad arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for a bad kind, non-positive or non-integer sizes, a negative scale, a bad frame mask, an out-of-range row, or levels and stabilization outside their ranges.",
         "code": _ORACLE + r"""
train, step, model = fns({fn})
def rejects(f, *args):
    try:
        f(*args)
    except ValueError:
        return True
    return False
g = torch.Generator().manual_seed(0)
for args in ((0, 3, "pyramid"), (3, 0, "pyramid"), (2.0, 3, "pyramid"), (True, 3, "pyramid"), (3, 3, "trapezoid"), (3, 3, "zigzag"), (3, 3, "pyramid", -0.5), (3, 3, "pyramid", "1")):
    assert rejects({fn}, *args), f"scheduling_matrix accepted {args!r}"
assert rejects(train, 0, 2, 10, g) and rejects(train, 2, 0, 10, g) and rejects(train, 2, 2, 0, g), "train_noise_levels accepted a zero size"
assert rejects(train, 2, 2, 10, g, torch.ones(2, 3, dtype=torch.bool)), "accepted a mask of the wrong shape"
assert rejects(train, 2, 2, 10, g, torch.ones(2, 2)), "accepted a float mask"
M = want_pyramid(3, 2, 1.0)
assert rejects(step, M, M.shape[0] - 1, 0), "accepted the last row as m"
assert rejects(step, M, -1, 0) and rejects(step, M, 0, -1), "accepted a negative m or n_context"
assert rejects(step, M.float(), 0, 0) and rejects(step, M[0], 0, 0), "accepted a non-long or 1-D matrix"
assert rejects(model, torch.tensor([0, 4]), 100, 3, 5), "accepted a level above S"
assert rejects(model, torch.tensor([-1, 0]), 100, 3, 5), "accepted a negative level"
assert rejects(model, torch.tensor([0.0, 1.0]), 100, 3, 5), "accepted float levels"
assert rejects(model, torch.tensor([0, 1]), 10, 20, 5), "accepted S > timesteps"
assert rejects(model, torch.tensor([0, 1]), 100, 3, 0) and rejects(model, torch.tensor([0, 1]), 100, 3, 101), "accepted a bad stabilization_level"
"""},
    ],
    "solution": '''import torch


def _check_int(name, value, low):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")


def train_noise_levels(num_frames, batch_size, timesteps, generator, frame_mask=None):
    _check_int("num_frames", num_frames, 1)
    _check_int("batch_size", batch_size, 1)
    _check_int("timesteps", timesteps, 1)
    if frame_mask is not None:
        if not torch.is_tensor(frame_mask) or frame_mask.dtype != torch.bool or frame_mask.shape != (num_frames, batch_size):
            raise ValueError("frame_mask must be a bool tensor of shape (num_frames, batch_size)")
    levels = torch.randint(0, timesteps, (num_frames, batch_size), generator=generator)
    if frame_mask is not None:
        # A missing frame carries no information: show it at pure noise.
        levels = torch.where(frame_mask, levels, torch.full_like(levels, timesteps - 1))
    return levels


def _pyramid(horizon, S, u):
    height = S + int((horizon - 1) * u) + 1
    matrix = torch.zeros(height, horizon, dtype=torch.long)
    for m in range(height):
        for t in range(horizon):
            matrix[m, t] = S + int(t * u) - m
    return matrix.clamp(0, S)


def scheduling_matrix(horizon, sampling_timesteps, kind, uncertainty_scale=1.0):
    _check_int("horizon", horizon, 1)
    _check_int("sampling_timesteps", sampling_timesteps, 1)
    if isinstance(uncertainty_scale, bool) or not isinstance(uncertainty_scale, (int, float)) or uncertainty_scale < 0:
        raise ValueError("uncertainty_scale must be a number >= 0")
    S = sampling_timesteps
    if kind == "pyramid":
        return _pyramid(horizon, S, uncertainty_scale)
    if kind == "autoregressive":
        return _pyramid(horizon, S, S)
    if kind == "full_sequence":
        return torch.arange(S, -1, -1, dtype=torch.long)[:, None].repeat(1, horizon)
    raise ValueError(f"unknown kind {kind!r}")


def step_noise_levels(matrix, m, n_context):
    if not torch.is_tensor(matrix) or matrix.ndim != 2 or matrix.dtype != torch.long:
        raise ValueError("matrix must be a 2-D torch.long tensor")
    _check_int("n_context", n_context, 0)
    if isinstance(m, bool) or not isinstance(m, int) or not 0 <= m <= matrix.shape[0] - 2:
        raise ValueError("m must be an integer in [0, height - 2]")
    context = torch.zeros(n_context, dtype=torch.long)
    from_levels = torch.cat([context, matrix[m]])
    to_levels = torch.cat([context, matrix[m + 1]])
    return from_levels, to_levels, from_levels != to_levels


def model_noise_levels(levels, timesteps, sampling_timesteps, stabilization_level):
    _check_int("timesteps", timesteps, 1)
    _check_int("sampling_timesteps", sampling_timesteps, 1)
    _check_int("stabilization_level", stabilization_level, 1)
    if sampling_timesteps > timesteps or stabilization_level > timesteps:
        raise ValueError("need sampling_timesteps <= timesteps and stabilization_level <= timesteps")
    if not torch.is_tensor(levels) or levels.dtype != torch.long:
        raise ValueError("levels must be a torch.long tensor")
    if levels.numel() and (int(levels.min()) < 0 or int(levels.max()) > sampling_timesteps):
        raise ValueError("levels must lie in [0, sampling_timesteps]")
    real_steps = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1).long()
    real = real_steps[levels]
    # Clean context (-1) is shown to the model as slightly noisy.
    return torch.where(real < 0, torch.full_like(real, stabilization_level - 1), real)
''',
    "interview_questions": interview(
        concept=[
            "How does Diffusion Forcing's training differ from teacher forcing and from full-sequence diffusion, and what does the per-frame noise level buy at sampling time?",
            "What does one row of a scheduling matrix represent, and how do the full-sequence, pyramid and autoregressive matrices differ in the order frames are denoised?",
        ],
        deep_dive=[
            "Write the pyramid scheduling matrix entry for row m and frame t with sampling steps S and uncertainty scale u. How many rows does it have, and why is int(t * u) taken before subtracting m?",
            "During one sampling step with clean context frames in front of the chunk, which frames are rewritten and which are copied through, and what noise level does the model see for the context?",
            "Why does stabilizing the context with a small noise level help long autoregressive rollouts, and how would you pick the level?",
        ],
        tradeoffs=[
            "A pyramid with uncertainty scale 1 versus the autoregressive schedule versus full-sequence denoising: compare the number of network calls per chunk and how far ahead each lets the model plan.",
            "Independent per-frame noise levels at training time versus one shared level per sequence: what does each cost in training difficulty and gain in sampling flexibility?",
        ],
    ),
}
