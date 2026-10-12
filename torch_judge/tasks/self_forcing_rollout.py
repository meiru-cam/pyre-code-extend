"""Self Forcing: the training-time autoregressive self-rollout with KV caching and gradient truncation."""

from ._interview import interview

# Fakes shared by the evaluator cases. The model is a tiny causal "video" network:
# denoise() reads the mean of every cached entry, kv() summarizes a block per frame.
_FAKES = r"""
import torch

SHIFT = 5.0
_SIG = torch.linspace(1.0, 0.0, 1001)[:-1]
_SIG = SHIFT * _SIG / (1 + (SHIFT - 1) * _SIG)
_TS = _SIG * 1000

def snap_sigma(t):
    # nearest entry of a shifted 1000-step flow-matching table: snap_sigma(0) is about 0.005, not 0
    return float(_SIG[torch.argmin((_TS - float(t)).abs())])

def linear_sigma(t):
    return float(t) / 1000.0

class FakeModel:
    def __init__(self, seed, channels=2, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(channels, 1, 1, generator=g, dtype=dtype) * 0.5 + 1.0)
        self.u = torch.nn.Parameter(torch.randn(channels, 1, 1, generator=g, dtype=dtype) * 0.5 + 1.0)
        self.calls = []

    def _context(self, cache, x):
        if len(cache) == 0:
            return torch.zeros(x.shape[0], dtype=x.dtype)
        return torch.stack([entry.mean(dim=1) for entry in cache]).mean(dim=0)

    def denoise(self, x, t, cache):
        self.calls.append({"kind": "denoise", "t": float(t), "n": len(cache), "grad": torch.is_grad_enabled(),
                           "x_requires_grad": x.requires_grad, "x": x.detach().clone(),
                           "cache_grad": any(e.requires_grad for e in cache)})
        ctx = self._context(cache, x)
        s = float(t) / 1000.0
        return self.w * torch.tanh(x) * (1 - 0.5 * s) + 0.1 * x + 0.3 * ctx[:, None, None, None, None]

    def kv(self, x, t, cache):
        self.calls.append({"kind": "kv", "t": float(t), "n": len(cache), "grad": torch.is_grad_enabled(),
                           "x": x.detach().clone()})
        ctx = self._context(cache, x)
        return (self.u * torch.tanh(x)).mean(dim=(2, 3, 4)) * (1 + float(t) / 1000.0) + 0.5 * ctx[:, None]

def manual_rollout(model, noise, timesteps, sigma_of, exits, fpb, seed, ctx_t=0, grad_from=0):
    # An independent step-by-step rollout used as the expected value.
    steps = [t for t in timesteps]
    if steps[-1] == 0:
        steps.pop()
    g = torch.Generator().manual_seed(seed)
    entries = []
    pieces = []
    for b in range(noise.shape[1] // fpb):
        x = noise[:, b * fpb:(b + 1) * fpb]
        k = 0
        while True:
            last = k == exits[b]
            if last and b * fpb >= grad_from:
                pred = model.denoise(x, steps[k], entries)
            else:
                with torch.no_grad():
                    pred = model.denoise(x, steps[k], entries)
            if last:
                break
            s = sigma_of(steps[k + 1])
            with torch.no_grad():
                x = (1 - s) * pred + s * torch.randn(pred.shape, generator=g, dtype=pred.dtype)
            k += 1
        pieces.append(pred)
        s = sigma_of(ctx_t)
        with torch.no_grad():
            noisy_ctx = (1 - s) * pred + s * torch.randn(pred.shape, generator=g, dtype=pred.dtype)
            entries.append(model.kv(noisy_ctx, ctx_t, entries))
    return torch.cat(pieces, dim=1)
"""

TASK = {
    "title": "Self Forcing Training Rollout",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "self_forcing_rollout",
    "description_en": r"""Implement the training-time rollout of Self Forcing for a few-step, block-autoregressive video diffusion model. Generate the video block by block from the model's own outputs, cache each finished block, and keep gradients only where Self Forcing keeps them.

**Signature:** `self_forcing_rollout(model, noise, timesteps, sigma_of, exit_steps, frames_per_block, generator, context_noise=0, grad_from_frame=0) -> Tensor`

**Inputs:**
- `noise` is a float tensor of shape `(B, F, C, H, W)`, the starting noise of every latent frame. `F` is a multiple of `frames_per_block`. Block `b` covers frames `b * frames_per_block` up to, but not including, `(b + 1) * frames_per_block`.
- `timesteps` lists the denoising timesteps from noisiest to least noisy, for example `[1000, 750, 500, 250]`. If its last entry is `0`, drop that entry first. Call the remaining list `steps`, indexed from 0.
- `exit_steps` has one int per block with `0 <= exit_steps[b] < len(steps)`. It is the index of the step whose prediction becomes block `b`'s output.
- `sigma_of(t)` returns the noise level of timestep `t`, a float or a 0-dim tensor.
- `generator` is the `torch.Generator` for every fresh noise draw.
- `model.denoise(x_t, t, cache)` returns the predicted clean block `x0_hat`, shaped like `x_t`.
- `model.kv(x, t, cache)` returns the cache entry for a block.
- `cache` is the list of entries of the earlier blocks, oldest first. Never pass an entry of the current block or a later one.

**Re-noising:** `renoise(x0, t)` draws `eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)` and returns `(1 - sigma_of(t)) * x0 + sigma_of(t) * eps`. Draw exactly where the steps below say, in that order, even when `sigma_of(t)` is 0.

**Each block `b`, in order,** with `e = exit_steps[b]` and `x` set to the block's slice of `noise`:
- For `j = 0, ..., e - 1`: `x0_hat = model.denoise(x, steps[j], cache)`, then `x = renoise(x0_hat, steps[j + 1])`. Both run under `torch.no_grad()`.
- Exit step: `x0_hat = model.denoise(x, steps[e], cache)`. If the block's first frame index is `>= grad_from_frame`, make this call outside `torch.no_grad()`, so it keeps gradients. Otherwise make it under `torch.no_grad()`. This `x0_hat` is block `b`'s output.
- Cache update: `ctx = renoise(x0_hat, context_noise)`, then append `model.kv(ctx, context_noise, cache)` to the cache. The `model.kv` call runs under `torch.no_grad()`.

**Output and errors:**
- Return the block outputs concatenated along the frame dimension, shape `(B, F, C, H, W)`.
- Raise `ValueError` when `noise` is not 5-D, `frames_per_block` is not a positive int, `F` is not a multiple of it, `len(exit_steps)` is not the number of blocks, an exit step is not an int in range, or no step is left after dropping the zero.
- Never modify `noise` in place.

**Example:** with `steps = [1000, 750, 500, 250]`, `exit_steps = [2, 0]` and two blocks:

    block 0: denoise at 1000, renoise to 750, denoise at 750, renoise to 500,
             denoise at 500 with gradient -> output; cache += kv(renoise(x0_hat, 0), 0, [])
    block 1: denoise at 1000 with the 1-entry cache, with gradient -> output; cache update
    model calls: 3 + 1 denoise, 2 kv; noise draws: sum(exit_steps) + 2 = 4

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Exposure bias.** An autoregressive video diffusion model trained with Teacher Forcing sees ground-truth context frames, and with Diffusion Forcing it sees noised ground-truth context. At inference it conditions on its own imperfect frames instead, so errors accumulate. Self Forcing (Huang et al., 2025) closes the gap by running the inference procedure during training: each block is denoised from the model's own earlier outputs through a KV cache, and a video-level distribution-matching loss (DMD, SiD or a GAN in the paper) scores the whole rollout.

**Exit step and gradient truncation.** Backpropagating through every denoising step of every block costs too much memory. Algorithm 1 samples one step `s` uniformly from `1..T` per training iteration and uses that step's prediction as the output. Its loop counts `j` down from `T`, so `s` corresponds to `e = T - s` in this exercise's 0-based, noisiest-first list. Only that step gets gradients. Because `s` is random, every intermediate step is supervised over training. The official code draws one index per block with `torch.randint(0, T)` and uses the first one for every block when `same_step_across_blocks` is true, which is the default config. Here the caller passes `exit_steps`, so both cases are covered.

**The cache carries no gradient.** The paper detaches earlier frames from the current one by stopping gradients into the KV cache embeddings. In Algorithm 1 the cache entry is computed from the clean prediction at timestep 0. The official code first re-noises the prediction to `context_noise` (default 0), and its scheduler snaps a timestep to the nearest entry of a 1000-entry table whose smallest noise level is about 0.005. So `context_noise = 0` still mixes in a little fresh noise there. `sigma_of` stands for that lookup.

**grad_from_frame.** The official code runs exit steps of blocks that start before `num_output_frames - 21` without gradients, so a longer rollout trains only the blocks that start within its last 21 latent frames. With the released 21-frame training rollouts this is 0, and every block gets gradients.

**Schedule.** The paper uses a uniform 4-step schedule `[1000, 750, 500, 250]` with timestep shift 5 on Wan2.1-T2V-1.3B, generating 3 latent frames per block in the chunk-wise variant. With `warp_denoising_step` the code maps these through the shifted table to about `[1000, 937.5, 833.3, 625]`.""",
    "advisory_prerequisites": ["kv_cache", "flow_matching", "ddim_step", "diffusion_forcing_schedule"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How many denoise calls does a block with exit index e make, and which of them is allowed to build an autograd graph? Which timestep does each re-noise use: the step just taken or the next one? What exactly goes into the cache: the prediction, the noisy input, or something else, and at which timestep? When should the trailing 0 be removed relative to checking the exit indices?"},
        {"level": 2, "kind": "analysis", "content": "Validate, then build steps by dropping a trailing 0. Keep a plain Python list as the cache. For each block, slice the noise, loop j over the first e steps under no_grad (denoise, then re-noise to steps[j + 1] with one randn draw from the generator), then make the exit call with or without no_grad depending on the block's start frame versus grad_from_frame. Store that prediction, re-noise it to context_noise with one more draw, and append model.kv of that under no_grad. Concatenate the stored predictions on dim 1."},
    ],
    "model_connections": [
        "Self-Forcing's SelfForcingTrainingPipeline.inference_with_trajectory is this loop for Wan2.1: the DMD, SiD and GAN trainers call it to produce the video their distribution-matching loss scores.",
        "CausVid also trains a few-step causal video model with DMD, but on Diffusion Forcing outputs; the Self Forcing paper argues that this training distribution deviates from the inference-time one and undermines its effectiveness.",
        "Self-Forcing's CausalInferencePipeline runs the same block loop through every step without gradients, but passes the clean prediction to the cache-update call at timestep context_noise, without the extra re-noise the training pipeline applies.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Training sees exactly the context distribution inference produces, so the model learns to recover from its own errors instead of only from ground truth.",
            "A random exit step plus a gradient-free cache bounds memory to one differentiable model call per block while still supervising every denoising step over training.",
            "The rollout uses ordinary full attention with a KV cache, so it needs no special causal masks; the paper reports per-iteration time comparable to Teacher and Diffusion Forcing.",
        ],
        "cons": [
            "The rollout is sequential across blocks, unlike Teacher or Diffusion Forcing, which train all frames in parallel.",
            "No gradient flows from a block into earlier blocks through the cache, which the paper notes may limit learning of long-range dependencies.",
            "Quality still degrades for videos much longer than the training rollout.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/guandeh17/Self-Forcing",
            "commit": "33593df3e81fa3ec10239271dd2c100facac6de1",
            "path": "pipeline/self_forcing_training.py",
            "symbol": "SelfForcingTrainingPipeline.inference_with_trajectory, SelfForcingTrainingPipeline.generate_and_sync_list",
            "license": "Apache-2.0",
            "adapted": "The block loop: drop a trailing zero timestep, denoise under no_grad up to the exit step, re-noise to the next timestep, make the exit call with gradients only for blocks at or after the gradient start frame, record the prediction, re-noise it to context_noise and rerun the model under no_grad to fill the cache.",
            "simplifications": "Exit indices are passed in instead of drawn with torch.randint on rank 0 and broadcast, so same_step_across_blocks and last_step_only become choices of exit_steps; "
                               "upstream's single generator call, which writes its keys and values into preallocated cache tensors at frame offsets, is split into denoise and kv callables and a Python list (upstream's denoising calls also write the current block's slots, which the context call then overwrites); "
                               "fresh noise comes from a passed torch.Generator instead of torch.randn_like, in the same order; "
                               "the hard-coded start_gradient_frame_index = num_output_frames - 21 becomes grad_from_frame; "
                               "initial_latent, independent_first_frame, conditioning and the cross-attention cache are dropped; "
                               "denoised_timestep_from and denoised_timestep_to are not returned; "
                               "bad inputs raise ValueError instead of failing an assert.",
        },
        {
            "kind": "code",
            "url": "https://github.com/guandeh17/Self-Forcing",
            "commit": "33593df3e81fa3ec10239271dd2c100facac6de1",
            "path": "utils/scheduler.py",
            "symbol": "FlowMatchScheduler.add_noise",
            "license": "Apache-2.0",
            "adapted": "The forward process (1 - sigma) * x0 + sigma * eps used for both re-noising calls.",
            "simplifications": "The nearest-timestep sigma lookup is passed in as sigma_of instead of being read from the scheduler's 1000-entry table.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2506.08009",
            "section": "Huang et al. 2025, Section 3.2 and Algorithm 1 (Self Forcing Training); Section 3.3; Appendix A (Noise schedule and model parameterization)",
            "note": "Self-rollout with KV caching during training, a uniformly sampled exit step with gradients only at that step, gradients stopped at the KV cache, and the 4-step shifted flow-matching schedule.",
        },
    ],
    "tests": [
        {"name": "Call order, timesteps and gradient modes for two blocks", "behavior": "events.ordering", "code": _FAKES + r"""
model = FakeModel(0, channels=1)
noise = torch.randn(1, 4, 1, 2, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
out = {fn}(model, noise, [1000, 750, 500, 250], linear_sigma, [2, 0], 2, torch.Generator().manual_seed(0))
assert out.shape == (1, 4, 1, 2, 2), out.shape
got = [(c["kind"], c["t"], c["n"], c["grad"]) for c in model.calls]
want = [("denoise", 1000.0, 0, False), ("denoise", 750.0, 0, False), ("denoise", 500.0, 0, True), ("kv", 0.0, 0, False),
        ("denoise", 1000.0, 1, True), ("kv", 0.0, 1, False)]
assert got == want, got
assert out.requires_grad, "the exit-step predictions should keep their gradients"
"""},
        {"name": "Re-noising draws from the generator in order", "behavior": "state.invariant", "code": r"""
import torch

class Constant:
    def __init__(self):
        self.seen = []
    def denoise(self, x, t, cache):
        self.seen.append(("denoise", x.clone()))
        return torch.full_like(x, 2.0)
    def kv(self, x, t, cache):
        self.seen.append(("kv", x.clone()))
        return x.mean(dim=(2, 3, 4))

model = Constant()
noise = torch.zeros(1, 1, 1, 1, 3, dtype=torch.float64)
out = {fn}(model, noise, [1000, 600, 0], lambda t: float(t) / 1000.0, [1], 1, torch.Generator().manual_seed(5), context_noise=100)
g = torch.Generator().manual_seed(5)
eps1 = torch.randn(1, 1, 1, 1, 3, generator=g, dtype=torch.float64)
eps2 = torch.randn(1, 1, 1, 1, 3, generator=g, dtype=torch.float64)
assert len(model.seen) == 3, [k for k, _ in model.seen]   # the trailing 0 is not a step
assert torch.allclose(model.seen[1][1], 0.4 * 2.0 + 0.6 * eps1), "step 1 input is renoise(x0_hat, 600)"
assert torch.allclose(model.seen[2][1], 0.9 * 2.0 + 0.1 * eps2), "cache input is renoise(x0_hat, context_noise)"
assert torch.allclose(out, torch.full_like(noise, 2.0)), out
"""},
        {"name": "Seeded rollouts match a step-by-step oracle", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Denoise steps[0..e-1] and re-noise each prediction to the next timestep, output the exit-step prediction, and cache kv of the prediction re-noised to context_noise.",
         "code": _FAKES + r"""
cases = [(0, 6, 3, [0, 3], 0), (1, 6, 2, [1, 2, 3], 0), (2, 4, 1, [3, 0, 2, 1], 250), (3, 6, 6, [2], 0), (4, 6, 3, [3, 3], 500)]
for seed, frames, fpb, exits, ctx_t in cases:
    noise = torch.randn(2, frames, 2, 2, 3, generator=torch.Generator().manual_seed(100 + seed), dtype=torch.float64)
    steps = [1000.0, 937.5, 833.3333, 625.0]
    out = {fn}(FakeModel(seed), noise, steps, snap_sigma, exits, fpb, torch.Generator().manual_seed(seed), context_noise=ctx_t)
    want = manual_rollout(FakeModel(seed), noise, steps, snap_sigma, exits, fpb, seed, ctx_t)
    assert out.shape == noise.shape, out.shape
    assert torch.allclose(out, want, atol=1e-10), (seed, (out - want).abs().max())
"""},
        {"name": "Each re-noise uses the next timestep", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After denoising at steps[j], re-noise the prediction to steps[j + 1], not to steps[j].",
         "code": _FAKES + r"""
model = FakeModel(7)
noise = torch.randn(1, 2, 2, 2, 2, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
{fn}(model, noise, [1000, 750, 500, 250], linear_sigma, [3], 2, torch.Generator().manual_seed(3))
den = [c for c in model.calls if c["kind"] == "denoise"]
assert [c["t"] for c in den] == [1000.0, 750.0, 500.0, 250.0], [c["t"] for c in den]
ref = FakeModel(7)
g = torch.Generator().manual_seed(3)
x = noise
with torch.no_grad():
    for j, nxt in enumerate([750.0, 500.0, 250.0]):
        pred = ref.denoise(x, den[j]["t"], [])
        s = nxt / 1000.0
        x = (1 - s) * pred + s * torch.randn(pred.shape, generator=g, dtype=pred.dtype)
        assert torch.allclose(den[j + 1]["x"], x, atol=1e-12), ("input of step", j + 1)
"""},
        {"name": "Only the exit step builds a graph", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Run every denoise call before the exit step and every model.kv call under torch.no_grad(); only the exit call keeps gradients.",
         "code": _FAKES + r"""
for exits in ([0, 3, 1], [2, 2, 2], [3, 0, 0]):
    model = FakeModel(11)
    noise = torch.randn(1, 6, 2, 2, 2, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    {fn}(model, noise, [1000, 750, 500, 250], snap_sigma, exits, 2, torch.Generator().manual_seed(0))
    den = [c for c in model.calls if c["kind"] == "denoise"]
    kvs = [c for c in model.calls if c["kind"] == "kv"]
    assert len(den) == sum(exits) + 3 and len(kvs) == 3, (exits, len(den), len(kvs))
    pos = 0
    for b, e in enumerate(exits):
        flags = [c["grad"] for c in den[pos:pos + e + 1]]
        assert flags == [False] * e + [True], (exits, b, flags)
        assert all(c["n"] == b for c in den[pos:pos + e + 1]), (exits, b, "cache size")
        assert not den[pos + e]["x_requires_grad"], (exits, b, "exit input should not carry a graph")
        pos += e + 1
    assert not any(c["grad"] for c in kvs), "model.kv must run under no_grad"
    assert not any(c["cache_grad"] for c in den), "cache entries must not require grad"
"""},
        {"name": "Gradients match the truncated rollout", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Gradients reach the model only through each block's exit call; no gradient may flow through the cache or earlier denoising steps.",
         "code": _FAKES + r"""
for seed, exits in ((21, [1, 3, 0]), (22, [2, 2, 2]), (23, [0, 0, 3])):
    noise = torch.randn(2, 6, 2, 2, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    weights = torch.randn(2, 6, 2, 2, 2, generator=torch.Generator().manual_seed(seed + 50), dtype=torch.float64)
    model = FakeModel(seed)
    out = {fn}(model, noise, [1000, 750, 500, 250], snap_sigma, exits, 2, torch.Generator().manual_seed(seed))
    (out * weights).sum().backward()
    ref = FakeModel(seed)
    want = manual_rollout(ref, noise, [1000, 750, 500, 250], snap_sigma, exits, 2, seed)
    (want * weights).sum().backward()
    assert model.w.grad is not None and torch.allclose(model.w.grad, ref.w.grad, atol=1e-10), (seed, "denoise parameter gradient")
    assert model.u.grad is None or torch.all(model.u.grad == 0), (seed, "the kv parameter must get no gradient")
"""},
        {"name": "Blocks before grad_from_frame carry no gradient", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Make the exit call under no_grad when the block's first frame index is below grad_from_frame, and with gradients from that frame on.",
         "code": _FAKES + r"""
for grad_from, want_flags in ((0, [True, True, True, True]), (2, [False, True, True, True]), (3, [False, False, True, True]), (5, [False, False, False, True]), (8, [False, False, False, False])):
    model = FakeModel(31)
    noise = torch.randn(1, 8, 2, 1, 2, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    out = {fn}(model, noise, [1000, 750, 500, 250], snap_sigma, [1, 0, 2, 3], 2, torch.Generator().manual_seed(1), grad_from_frame=grad_from)
    den = [c for c in model.calls if c["kind"] == "denoise"]
    ends = [1, 2, 5, 9]
    flags = [den[i]["grad"] for i in ends]
    assert flags == want_flags, (grad_from, flags)
    assert not any(c["grad"] for i, c in enumerate(den) if i not in ends), grad_from
    if any(want_flags):
        ref = FakeModel(31)
        want = manual_rollout(ref, noise, [1000, 750, 500, 250], snap_sigma, [1, 0, 2, 3], 2, 1, grad_from=grad_from)
        out.sum().backward(); want.sum().backward()
        assert torch.allclose(model.w.grad, ref.w.grad, atol=1e-10), grad_from
"""},
        {"name": "Cache entries come from the re-noised prediction at the context timestep", "visibility": "unshown", "behavior": "attention.cache",
         "failure_message": "Cache model.kv(renoise(x0_hat, context_noise), context_noise, cache) with x0_hat the exit-step prediction, drawing fresh noise even when the noise level is tiny.",
         "code": _FAKES + r"""
for ctx_t in (0, 300):
    model = FakeModel(41)
    noise = torch.randn(1, 4, 2, 2, 2, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    out = {fn}(model, noise, [1000, 750, 500, 250], snap_sigma, [0, 0], 2, torch.Generator().manual_seed(9), context_noise=ctx_t)
    kvs = [c for c in model.calls if c["kind"] == "kv"]
    assert [c["t"] for c in kvs] == [float(ctx_t)] * 2, [c["t"] for c in kvs]
    g = torch.Generator().manual_seed(9)
    s = snap_sigma(ctx_t)
    for b in range(2):
        pred = out[:, 2 * b:2 * b + 2].detach()
        eps = torch.randn(pred.shape, generator=g, dtype=pred.dtype)
        assert torch.allclose(kvs[b]["x"], (1 - s) * pred + s * eps, atol=1e-12), (ctx_t, b)
"""},
        {"name": "Each block uses its own exit step and a trailing zero is dropped", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Drop a trailing 0 from timesteps before indexing, and use exit_steps[b] for block b.",
         "code": _FAKES + r"""
model = FakeModel(51)
noise = torch.randn(1, 4, 2, 1, 2, generator=torch.Generator().manual_seed(51), dtype=torch.float64)
out = {fn}(model, noise, [1000, 500, 0], linear_sigma, [1, 0, 1, 0], 1, torch.Generator().manual_seed(2))
den = [(c["t"], c["n"]) for c in model.calls if c["kind"] == "denoise"]
assert den == [(1000.0, 0), (500.0, 0), (1000.0, 1), (1000.0, 2), (500.0, 2), (1000.0, 3)], den
want = manual_rollout(FakeModel(51), noise, [1000, 500, 0], linear_sigma, [1, 0, 1, 0], 1, 2)
assert torch.allclose(out, want, atol=1e-12), (out - want).abs().max()
try:
    {fn}(FakeModel(51), noise, [1000, 500, 0], linear_sigma, [2, 0, 0, 0], 1, torch.Generator().manual_seed(2))
except ValueError:
    pass
else:
    raise AssertionError("exit step 2 indexes the dropped zero timestep")
model = FakeModel(52)
{fn}(model, noise, [1000, 500, 0], linear_sigma, [1], 4, torch.Generator().manual_seed(2))
assert [c["n"] for c in model.calls] == [0, 0, 0], "one block of 4 frames: two denoise calls and one kv call"
"""},
        {"name": "Noise draw count and order", "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Draw one randn per re-noise (e per block for steps, one for the cache), from the passed generator, in rollout order.",
         "code": _FAKES + r"""
for exits in ([0, 0, 0], [3, 1, 2], [1, 3, 0]):
    g = torch.Generator().manual_seed(77)
    noise = torch.randn(1, 3, 2, 2, 2, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    {fn}(FakeModel(4), noise, [1000, 750, 500, 250], snap_sigma, exits, 1, g)
    probe = torch.Generator().manual_seed(77)
    for _ in range(sum(exits) + len(exits)):
        torch.randn((1, 1, 2, 2, 2), generator=probe, dtype=torch.float64)
    assert torch.equal(torch.randn(3, generator=g), torch.randn(3, generator=probe)), (exits, "wrong number of draws")
"""},
        {"name": "Inputs are kept and dtype follows the noise", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Do not modify noise in place, and draw fresh noise in the dtype of the prediction.",
         "code": _FAKES + r"""
model = FakeModel(61, dtype=torch.float32)
noise = torch.randn(2, 4, 2, 2, 2, generator=torch.Generator().manual_seed(61))
saved = noise.clone()
out = {fn}(model, noise, [1000, 750, 500, 250], snap_sigma, [3, 1], 2, torch.Generator().manual_seed(6))
assert torch.equal(noise, saved), "noise was modified"
assert out.dtype == torch.float32, out.dtype
assert all(c["x"].dtype == torch.float32 for c in model.calls), "a re-noised input changed dtype"
want = manual_rollout(FakeModel(61, dtype=torch.float32), noise, [1000, 750, 500, 250], snap_sigma, [3, 1], 2, 6)
assert torch.allclose(out, want, atol=1e-5), (out - want).abs().max()
"""},
        {"name": "Rejects malformed inputs", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for non-5-D noise, a bad frames_per_block, a frame count that is not a multiple of it, a wrong number of exit steps, an out-of-range exit step, or no steps.",
         "code": _FAKES + r"""
n = torch.zeros(1, 4, 1, 1, 1, dtype=torch.float64)
g = torch.Generator().manual_seed(0)
S = [1000, 750, 500, 250]
bad = [
    lambda: {fn}(FakeModel(0, 1), n[0], S, linear_sigma, [0, 0], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0, 0], 0, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0, 0], 3, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0, 4], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [-1, 0], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0, 1.0], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, [0], linear_sigma, [0, 0], 2, g),
    lambda: {fn}(FakeModel(0, 1), n, S, linear_sigma, [0, 0], 2.0, g),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch


def self_forcing_rollout(model, noise, timesteps, sigma_of, exit_steps, frames_per_block,
                         generator, context_noise=0, grad_from_frame=0):
    steps = list(timesteps)
    if steps and steps[-1] == 0:
        steps = steps[:-1]  # the clean end of the schedule is not a denoising step
    if not steps:
        raise ValueError("timesteps must contain at least one nonzero step")
    if noise.ndim != 5:
        raise ValueError("noise must have shape (B, F, C, H, W)")
    if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, int) or frames_per_block < 1:
        raise ValueError("frames_per_block must be a positive int")
    num_frames = noise.shape[1]
    if num_frames % frames_per_block != 0:
        raise ValueError("the frame count must be a multiple of frames_per_block")
    num_blocks = num_frames // frames_per_block
    exit_steps = list(exit_steps)
    if len(exit_steps) != num_blocks:
        raise ValueError("exit_steps needs one index per block")
    for e in exit_steps:
        if isinstance(e, bool) or not isinstance(e, int) or not 0 <= e < len(steps):
            raise ValueError(f"exit step {e!r} is not an index into {len(steps)} steps")

    def renoise(x0, t):
        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
        sigma = sigma_of(t)
        return (1 - sigma) * x0 + sigma * eps

    cache = []
    outputs = []
    for b, exit_step in enumerate(exit_steps):
        start = b * frames_per_block
        x = noise[:, start:start + frames_per_block]
        with torch.no_grad():
            for j in range(exit_step):
                x0_hat = model.denoise(x, steps[j], cache)
                x = renoise(x0_hat, steps[j + 1])
        if start >= grad_from_frame:
            x0_hat = model.denoise(x, steps[exit_step], cache)  # the only differentiable call
        else:
            with torch.no_grad():
                x0_hat = model.denoise(x, steps[exit_step], cache)
        outputs.append(x0_hat)
        with torch.no_grad():
            context = renoise(x0_hat, context_noise)
            cache.append(model.kv(context, context_noise, cache))
    return torch.cat(outputs, dim=1)
''',
    "interview_questions": interview(
        concept=[
            "What is exposure bias in autoregressive video diffusion, and how does training on a self-rollout remove the mismatch that Teacher Forcing and Diffusion Forcing leave?",
            "Why does Self Forcing train a few-step diffusion model rather than a many-step one, given that it unrolls the whole video during training?",
        ],
        deep_dive=[
            "Self Forcing samples one exit step per iteration and backpropagates only through that step's prediction. Why does randomizing the exit step still supervise every denoising step over training?",
            "Why is the cache entry computed under no_grad from a re-noised copy of the prediction, and what would change in memory and in the learning signal if gradients flowed through the cache?",
            "Count the denoise calls, cache calls and fresh noise draws in one training rollout of N blocks with exit index e, and compare them with a 4-step inference rollout of the same video.",
        ],
        tradeoffs=[
            "Self Forcing versus Diffusion Forcing for training a causal video world model: compare per-iteration cost, attention implementation and robustness to error accumulation.",
            "Gradient truncation keeps memory bounded. What long-range learning signal does it give up, and how might you recover some of it?",
        ],
    ),
}
