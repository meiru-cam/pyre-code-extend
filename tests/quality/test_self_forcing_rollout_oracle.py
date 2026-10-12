"""The self_forcing_rollout reference, checked against the Self-Forcing training pipeline.

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied_frontier/vendored/self_forcing_training.py`.

Upstream's generator is one module that both predicts the clean block and writes its
keys and values into preallocated cache tensors at ``current_start``. The fakes below
compute the same arithmetic in both shapes: ``UpstreamGenerator`` writes a per-frame
summary into ``kv_cache[0]["k"][:, frame, 0, 0]`` and reads every earlier frame;
``ExerciseModel`` returns that summary as a list entry from ``kv`` and reads the list
in ``denoise``. Upstream's denoising calls also write the current block's slots, which
the context call then overwrites, so those writes never reach a later block.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.self_forcing_training import (
    FlowMatchScheduler,
    SelfForcingTrainingPipeline,
    warp_denoising_step_list,
)
from torch_judge.tasks import get_task

TASK_ID = "self_forcing_rollout"
SEEDS = (0, 1, 2, 3, 4, 5, 6)


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def wan_scheduler() -> FlowMatchScheduler:
    """The scheduler WanDiffusionWrapper builds (utils/wan_wrapper.py)."""
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def sigma_lookup(scheduler: FlowMatchScheduler):
    """add_noise's nearest-timestep lookup, as the exercise's sigma_of."""

    def sigma_of(t):
        t = torch.as_tensor(t, dtype=torch.float32).reshape(1)
        index = torch.argmin((scheduler.timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(), dim=1)
        return scheduler.sigmas[index][0]

    return sigma_of


class Params:
    def __init__(self, seed: int, channels: int):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(channels, 1, 1, generator=g, dtype=torch.float64).mul(0.5).add(1.0).requires_grad_()
        self.u = torch.randn(channels, 1, 1, generator=g, dtype=torch.float64).mul(0.5).add(1.0).requires_grad_()

    def predict(self, x, t, context):
        s = t / 1000.0
        return self.w * torch.tanh(x) * (1 - 0.5 * s) + 0.1 * x + 0.3 * context[:, None, None, None, None]

    def summary(self, x, t):
        return (self.u * torch.tanh(x)).mean(dim=(2, 3, 4)) * (1 + t / 1000.0)


class UpstreamGenerator:
    """Called like WanDiffusionWrapper: returns (flow_pred, x0) and writes the KV cache."""

    def __init__(self, params: Params):
        self.params = params

    def __call__(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start):
        t = float(timestep.reshape(-1)[0])
        keys = kv_cache[0]["k"]
        batch = noisy_image_or_video.shape[0]
        context = (keys[:, :current_start, 0, 0].mean(dim=1) if current_start > 0
                   else torch.zeros(batch, dtype=keys.dtype))
        x0 = self.params.predict(noisy_image_or_video, t, context)
        frames = noisy_image_or_video.shape[1]
        entry = self.params.summary(noisy_image_or_video, t) + 0.5 * context[:, None]
        keys[:, current_start:current_start + frames, 0, 0] = entry.detach()
        return None, x0


class ExerciseModel:
    def __init__(self, params: Params):
        self.params = params

    @staticmethod
    def _context(cache, batch, dtype):
        if not cache:
            return torch.zeros(batch, dtype=dtype)
        return torch.cat(cache, dim=1).mean(dim=1)

    def denoise(self, x, t, cache):
        return self.params.predict(x, float(t), self._context(cache, x.shape[0], x.dtype))

    def kv(self, x, t, cache):
        return self.params.summary(x, float(t)) + 0.5 * self._context(cache, x.shape[0], x.dtype)[:, None]


def run_both(seed, *, frames, fpb, exit_flags, same_step=False, context_noise=0, warp=True,
             step_list=(1000, 750, 500, 250), channels=2):
    scheduler = wan_scheduler()
    steps = warp_denoising_step_list(scheduler, list(step_list)) if warp else torch.tensor(step_list, dtype=torch.long)
    # Batch 1: upstream's context re-noise multiplies a (B, F) timestep by a (B * F,) tensor,
    # which only broadcasts when B == 1 (the released configs use batch_size 1 per GPU).
    noise = torch.randn(1, frames, channels, 2, 3, generator=torch.Generator().manual_seed(1000 + seed), dtype=torch.float64)

    up_params = Params(seed, channels)
    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=steps,
        scheduler=scheduler,
        generator=UpstreamGenerator(up_params),
        num_frame_per_block=fpb,
        same_step_across_blocks=same_step,
        num_max_frames=frames,
        context_noise=context_noise,
        num_transformer_blocks=1,
        frame_seq_length=1,
        noise_generator=torch.Generator().manual_seed(seed),
    )
    up_out = pipeline.inference_with_trajectory(noise, exit_flags)[0]

    ex_params = Params(seed, channels)
    exits = [exit_flags[0]] * len(exit_flags) if same_step else list(exit_flags)
    ref_out = reference()(
        ExerciseModel(ex_params), noise, [float(t) for t in steps], sigma_lookup(scheduler), exits, fpb,
        torch.Generator().manual_seed(seed), context_noise=context_noise,
        grad_from_frame=frames - 21,
    )
    return up_out, ref_out, up_params, ex_params


def assert_same(up_out, ref_out, up_params, ex_params, weights_seed):
    assert ref_out.shape == up_out.shape
    assert torch.allclose(ref_out, up_out, atol=1e-12, rtol=0), (ref_out - up_out).abs().max()
    assert ref_out.requires_grad == up_out.requires_grad
    if up_out.requires_grad:
        weights = torch.randn(up_out.shape, generator=torch.Generator().manual_seed(weights_seed), dtype=torch.float64)
        (up_out * weights).sum().backward()
        (ref_out * weights).sum().backward()
        assert torch.allclose(ex_params.w.grad, up_params.w.grad, atol=1e-12, rtol=0)
        assert up_params.u.grad is None and ex_params.u.grad is None


@pytest.mark.parametrize("seed", SEEDS)
def test_per_block_exit_steps_match_upstream(seed):
    g = torch.Generator().manual_seed(seed)
    exit_flags = torch.randint(0, 4, (3,), generator=g).tolist()
    assert_same(*run_both(seed, frames=9, fpb=3, exit_flags=exit_flags), weights_seed=seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_same_step_across_blocks_matches_upstream(seed):
    """The default config: one exit index for the whole rollout, as in Algorithm 1."""
    exit_flags = [seed % 4, 0, 3]  # upstream reads only exit_flags[0]
    assert_same(*run_both(seed, frames=6, fpb=2, exit_flags=exit_flags, same_step=True), weights_seed=seed)


@pytest.mark.parametrize("context_noise", (0, 125, 500))
def test_context_noise_matches_upstream(context_noise):
    assert_same(*run_both(3, frames=6, fpb=3, exit_flags=[1, 2], context_noise=context_noise), weights_seed=3)


def test_context_noise_zero_still_mixes_in_noise_upstream():
    """Pin the quirk the description states: the Wan table's smallest sigma is about 0.005."""
    scheduler = wan_scheduler()
    sigma = sigma_lookup(scheduler)(0)
    assert 0.004 < float(sigma) < 0.006


@pytest.mark.parametrize("exit_flags", ([0, 0, 0], [3, 3, 3], [3, 0, 2]))
def test_extreme_exit_steps_match_upstream(exit_flags):
    assert_same(*run_both(11, frames=3, fpb=1, exit_flags=exit_flags), weights_seed=11)


def test_unwarped_list_with_trailing_zero_matches_upstream():
    """Upstream drops a trailing zero timestep before indexing exit flags."""
    assert_same(*run_both(5, frames=4, fpb=2, exit_flags=[2, 1], warp=False, step_list=(1000, 500, 250, 0)),
                weights_seed=5)


def test_long_rollout_truncates_gradient_like_upstream():
    """Past 21 frames, upstream runs exit calls of blocks before num_frames - 21 without gradients."""
    up_out, ref_out, up_params, ex_params = run_both(8, frames=27, fpb=3, exit_flags=[1] * 9, channels=1)
    for out, params in ((up_out, up_params), (ref_out, ex_params)):
        early = torch.autograd.grad(out[:, :6].sum(), params.w, retain_graph=True)[0]
        late = torch.autograd.grad(out[:, 6:9].sum(), params.w, retain_graph=True)[0]
        assert torch.all(early == 0) and torch.any(late != 0)
    assert_same(up_out, ref_out, up_params, ex_params, weights_seed=8)


def test_warped_schedule_values():
    """The 4-step list [1000, 750, 500, 250] maps to the shifted table as the description says."""
    steps = warp_denoising_step_list(wan_scheduler(), [1000, 750, 500, 250])
    assert torch.allclose(steps, torch.tensor([1000.0, 937.5, 833.3333, 625.0]), atol=1e-3)
