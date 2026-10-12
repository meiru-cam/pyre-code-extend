"""Vendored Self Forcing training rollout and flow-matching scheduler from Self-Forcing.

Upstream:   https://github.com/guandeh17/Self-Forcing
Commit:     33593df3e81fa3ec10239271dd2c100facac6de1
Files:      pipeline/self_forcing_training.py, utils/scheduler.py, model/base.py
Symbols:    SelfForcingTrainingPipeline (generate_and_sync_list, inference_with_trajectory,
            _initialize_kv_cache, _initialize_crossattn_cache), FlowMatchScheduler
            (set_timesteps, add_noise), the warp_denoising_step branch of BaseModel.__init__
Retrieved:  2026-10-12
License:    Apache License 2.0
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0 (repository LICENSE file)

WHY THIS FILE EXISTS
--------------------
Ground truth for self_forcing_rollout, so that exercise is not checked only
against an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic of ``inference_with_trajectory`` and ``FlowMatchScheduler`` is
transcribed line for line. Changes:

  * Distributed code removed. ``generate_and_sync_list`` drew the exit indices on
    rank 0 and ``dist.broadcast`` them; here it is single-process and takes an
    optional ``torch.Generator``. ``inference_with_trajectory`` takes the exit
    indices as an ``exit_flags`` argument instead of calling it, so a test can fix
    them, and asserts there is one per block.
  * CUDA code removed. The three ``.cuda()`` calls in the returned
    ``denoised_timestep_from`` / ``denoised_timestep_to`` computation are dropped;
    the arithmetic is unchanged.
  * ``torch.randn_like(x)`` in the two re-noising calls became
    ``torch.randn(x.shape, generator=self.noise_generator, dtype=x.dtype,
    device=x.device)``. Same shape, same order of draws, but from a generator the
    caller seeds, so the rollout is reproducible.
  * The Wan-specific constants ``num_transformer_blocks = 30`` and
    ``frame_seq_length = 1560`` became constructor arguments with those defaults,
    so a tiny fake generator can use the same cache layout. The 12-head, 128-dim
    cache tensors are kept as upstream allocates them.
  * Type hints that imported ``WanDiffusionWrapper`` and ``SchedulerInterface``
    were dropped, and a commented-out block of dead cache-reset code was removed.
  * ``FlowMatchScheduler`` keeps ``__init__``, ``set_timesteps`` and ``add_noise``;
    ``step``, ``training_target`` and ``training_weight`` are not used by the rollout.
  * ``warp_denoising_step_list`` is the three-line ``warp_denoising_step`` branch of
    ``BaseModel.__init__`` (model/base.py), lifted into a function.

Kept as upstream, including one quirk: the context re-noise multiplies a
``(B, F)`` timestep by a ``(B * F,)`` tensor, which only broadcasts when the batch
size is 1. The released configs train with ``batch_size: 1`` per GPU, and the
oracle test uses batch 1.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import List, Optional

import torch

__all__ = ["FlowMatchScheduler", "SelfForcingTrainingPipeline", "warp_denoising_step_list"]


class FlowMatchScheduler():
    """Transcribed from utils/scheduler.py::FlowMatchScheduler (rollout methods only)."""

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)


def warp_denoising_step_list(scheduler: FlowMatchScheduler, denoising_step_list) -> torch.Tensor:
    """The warp_denoising_step branch of model/base.py::BaseModel.__init__."""
    denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long)
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    return timesteps[1000 - denoising_step_list]


class SelfForcingTrainingPipeline:
    """Transcribed from pipeline/self_forcing_training.py::SelfForcingTrainingPipeline."""

    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler,
                 generator,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 num_transformer_blocks: int = 30,
                 frame_seq_length: int = 1560,
                 noise_generator: Optional[torch.Generator] = None,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = num_transformer_blocks
        self.frame_seq_length = frame_seq_length
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.noise_generator = noise_generator

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device, generator=None):
        # Generate random indices (single process: no broadcast)
        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_blocks,),
            device=device,
            generator=generator,
        )
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        return indices.tolist()

    def _randn_like(self, x):
        return torch.randn(x.shape, generator=self.noise_generator, dtype=x.dtype, device=x.device)

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            exit_flags: List[int],
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ):
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = list(exit_flags)  # upstream: self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, ...)
        assert len(exit_flags) == len(all_num_frames)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            self._randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                self._randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps - self.denoising_step_list[exit_flags[0] + 1]).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
