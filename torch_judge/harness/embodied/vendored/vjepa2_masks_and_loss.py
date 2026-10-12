"""Vendored V-JEPA 2 oracles: multi-block 3D masks, apply_masks, the latent loss and the EMA target.

Upstream:   https://github.com/facebookresearch/vjepa2
Commit:     204698b45b3712590f06245fbfba32d3be539812
Files:      src/masks/multiseq_multiblock3d.py
            src/masks/utils.py
            app/vjepa/train.py
Symbols:    _MaskGenerator._sample_block_mask (line 173), _MaskGenerator.__call__ (line 190);
            apply_masks;
            momentum_scheduler (lines 298-302), forward_target (line 429),
            loss_fn (lines 440-449), the momentum update of the target encoder (lines 472-480)
License:    MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.
            (The repository README names three other files as Apache-2.0; none of
            them is used here.)
Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for jepa_block_masks_and_loss, so that exercise is not checked only
against an oracle written by the same author as its reference solution. Only the
MIT-licensed V-JEPA 2 code is used; I-JEPA and V-JEPA (v1) are CC BY-NC 4.0 and
are not vendored.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * ``_MaskGenerator`` draws the block size from a seeded generator shared by
    the batch and each block position with the global RNG. The exercise passes
    block placements in, so ``vjepa2_block_mask`` is ``_sample_block_mask`` with
    ``start``, ``top`` and ``left`` (and the size ``t, h, w``) as arguments
    instead of ``torch.randint`` draws. ``_sample_block_size``, ``step`` and the
    shared-counter seeding are dropped.
  * ``vjepa2_collate_masks`` is ``_MaskGenerator.__call__`` with the block
    placements of every sample passed in. Upstream redraws the blocks of a
    sample whose context comes out empty (``while empty_context``); with fixed
    placements a redraw would loop forever, so this transcription raises
    ``ValueError`` at the point where upstream would redraw.
  * Blocks of one sample may have different sizes here; upstream draws one size
    per batch and mask generator. The per-block code is unchanged.
  * The ``full_complement``, ``pred_full_complement`` and ``inv_block`` branches
    are dropped (all three are off in the released pretraining configs).
  * ``loss_fn`` and the ``forward_target`` layer norm are closures inside
    ``train_step``; they are lifted into ``vjepa2_loss`` with the encoder,
    predictor and masks passed in as tensors. Mixed precision is dropped.
  * The momentum update is lifted into ``vjepa2_ema_update`` with the two
    parameter lists passed in; the DDP-wrapped modules are dropped.
  * ``momentum_scheduler`` is lifted into a function returning the same
    generator.

UPSTREAM QUIRKS THE EXERCISE DOES NOT COPY
------------------------------------------
  * ``torch.argwhere(...).squeeze()`` and ``torch.nonzero(...).squeeze()``
    collapse a one-token mask to a 0-d tensor, so ``len()`` raises ``TypeError``.
    The exercise always returns ``(B, K)`` index tensors, including ``K = 1``.
  * ``zip`` over the two parameter lists silently stops at the shorter one; the
    exercise raises on a length or shape mismatch.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "apply_masks",
    "vjepa2_block_mask",
    "vjepa2_collate_masks",
    "vjepa2_ema_update",
    "vjepa2_loss",
    "vjepa2_momentum_scheduler",
]


def vjepa2_block_mask(duration, height, width, b_size, start, top, left, max_context_duration=None):
    """``_MaskGenerator._sample_block_mask`` with the block position passed in."""
    if max_context_duration is None:
        max_context_duration = duration
    t, h, w = b_size

    mask = torch.ones((duration, height, width), dtype=torch.int32)
    mask[start : start + t, top : top + h, left : left + w] = 0

    # Context mask will only span the first X frames
    # (X=self.max_context_frames)
    if max_context_duration < duration:
        mask[max_context_duration:, :, :] = 0

    # --
    return mask


def vjepa2_collate_masks(duration, height, width, placements, max_keep=None, max_context_duration=None):
    """``_MaskGenerator.__call__`` with every block given as ``(start, t, top, h, left, w)``.

    ``placements[b]`` lists the blocks of sample ``b``. Returns ``(masks_enc, masks_pred)``.
    """
    collated_masks_pred, collated_masks_enc = [], []
    min_keep_enc = min_keep_pred = duration * height * width
    for blocks in placements:

        mask_e = torch.ones((duration, height, width), dtype=torch.int32)
        for start, t, top, h, left, w in blocks:
            mask_e *= vjepa2_block_mask(duration, height, width, (t, h, w), start, top, left, max_context_duration)
        mask_e = mask_e.flatten()

        mask_p = torch.argwhere(mask_e == 0).squeeze()
        mask_e = torch.nonzero(mask_e).squeeze()

        empty_context = len(mask_e) == 0
        if empty_context:
            # Upstream redraws the blocks here; fixed placements cannot be redrawn.
            raise ValueError("empty context")
        min_keep_pred = min(min_keep_pred, len(mask_p))
        min_keep_enc = min(min_keep_enc, len(mask_e))
        collated_masks_pred.append(mask_p)
        collated_masks_enc.append(mask_e)

    if max_keep is not None:
        min_keep_enc = min(min_keep_enc, max_keep)

    collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
    collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

    collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
    collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

    return collated_masks_enc, collated_masks_pred


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


def vjepa2_loss(z, h, masks_pred, loss_exp):
    """``forward_target``'s layer norm followed by ``loss_fn``.

    ``z[i][j]`` is the predictor output for clip group ``i`` and mask ``j``; ``h[i]`` is the
    raw target-encoder output of clip group ``i``; ``masks_pred[i][j]`` is the ``(B, K)``
    index tensor of mask ``j``.
    """
    with torch.no_grad():
        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]

    # Assumption: predictor will have returned only masked tokens for z
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

    loss, n = 0, 0
    for zi, hi in zip(z, h):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    loss /= n
    return loss


def vjepa2_ema_update(encoder_params, target_params, m):
    """Step 3 of ``train_step``: momentum update of the target encoder, in place."""
    with torch.no_grad():
        params_k = []
        params_q = []
        for param_q, param_k in zip(encoder_params, target_params):
            params_k.append(param_k)
            params_q.append(param_q)
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1 - m)


def vjepa2_momentum_scheduler(ema, ipe, num_epochs, ipe_scale=1.0):
    """The ``momentum_scheduler`` generator: one momentum per optimizer step, both ends included."""
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )
    return momentum_scheduler
