"""Vendored knowledge-insulation attention and FAST/action masking from LeRobot.

Upstream:   https://github.com/huggingface/lerobot
Commit:     75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee
Files:      src/lerobot/policies/pi05/modeling_pi05.py,
            src/lerobot/policies/fineart_vla/modeling_fineart_vla.py,
            src/lerobot/policies/common/vla_utils.py,
            src/lerobot/utils/constants.py
Symbols:    compute_layer_complete (knowledge_insulation branch),
            PI05Pytorch.embed_suffix (suffix att_masks),
            FineARTVLAPytorch._embed_supervised_prefix (FAST att_masks),
            FineARTVLAPytorch._combined_prefix_and_flow (suffix-to-FAST block, position_ids),
            make_att_2d_masks, prepare_attention_masks_4d, OPENPI_ATTENTION_MASK_VALUE
            and, inlined, transformers v5.4.0 models/gemma/modeling_gemma.py::eager_attention_forward
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2025-2026 Physical Intelligence and The HuggingFace Inc. team.
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for knowledge_insulation_attention, so that exercise is not checked
only against an oracle written by the same author as its reference solution.

Where upstream puts the two knowledge-insulation pieces, at this commit:

  * The stop-gradient lives in pi05's ``compute_layer_complete``. When
    ``knowledge_insulation`` is on, the action-expert queries attend to keys and
    values whose first ``prefix_len`` positions (every token of the VLM stream,
    FAST tokens included) are ``.detach()``-ed. The VLM queries attend to the
    original keys and values. The flag is off in stock pi0.5; FineART-VLA turns it on.
  * The FAST/action mask lives in FineART-VLA. FAST tokens are appended to the
    VLM prefix with ``att_masks = 1`` each (causal among themselves, invisible to
    the prompt), the action suffix gets ``[1, 0, ..., 0]`` (one bidirectional
    block), and ``att_2d_masks[:, fast_end:, fast_start:fast_end] = False`` then
    removes suffix-to-FAST attention. Action RoPE positions restart right after
    the non-FAST prefix, parallel to the FAST tokens.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed or replaced infrastructure:

  * Only the attention core of ``compute_layer_complete`` is kept: the layer
    norms, Q/K/V/O projections, RoPE, MLPs, gated residuals, ``attention_fn`` /
    ``norm_backend`` hooks and ``suppress_prefix_grads`` are dropped. The function
    takes query, key and value states directly, shape (B, heads, N, head_dim).
  * ``modeling_gemma.eager_attention_forward`` (transformers, Apache-2.0) is
    inlined. ``repeat_kv`` is dropped (no grouped-query attention), dropout is 0,
    and the softmax runs in the input dtype instead of float32 so float64
    comparisons are exact.
  * ``prepare_attention_masks_4d`` is inlined with the input dtype.
  * The mask builder takes pad masks directly instead of embedding images,
    language and FAST tokens. The non-FAST prefix uses ``att_masks = 0``
    everywhere (stock pi05 ``embed_prefix``); FineART's causal language-target
    marks (``_mark_target_span_causal``) are dropped.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch

__all__ = [
    "OPENPI_ATTENTION_MASK_VALUE",
    "lerobot_fineart_mask_and_positions",
    "lerobot_ki_attention",
    "lerobot_make_att_2d_masks",
]

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def lerobot_make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Transcribed from vla_utils.py::make_att_2d_masks (itself copied from big_vision)."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def lerobot_fineart_mask_and_positions(
    prefix_pad: torch.Tensor, fast_pad: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """FineART-VLA's joint prefix + FAST + action-suffix mask and RoPE positions.

    Transcribed from _embed_supervised_prefix (FAST att = ones), pi05 embed_prefix
    (prefix att = zeros), pi05 embed_suffix (suffix att = [1, 0, ...], pad = ones) and
    _combined_prefix_and_flow (suffix-to-FAST block and parallel suffix positions).
    """
    bsize = prefix_pad.shape[0]
    device = prefix_pad.device
    non_fast_prefix_len = prefix_pad.shape[1]

    # pi05 embed_prefix: images, state and language all get att_masks 0.
    prefix_att = torch.zeros(bsize, non_fast_prefix_len, dtype=torch.bool, device=device)

    # _embed_supervised_prefix: FAST tokens join the prefix with att_masks 1 each.
    fast_len = fast_pad.shape[1]
    prefix_pad = torch.cat([prefix_pad, fast_pad.to(prefix_pad.dtype)], dim=1)
    prefix_att = torch.cat(
        [prefix_att, torch.ones((bsize, fast_len), dtype=torch.bool, device=device)], dim=1
    )

    # pi05 embed_suffix: every action token is valid; one bidirectional block.
    suffix_pad = torch.ones(bsize, chunk_size, dtype=torch.bool, device=device)
    att_masks = [1] + ([0] * (chunk_size - 1))
    suffix_att = torch.tensor(att_masks, dtype=torch.float32, device=device)[None, :].expand(bsize, chunk_size)

    # _combined_prefix_and_flow.
    pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d_masks = lerobot_make_att_2d_masks(pad_masks, att_masks)

    # Block suffix-to-FAST attention to prevent trivial action leakage.
    if fast_len > 0:
        fast_start = non_fast_prefix_len
        fast_end = non_fast_prefix_len + fast_len  # = prefix_pad.shape[1]
        att_2d_masks[:, fast_end:, fast_start:fast_end] = False

    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    if fast_len > 0:
        # Position flow parallel to FAST so its RoPE offsets match inference without FAST.
        non_fast_valid = prefix_pad[:, :non_fast_prefix_len].sum(dim=1, keepdim=True)
        suffix_pos = non_fast_valid + torch.cumsum(suffix_pad, dim=1) - 1
        position_ids = torch.cat([position_ids[:, : prefix_pad.shape[1]], suffix_pos], dim=1)
    return att_2d_masks, position_ids


def _eager_attention(query, key_states, value_states, attention_mask, scaling):
    """transformers modeling_gemma.eager_attention_forward without repeat_kv or dropout."""
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def lerobot_ki_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    att_2d_masks: torch.Tensor,
    prefix_len: int,
    knowledge_insulation: bool = True,
) -> torch.Tensor:
    """The attention core of modeling_pi05.py::compute_layer_complete.

    Inputs are (B, heads, N, head_dim); the boolean (B, N, N) mask is turned into the
    additive 4-D mask as prepare_attention_masks_4d does. Returns (B, heads, N, head_dim)
    (upstream returns (B, N, heads, head_dim) and reshapes it for o_proj).
    """
    attention_mask = torch.where(att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE).to(
        query_states.dtype
    )
    scaling = query_states.shape[-1] ** -0.5

    def attend(query, key, value, mask):
        return _eager_attention(query, key, value, mask, scaling)

    if knowledge_insulation:
        # Detach VLM K/V only on the path the action queries use (pi0.5 knowledge insulation).
        key_for_action = torch.cat(
            [key_states[:, :, :prefix_len].detach(), key_states[:, :, prefix_len:]], dim=2
        )
        value_for_action = torch.cat(
            [value_states[:, :, :prefix_len].detach(), value_states[:, :, prefix_len:]], dim=2
        )
        vlm_mask = attention_mask[:, :, :prefix_len] if attention_mask is not None else None
        action_mask = attention_mask[:, :, prefix_len:] if attention_mask is not None else None
        att_vlm = attend(query_states[:, :, :prefix_len], key_states, value_states, vlm_mask)
        att_action = attend(query_states[:, :, prefix_len:], key_for_action, value_for_action, action_mask)
        att_output = torch.cat([att_vlm, att_action], dim=1)
    else:
        att_output = attend(query_states, key_states, value_states, attention_mask)
    return att_output.transpose(1, 2)
