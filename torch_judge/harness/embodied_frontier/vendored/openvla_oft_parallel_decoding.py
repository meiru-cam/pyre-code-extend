"""Vendored OpenVLA-OFT parallel decoding: input packing, action masks, bidirectional mask, readout.

Upstream:   https://github.com/moojink/openvla-oft
Commit:     e4287e94541f459edc4feabc4e181f537cd569a8
Files:      prismatic/extern/hf/modeling_prismatic.py,
            prismatic/models/action_heads.py,
            prismatic/training/train_utils.py,
            prismatic/vla/constants.py,
            vla-scripts/finetune.py
Symbols:    OpenVLAForActionPrediction.predict_action,
            OpenVLAForActionPrediction._prepare_input_for_action_prediction,
            OpenVLAForActionPrediction._prepare_labels_for_action_prediction,
            OpenVLAForActionPrediction._regression_or_discrete_prediction,
            PrismaticForConditionalGeneration._process_action_masks,
            PrismaticForConditionalGeneration._build_multimodal_attention,
            L1RegressionActionHead.predict_action (the reshape),
            get_current_action_mask, get_next_actions_mask,
            IGNORE_INDEX, ACTION_TOKEN_BEGIN_IDX, STOP_INDEX,
            run_forward_pass (the continuous-action hidden-state readout)
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.

Upstream:   https://github.com/moojink/transformers-openvla-oft
Commit:     bc339d9ad707454c0c115970db43c260067c61ab
Files:      src/transformers/models/llama/modeling_llama.py
Symbols:    LlamaModel._update_causal_mask (eager 2D-mask branch),
            LlamaSdpaAttention.forward (the "Full bidirectional attention" patch)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2022 EleutherAI and the HuggingFace Inc. team. Licensed
            under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for oft_parallel_action_decoding, so that exercise is not checked
only against an oracle written by the same author as its reference solution.

The bidirectional mask is not in the openvla-oft repo: OFT installs a patched
transformers (the second upstream above) whose SDPA attention takes the last row
of Hugging Face's causal-plus-padding mask and copies it to every row, then calls
scaled_dot_product_attention with is_causal=False. OFT paper Section IV-B states
the same design in prose ("replace the causal attention mask with bidirectional
attention"); the patch is transcribed here so the exercise is checked against
code, not only against the prose.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Changes:

  * Module-level constants that upstream picks from ``sys.argv`` (ACTION_DIM,
    NUM_ACTIONS_CHUNK) are function parameters. ACTION_TOKEN_BEGIN_IDX is a
    parameter with the Llama 2 default 31743.
  * The Llama decoder, vision backbone, projector and tokenizer need weights. The
    caller passes ``embed(input_ids)`` and ``language_model(embeds, mask4d)``
    callables instead, plus already-projected ``patch_embeddings``. The model's
    own 2D-to-4D mask step is made explicit: ``predict_action`` builds the 4D
    mask with ``hf_causal_mask`` and ``oft_bidirectional_mask`` and passes it in.
  * ``hf_causal_mask`` keeps only the dynamic-cache, 2D ``attention_mask`` branch
    of ``_update_causal_mask`` with ``cache_position = arange(S)``. The SDPA
    shortcut that returns ``None`` for an all-ones mask is dropped: with the OFT
    patch, ``None`` means "no mask, is_causal=False", which equals an all-zero
    additive mask, which is what the eager branch plus the patch produce.
  * ``_regression_or_discrete_prediction`` returns torch tensors instead of
    numpy arrays; the bin centers are computed with ``np.linspace`` as upstream
    and converted with ``torch.from_numpy`` (float64).
  * Vision, proprio, FiLM and diffusion branches are dropped; ``NUM_PATCHES`` is
    ``patch_embeddings.shape[1]``.
  * The 29871 append in ``predict_action`` concatenates a (1, 1) tensor, so like
    upstream it only works for batch size 1.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "IGNORE_INDEX",
    "ACTION_TOKEN_BEGIN_IDX",
    "STOP_INDEX",
    "get_current_action_mask",
    "get_next_actions_mask",
    "process_action_masks",
    "prepare_input_for_action_prediction",
    "prepare_labels_for_action_prediction",
    "build_multimodal_attention",
    "hf_causal_mask",
    "oft_bidirectional_mask",
    "l1_head_rearrange",
    "finetune_actions_hidden_states",
    "predict_action",
]

# prismatic/vla/constants.py -- Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# --- prismatic/training/train_utils.py ------------------------------------------------


def get_current_action_mask(token_ids, action_dim, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= action_dim)

    # Extract the action part only
    action_tokens_only_mask = token_ids > action_token_begin_idx
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids, action_dim, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > action_dim

    # Extract the action part only
    action_tokens_only_mask = token_ids > action_token_begin_idx
    mask = action_tokens_only_mask * mask

    return mask


# --- prismatic/extern/hf/modeling_prismatic.py -----------------------------------------


def process_action_masks(labels, action_dim, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    """PrismaticForConditionalGeneration._process_action_masks"""
    current_action_mask = get_current_action_mask(labels, action_dim, action_token_begin_idx)
    next_actions_mask = get_next_actions_mask(labels, action_dim, action_token_begin_idx)
    all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
    return all_actions_mask


def prepare_input_for_action_prediction(input_ids, attention_mask, action_dim, num_actions_chunk):
    """OpenVLAForActionPrediction._prepare_input_for_action_prediction"""
    # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
    placeholder_action_token_ids = (
        torch.ones((input_ids.shape[0], action_dim * num_actions_chunk)).to(input_ids.device).to(input_ids.dtype)
    )
    input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

    # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
    stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
    input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

    # Extend the attention mask to fit the new shape of input
    # Note: Only batch size == 1 supported right now
    mask_extension = (
        torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
        .to(attention_mask.device)
        .to(attention_mask.dtype)
    )
    attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

    return input_ids, attention_mask


def prepare_labels_for_action_prediction(labels, input_ids, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    """OpenVLAForActionPrediction._prepare_labels_for_action_prediction"""
    # Extend labels tensor with fake action labels
    ARBITRARY_ACTION_TOKEN_IDX = action_token_begin_idx + 1
    labels_extension = (
        torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
        * ARBITRARY_ACTION_TOKEN_IDX
    )
    labels = torch.cat([labels, labels_extension], dim=-1)

    # Replace last label token with stop token
    labels[:, -1] = STOP_INDEX

    return labels


def build_multimodal_attention(input_embeddings, projected_patch_embeddings, attention_mask):
    """PrismaticForConditionalGeneration._build_multimodal_attention"""
    # Update attention mask
    projected_patch_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
    )

    multimodal_attention_mask = None
    if attention_mask is not None:
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
        )

    return multimodal_embeddings, multimodal_attention_mask


# --- transformers-openvla-oft: src/transformers/models/llama/modeling_llama.py -------------


def hf_causal_mask(attention_mask, input_tensor):
    """LlamaModel._update_causal_mask, dynamic cache, 2D attention_mask, no past tokens."""
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1]
    cache_position = torch.arange(sequence_length, device=device)

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    return causal_mask


def oft_bidirectional_mask(causal_mask):
    """LlamaSdpaAttention.forward, the OFT patch: copy the last row to every row."""
    # Trick: To get this mask, we just take the last row of the old `causal_mask` and duplicate it all the way
    # through to get the new mask.
    if causal_mask is not None:
        D = causal_mask.shape[-1]
        last_row = causal_mask[:, :, -1, :].clone()
        new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
        causal_mask = new_mask
    return causal_mask


# --- prismatic/models/action_heads.py --------------------------------------------------


def l1_head_rearrange(actions_hidden_states, num_actions_chunk):
    """L1RegressionActionHead.predict_action, the input reshape before the MLP."""
    batch_size = actions_hidden_states.shape[0]
    rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, num_actions_chunk, -1)
    return rearranged_actions_hidden_states


# --- vla-scripts/finetune.py -----------------------------------------------------------


def finetune_actions_hidden_states(last_hidden_states, labels, num_patches, action_dim, num_actions_chunk,
                                   action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    """run_forward_pass, continuous-action branch: hidden states that feed the action head."""
    ground_truth_token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids, action_dim, action_token_begin_idx)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids, action_dim, action_token_begin_idx)
    # Get hidden states for text portion of prompt+response (after the vision patches)
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    # Get hidden states for action portion of response
    batch_size = labels.shape[0]
    actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
        batch_size, num_actions_chunk * action_dim, -1
    )  # (B, act_chunk_len, D)
    return actions_hidden_states


# --- OpenVLAForActionPrediction.predict_action + _regression_or_discrete_prediction ------


def predict_action(input_ids, attention_mask, patch_embeddings, embed, language_model, action_dim,
                   num_actions_chunk, vocab_size=32000, n_action_bins=256,
                   action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX, action_head=None):
    """Returns (normalized_actions (NUM_ACTIONS_CHUNK, ACTION_DIM), actions_hidden_states)."""
    bins = np.linspace(-1, 1, n_action_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )

    # Create fake labels tensor (needed for action mask)
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    # Get number of tokens in prompt (excluding the start token)
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

    # Prepare inputs by adding necessary tokens
    input_ids, attention_mask = prepare_input_for_action_prediction(
        input_ids, attention_mask, action_dim, num_actions_chunk
    )

    # Update labels tensor for action mask computation later
    labels = prepare_labels_for_action_prediction(labels, input_ids, action_token_begin_idx)

    # Get input embeddings and action masks
    input_embeddings = embed(input_ids)
    all_actions_mask = process_action_masks(labels, action_dim, action_token_begin_idx)

    projected_patch_embeddings = patch_embeddings
    NUM_PATCHES = projected_patch_embeddings.shape[1]

    # _regression_or_discrete_prediction
    # Zero out action token embeddings
    all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    input_embeddings = input_embeddings * ~all_actions_mask

    # Build multimodal embeddings and attention mask
    multimodal_embeddings, multimodal_attention_mask = build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    # Forward pass through language model (the patched model builds this 4D mask internally)
    mask4d = oft_bidirectional_mask(hf_causal_mask(multimodal_attention_mask, multimodal_embeddings))
    last_hidden_states, logits = language_model(multimodal_embeddings, mask4d)

    # Extract hidden states for action tokens
    actions_hidden_states = last_hidden_states[
        :,
        NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + action_dim * num_actions_chunk,
        :,
    ]  # (B, act_chunk_len, D)

    # Handle different prediction methods
    if action_head is not None:
        # L1 regression prediction
        normalized_actions = action_head(l1_head_rearrange(actions_hidden_states, num_actions_chunk))
        normalized_actions = normalized_actions.reshape(num_actions_chunk, action_dim)
    else:
        # Discrete token-based prediction
        predicted_action_token_ids = (
            logits[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + action_dim * num_actions_chunk,
            ]
            .argmax(dim=2)
            .cpu()
            .numpy()
        )
        discretized_actions = vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
        normalized_actions = torch.from_numpy(bin_centers[discretized_actions])
        normalized_actions = normalized_actions.reshape(num_actions_chunk, action_dim)

    return normalized_actions, actions_hidden_states
