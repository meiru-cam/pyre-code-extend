"""Vendored Coconut forward pass: continuous thoughts fed back through a KV-cached loop.

Upstream:   https://github.com/facebookresearch/coconut
Commit:     27273cb8cca4bb763c041a63b036d0c3b7cbbb48
Files:      coconut.py
Symbols:    Coconut.__init__ (line 16), Coconut.forward (lines 39-193)
Retrieved:  2026-10-12
License:    MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

            Permission is hereby granted, free of charge, to any person obtaining
            a copy of this software and associated documentation files (the
            "Software"), to deal in the Software without restriction, including
            without limitation the rights to use, copy, modify, merge, publish,
            distribute, sublicense, and/or sell copies of the Software, and to
            permit persons to whom the Software is furnished to do so, subject to
            the following conditions: The above copyright notice and this
            permission notice shall be included in all copies or substantial
            portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
            WARRANTY OF ANY KIND.

WHY THIS FILE EXISTS
--------------------
Ground truth for coconut_latent_forward, so that exercise is not checked only
against an oracle written by the same author as its reference solution. The
exercise asks for full recomputation on every pass; upstream reuses a KV cache
and only computes the new positions. The oracle test shows that both give the
same logits and the same gradients on the batches upstream supports.

WHAT WAS CHANGED, AND WHY
-------------------------
The body of ``forward`` is transcribed line for line. Removed infrastructure:

  * ``from transformers.models.gpt2 import GPT2LMHeadModel`` and the
    ``isinstance(..., GPT2LMHeadModel)`` branch in ``__init__`` are gone, because
    ``transformers`` is not a dependency. The constructor takes the input
    embedding module directly as ``embedding``. ``base_causallm`` is any module
    with the Hugging Face causal-LM call signature (``inputs_embeds``,
    ``attention_mask``, ``position_ids``, ``past_key_values``,
    ``output_hidden_states``) that returns an object with ``logits``,
    ``hidden_states`` and ``past_key_values``, the latter as a list of
    ``(key, value)`` tensors of shape ``(B, heads, T, head_dim)``. The oracle
    test supplies a small torch-only GPT-2 stub with that interface.
  * ``train``, ``eval`` and ``generate`` (lines 195 onward) are not used by the
    exercise and are dropped.
  * ``MAX_N_LATENT`` is only used by ``generate`` and is dropped.

UPSTREAM ASSUMPTIONS THE EXERCISE DOES NOT SHARE
------------------------------------------------
The KV-cached loop computes ``[0, first latent)`` and then one new position per
pass, with ``hidden_states_offset`` counting from the earliest latent in the
whole batch. It is therefore only correct when every item's latents are
contiguous and start at the same index (upstream's collator left-pads batches
to align them). On other batches it raises ``IndexError``. A latent at position
0 has no previous position to read. The exercise recomputes the full sequence on
every pass, accepts any latent positions, and rejects a latent at position 0.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

__all__ = ["Coconut", "Outputs"]

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        embedding,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.embedding = embedding

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
