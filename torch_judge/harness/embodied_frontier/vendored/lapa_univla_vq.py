"""Vendored latent-action vector quantizers from LAPA (NSVQ) and UniVLA (straight-through VQ).

This module carries two upstreams, each with its own license.

[1] LAPA: noise-substitution VQ and dead-code replacement
Upstream:   https://github.com/LatentActionPretraining/LAPA
Commit:     46aca51d7faebcec02d7d323bbb3820c2df07bc6
Files:      laq/laq_model/nsvq.py
Symbols:    NSVQ.forward (lines 112-175), NSVQ.replace_unused_codebooks (lines 177-229)
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2024 Seonghyeon Ye and Joel Jang.
            Permission is hereby granted, free of charge, to any person obtaining a
            copy of this software, to deal in the Software without restriction,
            subject to including the above copyright notice and this permission
            notice in all copies or substantial portions of the Software.

[2] UniVLA: straight-through VQ and the VQ-VAE codebook/commitment losses
Upstream:   https://github.com/OpenDriveLab/UniVLA
Commit:     0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352
Files:      latent_action_model/genie/modules/blocks.py, latent_action_model/genie/model.py
Symbols:    VectorQuantizer.forward (blocks.py lines 398-413),
            DINO_LAM.shared_step loss lines (model.py lines 97-101)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Licensed under the Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for the latent_action_quantizer exercise, so its reference solution is
not checked only against oracles written by the same author.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed or injected infrastructure:

  * LAPA ``NSVQ.forward``: ``encode`` (project_in + CNN) and ``decode``
    (project_out) are dropped; the function takes the already-encoded features
    ``input_data_first`` / ``input_data_last`` of shape (N, D) and returns the
    quantized vector before ``decode``. The ``normal_dist.Normal(0, 1).sample``
    draw is replaced by an injected ``random_vector`` so the oracle is
    deterministic. The ``codebook_training_only`` branch, the ``print`` and the
    ``codebooks_used`` side effect are dropped; the usage counts are returned.
  * LAPA ``replace_unused_codebooks``: works on tensors passed in instead of
    module state and returns the new codebook. ``torch.randperm`` and
    ``torch.randn`` are injected as callables so a test can make them
    deterministic. ``print`` and the ``codebooks_used`` reset are dropped.
  * UniVLA ``VectorQuantizer.forward``: ``nn.Embedding`` lookup becomes indexing
    into a weight tensor; the usage-buffer update is dropped.
  * UniVLA loss: only the ``q_loss`` and ``commit_loss`` lines are kept; the
    reconstruction loss belongs to the decoder and is out of scope.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch

__all__ = [
    "LAPA_NSVQ_EPS",
    "UNIVLA_VQ_BETA",
    "lapa_nsvq_forward",
    "lapa_replace_unused_codebooks",
    "univla_vector_quantize",
    "univla_vq_losses",
]

# nsvq.py line 35: self.eps = 1e-12
LAPA_NSVQ_EPS = 1e-12
# model.py line 41: vq_beta: float = 0.25
UNIVLA_VQ_BETA = 0.25


def lapa_nsvq_forward(input_data_first, input_data_last, codebooks, random_vector, eps=LAPA_NSVQ_EPS):
    """Transcribed from nsvq.py::NSVQ.forward with encode/decode removed and the noise injected.

    Returns ``(quantized_input, perplexity, min_indices, usage)``.
    """
    num_embeddings = codebooks.shape[0]

    input_data = input_data_last - input_data_first

    distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                 - 2 * (torch.matmul(input_data, codebooks.t()))
                 + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))

    min_indices = torch.argmin(distances, dim=1)

    hard_quantized_input = codebooks[min_indices]

    norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
    norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

    # defining vector quantization error
    vq_error = (norm_quantization_residual / norm_random_vector + eps) * random_vector

    quantized_input = input_data + vq_error

    # claculating the perplexity (average usage of codebook entries)
    encodings = torch.zeros(input_data.shape[0], num_embeddings, device=input_data.device)
    encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))

    usage = torch.zeros(num_embeddings, dtype=torch.int32)
    with torch.no_grad():
        usage[min_indices.cpu()] += 1

    return quantized_input, perplexity, min_indices, usage


def lapa_replace_unused_codebooks(codebooks, codebooks_used, num_batches, discarding_threshold,
                                  eps=LAPA_NSVQ_EPS, randperm=torch.randperm, randn=torch.randn):
    """Transcribed from nsvq.py::NSVQ.replace_unused_codebooks; returns the new codebook."""
    codebooks = codebooks.detach().clone()
    embedding_dim = codebooks.shape[1]
    with torch.no_grad():

        unused_indices = torch.where((codebooks_used.cpu() / num_batches) < discarding_threshold)[0]
        used_indices = torch.where((codebooks_used.cpu() / num_batches) >= discarding_threshold)[0]

        unused_count = unused_indices.shape[0]
        used_count = used_indices.shape[0]

        if used_count == 0:
            codebooks += eps * randn(codebooks.size()).clone()
        else:
            used = codebooks[used_indices].clone()
            if used_count < unused_count:
                used_codebooks = used.repeat(int((unused_count / (used_count + eps)) + 1), 1)
                used_codebooks = used_codebooks[randperm(used_codebooks.shape[0])]
            else:
                used_codebooks = used

            codebooks[unused_indices] *= 0
            codebooks[unused_indices] += used_codebooks[range(unused_count)] + eps * randn(
                (unused_count, embedding_dim)).clone()

    return codebooks


def univla_vector_quantize(x, codebook_weight):
    """Transcribed from blocks.py::VectorQuantizer.forward; returns ``(z_q, z, x, indices)``."""
    # Compute distances
    distance = torch.cdist(x, codebook_weight)

    # Get indices and embeddings
    indices = torch.argmin(distance, dim=-1)
    z = codebook_weight[indices]

    # Straight through estimator
    z_q = x + (z - x).detach()
    return z_q, z, x, indices


def univla_vq_losses(emb, z, vq_beta=UNIVLA_VQ_BETA):
    """Transcribed from model.py::shared_step lines 98-101 without the reconstruction term."""
    q_loss = ((emb.detach() - z) ** 2).mean()
    commit_loss = ((emb - z.detach()) ** 2).mean()
    return q_loss + vq_beta * commit_loss, q_loss, commit_loss
