"""Vendored per-token adaptive-exit evaluators from the Huginn recurrent-depth model.

Upstream:   https://github.com/seal-rg/recurrent-pretraining
Commit:     1ea7220ec7eb42d13e89db0663df254d0bcdc28e
Files:      recpre/raven_modeling_minimal.py
Symbols:    PerIterationExitEvaluator (line 2106), EntropyDiffExitEvaluator (line 2128),
            LatentDiffExitEvaluator (line 2149), KLExitEvaluator (line 2167),
            ArgmaxStabilityExitEvaluator (line 2218), get_adaptive_exit_evaluator (line 2284),
            RavenForCausalLM.generate_with_adaptive_compute (line 1552, exit bookkeeping only)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright the recurrent-pretraining authors (seal-rg). Licensed under the
            Apache License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for latent_exit_criteria, so that exercise is not checked only
against an oracle written by the same author as its reference solution.
Upstream evaluates the criteria one recurrent step at a time during generation;
the exercise computes the first exit step for every token at once. The oracle
test drives these evaluators step by step and compares.

WHAT WAS CHANGED, AND WHY
-------------------------
The exit arithmetic in every ``__init__``, ``init`` and ``check`` is
transcribed line for line. Removed infrastructure:

  * ``check(model, latents, aux_inputs)`` becomes ``check(latents, logits)``.
    Upstream calls ``model.predict_from_latents(latents, **aux_inputs).logits``
    to get the logits; the caller now passes those logits directly. The second
    tuple element, upstream's model outputs, is returned as those logits.
  * ``KLExitEvaluator.__init__`` takes ``vocab_size`` instead of the model and
    reads it where upstream reads ``model.config.padded_vocab_size``.
  * ``get_adaptive_exit_evaluator`` keeps only the four criteria the exercise
    teaches; ``cosine``, ``minp-kl`` and ``none`` are dropped.
  * ``huginn_exit_steps`` lifts the per-token bookkeeping out of
    ``generate_with_adaptive_compute`` for one generated token, with upstream's
    defaults ``min_steps=0``, ``check_criterion_every_n_steps=1`` and
    ``do_not_exit_in_prefill=False``, and every sequence unfinished. The
    recurrent core is replaced by the precomputed per-step latents and logits,
    and sampling, KV caches and continuous compute are dropped. A sequence that
    exits at loop index ``compute_step`` records ``compute_step + 1``; one that
    never exits records ``max_steps``, exactly as upstream.

Tensors keep upstream's layout ``(batch, seq, ...)``; with ``reduce=True`` the
logit-based criteria read only the last sequence position, and latent-diff
averages over the sequence.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "ArgmaxStabilityExitEvaluator",
    "EntropyDiffExitEvaluator",
    "KLExitEvaluator",
    "LatentDiffExitEvaluator",
    "PerIterationExitEvaluator",
    "get_adaptive_exit_evaluator",
    "huginn_exit_steps",
]

Exit = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]


class PerIterationExitEvaluator:
    """Base class for exit evaluators that check after each recurrent step."""

    def init(self, initial_latents: torch.Tensor):
        """Initialize evaluator state."""

    def check(self, latents: torch.Tensor, logits: torch.Tensor) -> Exit:
        """Returns (should_exit, outputs (or None), exit_values)"""
        raise NotImplementedError()


class EntropyDiffExitEvaluator(PerIterationExitEvaluator):
    """Exit based on change in output entropy."""

    def __init__(self, exit_threshold: Union[str, float] = "auto", reduce: bool = True):
        self.exit_threshold = 1e-3 if exit_threshold == "auto" else float(exit_threshold)
        self.reduce = reduce

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_entropy = torch.ones(batch_size, device=initial_latents.device) * 100.0

    def check(self, latents: torch.Tensor, logits: torch.Tensor) -> Exit:
        probs = F.softmax(logits[:, -1, :], dim=-1) if self.reduce else F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        exit_values = (entropy - self.prev_entropy).abs()
        self.prev_entropy = entropy
        return exit_values < self.exit_threshold, logits, exit_values


class LatentDiffExitEvaluator(PerIterationExitEvaluator):
    """Exit based on change in latent states."""

    def __init__(self, exit_threshold: Union[str, float] = "auto", reduce: bool = True):
        self.exit_threshold = 0.03 if exit_threshold == "auto" else float(exit_threshold)
        self.reduce = reduce

    def init(self, initial_latents: torch.Tensor):
        self.prev_latents = initial_latents.clone().detach()

    def check(self, latents: torch.Tensor, logits: torch.Tensor) -> Exit:
        exit_values = (latents - self.prev_latents).norm(dim=-1) / latents.norm(dim=-1)
        if self.reduce:
            exit_values = exit_values.mean(dim=-1)
        self.prev_latents = latents.clone().detach()
        return exit_values < self.exit_threshold, None, exit_values


class KLExitEvaluator(PerIterationExitEvaluator):
    """Exit based on KL divergence between successive outputs."""

    def __init__(self, vocab_size: int, exit_threshold: Union[str, float] = "auto", reduce: bool = True):
        self.exit_threshold = 0.001 if exit_threshold == "auto" else float(exit_threshold)
        self.V = vocab_size
        self.reduce = reduce

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_log_probs = ((1 / self.V) * torch.ones(batch_size, self.V, device=initial_latents.device)).log()

    def check(self, latents: torch.Tensor, logits: torch.Tensor) -> Exit:
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1) if self.reduce else F.log_softmax(logits, dim=-1)
        exit_values = F.kl_div(log_probs, self.prev_log_probs, reduction="none", log_target=True).sum(dim=-1)
        self.prev_log_probs = log_probs
        return exit_values < self.exit_threshold, logits, exit_values


class ArgmaxStabilityExitEvaluator(PerIterationExitEvaluator):
    """Exit based on argmax stability over consecutive steps."""

    def __init__(self, exit_threshold: Union[str, int] = "auto", reduce: bool = True):
        self.exit_threshold = 5 if exit_threshold == "auto" else int(exit_threshold)
        self.reduce = reduce

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_argmax = torch.ones(batch_size, dtype=torch.long, device=initial_latents.device) * -1
        self.stable_for_n_steps = torch.zeros(batch_size, dtype=torch.long, device=initial_latents.device)

    def check(self, latents: torch.Tensor, logits: torch.Tensor) -> Exit:
        current_argmax = logits[:, -1, :].argmax(dim=-1) if self.reduce else logits.argmax(dim=-1)
        stable_for_n_steps = torch.where(
            current_argmax == self.prev_argmax, self.stable_for_n_steps + 1, torch.zeros_like(self.stable_for_n_steps)
        )
        exit_values = stable_for_n_steps
        self.prev_argmax = current_argmax
        self.stable_for_n_steps = stable_for_n_steps
        return exit_values >= self.exit_threshold, logits, exit_values


def get_adaptive_exit_evaluator(
    criterion: str, exit_threshold: Union[str, float, int], vocab_size: int, reduce: bool = True
) -> PerIterationExitEvaluator:
    """Factory function to create appropriate exit evaluator."""
    if criterion == "entropy-diff":
        return EntropyDiffExitEvaluator(exit_threshold, reduce)
    elif criterion == "latent-diff":
        return LatentDiffExitEvaluator(exit_threshold, reduce)
    elif "kl" in criterion:
        return KLExitEvaluator(vocab_size, exit_threshold, reduce)
    elif criterion == "argmax-stability":
        return ArgmaxStabilityExitEvaluator(exit_threshold, reduce)  # type: ignore
    else:
        raise ValueError(f"Invalid adaptive compute strategy: {criterion}")


def huginn_exit_steps(
    exit_evaluator: PerIterationExitEvaluator,
    initial_latents: torch.Tensor,
    latents_per_step: list,
    logits_per_step: list,
) -> list:
    """Exit bookkeeping of generate_with_adaptive_compute for one generated token.

    ``initial_latents`` is ``(batch, seq, h)``; ``latents_per_step[k]`` and
    ``logits_per_step[k]`` are the state and logits after loop index ``k``.
    Returns ``compute_steps_per_seq``.
    """
    max_steps = len(latents_per_step)
    batch_size = initial_latents.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=initial_latents.device)
    compute_steps_per_seq = [0] * batch_size
    exit_reached = torch.zeros(batch_size, dtype=torch.bool, device=initial_latents.device)
    exit_evaluator.init(initial_latents)

    for compute_step in range(max_steps):
        current_latents = latents_per_step[compute_step]
        new_exits, outputs, exit_values = exit_evaluator.check(current_latents, logits_per_step[compute_step])
        new_exits = new_exits & ~exit_reached & unfinished_sequences.bool()
        if new_exits.any():
            exit_reached = exit_reached | new_exits
            for i in range(batch_size):
                if new_exits[i]:
                    compute_steps_per_seq[i] = compute_step + 1
        if (exit_reached | ~unfinished_sequences.bool()).all():
            break
    else:
        for i in range(batch_size):
            if not exit_reached[i] and unfinished_sequences[i].bool():
                compute_steps_per_seq[i] = max_steps
    return compute_steps_per_seq
