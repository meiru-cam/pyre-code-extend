"""Vendored GRPO oracles transcribed from nano-aha-moment.

Upstream:   https://github.com/McGill-NLP/nano-aha-moment
Commit:     5314e6f8fc60efaa0f4b8fdb62353e9bd451638a
Files:      nano_r1_script.py, utils.py
Symbols:    log_softmax_and_gather, create_training_episodes (advantage block),
            compute_pg_loss, create_vineppo_training_episodes.get_tokens_advantages
Retrieved:  2026-09-13
License:    MIT License, Copyright (c) 2025 McGill NLP

WHY THIS FILE EXISTS
--------------------
Ground truth for the GRPO-family exercises. An oracle written by the same author
as the reference solution can be wrong in the same way twice; transcribing the
upstream arithmetic gives an independent check.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed is everything that cannot
run in the offline grading sandbox (torch and numpy only):

  * The model forward pass. ``compute_token_log_probs`` calls a DeepSpeed or
    HuggingFace model; the oracles here take log-probabilities as arguments.
  * The causal shift. Upstream applies ``logits[..., :-1, :]`` against
    ``labels[..., 1:]`` inside ``compute_token_log_probs``; the pure gather it
    then calls is ``log_softmax_and_gather``, which is what is vendored here.
  * Temperature scaling, the tokenizer, the vLLM inference engine, and the
    episode/stats bookkeeping.
  * ``numpy`` is replaced by ``torch`` in the advantage block. This is
    behaviour-preserving: upstream calls ``ndarray.std()``, whose default
    ``ddof=0`` is the population standard deviation, matching
    ``torch.std(unbiased=False)``. The ``1e-4`` epsilon is upstream's.

NOT VENDORED: ``format_reward_func``. Its three tiers are specific to the
Countdown task — the 0.5 tier means "well formed, but the answer body is not a
pure arithmetic expression", not "partially formed". It also synthetically
prepends ``<think>`` and requires a literal newline between the blocks. The
``rlvr_format_reward`` exercise borrows the idea of a graded structural reward
and defines its own rules, so there is no shared contract to compare against.
See that task's ``sources`` entry.

NOTHING HERE IMPORTS A TASK SOLUTION.
"""

from __future__ import annotations

import torch

__all__ = [
    "NANO_ADVANTAGE_EPS",
    "nano_group_advantages",
    "nano_k3_kl_penalty",
    "nano_log_softmax_and_gather",
    "nano_pg_loss",
    "nano_vineppo_token_advantages",
]

# create_training_episodes: (rewards - rewards.mean()) / (rewards.std() + 1e-4)
NANO_ADVANTAGE_EPS = 1e-4


def nano_log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Transcribed from utils.py::log_softmax_and_gather.

    Upstream notes it is itself copied from allenai/open-instruct and normally
    runs under ``torch.compile``; the eager body is identical.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def nano_group_advantages(
    rewards: torch.Tensor, eps: float = NANO_ADVANTAGE_EPS
) -> torch.Tensor:
    """Transcribed from nano_r1_script.py::create_training_episodes.

    Upstream operates on one group at a time inside a loop over prompts, with
    ``rewards`` a numpy array of that group's rewards. Pass one group here.
    """
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def nano_k3_kl_penalty(logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Transcribed from nano_r1_script.py::compute_pg_loss.

        ref_logratio = ref_logps - logps
        kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio
    """
    ref_logratio = ref_logps - logps
    return torch.exp(ref_logratio) - 1 - ref_logratio


def nano_pg_loss(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    labels_mask: torch.Tensor,
    kl_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transcribed from nano_r1_script.py::compute_pg_loss.

    ``advantages`` is per token upstream, already broadcast from one scalar per
    response by ``create_training_episodes``. ``total_response_len`` upstream is
    the batch-wide count of unmasked tokens, computed here as ``labels_mask.sum()``.
    """
    mask = labels_mask.to(logps.dtype)

    ref_logratio = ref_logps - logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio
    kl_penalty = kl_penalty * mask

    policy_loss = -logps * advantages
    policy_loss = policy_loss * mask

    total_response_len = mask.sum()
    loss = (policy_loss + kl_coefficient * kl_penalty).sum() / total_response_len

    metrics = {
        "policy_loss": policy_loss.sum() / total_response_len,
        "kl_penalty": kl_penalty.sum() / total_response_len,
    }
    return loss, metrics


def nano_vineppo_token_advantages(
    states: list[int], value_estimates: list[float]
) -> list[float]:
    """Transcribed from nano_r1_script.py::create_vineppo_training_episodes.

        for i in range(len(states) - 1):
            length = states[i + 1] - states[i]
            advantage = value_estimates[i + 1] - value_estimates[i]
            tokens_advantages.extend([advantage] * length)
    """
    if sorted(states) != states:
        raise ValueError("states must be sorted")
    tokens_advantages: list[float] = []
    for i in range(len(states) - 1):
        length = states[i + 1] - states[i]
        advantage = value_estimates[i + 1] - value_estimates[i]
        tokens_advantages.extend([advantage] * length)
    return tokens_advantages
