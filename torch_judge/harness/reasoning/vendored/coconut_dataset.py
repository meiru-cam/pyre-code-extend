"""Vendored Coconut curriculum sample builder (one training sample per stage).

Upstream:   https://github.com/facebookresearch/coconut
Commit:     27273cb8cca4bb763c041a63b036d0c3b7cbbb48
Files:      dataset.py, args/gsm_coconut.yaml
Symbols:    get_cot_latent_dataset (dataset.py line 230) and its inner
            process_dataset (line 243); GSM8k settings c_thought,
            max_latent_stage, pad_latent_to_max, uniform_prob
            (args/gsm_coconut.yaml)
License:    MIT License, Copyright (c) Meta Platforms, Inc. and affiliates

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for coconut_stage_sample, so that exercise is not checked only
against an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The per-sample arithmetic of ``process_dataset`` is transcribed line for line
into ``process_sample``. Removed infrastructure:

  * ``get_cot_latent_dataset`` maps ``process_dataset`` over a Hugging Face
    ``datasets.Dataset``, broadcasts it across ranks with ``torch.distributed``
    and optionally shuffles it. All of that is dropped; ``process_sample``
    takes one already-tokenized sample dict (``question_tokenized``,
    ``steps_tokenized``, ``answer_tokenized``, ``idx``) and returns one dict.
  * The closure variables ``scheduled_stage``, ``configs``, ``start_id``,
    ``latent_id``, ``end_id`` and ``no_special_marker`` become arguments.
    ``configs`` is any object with the attributes upstream reads
    (``uniform_prob``, ``max_latent_stage``, ``pad_latent_to_max``,
    ``no_cot``, ``c_thought``); ``make_configs`` builds one.
  * The ``uniform_prob`` branch draws from the module-level ``random`` as
    upstream does. The exercise and its tests use ``uniform_prob = 0``
    (the GSM8k value), which never enters it.
  * Outputs stay Python lists, as upstream. ``attention_mask`` and ``idx``
    are kept so the dict matches upstream key for key.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

__all__ = ["GSM_COCONUT_CONFIG", "make_configs", "process_sample"]


# args/gsm_coconut.yaml, the keys process_dataset reads.
GSM_COCONUT_CONFIG = {
    "c_thought": 2,
    "max_latent_stage": 3,
    "pad_latent_to_max": True,
    "uniform_prob": 0.0,
    "no_cot": False,
}


def make_configs(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**GSM_COCONUT_CONFIG, **overrides})


def process_sample(
    sample,
    scheduled_stage,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    n_additional_tokens = 0 if no_special_marker else 2

    if (
        random.random() < configs.uniform_prob
    ):  # with some prob, randomly sample stage
        scheduled_stage_to_train = random.choice(
            list(range(len(sample["steps_tokenized"]) + 1))
        )
    else:
        scheduled_stage_to_train = scheduled_stage

    if scheduled_stage_to_train > configs.max_latent_stage:
        n_skip_steps = 10000  # skip all
        if configs.pad_latent_to_max:
            n_latent_tokens = configs.max_latent_stage
        else:
            n_latent_tokens = min(
                len(sample["steps_tokenized"]), configs.max_latent_stage
            )

    else:
        n_skip_steps, n_latent_tokens = (
            scheduled_stage_to_train,
            scheduled_stage_to_train,
        )

    if configs.no_cot:
        n_skip_steps = 100  # skip all step
        n_latent_tokens = 0

    n_latent_tokens *= configs.c_thought

    tokens = (
        sample["question_tokenized"]
        + ([] if no_special_marker else [start_id])
        + [latent_id] * n_latent_tokens
        + ([] if no_special_marker else [end_id])
        + list(
            itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
        )
        + sample["answer_tokenized"]
    )

    return {
        "input_ids": tokens,
        "labels": [-100]
        * (
            len(sample["question_tokenized"])
            + n_latent_tokens
            + n_additional_tokens
        )
        + tokens[
            n_latent_tokens
            + n_additional_tokens
            + len(sample["question_tokenized"]) :
        ],
        "attention_mask": [1] * len(tokens),
        "idx": sample["idx"],
        "position_ids": list(range(len(tokens))),
    }
