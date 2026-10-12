"""Vendored action normalization and uniform-binning oracles from OpenVLA and RT-1.

Upstream:   https://github.com/openvla/openvla
Commit:     c8f03f48af692657d3060c19588038c7220e9af9
Files:      prismatic/vla/action_tokenizer.py,
            prismatic/vla/datasets/rlds/utils/data_utils.py,
            prismatic/extern/hf/modeling_prismatic.py
Symbols:    ActionTokenizer.__init__, ActionTokenizer.__call__,
            ActionTokenizer.decode_token_ids_to_actions,
            normalize_action_and_proprio (BOUNDS_Q99 branch),
            OpenVLAForActionPrediction.predict_action (unnormalize step)
Retrieved:  2026-10-12
License:    MIT License
            Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti.

Upstream:   https://github.com/google-research/robotics_transformer
Commit:     4569641b8111f3f402c32d8e24becd2a6e952ecc
Files:      tokenizers/action_tokenizer.py
Symbols:    RT1ActionTokenizer.tokenize, RT1ActionTokenizer.detokenize (float branch)
Retrieved:  2026-10-12
License:    Apache License 2.0
            Copyright 2022 Google LLC. Licensed under the Apache License, Version
            2.0; you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for vla_action_bins, so that exercise is not checked only against an
oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Changes:

  * ``ActionTokenizer`` no longer wraps a Hugging Face tokenizer. It takes the
    integer ``vocab_size`` directly, and ``__call__`` returns the token ids
    ``vocab_size - digitize(action)`` instead of decoding them into a string.
    The ids are what the string would re-encode to.
  * ``normalize_action_and_proprio`` is TensorFlow over a trajectory dict. Only
    the ``BOUNDS_Q99`` arithmetic for one array is kept, written with numpy
    (``tf.where`` -> ``np.where``, ``tf.clip_by_value`` -> ``np.clip``). The
    upstream zeroing mask compares the dataset ``min`` and ``max``, not ``q01``
    and ``q99``, so both pairs are parameters here.
  * The unnormalize step is lifted out of ``predict_action`` (it runs after
    ``generate``, which needs model weights).
  * RT-1 is TensorFlow over a ``TensorSpecStruct``. Only the float-action branch
    is kept, for one ``(..., D)`` array with per-dimension ``minimum`` and
    ``maximum``, written with numpy (``tf.cast(x, tf.int32)`` -> ``astype(np.int32)``,
    both truncate toward zero). The per-dimension detokenize loop is
    vectorized, and it runs in float64 instead of float32.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OpenVLAActionTokenizer",
    "RT1FloatActionTokenizer",
    "openvla_normalize_bounds_q99",
    "openvla_unnormalize",
]


def openvla_normalize_bounds_q99(x, q01, q99, data_min, data_max, mask=None):
    """Transcribed from data_utils.py::normalize_action_and_proprio, BOUNDS_Q99 branch."""
    low = q01
    high = q99
    if mask is None:
        mask = np.ones_like(data_min, dtype=bool)
    x = np.where(
        mask,
        np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
        x,
    )

    # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
    zeros_mask = data_min == data_max
    x = np.where(zeros_mask, 0.0, x)
    return x


def openvla_unnormalize(normalized_actions, q01, q99, mask=None):
    """Transcribed from modeling_prismatic.py::predict_action, the unnormalize step."""
    if mask is None:
        mask = np.ones_like(q01, dtype=bool)
    action_high, action_low = np.array(q99), np.array(q01)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


class OpenVLAActionTokenizer:
    """Transcribed from action_tokenizer.py::ActionTokenizer, without the wrapped LLM tokenizer."""

    def __init__(self, vocab_size: int, bins: int = 256, min_action: int = -1, max_action: int = 1) -> None:
        self.vocab_size, self.n_bins, self.min_action, self.max_action = vocab_size, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """Clip & bin actions to the last `n_bins` token ids of the vocabulary."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        return self.vocab_size - discretized_action

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """Digitize returns [1, n_bins]; shift by one and clip so the top index maps to the last center."""
        discretized_actions = self.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]


class RT1FloatActionTokenizer:
    """Transcribed from tokenizers/action_tokenizer.py::RT1ActionTokenizer, float-action branch."""

    def __init__(self, minimum, maximum, vocab_size: int) -> None:
        self.minimum = np.asarray(minimum, dtype=np.float64)
        self.maximum = np.asarray(maximum, dtype=np.float64)
        self._vocab_size = vocab_size

    def tokenize(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, self.minimum, self.maximum)
        # Normalize the action [batch, actions_size]
        token = (a - self.minimum) / (self.maximum - self.minimum)
        # Bucket and discretize the action to vocab_size, [batch, actions_size]
        token = (token * (self._vocab_size - 1)).astype(np.int32)
        return token

    def detokenize(self, action_tokens: np.ndarray) -> np.ndarray:
        a = action_tokens.astype(np.float64)
        a = a / (self._vocab_size - 1)
        a = (a * (self.maximum - self.minimum)) + self.minimum
        return a
