"""Vendored FAST action-tokenizer arithmetic (DCT, quantization, flattening) without BPE.

Upstream:   https://huggingface.co/physical-intelligence/fast
Commit:     ec4d7aa71691cac0b8bed6942be45684db2110f4 (Hugging Face revision)
Files:      processing_action_tokenizer.py
Symbols:    UniversalActionProcessor.__call__, UniversalActionProcessor.decode
Retrieved:  2026-10-12
License:    Apache License 2.0 (model card metadata `license: apache-2.0`).
            Licensed under the Apache License, Version 2.0; you may obtain a copy
            at http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for fast_dct_action_tokens, so the exercise is not checked only
against an oracle written by the same author as its reference solution. The
upstream calls ``scipy.fft.dct`` / ``scipy.fft.idct`` with ``norm="ortho"``;
the reference solution builds the DCT as a torch matrix instead. scipy is a
test-only dependency of pyre-code, so this module is imported only by
``tests/quality/`` and never by the grading sandbox.

WHAT WAS CHANGED, AND WHY
-------------------------
The DCT, rounding, flattening, ``min_token`` shift and clamp are transcribed
line for line, still in numpy and scipy. Removed infrastructure:

  * The BPE step is dropped. ``__call__`` returns the integer array that upstream
    turns into a string with ``map(chr, ...)`` and feeds to ``bpe_tokenizer``;
    ``decode`` takes that integer array instead of BPE ids and skips
    ``bpe_tokenizer.decode`` and ``map(ord, ...)``. ``chr`` / ``ord`` are an exact
    round trip for non-negative ints, so the arithmetic is unchanged. BPE is
    covered by the separate ``bpe`` exercise.
  * ``ProcessorMixin``, ``fit`` and the cached ``time_horizon`` / ``action_dim``
    state are dropped; ``time_horizon`` and ``action_dim`` are passed explicitly.
  * ``scale`` and ``min_token`` become function arguments instead of attributes.
  * The two ``print`` calls in the decode error path are dropped. The fallback
    itself is kept: a token row of the wrong length decodes to all-zero actions.
    The exercise raises ``ValueError`` instead, and the oracle test pins that
    divergence.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

__all__ = ["fast_call_tokens", "fast_decode_tokens"]


def fast_call_tokens(action_chunk: np.ndarray, scale: float = 10, min_token: int = 0) -> list[np.ndarray]:
    """Transcribed from UniversalActionProcessor.__call__, stopping before BPE.

    Returns one 1-D int array per batch item: the code points upstream passes to
    ``chr`` before BPE.
    """
    assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
    if action_chunk.ndim == 2:
        action_chunk = action_chunk[None, ...]

    dct_coeff = dct(action_chunk, axis=1, norm="ortho")
    dct_coeff = np.around(dct_coeff * scale)
    tokens = []
    for elem in dct_coeff:
        tokens.append(np.maximum(elem.flatten() - min_token, 0).astype(int))
    return tokens


def fast_decode_tokens(
    tokens: list[np.ndarray], time_horizon: int, action_dim: int, scale: float = 10, min_token: int = 0
) -> np.ndarray:
    """Transcribed from UniversalActionProcessor.decode, starting after BPE decoding."""
    decoded_actions = []
    for token in tokens:
        try:
            decoded_dct_coeff = np.array(list(token)) + min_token
            decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
            assert (
                decoded_dct_coeff.shape
                == (
                    time_horizon,
                    action_dim,
                )
            ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({time_horizon}, {action_dim})"
        except Exception:
            decoded_dct_coeff = np.zeros((time_horizon, action_dim))
        decoded_actions.append(idct(decoded_dct_coeff / scale, axis=0, norm="ortho"))
    return np.stack(decoded_actions)
