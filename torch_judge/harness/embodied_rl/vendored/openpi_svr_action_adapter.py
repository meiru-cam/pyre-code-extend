"""Vendored VLA action post-processing: openpi output transforms plus SimpleVLA-RL's LIBERO gripper step.

Upstream:   https://github.com/Physical-Intelligence/openpi
Commit:     215abfb217dbac7d5f1273282331b9b1866c0479
Files:      src/openpi/transforms.py, src/openpi/policies/libero_policy.py
Symbols:    Unnormalize._unnormalize_quantile (L175-181), AbsoluteActions.__call__ (L226-244 (__call__ at L234-244)),
            pad_to_dim (L423-430), make_bool_mask (L433-452), LiberoOutputs (L87-100)
License:    Apache License 2.0, Physical Intelligence. Licensed under the Apache
            License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

Upstream:   https://github.com/PRIME-RL/SimpleVLA-RL
Commit:     7c51662df27b586f9e8a1ab35fcf849f2b8852f9
Files:      verl/utils/libero_utils.py (call site: verl/workers/rollout/rob_rollout.py L398-399)
Symbols:    normalize_gripper_action (L156-184), invert_gripper_action (L196-215)
License:    MIT License, Copyright (c) 2025 PRIME-RL

Retrieved:  2026-10-12

WHY THIS FILE EXISTS
--------------------
Ground truth for vla_action_adapter, so the exercise is not checked only against
an oracle written by the same author as its reference solution.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line and stays in numpy. Removed
infrastructure:

  * ``Unnormalize`` / ``AbsoluteActions`` are frozen dataclasses applied to a data
    dict through ``apply_tree``; here they are plain functions on arrays. The
    z-score branch and the ``NormStats`` container are dropped.
  * ``AbsoluteActions`` adds to ``data["actions"]`` in place; here it works on a
    copy so the caller's array is untouched. The values are identical.
  * ``LiberoOutputs`` hard-codes ``[..., :7]``; here the kept width is an argument.
  * ``openpi_svr_adapt`` composes the steps in openpi's output order
    (``policy_config.py`` L84-89: ``Unnormalize``, then the data outputs, where
    ``Group.push`` puts ``AbsoluteActions`` before ``LiberoOutputs``), followed by
    SimpleVLA-RL's rollout call order (``normalize_gripper_action`` with
    ``binarize=True``, then ``invert_gripper_action``). Chaining the two projects
    is the exercise's design: openpi's own LIBERO client does not binarize the
    gripper, and SimpleVLA-RL unnormalizes OpenVLA bins with its own masked
    formula without the ``+ 1e-6``.
  * In openpi the state reaching ``AbsoluteActions`` is the model input state run
    back through ``Unnormalize``; here the caller passes the raw state directly.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "absolute_actions",
    "invert_gripper_action",
    "libero_outputs",
    "make_bool_mask",
    "normalize_gripper_action",
    "openpi_svr_adapt",
    "pad_to_dim",
    "unnormalize_quantile",
]


def unnormalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """transforms.py::Unnormalize._unnormalize_quantile."""
    if (dim := q01.shape[-1]) < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def absolute_actions(state: np.ndarray, actions: np.ndarray, mask: Sequence[bool] | None) -> np.ndarray:
    """transforms.py::AbsoluteActions.__call__ (on a copy)."""
    actions = np.array(actions, copy=True)
    if mask is None:
        return actions
    mask = np.asarray(mask)
    dims = mask.shape[-1]
    actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    return actions


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """transforms.py::pad_to_dim."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """transforms.py::make_bool_mask."""
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def libero_outputs(actions: np.ndarray, action_dim: int = 7) -> np.ndarray:
    """libero_policy.py::LiberoOutputs.__call__, with the width as an argument."""
    return np.asarray(actions[..., :action_dim])


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """libero_utils.py::normalize_gripper_action."""
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """libero_utils.py::invert_gripper_action."""
    inverted_action = action.copy()
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0
    return inverted_action


def openpi_svr_adapt(
    actions: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    state: np.ndarray,
    delta_mask: Sequence[bool] | None,
    binarize: bool = True,
    invert: bool = True,
) -> np.ndarray:
    """Unnormalize -> AbsoluteActions -> LiberoOutputs -> gripper normalize/binarize -> invert."""
    x = unnormalize_quantile(actions, q01, q99)
    x = absolute_actions(state, x, delta_mask)
    x = libero_outputs(x, q01.shape[-1])
    x = normalize_gripper_action(x, binarize=binarize)
    if invert:
        x = invert_gripper_action(x)
    return x
