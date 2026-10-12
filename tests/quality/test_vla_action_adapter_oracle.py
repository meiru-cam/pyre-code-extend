"""vla_action_adapter's reference solution, checked against openpi and SimpleVLA-RL.

The oracle chains openpi's output transforms (quantile Unnormalize, AbsoluteActions,
LiberoOutputs) with SimpleVLA-RL's LIBERO gripper step (normalize_gripper_action
with binarize, then invert_gripper_action), all in numpy at the pinned commits.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.openpi_svr_action_adapter import (
    absolute_actions,
    invert_gripper_action,
    make_bool_mask,
    normalize_gripper_action,
    openpi_svr_adapt,
    pad_to_dim,
    unnormalize_quantile,
)
from torch_judge.tasks import get_task

TASK_ID = "vla_action_adapter"
ATOL = 1e-12


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def case(seed: int, lead: tuple[int, ...], horizon: int, dim: int, width: int, mask: tuple[bool, ...]):
    rng = np.random.default_rng(seed)
    # The model emits `dim` real values; openpi pads them to the model width with pad_to_dim.
    real = rng.uniform(-1.2, 1.2, size=lead + (horizon, dim))
    actions = pad_to_dim(real, width, axis=-1)
    actions[..., dim:] = rng.normal(size=actions[..., dim:].shape) * 5
    q01 = rng.uniform(-1.5, 0.5, size=dim)
    q99 = q01 + rng.uniform(0.1, 2.0, size=dim)
    q01[-1], q99[-1] = 0.0, 1.0
    state = rng.normal(size=lead + (dim,)) * 3
    return actions, q01, q99, state, np.asarray(mask)


def run_reference(actions, q01, q99, state, mask, **flags):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    out = reference()(t(actions), t(q01), t(q99), t(state), torch.from_numpy(mask), **flags)
    return out.numpy()


@pytest.mark.parametrize(
    "seed,lead,horizon,width,mask",
    [
        (0, (), 50, 32, make_bool_mask(6, -1)),
        (1, (), 10, 8, make_bool_mask(-7)),
        (2, (4,), 5, 32, make_bool_mask(6, -1)),
        (3, (3,), 3, 7, make_bool_mask(3, -3, -1)),
        (4, (2,), 8, 16, (True, False, True, False, True, False, False)),
        (5, (), 1, 7, make_bool_mask(6, -1)),
        (6, (5,), 2, 10, make_bool_mask(-6, -1)),
    ],
)
@pytest.mark.parametrize("binarize,invert", [(True, True), (False, True), (True, False), (False, False)])
def test_matches_openpi_then_simplevla_rl(seed, lead, horizon, width, mask, binarize, invert):
    actions, q01, q99, state, mask = case(seed, lead, horizon, len(mask), width, mask)
    got = run_reference(actions, q01, q99, state, mask, binarize_gripper=binarize, invert_gripper=invert)
    want = openpi_svr_adapt(actions.copy(), q01, q99, state, mask, binarize=binarize, invert=invert)
    assert got.shape == want.shape == lead + (horizon, len(mask))
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=0)


def test_matches_aloha_style_bimanual_mask():
    """make_bool_mask(6, -1, 6, -1): two arms of six joints with absolute grippers, D = 14."""
    mask = make_bool_mask(6, -1, 6, -1)
    actions, q01, q99, state, mask = case(7, (2,), 4, 14, 32, mask)
    got = run_reference(actions, q01, q99, state, mask)
    want = openpi_svr_adapt(actions.copy(), q01, q99, state, mask)
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=0)


def test_exact_half_gripper_is_zero_in_both():
    """np.sign(0) = 0 upstream and torch.sign(0) = 0 in the reference: a 0.5 gripper is neither open nor close."""
    q01 = np.array([0.0, 0.5])
    q99 = np.array([1.0, 1.0])
    actions = np.array([[0.3, -1.0], [0.3, 1.0]])  # gripper -1 unnormalizes to q01 = 0.5 exactly
    state = np.zeros(2)
    mask = np.array([False, False])
    want = openpi_svr_adapt(actions.copy(), q01, q99, state, mask)
    got = run_reference(actions, q01, q99, state, mask)
    assert want[0, 1] == 0.0 and got[0, 1] == 0.0
    assert want[1, 1] == got[1, 1] == -1.0
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=0)


def test_flip_commutes_with_sign_upstream():
    """invert_gripper_action before or after the sign gives the same command."""
    rng = np.random.default_rng(9)
    a = rng.uniform(0, 1, size=(20, 7))
    a[0, -1] = 0.5
    after = invert_gripper_action(normalize_gripper_action(a, binarize=True))
    flipped_first = a.copy()
    flipped_first[..., -1] = -(2 * flipped_first[..., -1] - 1)
    before = flipped_first.copy()
    before[..., -1] = np.sign(before[..., -1])
    np.testing.assert_array_equal(after, before)


def test_padded_dims_pass_through_upstream_and_are_dropped_by_the_exercise():
    """openpi's quantile unnormalize leaves dims beyond q01 untouched; LiberoOutputs then slices them off."""
    actions, q01, q99, state, mask = case(10, (), 3, 4, 9, make_bool_mask(3, -1))
    unnorm = unnormalize_quantile(actions, q01, q99)
    np.testing.assert_array_equal(unnorm[..., 4:], actions[..., 4:])
    absolute = absolute_actions(state, unnorm, mask)
    np.testing.assert_array_equal(absolute[..., 4:], actions[..., 4:])
    got = run_reference(actions, q01, q99, state, mask)
    assert got.shape == (3, 4)


def test_reference_does_not_modify_inputs_unlike_upstream_absolute_actions():
    """Upstream AbsoluteActions adds in place on the data dict; the exercise returns a new tensor (pinned divergence)."""
    actions, q01, q99, state, mask = case(12, (), 4, 7, 7, make_bool_mask(6, -1))
    before = actions.copy()
    run_reference(actions, q01, q99, state, mask)
    np.testing.assert_array_equal(actions, before)

    # The upstream in-place add, reproduced directly on an array.
    upstream = actions.copy()
    upstream[..., :7] += np.expand_dims(np.where(mask, state[..., :7], 0), axis=-2)
    np.testing.assert_allclose(upstream, absolute_actions(state, actions, mask), atol=0)
    assert not np.array_equal(upstream, actions)
