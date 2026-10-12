"""The vla_action_bins reference solution, checked against OpenVLA and RT-1.

The mutation gate proves the evaluator rejects wrong implementations; it cannot
prove the contract matches upstream. These tests compare the reference against
code transcribed from OpenVLA (`prismatic/vla/action_tokenizer.py`, the
BOUNDS_Q99 normalization and the predict_action unnormalize step) and RT-1
(`tokenizers/action_tokenizer.py`), in
`torch_judge/harness/embodied/vendored/openvla_action_tokenizer.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied.vendored.openvla_action_tokenizer import (
    OpenVLAActionTokenizer,
    RT1FloatActionTokenizer,
    openvla_normalize_bounds_q99,
    openvla_unnormalize,
)
from torch_judge.tasks import get_task

SEEDS = [0, 1, 2, 3, 4, 5, 6]
BIN_COUNTS = [2, 3, 7, 256, 257]


def reference():
    task = get_task("vla_action_bins")
    assert task is not None
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def seeded_stats(seed: int, dims: int = 6, constant_dims=(2,)):
    """q01/q99 plus dataset min/max that are equal exactly on the constant dims."""
    rng = np.random.default_rng(seed)
    q01 = rng.uniform(-2.0, 0.5, dims)
    q99 = q01 + rng.uniform(0.1, 3.0, dims)
    data_min = q01 - rng.uniform(0.0, 0.5, dims)
    data_max = q99 + rng.uniform(0.0, 0.5, dims)
    for d in constant_dims:
        q99[d] = q01[d]
        data_min[d] = data_max[d] = q01[d]
    actions = q01 + (q99 - q01) * rng.uniform(-0.3, 1.3, (5, 4, dims))
    return q01, q99, data_min, data_max, actions


def _binner(q01, q99, n_bins, vocab, convention):
    return reference()(torch.from_numpy(q01), torch.from_numpy(q99), n_bins, vocab, convention=convention)


@pytest.mark.parametrize("seed", SEEDS)
def test_normalize_matches_openvla_bounds_q99(seed):
    q01, q99, data_min, data_max, actions = seeded_stats(seed)
    want = openvla_normalize_bounds_q99(actions, q01, q99, data_min, data_max)
    got = _binner(q01, q99, 256, 32000, "openvla").normalize(torch.from_numpy(actions)).numpy()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_bins", BIN_COUNTS)
def test_openvla_tokens_and_decode_match_upstream(seed, n_bins):
    q01, q99, data_min, data_max, actions = seeded_stats(seed)
    vocab = 32000 + 64 * seed
    tokenizer = OpenVLAActionTokenizer(vocab, bins=n_bins)
    normalized = openvla_normalize_bounds_q99(actions, q01, q99, data_min, data_max)
    want_ids = tokenizer(normalized)
    want_actions = openvla_unnormalize(tokenizer.decode_token_ids_to_actions(want_ids), q01, q99)

    binner = _binner(q01, q99, n_bins, vocab, "openvla")
    ids = binner.encode(torch.from_numpy(actions))
    assert ids.dtype == torch.long
    np.testing.assert_array_equal(ids.numpy(), want_ids)
    assert ids.min().item() > tokenizer.action_token_begin_idx
    np.testing.assert_allclose(binner.decode(ids).numpy(), want_actions, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_bins", BIN_COUNTS)
def test_rt1_tokens_and_decode_match_upstream(seed, n_bins):
    q01, q99, data_min, data_max, actions = seeded_stats(seed)
    normalized = openvla_normalize_bounds_q99(actions, q01, q99, data_min, data_max)
    rt1 = RT1FloatActionTokenizer(-np.ones_like(q01), np.ones_like(q01), n_bins)
    want_ids = rt1.tokenize(normalized)
    want_actions = openvla_unnormalize(rt1.detokenize(want_ids), q01, q99)

    binner = _binner(q01, q99, n_bins, 32000, "rt1")
    ids = binner.encode(torch.from_numpy(actions))
    np.testing.assert_array_equal(ids.numpy(), want_ids)
    np.testing.assert_allclose(binner.decode(ids).numpy(), want_actions, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_rt1_in_normalized_space_equals_rt1_on_raw_bounds(seed):
    """Applying RT-1 to the [-1, 1] value is RT-1 on the raw action with min = q01, max = q99."""
    q01, q99, _, _, actions = seeded_stats(seed, constant_dims=())
    n_bins = 256
    rt1 = RT1FloatActionTokenizer(q01, q99, n_bins)
    binner = _binner(q01, q99, n_bins, 32000, "rt1")
    ids = binner.encode(torch.from_numpy(actions))
    np.testing.assert_array_equal(ids.numpy(), rt1.tokenize(actions))
    np.testing.assert_allclose(binner.decode(ids).numpy(), rt1.detokenize(rt1.tokenize(actions)), atol=1e-12)


@pytest.mark.parametrize("n_bins", BIN_COUNTS)
def test_openvla_edges_and_out_of_range_ids(n_bins):
    """Clipped +1 digitizes to n_bins; -1 to 1; ids outside the tail clamp to the end centers."""
    vocab = 1000
    q01, q99 = np.zeros(1), np.full(1, 2.0)
    tokenizer = OpenVLAActionTokenizer(vocab, bins=n_bins)
    binner = _binner(q01, q99, n_bins, vocab, "openvla")

    actions = np.array([[-5.0], [0.0], [5.0], [2.0]])
    want_ids = tokenizer(openvla_normalize_bounds_q99(actions, q01, q99, q01, q99))
    ids = binner.encode(torch.from_numpy(actions))
    np.testing.assert_array_equal(ids.numpy(), want_ids)
    assert ids[0, 0].item() == vocab - 1 and ids[2, 0].item() == vocab - n_bins

    weird = np.array([[vocab + 50], [vocab], [vocab - n_bins - 3], [0]])
    want = openvla_unnormalize(tokenizer.decode_token_ids_to_actions(weird), q01, q99)
    np.testing.assert_allclose(binner.decode(torch.from_numpy(weird)).numpy(), want, atol=1e-12)


def test_divergence_constant_quantiles_with_moving_extremes():
    """Pinned divergence: upstream zeroes dims where the dataset min == max, the exercise where q01 == q99.

    For a dimension whose q01 equals q99 but whose min and max differ, upstream divides by 1e-8 and
    clips to -1 or +1; the exercise sets it to 0. They agree whenever min == max.
    """
    q01 = np.array([0.0, 1.0])
    q99 = np.array([2.0, 1.0])
    data_min = np.array([-1.0, 0.5])
    data_max = np.array([3.0, 1.5])
    actions = np.array([[1.0, 1.2], [1.0, 0.8]])
    upstream = openvla_normalize_bounds_q99(actions, q01, q99, data_min, data_max)
    ours = _binner(q01, q99, 256, 32000, "openvla").normalize(torch.from_numpy(actions)).numpy()
    np.testing.assert_allclose(ours[:, 0], upstream[:, 0], atol=1e-12)
    np.testing.assert_array_equal(upstream[:, 1], [1.0, -1.0])
    np.testing.assert_array_equal(ours[:, 1], [0.0, 0.0])

    agree = openvla_normalize_bounds_q99(actions, q01, q99, data_min=np.array([-1.0, 1.0]), data_max=np.array([3.0, 1.0]))
    np.testing.assert_allclose(ours, agree, atol=1e-12)
