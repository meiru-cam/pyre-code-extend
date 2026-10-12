"""The fast_dct_action_tokens reference, checked against the upstream FAST processor.

The mutation gate proves the evaluator rejects wrong implementations; it cannot
prove the contract is right. Here the reference solution (a torch DCT matrix) is
compared against `UniversalActionProcessor.__call__` / `decode` from
physical-intelligence/fast, transcribed in
`torch_judge/harness/embodied/vendored/fast_action_tokenizer.py`, which calls
`scipy.fft.dct` / `idct` with `norm="ortho"` directly. scipy is a test-only
dependency, so it is used here and never in the task.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied.vendored.fast_action_tokenizer import fast_call_tokens, fast_decode_tokens
from torch_judge.tasks import get_task

TASK_ID = "fast_dct_action_tokens"


def reference():
    """Load the reference solution's encode and decode functions."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]], namespace["fast_decode"]


def upstream_min_token(chunks: np.ndarray, scale: float) -> int:
    """The min_token upstream `fit` would pick: the smallest rounded coefficient."""
    from scipy.fft import dct

    return int(np.around(dct(chunks, axis=1, norm="ortho") * scale).min())


def assert_matches_upstream(chunks: np.ndarray, scale: float, min_token: int) -> None:
    encode, decode = reference()
    B, T, D = chunks.shape
    want_tokens = fast_call_tokens(chunks, scale=scale, min_token=min_token)
    got_tokens = encode(torch.from_numpy(chunks), scale=scale, min_token=min_token)
    assert got_tokens.dtype == torch.long
    assert got_tokens.shape == (B, T * D)
    np.testing.assert_array_equal(got_tokens.numpy(), np.stack(want_tokens))

    want_actions = fast_decode_tokens(want_tokens, T, D, scale=scale, min_token=min_token)
    got_actions = decode(got_tokens, T, D, scale=scale, min_token=min_token)
    assert got_actions.dtype == torch.float64
    np.testing.assert_allclose(got_actions.numpy(), want_actions, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6])
def test_seeded_chunks_match_upstream(seed):
    rng = np.random.default_rng(seed)
    B, T, D = int(rng.integers(1, 5)), int(rng.integers(1, 17)), int(rng.integers(1, 8))
    scale = float(rng.choice([5.0, 10.0, 37.5, 100.0]))
    # Smooth trajectories, like real normalized action chunks, plus noise.
    t = np.linspace(0.0, 1.0, T)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(B, 1, D))
    chunks = np.clip(np.sin(2 * np.pi * t + phase) + 0.1 * rng.standard_normal((B, T, D)), -1, 1)
    assert_matches_upstream(chunks, scale, upstream_min_token(chunks, scale))


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_clamped_tokens_match_upstream(seed):
    """min_token above the smallest coefficient: both clamp at 0, and decode stays lossy the same way."""
    rng = np.random.default_rng(seed)
    chunks = rng.uniform(-1, 1, size=(3, 8, 4))
    min_token = upstream_min_token(chunks, 10.0) + 6
    tokens = fast_call_tokens(chunks, scale=10.0, min_token=min_token)
    assert any((row == 0).any() for row in tokens)
    assert_matches_upstream(chunks, 10.0, min_token)


def test_two_dimensional_chunk_matches_upstream():
    encode, _ = reference()
    chunk = np.random.default_rng(21).uniform(-1, 1, size=(10, 7))
    (want,) = fast_call_tokens(chunk, scale=10.0, min_token=-40)
    got = encode(torch.from_numpy(chunk), scale=10.0, min_token=-40)
    assert got.shape == (70,)
    np.testing.assert_array_equal(got.numpy(), want)


def test_single_step_matches_upstream():
    chunks = np.random.default_rng(22).uniform(-1, 1, size=(4, 1, 6))
    assert_matches_upstream(chunks, 10.0, -20)


def test_constant_chunk_matches_upstream():
    chunks = np.full((2, 9, 3), -0.4)
    chunks[1] = 0.7
    assert_matches_upstream(chunks, 10.0, -20)
    (tokens, _) = fast_call_tokens(chunks, scale=10.0, min_token=-20)
    assert (tokens[3:] == 20).all()


def test_exact_ties_are_not_part_of_the_contract():
    """Divergence: np.around rounds ties to even, but scipy's T = 1 orthonormal DCT is one ulp
    off the identity, so upstream never sees an exact tie there and rounds these away from zero.
    The reference's DCT matrix is exactly 1 at T = 1 and rounds them to even. The evaluator has
    no case that depends on an exact tie, and the description says so."""
    encode, _ = reference()
    chunks = np.array([[[0.5, 1.5, 2.5, -0.5, -1.5, -2.5]]])
    (upstream,) = fast_call_tokens(chunks, scale=1.0, min_token=-10)
    assert upstream.tolist() == [11, 12, 13, 9, 8, 7]
    got = encode(torch.from_numpy(chunks), scale=1.0, min_token=-10)
    assert got.tolist() == [[10, 12, 12, 10, 8, 8]]
    # Off the tie by more than an ulp, both agree.
    nudged = chunks + np.sign(chunks) * 1e-9
    assert_matches_upstream(nudged, 1.0, -10)


def test_float32_chunk_matches_upstream_in_float64():
    """Divergence: upstream keeps float32 through scipy; the exercise always computes in float64."""
    encode, _ = reference()
    chunk = np.random.default_rng(23).uniform(-1, 1, size=(2, 12, 5)).astype(np.float32)
    got = encode(torch.from_numpy(chunk), scale=1000.0, min_token=-5000)
    want = np.stack(fast_call_tokens(chunk.astype(np.float64), scale=1000.0, min_token=-5000))
    np.testing.assert_array_equal(got.numpy(), want)


def test_wrong_length_tokens_diverge_from_upstream():
    """Divergence: upstream decodes a wrong-length token row to all zeros; the exercise raises."""
    _, decode = reference()
    upstream = fast_decode_tokens([np.arange(5)], 2, 3, scale=10.0, min_token=0)
    np.testing.assert_array_equal(upstream, np.zeros((1, 2, 3)))
    with pytest.raises(ValueError):
        decode(torch.arange(5).unsqueeze(0), 2, 3, scale=10.0, min_token=0)
