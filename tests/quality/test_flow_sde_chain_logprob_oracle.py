"""The flow_sde_chain_logprob reference, checked against RLinf.

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied_rl/vendored/rlinf_flow_sde.py`.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.rlinf_flow_sde import (
    rlinf_gaussian_logprob,
    rlinf_recompute_logprobs,
    rlinf_reduce_logprobs,
    rlinf_sample_mean_var,
)
from torch_judge.tasks import get_task

TASK_ID = "flow_sde_chain_logprob"
SEEDS = (0, 1, 2, 3, 4, 5, 6)
LEVELS = ("token_level", "action_level", "chunk_level")


def reference():
    """Load the reference solution's namespace."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace


@contextlib.contextmanager
def float64_default():
    """Upstream builds its grid with a default-dtype linspace; make it float64 to compare exactly."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def field(horizon: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1000)
    weights = torch.randn(horizon, horizon, generator=g, dtype=torch.float64) / horizon

    def v(x, t):
        tt = t.to(x.dtype)[:, None, None]
        return torch.einsum("hk,bkd->bhd", weights.to(x.dtype), torch.tanh(x)) * (1 + tt) - 0.5 * x * tt

    return v


def problem(seed: int, num_steps: int, batch: int = 4, horizon: int = 6, dim: int = 3):
    g = torch.Generator().manual_seed(seed)
    chain = torch.randn(batch, num_steps + 1, horizon, dim, generator=g, dtype=torch.float64)
    sde_index = torch.randint(0, num_steps, (batch,), generator=g)
    return chain, sde_index


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_steps", (1, 2, 5, 10))
@pytest.mark.parametrize("method", ("flow_ode", "flow_sde"))
def test_step_matches_sample_mean_var(seed, num_steps, method):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 4, 3, generator=g, dtype=torch.float64)
    v = torch.randn(5, 4, 3, generator=g, dtype=torch.float64)
    idx = torch.randint(0, num_steps, (5,), generator=g)
    idx[0] = 0  # always exercise the t = 1 guard
    idx[1] = num_steps - 1
    noise_level = 0.2 + 0.15 * seed
    got_mean, got_std = reference()["flow_sde_mean_std"](x, v, idx, num_steps, method, noise_level)
    with float64_default():
        want_mean, want_std = rlinf_sample_mean_var(
            x, v, idx, noise_method=method, noise_level=noise_level, num_steps=num_steps
        )
    torch.testing.assert_close(got_mean, want_mean, rtol=0, atol=1e-12)
    torch.testing.assert_close(got_std, want_std.expand_as(x), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_steps", (1, 3, 10))
@pytest.mark.parametrize("joint", (False, True))
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("truncate", (False, True))
def test_chain_logprob_matches_rlinf(seed, num_steps, joint, level, truncate):
    chain, sde_index = problem(seed, num_steps)
    batch, _, horizon, dim = chain.shape
    action_chunk, action_dim = (4, 2) if truncate else (horizon, dim)
    v = field(horizon, seed)
    got = reference()["flow_chain_logprob"](
        chain, v, None if joint else sde_index, 0.7, joint=joint, level=level,
        action_chunk=action_chunk if truncate else None, action_dim=action_dim if truncate else None,
    )
    # Upstream stores denoise_inds as [B, K]: arange rows in joint mode, the chosen index repeated otherwise.
    denoise_inds = (
        torch.arange(num_steps)[None].expand(batch, -1) if joint else sde_index[:, None].expand(batch, num_steps)
    )
    with float64_default():
        per_element = rlinf_recompute_logprobs(
            chain, denoise_inds, v, num_steps=num_steps, noise_method="flow_sde", noise_level=0.7,
            joint_logprob=joint, action_chunk=action_chunk, action_env_dim=action_dim,
        )
    want = rlinf_reduce_logprobs(per_element.reshape(batch, -1), level, action_dim)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("joint", (False, True))
def test_zero_noise_level_matches_rlinf_zero_mask(joint):
    """noise_level = 0 makes every step std 0; both sides must return exactly 0 for those terms."""
    chain, sde_index = problem(11, 4)
    v = field(chain.shape[2], 11)
    got = reference()["flow_chain_logprob"](chain, v, None if joint else sde_index, 0.0, joint=joint, level="token_level")
    denoise_inds = torch.arange(4)[None].expand(4, -1) if joint else sde_index[:, None].expand(4, 4)
    with float64_default():
        want = rlinf_recompute_logprobs(
            chain, denoise_inds, v, num_steps=4, noise_method="flow_sde", noise_level=0.0,
            joint_logprob=joint, action_chunk=6, action_env_dim=3,
        )
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    if not joint:
        assert torch.equal(got, torch.zeros_like(got))


def test_gaussian_logprob_is_zero_where_sigma_is_zero():
    sample = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    mu = torch.zeros(3, dtype=torch.float64)
    sigma = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
    want = rlinf_gaussian_logprob(sample, mu, sigma)
    got = reference()["_gaussian_logprob"](sample, mu, sigma)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert got[0] == 0 and got[2] == 0


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_float32_matches_upstream_float32_grid(seed):
    """At the default float32 dtype upstream's linspace grid and the exercise's 1 - k/K agree to rounding."""
    chain, sde_index = problem(seed, 10)
    chain = chain.float()
    v = field(chain.shape[2], seed)
    for joint in (False, True):
        got = reference()["flow_chain_logprob"](chain, v, None if joint else sde_index, 0.5, joint=joint, level="token_level")
        denoise_inds = torch.arange(10)[None].expand(4, -1) if joint else sde_index[:, None].expand(4, 10)
        want = rlinf_recompute_logprobs(
            chain, denoise_inds, v, num_steps=10, noise_method="flow_sde", noise_level=0.5,
            joint_logprob=joint, action_chunk=6, action_env_dim=3,
        )
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
