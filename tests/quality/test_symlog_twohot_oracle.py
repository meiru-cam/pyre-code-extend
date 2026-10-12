"""The symlog_twohot reference, checked against DreamerV3 (and the NM512 PyTorch port).

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied/vendored/dreamerv3_twohot.py`.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied.vendored.dreamerv3_twohot import (
    NM512DiscDist,
    TwoHot,
    danijar_twohot_bins,
    nm512_symexp,
    nm512_symlog,
    symexp,
    symlog,
)
from torch_judge.tasks import get_task

TASK_ID = "symlog_twohot"
SEEDS = (0, 1, 2, 3, 4, 5, 6)


def reference():
    """Execute the reference solution and return its namespace."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace


def identity(x):
    return x


def upstream(logits, bins):
    """danijar's TwoHot in the symlog-space configuration the exercise follows."""
    return TwoHot(logits, bins, squash=symlog, unsquash=symexp)


def batch(seed: int, K: int, shape=(4, 3), scale: float = 3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(*shape, K, generator=g, dtype=torch.float64) * scale
    # Raw targets spread over many magnitudes, a few beyond the symlog range of 20.
    y = torch.randn(*shape, generator=g, dtype=torch.float64) * torch.exp(
        torch.rand(*shape, generator=g, dtype=torch.float64) * 24 - 2)
    return logits, y


@pytest.mark.parametrize("seed", SEEDS)
def test_symlog_symexp_match_danijar(seed):
    ns = reference()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(500, generator=g, dtype=torch.float64) * torch.logspace(-8, 8, 500, dtype=torch.float64)
    torch.testing.assert_close(ns["symlog"](x), symlog(x), rtol=0, atol=0)
    torch.testing.assert_close(ns["symexp"](x / 1e6), symexp(x / 1e6), rtol=0, atol=0)


@pytest.mark.parametrize("K", (3, 5, 41, 255))
def test_odd_bins_match_danijar_symlog_positions(K):
    ns = reference()
    got = ns["symmetric_bins"](K)
    want = danijar_twohot_bins(K, transform=identity)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    # Both constructions mirror exactly, and the middle bin is exactly zero.
    assert torch.equal(got, -got.flip(0)) and torch.equal(want, -want.flip(0))
    assert got[K // 2] == 0 and want[K // 2] == 0


@pytest.mark.parametrize("K", (4, 254, 256))
def test_even_bins_diverge_from_danijar(K):
    """Pinned divergence: upstream's even K has two zero bins and a wider spacing."""
    ns = reference()
    got = ns["symmetric_bins"](K)
    want = danijar_twohot_bins(K, transform=identity)
    assert int((want == 0).sum()) == 2 and int((got == 0).sum()) == 0
    assert want[0] == got[0] == -20.0 and want[-1] == got[-1] == 20.0
    assert (want[1] - want[0]) > (got[1] - got[0])
    torch.testing.assert_close(got, torch.linspace(-20, 20, K), rtol=0, atol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("K", (5, 6, 41, 255))
def test_target_and_loss_match_danijar(seed, K):
    ns = reference()
    bins = ns["symmetric_bins"](K).double()
    logits, y = batch(seed, K)
    oracle = upstream(logits, bins)
    torch.testing.assert_close(ns["twohot_target"](y, bins), oracle.target(y), rtol=0, atol=1e-12)
    torch.testing.assert_close(ns["twohot_loss"](logits, y, bins), oracle.loss(y), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("K", (4, 5, 6, 41, 254, 255))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mean_matches_danijar_bitwise(seed, K, dtype):
    ns = reference()
    bins = ns["symmetric_bins"](K).to(dtype)
    logits, _ = batch(seed, K)
    logits = logits.to(dtype)
    assert torch.equal(ns["twohot_mean"](logits, bins), upstream(logits, bins).pred())


@pytest.mark.parametrize("K", (5, 6, 255))
def test_edge_cases_match_danijar(K):
    ns = reference()
    bins = ns["symmetric_bins"](K, 2.0).double()
    # On a bin, y = 0, outside the range on both sides, and exactly on the edges.
    y = torch.cat([symexp(bins), torch.tensor([0.0, 1e9, -1e9, 7.0, -7.0], dtype=torch.float64)])
    logits = torch.zeros(len(y), K, dtype=torch.float64)
    oracle = upstream(logits, bins)
    got = ns["twohot_target"](y, bins)
    torch.testing.assert_close(got, oracle.target(y), rtol=0, atol=1e-12)
    torch.testing.assert_close(got[:K], torch.eye(K, dtype=torch.float64), rtol=0, atol=1e-12)
    assert torch.equal(ns["twohot_mean"](logits, bins), torch.zeros(len(y), dtype=torch.float64))


@pytest.mark.parametrize("seed", SEEDS)
def test_raw_space_default_head_diverges_between_bins(seed):
    """Pinned divergence: danijar's default head interpolates in raw space between symexp bins."""
    ns = reference()
    bins = ns["symmetric_bins"](255).double()
    raw_bins = danijar_twohot_bins(255).double()
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(130, 200, (6,), generator=g)
    frac = torch.rand(6, generator=g, dtype=torch.float64) * 0.8 + 0.1
    y = symexp(bins[k] + frac * (bins[k + 1] - bins[k]))
    ours = ns["twohot_target"](y, bins)
    raw = TwoHot(torch.zeros(6, 255, dtype=torch.float64), raw_bins).target(y)
    # Same two neighbouring bins, different split of the mass.
    assert torch.equal(ours > 0, raw > 0)
    assert (ours - raw).abs().max() > 1e-3
    # Both decode the target exactly: ours via symlog-space bins, upstream via raw bins.
    torch.testing.assert_close(symexp((ours * bins).sum(-1)), y, rtol=1e-9, atol=0)
    torch.testing.assert_close((raw * raw_bins).sum(-1), y, rtol=1e-9, atol=0)


@pytest.mark.parametrize("seed", SEEDS)
def test_target_and_loss_match_nm512(seed):
    ns = reference()
    logits, y = batch(seed, 255)
    port = NM512DiscDist(logits)
    # The port's float64 linspace and the exercise's float32 bins agree to float32 precision;
    # the arithmetic is compared on the port's own buckets.
    torch.testing.assert_close(port.buckets, ns["symmetric_bins"](255).double(), rtol=0, atol=1e-5)
    bins = port.buckets
    # The port returns the log-probability (negative loss) and wants a trailing singleton dim.
    torch.testing.assert_close(-port.log_prob(y[..., None]), ns["twohot_loss"](logits, y, bins), rtol=0, atol=1e-9)
    torch.testing.assert_close(port.mean()[..., 0], ns["twohot_mean"](logits, bins), rtol=1e-9, atol=1e-9)


def test_nm512_plain_sum_mean_is_not_exactly_zero():
    """Pinned divergence: the port's left-to-right sum is not zero for uniform logits in float32."""
    ns = reference()
    logits = torch.zeros(3, 255)
    port = NM512DiscDist(logits)
    assert torch.all(port.mean() != 0)
    assert torch.equal(ns["twohot_mean"](logits, ns["symmetric_bins"](255)), torch.zeros(3))


def test_nm512_symlog_loses_tiny_values():
    """Pinned divergence: log(|x| + 1) and exp(|x|) - 1 round tiny inputs to zero."""
    ns = reference()
    x = torch.tensor([1e-20, -1e-18], dtype=torch.float64)
    assert torch.all(nm512_symlog(x) == 0) and torch.all(nm512_symexp(x) == 0)
    torch.testing.assert_close(ns["symlog"](x), x, rtol=1e-12, atol=0)
    torch.testing.assert_close(ns["symexp"](x), x, rtol=1e-12, atol=0)
