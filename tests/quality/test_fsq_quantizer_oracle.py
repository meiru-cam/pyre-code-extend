"""The fsq_quantizer reference solution, checked against NVIDIA Cosmos-Tokenizer's FSQuantizer.

The oracle is transcribed in
`torch_judge/harness/embodied_frontier/vendored/cosmos_tokenizer_quantizers.py`.
Deliberate divergences, listed in the task's `sources[].simplifications`, are pinned
below: int64 rather than int32 indices, no bfloat16 output cast, and rounding codes to
digits before the dot product rather than truncating (identical on every exact code).
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.cosmos_tokenizer_quantizers import CosmosFSQuantizer
from torch_judge.tasks import get_task

LEVEL_SETS = [
    [8, 8, 8, 5, 5, 5],  # Cosmos discrete image and video configs
    [8, 6, 5],           # FSQ paper Table 1, about 2^8
    [8, 5, 5, 5],        # about 2^10
    [7, 5, 5, 5, 5],     # about 2^12
    [8, 8, 8, 6, 5],     # about 2^14
    [3, 4, 2, 9],        # small and odd sizes
]


def reference_class():
    task = get_task("fsq_quantizer")
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


@pytest.mark.parametrize("levels", LEVEL_SETS)
@pytest.mark.parametrize("seed", range(5))
def test_forward_matches_cosmos(levels, seed):
    ours = reference_class()(levels)
    oracle = CosmosFSQuantizer(levels)
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(4, 6, len(levels), generator=generator) * (0.5 + seed)

    torch.testing.assert_close(ours.bound(z), oracle.bound(z), rtol=0, atol=1e-6)
    zhat, indices = ours(z)
    oracle_indices, oracle_codes = oracle(z)
    torch.testing.assert_close(zhat, oracle_codes, rtol=0, atol=0)
    assert indices.dtype == torch.int64 and oracle_indices.dtype == torch.int32
    assert torch.equal(indices, oracle_indices.long())
    assert ours.codebook_size == oracle.codebook_size


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_implicit_codebook_matches_cosmos(levels):
    ours = reference_class()(levels)
    oracle = CosmosFSQuantizer(levels)
    every_index = torch.arange(oracle.codebook_size)
    codes = ours.indices_to_codes(every_index)
    torch.testing.assert_close(codes, oracle.implicit_codebook.float(), rtol=0, atol=0)
    assert torch.equal(ours.codes_to_indices(codes), oracle.codes_to_indices(codes).long())


@pytest.mark.parametrize("seed", range(5))
def test_straight_through_gradient_matches_cosmos(seed):
    levels = [8, 8, 8, 5, 5, 5]
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(3, 5, len(levels), generator=generator, dtype=torch.float64)
    weights = torch.randn(z.shape, generator=generator, dtype=torch.float64)

    z_ours = z.clone().requires_grad_()
    (reference_class()(levels).quantize(z_ours) * weights).sum().backward()
    z_oracle = z.clone().requires_grad_()
    (CosmosFSQuantizer(levels).quantize(z_oracle) * weights).sum().backward()
    torch.testing.assert_close(z_ours.grad, z_oracle.grad, rtol=0, atol=1e-6)


def test_edge_inputs_match_cosmos():
    levels = [8, 5, 4, 7, 3, 6]
    ours = reference_class()(levels)
    oracle = CosmosFSQuantizer(levels)
    z = torch.tensor([[0.0] * 6, [50.0] * 6, [-50.0] * 6, [1e-4] * 6, [-1e-4] * 6])
    zhat, indices = ours(z)
    oracle_indices, oracle_codes = oracle(z)
    torch.testing.assert_close(zhat, oracle_codes, rtol=0, atol=0)
    assert torch.equal(indices, oracle_indices.long())
    # bound(0) is 0 on every channel because of the atanh shift.
    assert ours.bound(torch.zeros(6)).abs().max() < 1e-6
    assert oracle.bound(torch.zeros(6)).abs().max() < 1e-6


def test_divergence_upstream_bfloat16_cast_is_dropped():
    """Upstream defaults to casting outputs to bfloat16; the exercise keeps z's dtype."""
    levels = [8, 8, 8, 5, 5, 5]
    z = torch.randn(10, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    zhat, _ = reference_class()(levels)(z)
    _, upstream_codes = CosmosFSQuantizer(levels, dtype=torch.bfloat16)(z)
    assert zhat.dtype == torch.float64 and upstream_codes.dtype == torch.bfloat16
    # The level grids are dyadic, so the cast is lossless on these codes.
    torch.testing.assert_close(zhat, upstream_codes.double(), rtol=0, atol=0)
