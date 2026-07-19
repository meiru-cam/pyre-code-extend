"""Contract tests for MoE-specific deterministic evaluator helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_judge.harness.tensor import HarnessFailure
from torch_judge.harness.tensor.moe import (
    CountingExpert,
    assert_expert_gradient_partition,
    dense_capacity_dispatch_oracle,
    permute_expert_routes,
    seeded_domain_batch,
)


def _linear(scale: float) -> nn.Linear:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2) * scale)
    return layer


def test_dense_capacity_dispatch_oracle_applies_token_major_capacity_and_weights():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    indices = torch.tensor([[0, 1], [0, 1]])
    weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]])

    output, stats = dense_capacity_dispatch_oracle(
        x, indices, weights, [_linear(1.0), _linear(10.0)], capacity=1
    )

    assert torch.allclose(output, torch.tensor([[3.25, 6.5], [0.0, 0.0]]))
    assert stats == {"accepted": 2, "dropped": 2, "loads": [1, 1]}


def test_dense_capacity_dispatch_oracle_evaluates_each_expert_on_the_full_batch():
    experts = [CountingExpert(_linear(1.0)), CountingExpert(_linear(2.0))]
    x = torch.randn(4, 2)
    indices = torch.tensor([[0], [0], [1], [0]])
    weights = torch.ones(4, 1)

    dense_capacity_dispatch_oracle(x, indices, weights, experts, capacity=2)

    assert (experts[0].calls, experts[0].tokens) == (1, 4)
    assert (experts[1].calls, experts[1].tokens) == (1, 4)


def test_dense_capacity_dispatch_oracle_capacity_zero_masks_all_dense_outputs():
    experts = [CountingExpert(_linear(1.0))]
    output, stats = dense_capacity_dispatch_oracle(
        torch.randn(2, 2), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1), experts, 0
    )
    assert torch.equal(output, torch.zeros_like(output))
    assert stats == {"accepted": 0, "dropped": 2, "loads": [0]}
    assert (experts[0].calls, experts[0].tokens) == (1, 2)


@pytest.mark.parametrize(
    ("indices", "weights", "capacity", "match"),
    [
        (torch.tensor([[0]]), torch.ones(2, 1), 1, "leading dimensions"),
        (torch.tensor([[2]]), torch.ones(1, 1), 1, "expert index"),
        (torch.tensor([[0]]), torch.ones(1, 1), -1, "capacity"),
    ],
)
def test_dense_capacity_dispatch_oracle_rejects_invalid_author_fixtures(
    indices, weights, capacity, match
):
    with pytest.raises(HarnessFailure, match=match):
        dense_capacity_dispatch_oracle(
            torch.ones(1, 2), indices, weights, [_linear(1.0), _linear(2.0)], capacity
        )


def test_dense_capacity_dispatch_oracle_requires_floating_weights():
    with pytest.raises(HarnessFailure, match="expert_weights.*floating"):
        dense_capacity_dispatch_oracle(
            torch.ones(1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.long),
            [_linear(1.0)],
            1,
        )


def test_permute_expert_routes_preserves_route_meaning():
    experts = [_linear(1.0), _linear(2.0), _linear(3.0)]
    indices = torch.tensor([[0, 2], [1, 0]])
    remapped, reordered = permute_expert_routes(indices, experts, [2, 0, 1])
    assert torch.equal(remapped, torch.tensor([[1, 0], [2, 1]]))
    assert reordered == [experts[2], experts[0], experts[1]]


def test_assert_expert_gradient_partition_accepts_selected_only_and_reports_leaks():
    experts = [_linear(1.0), _linear(2.0)]
    experts[0](torch.ones(1, 2)).sum().backward()
    assert_expert_gradient_partition(experts, {0})

    experts[1].weight.grad = torch.ones_like(experts[1].weight)
    with pytest.raises(AssertionError, match="unselected expert 1"):
        assert_expert_gradient_partition(experts, {0})


def test_assert_expert_gradient_partition_rejects_any_nonfinite_selected_parameter():
    expert = nn.Linear(2, 2, bias=True)
    expert.weight.grad = torch.ones_like(expert.weight)
    expert.bias.grad = torch.full_like(expert.bias, float("nan"))

    with pytest.raises(AssertionError, match="selected expert 0.*finite"):
        assert_expert_gradient_partition([expert], {0})


def test_seeded_domain_batch_is_repeatable_separable_and_local_rng_only():
    torch.manual_seed(99)
    before = torch.random.get_rng_state().clone()
    first = seeded_domain_batch(7, tokens_per_domain=3, input_dim=4, num_domains=3)
    after = torch.random.get_rng_state()
    second = seeded_domain_batch(7, tokens_per_domain=3, input_dim=4, num_domains=3)

    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(before, after)
    x, targets, domains = first
    assert x.shape == (9, 4) and targets.shape == domains.shape == (9,)
    assert torch.equal(targets, domains)
    assert x[domains == 2, 2].mean() > x[domains == 2, 0].mean() + 2
