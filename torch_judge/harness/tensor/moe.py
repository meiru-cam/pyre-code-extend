"""Independent, deterministic helpers for authoring MoE evaluator cases."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from torch_judge.harness import HarnessFailure


class CountingExpert(nn.Module):
    """Wrap an expert and record invocation and token counts without changing its result."""

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.calls = 0
        self.tokens = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.tokens += x.shape[0]
        return self.expert(x)


def _validate_dispatch_fixture(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: Sequence[nn.Module],
    capacity: int,
) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise HarnessFailure("x must be a rank-2 torch tensor")
    if not isinstance(expert_indices, torch.Tensor) or expert_indices.ndim != 2:
        raise HarnessFailure("expert_indices must be a rank-2 torch tensor")
    if not isinstance(expert_weights, torch.Tensor) or expert_weights.ndim != 2:
        raise HarnessFailure("expert_weights must be a rank-2 torch tensor")
    if not x.is_floating_point():
        raise HarnessFailure("x must be a floating-point tensor")
    if not expert_weights.is_floating_point():
        raise HarnessFailure("expert_weights must be floating point")
    if expert_indices.shape != expert_weights.shape or expert_indices.shape[0] != x.shape[0]:
        raise HarnessFailure("route tensors must have matching token/slot leading dimensions")
    if expert_indices.dtype != torch.long:
        raise HarnessFailure("expert_indices must have dtype torch.long")
    if expert_indices.device != x.device or expert_weights.device != x.device:
        raise HarnessFailure("x and route tensors must use the same device")
    if not experts:
        raise HarnessFailure("at least one expert is required")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise HarnessFailure("capacity must be a non-negative integer")
    if expert_indices.numel() and (
        int(expert_indices.min()) < 0 or int(expert_indices.max()) >= len(experts)
    ):
        raise HarnessFailure("expert index is outside the supplied expert range")


def dense_capacity_dispatch_oracle(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: Sequence[nn.Module],
    capacity: int,
) -> tuple[torch.Tensor, dict[str, int | list[int]]]:
    """Algorithmically independent dense reference with token-major capacity acceptance.

    Every expert evaluates the full token batch; an independently constructed acceptance mask
    then gathers and weights selected rows. Sparse invocation accounting belongs in separate
    candidate checks. Experts must preserve the input row shape, dtype, and device.
    """
    _validate_dispatch_fixture(x, expert_indices, expert_weights, experts, capacity)
    flat_indices = expert_indices.reshape(-1)
    one_hot = torch.nn.functional.one_hot(flat_indices, num_classes=len(experts))
    ordinals = one_hot.cumsum(dim=0).gather(1, flat_indices.unsqueeze(1)).squeeze(1)
    accepted_flat = ordinals <= capacity
    accepted_mask = accepted_flat.reshape_as(expert_indices)

    dense_outputs = []
    for expert in experts:
        expert_output = expert(x)
        if (
            not isinstance(expert_output, torch.Tensor)
            or expert_output.shape != x.shape
            or expert_output.dtype != x.dtype
            or expert_output.device != x.device
        ):
            raise HarnessFailure(
                "each expert must preserve the input row shape, dtype, and device"
            )
        dense_outputs.append(expert_output)
    expert_outputs = torch.stack(dense_outputs, dim=0)
    token_ids = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(expert_indices)
    routed = expert_outputs[expert_indices, token_ids]
    output = (
        routed
        * expert_weights.unsqueeze(-1)
        * accepted_mask.unsqueeze(-1).to(expert_weights.dtype)
    ).sum(dim=1)

    loads_tensor = torch.bincount(flat_indices[accepted_flat], minlength=len(experts))
    loads = [int(load) for load in loads_tensor]
    accepted_count = int(accepted_flat.sum())
    total = expert_indices.numel()
    return output, {
        "accepted": accepted_count,
        "dropped": total - accepted_count,
        "loads": loads,
    }


def permute_expert_routes(
    expert_indices: torch.Tensor,
    experts: Sequence[nn.Module],
    order: Sequence[int],
) -> tuple[torch.Tensor, list[nn.Module]]:
    """Reorder experts and remap route ids so every route keeps its original meaning."""
    if sorted(order) != list(range(len(experts))):
        raise HarnessFailure("order must be a permutation of every expert index")
    if not isinstance(expert_indices, torch.Tensor) or expert_indices.dtype != torch.long:
        raise HarnessFailure("expert_indices must be a torch.long tensor")
    inverse = torch.empty(len(order), dtype=torch.long, device=expert_indices.device)
    inverse[torch.tensor(order, dtype=torch.long, device=expert_indices.device)] = torch.arange(
        len(order), device=expert_indices.device
    )
    return inverse[expert_indices], [experts[index] for index in order]


def assert_expert_gradient_partition(
    experts: Sequence[nn.Module], selected: set[int]
) -> None:
    """Require finite nonzero gradients on selected experts and none on unselected experts."""
    if any(index < 0 or index >= len(experts) for index in selected):
        raise HarnessFailure("selected contains an invalid expert index")
    for index, expert in enumerate(experts):
        parameters = list(expert.parameters())
        if not parameters:
            raise HarnessFailure(f"expert {index} has no parameters to inspect")
        gradients = [parameter.grad for parameter in parameters]
        if index in selected:
            assert all(gradient is not None for gradient in gradients), (
                f"selected expert {index} did not receive gradients for every parameter"
            )
            assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients), (
                f"selected expert {index} gradients must all be finite"
            )
            assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients), (
                f"selected expert {index} did not receive a non-zero gradient"
            )
        else:
            assert all(gradient is None or not bool(torch.count_nonzero(gradient)) for gradient in gradients), (
                f"unselected expert {index} received a gradient"
            )


def seeded_domain_batch(
    seed: int,
    *,
    tokens_per_domain: int,
    input_dim: int,
    num_domains: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a repeatable, separable toy domain-classification batch using a local RNG."""
    if min(tokens_per_domain, input_dim, num_domains) < 1 or input_dim < num_domains:
        raise HarnessFailure("positive sizes and input_dim >= num_domains are required")
    generator = torch.Generator().manual_seed(seed)
    domains = torch.arange(num_domains).repeat_interleave(tokens_per_domain)
    centers = torch.eye(input_dim)[:num_domains] * 4.0
    noise = torch.randn(
        num_domains * tokens_per_domain, input_dim, generator=generator
    ) * 0.1
    x = centers.index_select(0, domains) + noise
    return x, domains.clone(), domains
