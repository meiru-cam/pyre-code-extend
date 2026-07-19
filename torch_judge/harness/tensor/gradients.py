"""Autograd reachability assertions."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from torch_judge.harness import HarnessFailure


def assert_gradients_reach(
    loss: torch.Tensor,
    named_tensors: Mapping[str, torch.Tensor],
    *,
    require_nonzero: bool = True,
) -> None:
    """Assert a scalar loss has finite gradients to every named tensor."""
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise HarnessFailure("loss must be a scalar torch tensor")
    if not named_tensors:
        raise HarnessFailure("at least one gradient target is required")
    names = list(named_tensors)
    tensors = list(named_tensors.values())
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise HarnessFailure("every gradient target must be a torch tensor")
    gradients = torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
    for name, gradient in zip(names, gradients):
        assert gradient is not None, f"{name} did not receive a gradient"
        assert bool(torch.isfinite(gradient).all()), f"{name} gradient must be finite"
        if require_nonzero:
            assert bool(torch.count_nonzero(gradient)), f"{name} gradient must be non-zero"
