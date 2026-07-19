"""Small, explicit tensor assertions with useful learner-facing errors."""

from __future__ import annotations

import torch

from torch_judge.harness import HarnessFailure


def _require_expected_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise HarnessFailure(f"expected oracle value must be a torch tensor, got {type(value).__name__}")
    return value


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    name: str = "output",
    check_dtype: bool = True,
    check_device: bool = True,
) -> None:
    """Assert tensor structure before numerical closeness."""
    expected = _require_expected_tensor(expected)
    assert isinstance(actual, torch.Tensor), (
        f"actual learner output must be a torch tensor, got {type(actual).__name__}"
    )
    assert actual.shape == expected.shape, (
        f"{name} shape differs: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
    )
    if check_dtype:
        assert actual.dtype == expected.dtype, (
            f"{name} dtype differs: expected {expected.dtype}, got {actual.dtype}"
        )
    if check_device:
        assert actual.device == expected.device, (
            f"{name} device differs: expected {expected.device}, got {actual.device}"
        )
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{name} values differ: {error}") from error


def assert_finite(tensor: torch.Tensor, *, name: str = "tensor") -> None:
    """Assert every tensor element is finite."""
    if not isinstance(tensor, torch.Tensor):
        raise HarnessFailure(f"{name} must be a torch tensor, got {type(tensor).__name__}")
    assert bool(torch.isfinite(tensor).all()), f"{name} must contain only finite values"
