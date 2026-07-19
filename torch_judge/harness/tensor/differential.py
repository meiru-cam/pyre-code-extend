"""Independent candidate-versus-oracle comparisons."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from torch_judge.harness import HarnessFailure
from torch_judge.harness.tensor.assertions import assert_tensor_close


@dataclass(frozen=True)
class DifferentialCase:
    """One named invocation shared by a candidate and an independent oracle."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise HarnessFailure("differential case name must be non-empty")


def _assert_output_close(actual: Any, expected: Any, *, name: str, atol: float, rtol: float) -> None:
    if isinstance(expected, torch.Tensor):
        assert_tensor_close(actual, expected, atol=atol, rtol=rtol, name=name)
        return
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError(f"{name} output tuple structure differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_output_close(
                actual_item, expected_item, name=f"{name}[{index}]", atol=atol, rtol=rtol
            )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{name} output mapping structure differs")
        for key in expected:
            _assert_output_close(
                actual[key], expected[key], name=f"{name}.{key}", atol=atol, rtol=rtol
            )
        return
    assert actual == expected, f"{name} value differs: expected {expected!r}, got {actual!r}"


def _isolated(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        clone.requires_grad_(value.requires_grad)
        return clone
    if isinstance(value, tuple):
        return tuple(_isolated(item) for item in value)
    if isinstance(value, list):
        return [_isolated(item) for item in value]
    if isinstance(value, dict):
        return {key: _isolated(item) for key, item in value.items()}
    return copy.deepcopy(value)


def assert_differential(
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    cases: list[DifferentialCase] | tuple[DifferentialCase, ...],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Run named cases against independently supplied candidate and oracle callables."""
    if not callable(candidate) or not callable(reference):
        raise HarnessFailure("candidate and reference must be callable")
    if not cases:
        raise HarnessFailure("differential comparison requires at least one case")
    for case in cases:
        candidate_args = _isolated(case.args)
        candidate_kwargs = _isolated(dict(case.kwargs))
        reference_args = _isolated(case.args)
        reference_kwargs = _isolated(dict(case.kwargs))
        actual = candidate(*candidate_args, **candidate_kwargs)
        try:
            expected = reference(*reference_args, **reference_kwargs)
        except Exception as error:
            raise HarnessFailure(f"differential oracle failed for case {case.name!r}: {error}") from error
        _assert_output_close(actual, expected, name=case.name, atol=atol, rtol=rtol)
