"""Contract tests for the reusable tensor evaluation harness."""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.tensor import (
    DifferentialCase,
    HarnessFailure,
    assert_differential,
    assert_finite,
    assert_gradients_reach,
    assert_tensor_close,
)


def test_assert_tensor_close_accepts_matching_tensors():
    actual = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert_tensor_close(actual, actual + 1e-7)


@pytest.mark.parametrize(
    ("actual", "expected", "match"),
    [
        (torch.ones(2), torch.ones(1, 2), "shape"),
        (torch.ones(2), torch.ones(2, dtype=torch.float64), "dtype"),
        (torch.ones(2), torch.tensor([1.0, 2.0]), "values"),
    ],
)
def test_assert_tensor_close_reports_structural_and_value_failures(actual, expected, match):
    with pytest.raises(AssertionError, match=match):
        assert_tensor_close(actual, expected)


def test_assert_finite_rejects_nan_and_infinity():
    assert_finite(torch.tensor([0.0, 1.0]), name="output")
    with pytest.raises(AssertionError, match="output.*finite"):
        assert_finite(torch.tensor([float("nan")]), name="output")
    with pytest.raises(AssertionError, match="gradient.*finite"):
        assert_finite(torch.tensor([float("inf")]), name="gradient")


def test_assert_differential_runs_seeded_cases_without_task_solution_imports():
    cases = [
        DifferentialCase("positive", args=(torch.tensor([1.0, 2.0]),)),
        DifferentialCase("negative", args=(torch.tensor([-2.0, 3.0]),)),
    ]
    assert_differential(lambda x: x.square(), torch.square, cases)

    def wrong_for_negative(x):
        return x.square() if bool((x >= 0).all()) else x.abs()

    with pytest.raises(AssertionError, match="negative"):
        assert_differential(wrong_for_negative, torch.square, cases)


def test_assert_differential_supports_nested_outputs():
    cases = [DifferentialCase("pair", args=(torch.tensor([2.0]),))]
    assert_differential(lambda x: (x, {"square": x * x}), lambda x: (x, {"square": x.square()}), cases)


def test_assert_differential_isolates_oracle_inputs_from_candidate_mutation():
    source = torch.tensor([2.0, 3.0])

    def mutating_candidate(x):
        x.zero_()
        return x

    with pytest.raises(AssertionError, match="mutation-isolation"):
        assert_differential(
            mutating_candidate,
            torch.square,
            [DifferentialCase("mutation-isolation", args=(source,))],
        )
    assert torch.equal(source, torch.tensor([2.0, 3.0]))


def test_assert_gradients_reach_requires_finite_nonzero_gradients():
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    assert_gradients_reach((x * y).sum(), {"x": x, "y": y})

    detached = torch.tensor([4.0], requires_grad=True)
    with pytest.raises(AssertionError, match="detached.*gradient"):
        assert_gradients_reach((x * 2).sum(), {"detached": detached})


def test_harness_rejects_non_tensor_oracle_as_author_error():
    with pytest.raises(HarnessFailure, match="tensor"):
        assert_tensor_close(torch.tensor([1.0]), [1.0])


def test_non_tensor_candidate_output_is_a_learner_failure():
    with pytest.raises(AssertionError, match="actual.*tensor"):
        assert_tensor_close([1.0], torch.tensor([1.0]))
