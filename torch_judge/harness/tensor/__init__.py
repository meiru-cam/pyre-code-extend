"""Public tensor-harness API used by authored evaluator cases."""

from torch_judge.harness import HarnessFailure
from torch_judge.harness.tensor.assertions import assert_finite, assert_tensor_close
from torch_judge.harness.tensor.differential import DifferentialCase, assert_differential
from torch_judge.harness.tensor.gradients import assert_gradients_reach

__all__ = [
    "DifferentialCase",
    "HarnessFailure",
    "assert_differential",
    "assert_finite",
    "assert_gradients_reach",
    "assert_tensor_close",
]
