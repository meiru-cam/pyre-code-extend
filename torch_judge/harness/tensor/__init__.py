"""Public tensor-harness API used by authored evaluator cases."""

from torch_judge.harness import HarnessFailure
from torch_judge.harness.tensor.assertions import assert_finite, assert_tensor_close
from torch_judge.harness.tensor.differential import DifferentialCase, assert_differential
from torch_judge.harness.tensor.gradients import assert_gradients_reach
from torch_judge.harness.tensor.moe import (
    CountingExpert,
    assert_expert_gradient_partition,
    dense_capacity_dispatch_oracle,
    permute_expert_routes,
    seeded_domain_batch,
)

__all__ = [
    "DifferentialCase",
    "HarnessFailure",
    "CountingExpert",
    "assert_expert_gradient_partition",
    "assert_differential",
    "assert_finite",
    "assert_gradients_reach",
    "assert_tensor_close",
    "dense_capacity_dispatch_oracle",
    "permute_expert_routes",
    "seeded_domain_batch",
]
