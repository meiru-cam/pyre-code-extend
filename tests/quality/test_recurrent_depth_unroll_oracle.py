"""recurrent_depth_unroll's reference solution, checked against Huginn's upstream unroll.

Upstream iterate_forward is the contract for values and gradients. Two divergences
are pinned below: upstream applies ln_f to the returned state (the exercise returns
the raw state), and upstream always samples p >= 1 and returns long tensors (the
exercise takes p as a Python int, including 0, and returns ints).
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.reasoning.vendored.huginn_recurrent_depth import (
    huginn_initialize_state,
    huginn_iterate_forward,
    huginn_randomized_iteration_sampler,
    huginn_split_total,
)
from torch_judge.tasks import get_task

TASK_ID = "recurrent_depth_unroll"
ATOL = 1e-10


def reference(name: str | None = None):
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[name or task["function_name"]]


def build(seed: int, h: int = 4, shape: tuple[int, ...] = (2, 3)):
    """A prelude, an adapter and a two-block core in float64, plus prelude input and s0."""
    torch.manual_seed(seed)
    prelude = torch.nn.Linear(5, h).double()
    adapter = torch.nn.Linear(2 * h, h, bias=False).double()
    blocks = [torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.Tanh()).double() for _ in range(2)]
    x = torch.randn(*shape, 5, dtype=torch.float64)
    s0 = huginn_initialize_state(torch.zeros(*shape, h, dtype=torch.float64), init_std=0.5)
    return prelude, adapter, blocks, x, s0


def params(prelude, adapter, blocks):
    return list(prelude.parameters()) + list(adapter.parameters()) + [p for b in blocks for p in b.parameters()]


def core_of(blocks):
    def core(x):
        for block in blocks:
            x = block(x)
        return x

    return core


@pytest.mark.parametrize("seed,n,k", [(0, 3, 2), (1, 0, 4), (2, 6, 1), (3, 2, 0), (4, 5, 5), (5, 1, 3), (6, 0, 0)])
def test_matches_upstream_values_and_gradients(seed, n, k):
    prelude, adapter, blocks, x, s0 = build(seed)
    ps = params(prelude, adapter, blocks)
    g = torch.randn(*s0.shape, dtype=torch.float64)

    got = reference()(prelude(x), s0, adapter, core_of(blocks), n, k)
    want, n_up, k_up, _ = huginn_iterate_forward(prelude(x), s0, adapter, blocks, lambda t: t, (n, k))
    assert (n_up, k_up) == (n, k)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.allclose(got, want, atol=ATOL, rtol=0)
    assert got.requires_grad == want.requires_grad

    if want.requires_grad:
        got_grads = torch.autograd.grad((got * g).sum(), ps, allow_unused=True)
        want_grads = torch.autograd.grad((want * g).sum(), ps, allow_unused=True)
        for a, b in zip(got_grads, want_grads):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.allclose(a, b, atol=ATOL, rtol=0)


@pytest.mark.parametrize("n", [0, 3])
def test_matches_upstream_gradient_into_input_states(n):
    """Upstream clones input_states, so the gradient reaches them only when no no-grad step runs."""
    prelude, adapter, blocks, x, s0 = build(7)
    s0 = s0.clone().requires_grad_(True)
    got = reference()(prelude(x), s0, adapter, core_of(blocks), n, 2)
    want, *_ = huginn_iterate_forward(prelude(x), s0, adapter, blocks, lambda t: t, (n, 2))
    (ga,) = torch.autograd.grad(got.sum(), [s0], allow_unused=True)
    (gb,) = torch.autograd.grad(want.sum(), [s0], allow_unused=True)
    if n == 0:
        assert torch.allclose(ga, gb, atol=ATOL, rtol=0)
    else:
        assert ga is None and gb is None


def test_divergence_upstream_applies_ln_f_to_the_returned_state():
    prelude, adapter, blocks, x, s0 = build(8)
    ln_f = torch.nn.LayerNorm(4).double()
    got = reference()(prelude(x), s0, adapter, core_of(blocks), 2, 3)
    want, *_ = huginn_iterate_forward(prelude(x), s0, adapter, blocks, ln_f, (2, 3))
    assert torch.allclose(ln_f(got), want, atol=ATOL, rtol=0)
    assert not torch.allclose(got, want, atol=1e-3)


def test_upstream_eval_mode_is_k_grad_zero():
    """Upstream eval runs mean_recurrence no-grad steps; the exercise expresses it as k_grad = 0."""
    prelude, adapter, blocks, x, s0 = build(9)
    n_up, k_up = huginn_randomized_iteration_sampler(6, 2, training=False)
    assert (int(n_up), int(k_up)) == (6, 0)
    want, *_ = huginn_iterate_forward(prelude(x), s0, adapter, blocks, lambda t: t, (n_up, k_up))
    got = reference()(prelude(x), s0, adapter, core_of(blocks), 6, 0)
    assert torch.allclose(got, want, atol=ATOL, rtol=0)
    assert not got.requires_grad and not want.requires_grad


@pytest.mark.parametrize("s", [0, 1, 4, 8, 12])
def test_split_matches_upstream_arithmetic(s):
    split = reference("split_iterations")
    for p in range(0, 60):
        n_up, k_up = huginn_split_total(torch.tensor([float(p)]), s)
        assert split(p, s) == (int(n_up), int(k_up)), (p, s)


@pytest.mark.parametrize("seed", range(6))
def test_split_reproduces_seeded_upstream_draws(seed):
    """The upstream split always sums to the sampled p, so split(n + k, s) must give back (n, k)."""
    split = reference("split_iterations")
    torch.manual_seed(seed)
    for mean_recurrence, s in ((32, 8), (4, 8), (10, 1)):
        for _ in range(40):
            n_up, k_up = huginn_randomized_iteration_sampler(mean_recurrence, s, training=True)
            n_up, k_up = int(n_up), int(k_up)
            assert n_up + k_up >= 1
            assert split(n_up + k_up, s) == (n_up, k_up)


def test_divergence_split_accepts_p_zero_and_returns_ints():
    """Upstream samples p >= 1 and returns long tensors; the exercise also maps p = 0 to (0, 0)."""
    split = reference("split_iterations")
    assert split(0, 8) == (0, 0)
    n, k = split(13, 8)
    assert type(n) is int and type(k) is int
    n_up, k_up = huginn_split_total(torch.tensor([0.0]), 8)
    assert (int(n_up), int(k_up)) == (0, 0)
