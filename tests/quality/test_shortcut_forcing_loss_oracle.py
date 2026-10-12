"""shortcut_forcing_loss's reference, checked against the Dreamer 4 equations.

Dreamer 4 released no code, so the contract is the paper (Equations 4 and 6-8),
transcribed with scalar loops in
`torch_judge/harness/embodied_frontier/vendored/dreamer4_shortcut_paper.py`.
The bootstrap target is cross-checked against Frans et al.'s shortcut-models code
(MIT), which defines it in velocity space: wrapping a velocity network as an
x-prediction network must give the same target, and the scaled bootstrap term must
equal the shortcut-models squared error times (1 - tau)^2. Upstream's [-4, 4]
clipping, which the exercise drops, is pinned at the bottom.
"""

from __future__ import annotations

import math

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.dreamer4_shortcut_paper import (
    frame_losses,
    ramp_weight,
    shortcut_models_bootstrap_target,
    step_level_support,
)
from torch_judge.tasks import get_task

TASK_ID = "shortcut_forcing_loss"
ATOL = 1e-10


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]], namespace["sample_step_and_level"]


def x_model(D: int, seed: int):
    """A deterministic x-prediction network that mixes frames causally."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, D, generator=g, dtype=torch.float64) / math.sqrt(D)
    a = torch.randn(D, generator=g, dtype=torch.float64)
    b = torch.randn(D, generator=g, dtype=torch.float64)

    def f(z, tau, d):
        h = torch.tanh(z @ W + tau[..., None] * a + 2.0 * d[..., None] * b)
        steps = torch.arange(1, z.shape[1] + 1, dtype=z.dtype).reshape(1, -1, 1)
        return 1.5 * h - 0.7 * torch.cumsum(h, dim=1) / steps
    return f


def v_model(D: int, seed: int):
    """A velocity network, as shortcut-models trains."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, D, generator=g, dtype=torch.float64) / math.sqrt(D)
    a = torch.randn(D, generator=g, dtype=torch.float64)

    def v(x, t, dt):
        return torch.sin(x @ W + t[..., None] * a) * (1 + dt[..., None])
    return v


def on_grid(B: int, T: int, k_max: int, seed: int, flow_frac: float = 0.3):
    g = torch.Generator().manual_seed(seed)
    n = k_max.bit_length()
    steps = 2 ** torch.randint(0, n, (B, T), generator=g)
    steps[torch.rand(B, T, generator=g) < flow_frac] = k_max
    d = 1.0 / steps.double()
    tau = torch.floor(torch.rand(B, T, generator=g, dtype=torch.float64) * steps) * d
    z1 = torch.randn(B, T, 3, generator=g, dtype=torch.float64)
    z0 = torch.randn(B, T, 3, generator=g, dtype=torch.float64)
    return z1, z0, tau, d


@pytest.mark.parametrize(
    "seed,B,T,k_max",
    [(0, 2, 5, 8), (1, 3, 4, 4), (2, 1, 8, 16), (3, 2, 6, 2), (4, 4, 3, 64), (5, 2, 7, 8), (6, 1, 4, 1)],
)
def test_reference_matches_paper_equations(seed, B, T, k_max):
    loss_fn, _ = reference()
    z1, z0, tau, d = on_grid(B, T, k_max, seed)
    model = x_model(3, seed)
    got = loss_fn(model, z1, z0, tau, d, 1.0 / k_max)
    want = torch.tensor(frame_losses(model, z1, z0, tau, d, 1.0 / k_max), dtype=torch.float64)
    assert torch.allclose(got, want, atol=ATOL, rtol=1e-9), (got - want).abs().max()


def test_reference_at_grid_boundaries():
    loss_fn, _ = reference()
    model = x_model(3, 9)
    g = torch.Generator().manual_seed(9)
    z1 = torch.randn(1, 6, 3, generator=g, dtype=torch.float64)
    z0 = torch.randn(1, 6, 3, generator=g, dtype=torch.float64)
    d = torch.tensor([[1.0, 0.5, 0.5, 0.125, 0.125, 0.25]], dtype=torch.float64)
    tau = torch.tensor([[0.0, 0.5, 0.0, 0.875, 0.0, 0.75]], dtype=torch.float64)
    got = loss_fn(model, z1, z0, tau, d, 0.125)
    want = torch.tensor(frame_losses(model, z1, z0, tau, d, 0.125), dtype=torch.float64)
    assert torch.allclose(got, want, atol=ATOL), (got, want)


def test_ramp_weight_multiplies_both_branches():
    """The paper never writes w(tau) * L; the exercise (and the oracle) weight both cases of Equation 7."""
    loss_fn, _ = reference()
    z1, z0, tau, d = on_grid(3, 8, 8, 21, flow_frac=0.5)
    model = x_model(3, 21)
    weighted = loss_fn(model, z1, z0, tau, d, 0.125)
    flow = d == 0.125
    assert flow.any() and (~flow).any()
    oracle = torch.tensor(frame_losses(model, z1, z0, tau, d, 0.125), dtype=torch.float64)
    w = torch.tensor([[ramp_weight(float(s)) for s in row] for row in tau], dtype=torch.float64)
    unweighted = oracle / w
    assert torch.allclose(weighted[flow], (w * unweighted)[flow], atol=ATOL)
    assert torch.allclose(weighted[~flow], (w * unweighted)[~flow], atol=ATOL)
    assert ramp_weight(0.0) == pytest.approx(0.1) and ramp_weight(1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bootstrap_target_matches_shortcut_models(seed):
    """x-prediction f(z) = z + (1 - tau) v(z) turns Dreamer 4's b', b'' back into shortcut-models' v_b1, v_b2."""
    loss_fn, _ = reference()
    v = v_model(3, seed)
    x = lambda z, tau, d: z + (1 - tau[..., None]) * v(z, tau, d)
    k_max = 16
    z1, z0, tau, d = on_grid(2, 6, k_max, seed + 50, flow_frac=0.0)
    d = torch.where(d == 1.0 / k_max, torch.full_like(d, 2.0 / k_max), d)  # every frame bootstraps
    tau = torch.floor(tau / d) * d
    t = tau[..., None]
    z_tilde = (1 - t) * z0 + t * z1
    target = shortcut_models_bootstrap_target(v, z_tilde, tau, d, clip=None)
    v_hat = v(z_tilde, tau, d)
    want = (0.9 * tau + 0.1) * (1 - tau) ** 2 * ((v_hat - target) ** 2).sum(-1)
    got = loss_fn(x, z1, z0, tau, d, 1.0 / k_max)
    assert torch.allclose(got, want, atol=ATOL), (got - want).abs().max()


def test_upstream_clipping_is_dropped():
    """shortcut-models clips x_t2 and the target to [-4, 4]; the exercise does not."""
    loss_fn, _ = reference()
    v = lambda x, t, dt: 10.0 * torch.ones_like(x)  # a velocity far outside the clip range
    x = lambda z, tau, d: z + (1 - tau[..., None]) * v(z, tau, d)
    z1 = torch.zeros(1, 1, 2, dtype=torch.float64)
    z0 = torch.zeros(1, 1, 2, dtype=torch.float64)
    tau = torch.tensor([[0.0]], dtype=torch.float64)
    d = torch.tensor([[0.5]], dtype=torch.float64)
    clipped = shortcut_models_bootstrap_target(v, z0, tau, d, clip=4.0)
    unclipped = shortcut_models_bootstrap_target(v, z0, tau, d, clip=None)
    assert torch.allclose(clipped, torch.full_like(clipped, 4.0))
    assert torch.allclose(unclipped, torch.full_like(unclipped, 10.0))
    # The model predicts the target exactly, so the unclipped loss is zero; a clipped target would not be.
    assert torch.allclose(loss_fn(x, z1, z0, tau, d, 0.25), torch.zeros(1, 1, dtype=torch.float64))


@pytest.mark.parametrize("k_max,seed", [(1, 0), (2, 1), (8, 2), (16, 3), (64, 4)])
def test_sampler_matches_equation_4(k_max, seed):
    _, sample = reference()
    support = step_level_support(k_max)
    assert sum(support.values()) == pytest.approx(1.0)
    n = 80000
    tau, d = sample((n,), k_max, torch.Generator().manual_seed(seed))
    pairs = list(zip(d.double().tolist(), tau.double().tolist()))
    counts: dict[tuple[float, float], int] = {}
    for pair in pairs:
        assert pair in support, f"{pair} is outside Equation 4's grid"
        counts[pair] = counts.get(pair, 0) + 1
    # Marginal over d is uniform over the step counts.
    by_d: dict[float, int] = {}
    for (step, _), c in counts.items():
        by_d[step] = by_d.get(step, 0) + c
    for step, frac in ((s, c / n) for s, c in by_d.items()):
        assert abs(frac - 1 / k_max.bit_length()) < 0.015, (step, frac)
    # Each (d, tau) cell for coarse step sizes matches its probability.
    for (step, level), p in support.items():
        if step >= 1 / 8:
            assert abs(counts.get((step, level), 0) / n - p) < 0.01, (step, level)


def test_printed_grid_is_read_as_multiples_of_d():
    """Equation 4 prints {0, 1/d, ..., 1 - 1/d}; for d < 1 that leaves [0, 1], so the grid is {0, d, ..., 1 - d}."""
    support = step_level_support(8)
    for step, level in support:
        assert 0 <= level <= 1 - step
        assert abs(level / step - round(level / step)) < 1e-12
