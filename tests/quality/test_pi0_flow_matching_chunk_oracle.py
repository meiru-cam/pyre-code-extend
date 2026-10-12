"""The pi0_flow_matching_chunk reference, checked against LeRobot (and openpi's schedule).

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied/vendored/lerobot_flow_matching.py`.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied.vendored.lerobot_flow_matching import (
    FlowConvention,
    lerobot_euler_integrate,
    lerobot_make_flow_matching_inputs,
    openpi_euler_times,
)
from torch_judge.tasks import get_task

TASK_ID = "pi0_flow_matching_chunk"
SEEDS = (0, 1, 2, 3, 4, 5, 6)
CONVENTIONS = ("noise_at_one", "noise_at_zero")


def reference():
    """Load the reference solution's class and return an instance."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]()


def chunk(seed: int, batch: int = 3, horizon: int = 6, dim: int = 4):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(batch, horizon, dim, generator=g, dtype=torch.float64)
    noise = torch.randn(batch, horizon, dim, generator=g, dtype=torch.float64)
    time = torch.rand(batch, generator=g, dtype=torch.float64) * 0.999 + 0.001
    mask = torch.rand(batch, horizon, generator=g) < 0.4
    weights = torch.randn(horizon, horizon, generator=g, dtype=torch.float64) / horizon
    return actions, noise, time, mask, weights


def mixing_field(weights):
    """A nonlinear, time-dependent field that couples horizon positions."""

    def v(x, t):
        tt = t.to(x.dtype).reshape(t.shape[0], -1, 1)
        return torch.einsum("hk,bkd->bhd", weights, torch.tanh(x)) * (1 + tt) - 0.5 * x * tt

    return v


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("with_prefix", (False, True))
def test_make_inputs_matches_lerobot(seed, convention, with_prefix):
    actions, noise, time, mask, _ = chunk(seed)
    prefix_mask = mask if with_prefix else None
    got = reference().make_inputs(actions, noise, time, convention, prefix_mask)
    want = lerobot_make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention(convention), prefix_mask=prefix_mask
    )
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.dtype == w.dtype
        torch.testing.assert_close(g, w, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("num_steps", (1, 4, 10))
@pytest.mark.parametrize("with_prefix", (False, True))
def test_euler_sample_matches_lerobot(seed, convention, num_steps, with_prefix):
    actions, noise, _, mask, weights = chunk(seed)
    field = mixing_field(weights)
    if with_prefix:
        got = reference().euler_sample(field, noise, num_steps, convention, actions, mask)
        # Upstream takes a mask shaped like the sample; the exercise's (B, H) mask covers every dim.
        want = lerobot_euler_integrate(
            field, noise, num_steps, convention=FlowConvention(convention),
            hard_prefix=actions, hard_prefix_mask=mask[..., None].expand_as(noise),
        )
    else:
        got = reference().euler_sample(field, noise, num_steps, convention)
        want = lerobot_euler_integrate(field, noise, num_steps, convention=FlowConvention(convention))
    # Upstream evaluates the field at float32 times; the exercise uses noise's dtype.
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_time_arguments_match_lerobot(convention):
    _, noise, _, mask, _ = chunk(11)
    for prefix in (None, noise.clone()):
        got_times, want_times = [], []

        def recorder(log):
            def v(x, t):
                log.append(t.clone())
                return torch.zeros_like(x)
            return v

        kwargs_ref = {} if prefix is None else {"hard_prefix": prefix, "hard_prefix_mask": mask}
        reference().euler_sample(recorder(got_times), noise, 5, convention, **kwargs_ref)
        kwargs_up = {} if prefix is None else {"hard_prefix": prefix, "hard_prefix_mask": mask[..., None].expand_as(noise)}
        lerobot_euler_integrate(recorder(want_times), noise, 5, convention=FlowConvention(convention), **kwargs_up)
        assert len(got_times) == len(want_times) == 5
        for g, w in zip(got_times, want_times):
            assert g.shape == w.shape
            torch.testing.assert_close(g, w.to(g.dtype), rtol=0, atol=1e-7)


@pytest.mark.parametrize("num_steps", (1, 2, 3, 7, 10, 33, 50))
def test_openpi_and_lerobot_schedules_agree(num_steps):
    """openpi accumulates time += dt in a while_loop; LeRobot computes 1 + k * dt."""
    dt = -1.0 / num_steps
    lerobot = [1.0 + k * dt for k in range(num_steps)]
    openpi = openpi_euler_times(num_steps)
    assert len(openpi) == num_steps
    assert max(abs(a - b) for a, b in zip(openpi, lerobot)) < 1e-12


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_both_conventions_recover_the_chunk_with_the_exact_field(seed):
    actions, noise, _, _, _ = chunk(seed)
    for convention in CONVENTIONS:
        const = (noise - actions) if convention == "noise_at_one" else (actions - noise)
        for impl in (
            lambda f: reference().euler_sample(f, noise, 10, convention),
            lambda f: lerobot_euler_integrate(f, noise, 10, convention=FlowConvention(convention)),
        ):
            torch.testing.assert_close(impl(lambda x, t: const), actions, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Deliberate divergences from upstream, pinned so a change is noticed.


def test_divergence_time_dtype_follows_noise():
    """Upstream always passes float32 times; the exercise uses noise's dtype."""
    _, noise, _, _, _ = chunk(0)
    seen: dict[str, torch.dtype] = {}

    def spy(name):
        def v(x, t):
            seen[name] = t.dtype
            return torch.zeros_like(x)
        return v

    reference().euler_sample(spy("ref"), noise, 2, "noise_at_one")
    lerobot_euler_integrate(spy("up"), noise, 2, convention=FlowConvention.NOISE_AT_ONE)
    assert seen == {"ref": torch.float64, "up": torch.float32}


def test_divergence_noise_must_match_actions():
    """Upstream broadcasts noise against actions; the exercise requires the same shape."""
    actions, noise, time, _, _ = chunk(1)
    shared = noise[:1]
    lerobot_make_flow_matching_inputs(actions, shared, time, convention=FlowConvention.NOISE_AT_ONE)
    with pytest.raises(ValueError):
        reference().make_inputs(actions, shared, time, "noise_at_one")


def test_divergence_string_conventions():
    """Upstream requires the FlowConvention enum (TypeError); the exercise takes strings."""
    actions, noise, time, _, _ = chunk(2)
    with pytest.raises(TypeError):
        lerobot_make_flow_matching_inputs(actions, noise, time, convention="noise_at_one")
    reference().make_inputs(actions, noise, time, "noise_at_one")
    with pytest.raises(ValueError):
        reference().make_inputs(actions, noise, time, "forward")


def test_divergence_mask_without_prefix_raises():
    """Upstream ignores a hard_prefix_mask passed without hard_prefix; the exercise raises."""
    _, noise, _, mask, _ = chunk(3)
    zero = lambda x, t: torch.zeros_like(x)  # noqa: E731
    lerobot_euler_integrate(zero, noise, 2, hard_prefix_mask=mask[..., None].expand_as(noise))
    with pytest.raises(ValueError):
        reference().euler_sample(zero, noise, 2, "noise_at_one", hard_prefix_mask=mask)


def test_edge_all_false_and_all_true_prefix_masks_match_lerobot():
    actions, noise, time, _, weights = chunk(4)
    field = mixing_field(weights)
    for value in (False, True):
        mask = torch.full(actions.shape[:2], value)
        for convention in CONVENTIONS:
            got = reference().make_inputs(actions, noise, time, convention, mask)
            want = lerobot_make_flow_matching_inputs(
                actions, noise, time, convention=FlowConvention(convention), prefix_mask=mask
            )
            for g, w in zip(got, want):
                torch.testing.assert_close(g, w, rtol=0, atol=1e-12)
            got = reference().euler_sample(field, noise, 3, convention, actions, mask)
            want = lerobot_euler_integrate(
                field, noise, 3, convention=FlowConvention(convention),
                hard_prefix=actions, hard_prefix_mask=mask[..., None].expand_as(noise),
            )
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
