"""The rtc_prefix_guidance reference, checked against LeRobot's RTCProcessor.

The mutation gate proves the evaluator rejects wrong implementations; these tests
prove the contract itself agrees with upstream. The reference is compared against
`torch_judge/harness/embodied/vendored/lerobot_rtc.py`, transcribed from LeRobot at
75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee. Where the exercise deliberately differs from
LeRobot or from the original Kinetix code, the difference is pinned below.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from torch_judge.harness.embodied.vendored.lerobot_rtc import lerobot_denoise_step, lerobot_get_prefix_weights
from torch_judge.tasks import get_task

TASK_ID = "rtc_prefix_guidance"
SCHEDULES = ("zeros", "ones", "linear", "exp")


def reference():
    task = get_task(TASK_ID)
    assert task is not None
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace["prefix_weights"], namespace["guided_velocity"]


def kinetix_prefix_weights(start, end, total, schedule):
    """Kinetix src/model.py::get_prefix_weights at 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b (MIT), jnp -> torch."""
    start = min(start, end)
    idx = torch.arange(total)
    if schedule == "ones":
        w = torch.ones(total)
    elif schedule == "zeros":
        w = (idx < start).float()
    else:
        w = torch.clip((start - 1 - idx) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    return torch.where(idx >= end, 0.0, w)


@pytest.mark.parametrize("seed", range(6))
def test_prefix_weights_match_lerobot(seed):
    prefix_weights, _ = reference()
    rng = random.Random(seed)
    for _ in range(80):
        total = rng.randint(1, 20)
        start, end = rng.randint(0, total + 3), rng.randint(0, total + 3)
        schedule = rng.choice(SCHEDULES)
        mine = prefix_weights(start, end, total, schedule)
        theirs = lerobot_get_prefix_weights(start, end, total, schedule)
        assert mine.shape == theirs.shape == (total,)
        assert torch.allclose(mine, theirs, atol=1e-7), (start, end, total, schedule)


@pytest.mark.parametrize(
    ("start", "end", "total"),
    [(0, 0, 5), (0, 5, 5), (3, 3, 5), (4, 2, 6), (2, 6, 10), (12, 15, 10), (5, 8, 6), (0, 1, 1)],
)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_prefix_weight_edges_match_lerobot(start, end, total, schedule):
    prefix_weights, _ = reference()
    assert torch.allclose(
        prefix_weights(start, end, total, schedule), lerobot_get_prefix_weights(start, end, total, schedule), atol=1e-7
    )


def test_prefix_weights_agree_with_kinetix_when_end_fits():
    prefix_weights, _ = reference()
    for total in range(1, 12):
        for end in range(total + 1):
            for start in range(total + 2):
                for schedule in SCHEDULES:
                    assert torch.allclose(
                        prefix_weights(start, end, total, schedule),
                        kinetix_prefix_weights(start, end, total, schedule),
                        atol=1e-6,
                    ), (start, end, total, schedule)


def test_divergence_end_past_total_follows_lerobot_not_kinetix():
    """Kinetix keeps the ramp slope from the unclamped end; LeRobot and the exercise clamp end first."""
    prefix_weights, _ = reference()
    mine = prefix_weights(2, 12, 10, "linear")
    assert torch.allclose(mine, lerobot_get_prefix_weights(2, 12, 10, "linear"))
    assert torch.allclose(mine[2:], torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]) / 9)
    kinetix = kinetix_prefix_weights(2, 12, 10, "linear")
    assert torch.allclose(kinetix[2:], torch.tensor([10, 9, 8, 7, 6, 5, 4, 3]) / 11)


def _case(seed):
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    batched = rng.random() < 0.5
    B, T, A = rng.randint(1, 3), rng.randint(2, 10), rng.randint(1, 4)
    P = rng.randint(1, T)
    shape = (B, T, A) if batched else (T, A)
    prev_shape = (B, P, A) if batched else (P, A)
    x_t = torch.randn(shape, generator=gen)
    prev = torch.randn(prev_shape, generator=gen)
    base = torch.randn(shape, generator=gen)
    return x_t, prev, base, rng


@pytest.mark.parametrize("seed", range(8))
def test_guided_velocity_matches_lerobot_for_state_independent_velocity(seed):
    """With a velocity that does not depend on x_t, both Jacobians are the identity and the two agree."""
    _, guided_velocity = reference()
    x_t, prev, base, rng = _case(seed)
    v_fn = lambda x: base * 1.5 - 0.25  # noqa: E731
    for time in (0.0, 0.1, 0.37, 0.5, 0.8, 0.999, 1.0):
        delay, horizon = rng.randint(0, x_t.shape[-2]), rng.randint(0, x_t.shape[-2] + 2)
        schedule, max_w = rng.choice(SCHEDULES), rng.choice([0.5, 5.0, 10.0])
        mine = guided_velocity(x_t, v_fn, prev, time, delay, horizon, max_w, schedule)
        theirs = lerobot_denoise_step(x_t, prev, delay, time, v_fn, horizon, max_w, schedule)
        assert mine.shape == theirs.shape == x_t.shape
        assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-5), (seed, time, delay, horizon, schedule)


def test_guided_velocity_without_previous_chunk_matches_lerobot():
    _, guided_velocity = reference()
    x_t = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(0))
    v_fn = torch.tanh
    assert torch.equal(
        guided_velocity(x_t, v_fn, None, 0.4, 2, 4, 10.0, "linear"),
        lerobot_denoise_step(x_t, None, 2, 0.4, v_fn, 4, 10.0, "linear"),
    )


def test_divergence_lerobot_drops_the_velocity_jacobian():
    """LeRobot runs the denoiser before x_t requires grad, so its correction is the weighted error.

    The exercise follows the paper and Kinetix (jax.vjp over the whole denoiser): the
    correction is J^T err with J = I - time * dv/dx.
    """
    _, guided_velocity = reference()
    gen = torch.Generator().manual_seed(4)
    M = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    x_t = torch.randn(6, 3, generator=gen, dtype=torch.float64)
    prev = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    v_fn = lambda x: x @ M  # noqa: E731
    time, max_w = 0.4, 10.0
    tau = 1 - time
    gain = min((1 - tau) / tau * ((1 - tau) ** 2 + tau**2) / (1 - tau) ** 2, max_w)
    w = lerobot_get_prefix_weights(2, 4, 6, "linear").double()[:, None]
    target = torch.cat([prev, torch.zeros(2, 3, dtype=torch.float64)])
    err = (target - (x_t - time * v_fn(x_t))) * w

    theirs = lerobot_denoise_step(x_t.float(), prev.float(), 2, time, lambda x: x @ M.float(), 5, max_w, "linear")
    assert torch.allclose(theirs.double(), v_fn(x_t) - gain * err, atol=1e-4)

    mine = guided_velocity(x_t, v_fn, prev, time, 2, 5, max_w, "linear")
    assert torch.allclose(mine, v_fn(x_t) - gain * (err - time * err @ M.T), atol=1e-6)
    assert not torch.allclose(mine, theirs.double(), atol=1e-3)


def test_divergence_narrow_action_dim_rejected_not_padded():
    _, guided_velocity = reference()
    x_t = torch.zeros(5, 3)
    prev = torch.ones(5, 2)
    v_fn = lambda x: torch.zeros_like(x)  # noqa: E731
    lerobot_denoise_step(x_t, prev, 1, 0.5, v_fn, 3, 10.0, "linear")  # upstream zero-pads the missing dim
    with pytest.raises(ValueError):
        guided_velocity(x_t, v_fn, prev, 0.5, 1, 3, 10.0, "linear")


def test_divergence_unknown_schedule():
    prefix_weights, _ = reference()
    with pytest.raises(ValueError):
        prefix_weights(1, 3, 5, "LINEAR")
