"""diffusion_forcing_schedule's reference solution, checked against its upstream counterparts.

df_base.py and diffusion.py from the official Diffusion Forcing repo are the contract.
open-oasis generate.py is an independent cross-check of the context stabilization
level and the schedule-to-real-step mapping. The trapezoid kind is left out of the
exercise; its upstream quirk is pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.diffusion_forcing_df_base import (
    df_generate_noise_levels,
    df_generate_scheduling_matrix,
    df_generate_trapezoid_scheduling_matrix,
    df_model_levels,
    df_step_levels,
    oasis_step_levels,
)
from torch_judge.tasks import get_task

TASK_ID = "diffusion_forcing_schedule"


def reference() -> dict:
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace


@pytest.mark.parametrize("seed", range(6))
def test_training_levels_match_df_generate_noise_levels(seed):
    ns = reference()
    gm = torch.Generator().manual_seed(500 + seed)
    T, B = int(torch.randint(1, 9, (1,), generator=gm)), int(torch.randint(1, 6, (1,), generator=gm))
    timesteps = [1, 2, 10, 1000, 7, 50][seed]
    mask = torch.rand(T, B, generator=gm) > 0.3
    for m in (None, mask):
        torch.manual_seed(seed)  # upstream draws from the global RNG
        want = df_generate_noise_levels(T, B, timesteps, None if m is None else m.float())
        got = ns["train_noise_levels"](T, B, timesteps, torch.Generator().manual_seed(seed), m)
        assert got.dtype == want.dtype == torch.long
        assert torch.equal(got, want), (seed, got, want)


@pytest.mark.parametrize("kind", ["pyramid", "autoregressive", "full_sequence"])
@pytest.mark.parametrize("seed", range(5))
def test_scheduling_matrix_matches_df_base(kind, seed):
    ns = reference()
    gm = torch.Generator().manual_seed(seed)
    for _ in range(6):
        horizon = int(torch.randint(1, 9, (1,), generator=gm))
        S = int(torch.randint(1, 12, (1,), generator=gm))
        u = [0, 0.25, 0.5, 1, 1.5, 2.0, 3, 0.75][int(torch.randint(0, 8, (1,), generator=gm))]
        want = torch.from_numpy(df_generate_scheduling_matrix(kind, horizon, S, u))
        got = ns["scheduling_matrix"](horizon, S, kind, u)
        assert got.dtype == torch.long
        assert torch.equal(got, want), (kind, horizon, S, u, got, want)


@pytest.mark.parametrize("horizon,S,u", [(1, 1, 1), (3, 5, 1), (4, 3, 0.5), (2, 50, 1), (8, 100, 1)])
def test_scheduling_matrix_edge_cases(horizon, S, u):
    ns = reference()
    for kind in ("pyramid", "autoregressive", "full_sequence"):
        want = torch.from_numpy(df_generate_scheduling_matrix(kind, horizon, S, u))
        assert torch.equal(ns["scheduling_matrix"](horizon, S, kind, u), want), (kind, horizon, S, u)


@pytest.mark.parametrize("seed", range(5))
def test_step_and_model_levels_match_df_sampling_loop(seed):
    ns = reference()
    gm = torch.Generator().manual_seed(seed)
    kind = ["pyramid", "autoregressive", "full_sequence"][seed % 3]
    horizon = int(torch.randint(1, 6, (1,), generator=gm))
    S = [2, 5, 10, 50, 4][seed]
    timesteps = S * [1, 2, 100, 20, 25][seed]
    stab = [1, 10, 15, 3, timesteps][seed]
    n_ctx = int(torch.randint(0, 4, (1,), generator=gm))
    np_matrix = df_generate_scheduling_matrix(kind, horizon, S, 1.0)
    matrix = ns["scheduling_matrix"](horizon, S, kind, 1.0)
    for m in range(np_matrix.shape[0] - 1):
        want_from, want_to = df_step_levels(n_ctx, np_matrix, m)
        got_from, got_to, got_update = ns["step_noise_levels"](matrix, m, n_ctx)
        assert torch.equal(got_from, want_from) and torch.equal(got_to, want_to), (m, got_from, want_from)
        clipped, _, _, want_update = df_model_levels(want_from, want_to, timesteps, S, stab)
        assert torch.equal(got_update, want_update), (m, got_update, want_update)
        got_model = ns["model_noise_levels"](got_from, timesteps, S, stab)
        assert torch.equal(got_model, clipped), (m, got_model, clipped)


@pytest.mark.parametrize("ddim_steps", [1, 4, 10, 20, 100])
def test_model_levels_match_open_oasis(ddim_steps):
    """Oasis generates one frame per chunk: a horizon-1 full-sequence matrix after i context frames."""
    ns = reference()
    matrix = ns["scheduling_matrix"](1, ddim_steps, "full_sequence")
    for i in (1, 3):
        for m in range(matrix.shape[0] - 1):
            noise_idx = ddim_steps - m
            from_levels, _, _ = ns["step_noise_levels"](matrix, m, i)
            got = ns["model_noise_levels"](from_levels, 1000, ddim_steps, 15)
            t, _ = oasis_step_levels(i, noise_idx, ddim_steps, max_noise_level=1000, stabilization_level=15)
            assert torch.equal(got, t[0]), (ddim_steps, i, m, got, t)


def test_upstream_trapezoid_leaves_the_middle_column_clean():
    """Pinned divergence: the exercise has no trapezoid kind because of this upstream quirk."""
    even = df_generate_trapezoid_scheduling_matrix(4, 1.0, 3)
    assert (even[:, 2] == 0).all() and even[0].tolist() == [3, 3, 0, 3]
    odd = df_generate_trapezoid_scheduling_matrix(5, 1.0, 3)
    assert (odd[0] == 3).all()
    ns = reference()
    with pytest.raises(ValueError):
        ns["scheduling_matrix"](4, 3, "trapezoid", 1.0)


def test_upstream_returns_none_for_unknown_kind_and_exercise_raises():
    assert df_generate_scheduling_matrix("zigzag", 3, 3, 1.0) is None
    with pytest.raises(ValueError):
        reference()["scheduling_matrix"](3, 3, "zigzag", 1.0)


def test_vendored_module_is_independent_of_tasks():
    import torch_judge.harness.embodied_frontier.vendored.diffusion_forcing_df_base as mod
    source = open(mod.__file__, encoding="utf-8").read()
    assert "torch_judge.tasks" not in source.split('"""', 2)[2]
    assert isinstance(df_generate_scheduling_matrix("pyramid", 2, 2, 1.0), np.ndarray)
