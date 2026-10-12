"""act_temporal_ensemble's reference solution, checked against its upstream counterparts.

LeRobot's online ACTTemporalEnsembler is the contract. The original ACT offline loop
agrees with it whenever no predicted action contains a zero, and the divergence on
exact zeros (the upstream `!= 0` populated check) is pinned below.
"""

from __future__ import annotations

import math

import pytest
import torch

from torch_judge.harness.embodied.vendored.lerobot_act_temporal_ensembler import (
    ACTTemporalEnsembler,
    act_original_temporal_agg,
)
from torch_judge.tasks import get_task

TASK_ID = "act_temporal_ensemble"
ATOL = 1e-5


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def chunks(seed: int, steps: int, batch: int, chunk_size: int, dim: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, chunk_size, dim, generator=generator) for _ in range(steps)]


def run(ensembler, sequence):
    return [ensembler.update(c) for c in sequence]


@pytest.mark.parametrize(
    "seed,coeff,chunk_size,batch,dim,steps",
    [
        (0, 0.01, 4, 2, 3, 15),
        (1, 0.5, 5, 3, 2, 18),
        (2, -0.3, 3, 1, 4, 12),
        (3, 0.0, 6, 2, 2, 20),
        (4, 1.5, 2, 4, 1, 10),
        (5, 0.01, 10, 1, 14, 30),
        (6, -1.0, 7, 2, 3, 25),
    ],
)
def test_matches_lerobot_online_ensembler(seed, coeff, chunk_size, batch, dim, steps):
    sequence = chunks(seed, steps, batch, chunk_size, dim)
    got = run(reference()(coeff, chunk_size), sequence)
    want = run(ACTTemporalEnsembler(coeff, chunk_size), [c.clone() for c in sequence])
    for t, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape == (batch, dim)
        assert torch.allclose(g, w, atol=ATOL, rtol=0), (t, g, w)


def test_matches_lerobot_with_chunk_size_one():
    sequence = chunks(8, 6, 2, 1, 3)
    got = run(reference()(0.2, 1), sequence)
    want = run(ACTTemporalEnsembler(0.2, 1), [c.clone() for c in sequence])
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=ATOL, rtol=0)


def test_matches_lerobot_across_reset():
    first, second = chunks(9, 7, 2, 4, 2), chunks(10, 9, 2, 4, 2)
    ours, theirs = reference()(0.05, 4), ACTTemporalEnsembler(0.05, 4)
    run(ours, first)
    run(theirs, [c.clone() for c in first])
    ours.reset()
    theirs.reset()
    for g, w in zip(run(ours, second), run(theirs, [c.clone() for c in second])):
        assert torch.allclose(g, w, atol=ATOL, rtol=0)


def test_matches_lerobot_on_exact_zero_actions():
    sequence = chunks(11, 12, 2, 3, 2)
    for c in sequence[::2]:
        c[:, 1] = 0.0
    got = run(reference()(0.3, 3), sequence)
    want = run(ACTTemporalEnsembler(0.3, 3), [c.clone() for c in sequence])
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=ATOL, rtol=0)


@pytest.mark.parametrize("seed,k,chunk_size,dim,steps", [(0, 0.01, 5, 3, 16), (1, 0.4, 3, 2, 10), (2, -0.2, 4, 4, 13), (3, 0.0, 6, 1, 18), (4, 1.0, 2, 2, 8)])
def test_matches_original_act_offline_loop_without_zeros(seed, k, chunk_size, dim, steps):
    sequence = [c.abs() + 0.1 for c in chunks(seed, steps, 1, chunk_size, dim)]  # no zero entries
    got = run(reference()(k, chunk_size), sequence)
    want = act_original_temporal_agg([c.clone() for c in sequence], chunk_size, k=k)
    for t, (g, w) in enumerate(zip(got, want)):
        assert torch.allclose(g.double(), w.double(), atol=ATOL, rtol=0), (t, g, w)


def test_pins_divergence_from_original_act_on_zero_actions():
    """The original drops any prediction with a zero entry; the exercise and LeRobot keep it."""
    sequence = [
        torch.tensor([[[1.0, 1.0], [0.0, 2.0], [5.0, 5.0]]]),
        torch.tensor([[[4.0, 4.0], [6.0, 6.0], [3.0, 3.0]]]),
    ]
    ours = run(reference()(0.0, 3), sequence)
    lerobot = run(ACTTemporalEnsembler(0.0, 3), [c.clone() for c in sequence])
    original = act_original_temporal_agg([c.clone() for c in sequence], 3, k=0.0)
    # Step 1 is predicted by [0, 2] (chunk 0) and [4, 4] (chunk 1).
    assert torch.allclose(ours[1], torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(lerobot[1], ours[1])
    # The original treats [0, 2] as unpopulated and executes only the newer prediction.
    assert torch.allclose(original[1].float(), torch.tensor([[4.0, 4.0]]))
    assert not torch.allclose(original[1].float(), ours[1])


def test_weights_follow_the_paper_convention():
    """w_0 belongs to the oldest prediction: positive coeff pulls the output toward it."""
    m = math.log(2.0)
    sequence = [torch.tensor([[[0.0], [0.0]]]), torch.tensor([[[3.0], [0.0]]])]
    ours = run(reference()(m, 2), sequence)
    lerobot = run(ACTTemporalEnsembler(m, 2), [c.clone() for c in sequence])
    assert torch.allclose(ours[1], torch.tensor([[1.0]]))  # (1 * 0 + 0.5 * 3) / 1.5
    assert torch.allclose(lerobot[1], ours[1])
