"""The latent_exit_criteria reference, checked against Huginn's step-by-step exit evaluators.

Upstream evaluates one recurrent step at a time inside generate_with_adaptive_compute.
These tests drive the transcribed evaluators and that loop's bookkeeping in
`torch_judge/harness/reasoning/vendored/huginn_exit_evaluators.py` step by step, and
compare the first exit step with the vectorized reference. Each token is its own
sequence of length 1, as during decoding. The one deliberate divergence, latent-diff
at step 1, is pinned below.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.reasoning.vendored.huginn_exit_evaluators import (
    ArgmaxStabilityExitEvaluator,
    EntropyDiffExitEvaluator,
    KLExitEvaluator,
    LatentDiffExitEvaluator,
    get_adaptive_exit_evaluator,
    huginn_exit_steps,
)
from torch_judge.tasks import get_task

TASK_ID = "latent_exit_criteria"
CRITERIA = ("kl", "entropy-diff", "latent-diff", "argmax-stability")
SEEDS = (0, 1, 2, 3, 4, 5, 6)


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def trajectories(seed: int, R: int = 16, N: int = 32, V: int = 11, h: int = 8):
    """Logits and latents that converge at token-dependent rates, as recurrent steps do."""
    g = torch.Generator().manual_seed(seed)
    decay = 0.6 ** torch.arange(R, dtype=torch.float32).view(R, 1, 1)
    scale = torch.rand(1, N, 1, generator=g) * 5
    logits = torch.randn(1, N, V, generator=g) * 2 + scale * decay * torch.randn(R, N, V, generator=g)
    latents = torch.randn(1, N, h, generator=g) + scale * decay * torch.randn(R, N, h, generator=g)
    return logits, latents


def upstream(logits, latents, criterion, threshold, initial_latents=None):
    """Run Huginn's evaluator one step at a time; tokens are (batch, seq=1) sequences."""
    R, N, V = logits.shape
    evaluator = get_adaptive_exit_evaluator(criterion, threshold, vocab_size=V)
    if initial_latents is None:
        # latent-diff only: a start state whose relative change to step 1 is exactly 2, so
        # step 1 never fires for thresholds <= 2 (the exercise has no initial state).
        initial_latents = -latents[0]
    return torch.tensor(
        huginn_exit_steps(
            evaluator,
            initial_latents.unsqueeze(1),
            [latents[r].unsqueeze(1) for r in range(R)],
            [logits[r].unsqueeze(1) for r in range(R)],
        )
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "criterion,thresholds",
    [
        ("kl", ("auto", 5e-4, 1e-2, 0.1)),
        ("entropy-diff", ("auto", 1e-4, 1e-2, 0.1)),
        ("latent-diff", ("auto", 1e-2, 0.1, 0.5)),
        ("argmax-stability", ("auto", 0, 1, 2, 3)),
    ],
)
def test_matches_huginn_step_by_step(seed, criterion, thresholds):
    logits, latents = trajectories(seed)
    for threshold in thresholds:
        got = reference()(logits, latents, criterion, threshold)
        want = upstream(logits, latents, criterion, threshold)
        assert got.dtype == torch.long
        assert torch.equal(got, want), (criterion, threshold, got.tolist(), want.tolist())


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_exit_steps_are_spread(seed):
    """The comparison above is only meaningful if tokens exit at different steps."""
    logits, latents = trajectories(seed)
    for criterion in CRITERIA:
        want = upstream(logits, latents, criterion, "auto")
        assert len(set(want.tolist())) >= 3, (criterion, want.tolist())


def test_auto_defaults_match_upstream():
    assert EntropyDiffExitEvaluator("auto").exit_threshold == 1e-3
    assert LatentDiffExitEvaluator("auto").exit_threshold == 0.03
    assert KLExitEvaluator(10, "auto").exit_threshold == 0.001
    assert ArgmaxStabilityExitEvaluator("auto").exit_threshold == 5
    logits, latents = trajectories(11)
    fn = reference()
    for criterion, default in (("kl", 1e-3), ("entropy-diff", 1e-3), ("latent-diff", 0.03), ("argmax-stability", 5)):
        assert torch.equal(fn(logits, latents, criterion, "auto"), fn(logits, latents, criterion, default))


def test_kl_uniform_start_matches_upstream():
    logits = torch.zeros(4, 3, 6)
    logits[:, 1, 0] = 9.0
    latents = torch.zeros(4, 3, 2)
    got = reference()(logits, latents, "kl", 1e-6)
    assert got.tolist() == [1, 2, 1]
    assert torch.equal(got, upstream(logits, latents, "kl", 1e-6))


@pytest.mark.parametrize("threshold", [1e-3, 98.0, 99.0, 150.0])
def test_entropy_sentinel_matches_upstream(threshold):
    logits = torch.zeros(3, 2, 4)
    logits[:, 0, 0] = 30.0  # entropy ~ 0
    latents = torch.zeros(3, 2, 2)
    assert torch.equal(reference()(logits, latents, "entropy-diff", threshold), upstream(logits, latents, "entropy-diff", threshold))


def test_ties_match_upstream():
    g = torch.Generator().manual_seed(0)
    same = torch.randn(1, 2, 5, generator=g).expand(4, 2, 5).clone()
    z = torch.ones(4, 2, 3)
    for criterion in ("kl", "entropy-diff"):
        assert reference()(same, z, criterion, 0.0).tolist() == [4, 4]
        assert torch.equal(reference()(same, z, criterion, 0.0), upstream(same, z, criterion, 0.0))
    lat = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]], [[4.0, 0.0]]])
    flat = torch.zeros(3, 1, 2)
    assert reference()(flat, lat, "latent-diff", 0.5).tolist() == [3]
    assert torch.equal(reference()(flat, lat, "latent-diff", 0.5), upstream(flat, lat, "latent-diff", 0.5))
    arg = torch.zeros(6, 1, 3)
    arg[..., 1] = 1.0
    for k in (0, 3, 5, 6):
        assert torch.equal(reference()(arg, torch.ones(6, 1, 2), "argmax-stability", k), upstream(arg, torch.ones(6, 1, 2), "argmax-stability", k))


def test_tied_logits_argmax_matches_upstream():
    logits = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]])
    latents = torch.ones(3, 1, 2)
    for k in (1, 2):
        assert torch.equal(reference()(logits, latents, "argmax-stability", k), upstream(logits, latents, "argmax-stability", k))


@pytest.mark.parametrize("criterion", CRITERIA)
def test_single_step_matches_upstream(criterion):
    logits, latents = trajectories(5, R=1)
    assert torch.equal(reference()(logits, latents, criterion, "auto"), upstream(logits, latents, criterion, "auto"))


def test_pins_latent_diff_step_one_divergence():
    """Upstream compares step 1 with the initial state; the exercise has none, so step 1 never fires."""
    _, latents = trajectories(3, R=4, N=6)
    logits = torch.zeros(4, 6, 5)
    ours = reference()(logits, latents, "latent-diff", 10.0)
    assert ours.tolist() == [2] * 6
    # From the random start that initialize_state would draw, the relative change at step 1
    # is large, so upstream also waits for step 2 under ordinary thresholds ...
    random_start = torch.randn(6, 8, generator=torch.Generator().manual_seed(1)) * 5
    assert torch.equal(upstream(logits, latents, "latent-diff", 0.03, initial_latents=random_start), reference()(logits, latents, "latent-diff", 0.03))
    # ... but if the start state already equals step 1, upstream exits at step 1 and the exercise does not.
    theirs = upstream(logits, latents, "latent-diff", 0.03, initial_latents=latents[0])
    assert theirs.tolist() == [1] * 6
    assert (reference()(logits, latents, "latent-diff", 0.03) >= 2).all()
