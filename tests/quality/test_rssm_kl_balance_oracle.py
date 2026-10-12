"""rssm_kl_balance's reference solution, checked against its upstream counterparts.

danijar/dreamerv3 (transcribed from JAX) is the contract. NM512/dreamerv3-torch
agrees with it once the betas are passed explicitly; its default dyn scale of 0.5
and the gradient at an exact free-bits tie are pinned below.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied.vendored.dreamerv3_rssm_kl import (
    danijar_rssm_loss,
    nm512_kl_loss,
)
from torch_judge.tasks import get_task

TASK_ID = "rssm_kl_balance"
ATOL = 1e-10


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def logits(seed: int, shape: tuple[int, ...], scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    post = scale * torch.randn(*shape, generator=generator, dtype=torch.float64)
    prior = scale * torch.randn(*shape, generator=generator, dtype=torch.float64)
    return post, prior


def routed_grads(fn, post, prior, weights):
    """Gradients of weighted dyn and rep with respect to (post, prior), None mapped to zeros."""
    post = post.clone().requires_grad_(True)
    prior = prior.clone().requires_grad_(True)
    loss, dyn, rep = fn(post, prior)
    out = {}
    for name, value in (("loss", loss), ("dyn", (weights * dyn).sum()), ("rep", (weights * rep).sum())):
        grads = torch.autograd.grad(value, [post, prior], allow_unused=True, retain_graph=True)
        out[name] = tuple(torch.zeros_like(post) if g is None else g for g in grads)
    return loss.detach(), dyn.detach(), rep.detach(), out


CASES = [
    # seed, (B, T, S, C), scale, free_nats, beta_dyn, beta_rep, unimix
    (0, (2, 3, 32, 16), 1.0, 1.0, 1.0, 0.1, 0.01),
    (1, (4, 5, 8, 8), 3.0, 1.0, 1.0, 0.1, 0.01),
    (2, (1, 7, 4, 32), 2.0, 0.5, 0.5, 0.1, 0.01),
    (3, (3, 2, 6, 5), 5.0, 0.0, 0.8, 0.2, 0.1),
    (4, (2, 2, 3, 4), 0.3, 1.0, 1.0, 0.1, 0.01),   # mostly below the floor
    (5, (2, 4, 16, 4), 6.0, 2.0, 1.0, 0.1, 0.5),
    (6, (5, 1, 2, 3), 4.0, 0.25, 0.3, 0.7, 0.0),
]


@pytest.mark.parametrize("seed,shape,scale,free,beta_dyn,beta_rep,unimix", CASES)
def test_matches_danijar_values_and_gradients(seed, shape, scale, free, beta_dyn, beta_rep, unimix):
    post, prior = logits(seed, shape, scale)
    weights = torch.rand(shape[:2], generator=torch.Generator().manual_seed(100 + seed), dtype=torch.float64)
    ours = routed_grads(
        lambda p, q: reference()(p, q, free_nats=free, beta_dyn=beta_dyn, beta_rep=beta_rep, unimix=unimix),
        post, prior, weights,
    )
    theirs = routed_grads(
        lambda p, q: danijar_rssm_loss(p, q, free_nats=free, unimix=unimix, dyn_scale=beta_dyn, rep_scale=beta_rep),
        post, prior, weights,
    )
    for got, want in zip(ours[:3], theirs[:3]):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=ATOL, rtol=0), (got, want)
    for name in ("loss", "dyn", "rep"):
        for got, want in zip(ours[3][name], theirs[3][name]):
            assert torch.allclose(got, want, atol=ATOL, rtol=0), name
    # Gradient routing, as upstream: dyn reaches only the prior, rep only the posterior.
    assert torch.all(ours[3]["dyn"][0] == 0) and torch.all(ours[3]["rep"][1] == 0)


@pytest.mark.parametrize("seed,shape,scale,free,beta_dyn,beta_rep,unimix", CASES)
def test_matches_nm512_with_explicit_betas(seed, shape, scale, free, beta_dyn, beta_rep, unimix):
    post, prior = logits(seed, shape, scale)
    loss, dyn, rep = reference()(post, prior, free_nats=free, beta_dyn=beta_dyn, beta_rep=beta_rep, unimix=unimix)
    per_step, _, nm_dyn, nm_rep = nm512_kl_loss(post, prior, free=free, dyn_scale=beta_dyn, rep_scale=beta_rep, unimix_ratio=unimix)
    assert torch.allclose(dyn, nm_dyn, atol=ATOL, rtol=0)
    assert torch.allclose(rep, nm_rep, atol=ATOL, rtol=0)
    assert torch.allclose(loss, per_step.mean(), atol=ATOL, rtol=0)


def test_identical_distributions_match_upstream_floor():
    post, _ = logits(7, (2, 3, 4, 6), 2.0)
    loss, dyn, rep = reference()(post, post.clone())
    want_loss, want_dyn, want_rep = danijar_rssm_loss(post, post.clone())
    assert torch.equal(dyn, want_dyn) and torch.equal(rep, want_rep)
    assert torch.allclose(loss, want_loss, atol=ATOL, rtol=0)
    assert torch.all(dyn == 1.0)


def test_extreme_logits_match_upstream():
    post = torch.tensor([30.0, -30.0, -30.0, -30.0], dtype=torch.float64).view(1, 1, 1, 4)
    prior = torch.tensor([-30.0, 30.0, -30.0, -30.0], dtype=torch.float64).view(1, 1, 1, 4)
    for unimix in (0.0, 0.01, 0.1):
        ours = reference()(post, prior, free_nats=0.0, unimix=unimix)
        theirs = danijar_rssm_loss(post, prior, free_nats=0.0, unimix=unimix)
        for got, want in zip(ours, theirs):
            assert torch.allclose(got, want, atol=1e-9, rtol=0), (unimix, got, want)


def test_float32_matches_upstream_precision():
    """Upstream casts to float32; float32 inputs reproduce it to float32 precision."""
    post, prior = logits(8, (2, 3, 32, 16), 2.0)
    ours = reference()(post.float(), prior.float())
    theirs = danijar_rssm_loss(post.float(), prior.float())
    for got, want in zip(ours, theirs):
        assert got.dtype == torch.float32
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_pins_nm512_default_dyn_scale():
    """The port's default dyn_scale is 0.5; upstream and the exercise default to 1.0."""
    post, prior = logits(9, (2, 3, 4, 8), 3.0)
    ours, dyn, rep = reference()(post, prior)
    upstream, _, _ = danijar_rssm_loss(post, prior)
    port_default, _, _, _ = nm512_kl_loss(post, prior)
    port_matched, _, _, _ = nm512_kl_loss(post, prior, dyn_scale=1.0)
    assert torch.allclose(ours, upstream, atol=ATOL, rtol=0)
    assert torch.allclose(ours, port_matched.mean(), atol=ATOL, rtol=0)
    assert torch.allclose(port_default.mean(), (0.5 * dyn + 0.1 * rep).mean(), atol=ATOL, rtol=0)
    assert not torch.allclose(port_default.mean(), ours)


def test_pins_gradient_at_an_exact_free_bits_tie():
    """jnp.maximum splits the gradient in half at a tie; torch.clip and the exercise pass all of it.

    The exercise never tests a tie. Each implementation is put exactly on its own floor.
    """
    post, prior = logits(10, (1, 1, 4, 5), 2.0)

    def prior_grad(fn, free):
        q = prior.clone().requires_grad_(True)
        return torch.autograd.grad(fn(post, q, free)[0].mean(), [q])[0]

    ours_fn = lambda p, q, free: reference()(p, q, free_nats=free, beta_dyn=1.0, beta_rep=0.0)
    danijar_fn = lambda p, q, free: danijar_rssm_loss(p, q, free_nats=free, dyn_scale=1.0, rep_scale=0.0)
    nm512_fn = lambda p, q, free: nm512_kl_loss(p, q, free=free, dyn_scale=1.0, rep_scale=0.0)

    ours_kl = reference()(post, prior, free_nats=0.0)[1].item()
    danijar_kl = danijar_rssm_loss(post, prior, free_nats=0.0)[1].item()
    nm512_kl = nm512_kl_loss(post, prior, free=0.0)[2].item()
    assert ours_kl > 0.0

    full = prior_grad(ours_fn, 0.0)
    assert torch.allclose(prior_grad(ours_fn, ours_kl), full, atol=ATOL, rtol=0)
    assert torch.allclose(prior_grad(nm512_fn, nm512_kl), full, atol=ATOL, rtol=0)
    assert torch.allclose(prior_grad(danijar_fn, danijar_kl), 0.5 * full, atol=ATOL, rtol=0)
