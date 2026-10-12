"""The latent_action_quantizer reference, checked against LAPA's NSVQ and UniVLA's VectorQuantizer.

The mutation gate proves the evaluator rejects wrong implementations; it cannot
prove the contract is right. Here the reference solution is compared against
code transcribed from LAPA (`nsvq.py`, MIT) and UniVLA (`blocks.py`, `model.py`,
Apache-2.0) in `torch_judge/harness/embodied_frontier/vendored/lapa_univla_vq.py`.

Pinned divergence: upstream LAPA shuffles the tiled donor list with
`torch.randperm` when live codes are fewer than dead ones, and draws its own
`eps * randn` perturbation. The exercise uses the unshuffled round-robin order
and a caller-supplied jitter; with an identity permutation and the same noise
the two agree exactly.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.lapa_univla_vq import (
    LAPA_NSVQ_EPS,
    UNIVLA_VQ_BETA,
    lapa_nsvq_forward,
    lapa_replace_unused_codebooks,
    univla_vector_quantize,
    univla_vq_losses,
)
from torch_judge.tasks import get_task

TASK_ID = "latent_action_quantizer"


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]], namespace["replace_dead_codes"]


def seeded(seed: int, n: int, d: int, k: int):
    g = torch.Generator().manual_seed(seed)
    codebook = torch.randn(k, d, generator=g)
    first = torch.randn(n, d, generator=g)
    last = first + torch.randn(n, d, generator=g) * 1.3
    noise = torch.randn(n, d, generator=g)
    return codebook, first, last, noise


@pytest.mark.parametrize("seed", range(7))
def test_nsvq_matches_lapa(seed):
    quantize, _ = reference()
    n, d, k = 3 + seed * 5, 2 + seed % 4, 4 + seed
    codebook, first, last, noise = seeded(seed, n, d, k)

    cb_up = codebook.clone().requires_grad_(True)
    want, want_ppl, want_idx, _ = lapa_nsvq_forward(first, last, cb_up, noise)
    cb_ref = codebook.clone().requires_grad_(True)
    got, idx, ppl, vq_loss = quantize(first, last, cb_ref, mode="nsvq", noise=noise, eps=LAPA_NSVQ_EPS)

    assert torch.equal(idx, want_idx)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ppl, want_ppl, rtol=1e-5, atol=1e-6)
    assert vq_loss.item() == 0.0

    w = torch.randn(n, d, generator=torch.Generator().manual_seed(100 + seed))
    (want * w).sum().backward()
    (got * w).sum().backward()
    torch.testing.assert_close(cb_ref.grad, cb_up.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("seed", range(7))
def test_ste_and_losses_match_univla(seed):
    quantize, _ = reference()
    codebook, first, last, _ = seeded(50 + seed, 4, 6, 9)
    first = first.reshape(2, 2, 6)  # UniVLA quantizes (B*T, n, E) tokens
    last = last.reshape(2, 2, 6)

    cb_up = codebook.clone().requires_grad_(True)
    x_up = (last - first).clone().requires_grad_(True)
    z_q_up, z_up, emb_up, idx_up = univla_vector_quantize(x_up, cb_up)
    loss_up, _, _ = univla_vq_losses(emb_up, z_up)

    cb_ref = codebook.clone().requires_grad_(True)
    last_ref = last.clone().requires_grad_(True)
    z_q, idx, _, vq_loss = quantize(first, last_ref, cb_ref, mode="ste", beta=UNIVLA_VQ_BETA)

    assert torch.equal(idx, idx_up)
    torch.testing.assert_close(z_q, z_q_up)
    torch.testing.assert_close(vq_loss, loss_up)

    w = torch.randn(2, 2, 6, generator=torch.Generator().manual_seed(seed))
    ((z_q_up * w).sum() + loss_up).backward()
    ((z_q * w).sum() + vq_loss).backward()
    torch.testing.assert_close(cb_ref.grad, cb_up.grad)
    torch.testing.assert_close(last_ref.grad, x_up.grad)


def test_perplexity_with_unused_codes_matches_lapa():
    quantize, _ = reference()
    codebook = torch.eye(6)
    last = codebook[torch.tensor([0, 0, 3, 3, 3, 5])]
    first = torch.zeros_like(last)
    noise = torch.ones_like(last)
    _, want_ppl, _, _ = lapa_nsvq_forward(first, last, codebook, noise)
    _, _, ppl, _ = quantize(first, last, codebook, mode="nsvq", noise=noise)
    torch.testing.assert_close(ppl, want_ppl)


def identity_perm(n):
    return torch.arange(n)


def fixed_randn(table):
    """A stand-in for torch.randn returning rows of a fixed table; upstream multiplies it by eps."""
    def randn(size):
        size = tuple(size)
        return table[: size[0], : size[1]].clone()
    return randn


@pytest.mark.parametrize("seed", range(7))
def test_dead_code_replacement_matches_lapa(seed):
    _, replace = reference()
    g = torch.Generator().manual_seed(seed)
    k, d = 6 + seed, 3
    codebook = torch.randn(k, d, generator=g)
    usage = torch.randint(0, 5, (k,), generator=g, dtype=torch.int32)
    usage[seed % k] = 9  # at least one live code
    num_batches, threshold = 10, 0.25
    table = torch.randn(k, d, generator=g)

    want = lapa_replace_unused_codebooks(
        codebook, usage, num_batches, threshold, randperm=identity_perm, randn=fixed_randn(table / LAPA_NSVQ_EPS),
    )
    dead = torch.where(usage / num_batches < threshold)[0]
    jitter = torch.zeros(k, d)
    jitter[dead] = table[: dead.numel()]
    got, got_dead = replace(codebook, usage, num_batches, threshold, jitter=jitter)

    assert torch.equal(got_dead, dead)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("live_mask", [[True, False, False, False, False], [True, True, True, False, False], [False] * 5, [True] * 5])
def test_replacement_edge_cases_match_lapa(live_mask):
    _, replace = reference()
    codebook = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    usage = torch.tensor([10 if live else 0 for live in live_mask], dtype=torch.int32)
    table = torch.full((5, 3), 0.5)
    want = lapa_replace_unused_codebooks(
        codebook, usage, 10, 0.5, randperm=identity_perm, randn=fixed_randn(table / LAPA_NSVQ_EPS),
    )
    dead = torch.tensor([i for i, live in enumerate(live_mask) if not live], dtype=torch.long)
    jitter = torch.zeros(5, 3)
    if not any(live_mask):
        jitter[:] = 0.5  # upstream perturbs every code when none is live
    else:
        jitter[dead] = 0.5
    got, got_dead = replace(codebook, usage, 10, 0.5, jitter=jitter)
    assert torch.equal(got_dead, dead)
    torch.testing.assert_close(got, want)


def test_upstream_shuffle_is_the_only_divergence():
    """With a non-identity permutation upstream may pick other donors; the exercise pins round-robin."""
    _, replace = reference()
    codebook = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    usage = torch.tensor([9, 0, 0, 9, 0], dtype=torch.int32)
    got, _ = replace(codebook, usage, 10, 0.5)
    assert got.tolist() == [[0, 1], [0, 1], [6, 7], [6, 7], [0, 1]]
    reversed_perm = lambda n: torch.arange(n - 1, -1, -1)  # noqa: E731
    shuffled = lapa_replace_unused_codebooks(
        codebook, usage, 10, 0.5, randperm=reversed_perm, randn=lambda size: torch.zeros(tuple(size)),
    )
    assert not torch.equal(shuffled, got)
