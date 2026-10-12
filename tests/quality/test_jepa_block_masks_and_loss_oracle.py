"""The jepa_block_masks_and_loss reference, checked against V-JEPA 2.

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied/vendored/vjepa2_masks_and_loss.py`.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied.vendored.vjepa2_masks_and_loss import (
    vjepa2_collate_masks,
    vjepa2_ema_update,
    vjepa2_loss,
    vjepa2_momentum_scheduler,
)
from torch_judge.tasks import get_task

TASK_ID = "jepa_block_masks_and_loss"
SEEDS = (0, 1, 2, 3, 4, 5, 6)


def reference():
    """Load the reference solution's class and return an instance."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]()


def draw(g, low, high):
    return int(torch.randint(low, high + 1, (1,), generator=g))


def upstream_style_blocks(seed, grid, batch, npred, shared_size=True):
    """Blocks as upstream draws them: one (t, h, w) per batch (if shared_size), a position per block.

    Redraws until every sample keeps at least two context and two target tokens, so
    upstream's ``.squeeze()`` never collapses a mask to 0-d.
    """
    T, H, W = grid
    g = torch.Generator().manual_seed(seed)
    while True:
        size = (draw(g, 1, T), draw(g, 1, H), draw(g, 1, W))
        placements = []
        for _ in range(batch):
            blocks = []
            for _ in range(npred):
                t, h, w = size if shared_size else (draw(g, 1, T), draw(g, 1, H), draw(g, 1, W))
                blocks.append((draw(g, 0, T - t), t, draw(g, 0, H - h), h, draw(g, 0, W - w), w))
            placements.append(blocks)
        covered = [
            len({(a * H + y) * W + x for (t0, t, top, h, left, w) in blocks
                 for a in range(t0, t0 + t) for y in range(top, top + h) for x in range(left, left + w)})
            for blocks in placements
        ]
        if all(2 <= c <= T * H * W - 2 for c in covered):
            return placements


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("grid", "batch", "npred"), [((8, 4, 4), 4, 2), ((2, 6, 5), 3, 4), ((4, 5, 7), 5, 1)])
@pytest.mark.parametrize("shared_size", (True, False))
@pytest.mark.parametrize("max_keep", (None, 3, 40))
def test_make_masks_matches_vjepa2(seed, grid, batch, npred, shared_size, max_keep):
    placements = upstream_style_blocks(seed, grid, batch, npred, shared_size)
    ctx, tgt = reference().make_masks(grid, placements, max_keep=max_keep)
    want_ctx, want_tgt = vjepa2_collate_masks(*grid, placements, max_keep=max_keep)
    assert ctx.dtype == want_ctx.dtype == torch.int64 and tgt.dtype == want_tgt.dtype == torch.int64
    assert torch.equal(ctx, want_ctx) and torch.equal(tgt, want_tgt)


def test_full_grid_block_raises_in_both():
    blocks = [[(0, 1, 0, 2, 0, 2)], [(0, 2, 0, 2, 0, 2)]]
    with pytest.raises(ValueError):
        reference().make_masks((2, 2, 2), blocks)
    with pytest.raises(ValueError):
        vjepa2_collate_masks(2, 2, 2, blocks)


def test_one_token_masks_diverge_deliberately():
    """Upstream's .squeeze() makes a one-token mask 0-d, and len() then fails; the exercise returns (B, 1)."""
    blocks = [[(0, 1, 0, 1, 0, 1)]]
    ctx, tgt = reference().make_masks((1, 2, 2), blocks)
    assert tgt.tolist() == [[0]] and ctx.tolist() == [[1, 2, 3]]
    with pytest.raises(TypeError):
        vjepa2_collate_masks(1, 2, 2, blocks)


def loss_inputs(seed, batch=3, tokens=24, dim=8, sizes=(5, 9)):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, tokens, dim, generator=g, dtype=torch.float64) * 4 - 2
    idx = [torch.stack([torch.randperm(tokens, generator=g)[:k] for _ in range(batch)]) for k in sizes]
    preds = [torch.randn(batch, k, dim, generator=g, dtype=torch.float64) for k in sizes]
    return preds, feats, idx


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("loss_exp", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("n_masks", (1, 2))
def test_jepa_loss_matches_vjepa2_value_and_gradient(seed, loss_exp, n_masks):
    preds, feats, idx = loss_inputs(seed)
    preds, idx = preds[:n_masks], idx[:n_masks]
    mine = [p.clone().requires_grad_(True) for p in preds]
    theirs = [p.clone().requires_grad_(True) for p in preds]
    arg_pred = mine[0] if n_masks == 1 else mine
    arg_idx = idx[0] if n_masks == 1 else idx
    got = reference().jepa_loss(arg_pred, feats, arg_idx, loss_exp=loss_exp)
    want = vjepa2_loss([theirs], [feats], [idx], loss_exp)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    got.backward()
    want.backward()
    for a, b in zip(mine, theirs):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-14)


def test_jepa_loss_float32_matches_vjepa2():
    preds, feats, idx = loss_inputs(9)
    preds, feats = [p.float() for p in preds], feats.float()
    got = reference().jepa_loss(preds, feats, idx)
    want = vjepa2_loss([preds], [feats], [idx], 1.0)
    assert got.dtype == want.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m", (0.0, 0.5, 0.996, 0.99925, 1.0))
def test_ema_update_matches_vjepa2(seed, m):
    torch.manual_seed(seed)
    online = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)).double()
    target_a = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)).double()
    target_b = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)).double()
    target_b.load_state_dict(target_a.state_dict())
    for step in range(3):
        reference().ema_update(target_a.parameters(), online.parameters(), m)
        vjepa2_ema_update(online.parameters(), target_b.parameters(), m)
        with torch.no_grad():
            for p in online.parameters():
                p.add_(0.1 * (step + 1))
    for a, b in zip(target_a.parameters(), target_b.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-14)


def test_ema_length_mismatch_diverges_deliberately():
    """Upstream's zip stops at the shorter list; the exercise raises instead."""
    target = [torch.ones(2), torch.ones(3)]
    vjepa2_ema_update([torch.zeros(2)], target, 0.5)
    assert torch.equal(target[0], torch.full((2,), 0.5)) and torch.equal(target[1], torch.ones(3))
    with pytest.raises(ValueError):
        reference().ema_update([torch.ones(2), torch.ones(3)], [torch.zeros(2)], 0.5)


@pytest.mark.parametrize(("ema", "ipe", "num_epochs", "ipe_scale"), [
    ((0.996, 1.0), 10, 3, 1.0),
    ((0.99925, 0.99925), 300, 10, 1.25),
    ((0.9, 0.999), 7, 5, 1.0),
    ((0.5, 0.4), 4, 1, 1.0),
])
def test_momentum_matches_vjepa2_schedule(ema, ipe, num_epochs, ipe_scale):
    want = list(vjepa2_momentum_scheduler(ema, ipe, num_epochs, ipe_scale))
    total = int(ipe * num_epochs * ipe_scale)
    assert len(want) == total + 1
    jepa = reference()
    got = [jepa.momentum(step, total, ema[0], ema[1]) for step in range(total + 1)]
    assert got[0] == ema[0]
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-15
