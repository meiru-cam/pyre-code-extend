"""The knowledge_insulation_attention reference, checked against LeRobot pi0.5 and FineART-VLA.

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied_frontier/vendored/lerobot_knowledge_insulation.py`.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.lerobot_knowledge_insulation import (
    lerobot_fineart_mask_and_positions,
    lerobot_ki_attention,
)
from torch_judge.tasks import get_task

TASK_ID = "knowledge_insulation_attention"
SEEDS = (0, 1, 2, 3, 4, 5, 6)
LAYOUTS = ((2, 5, 4, 3), (3, 3, 6, 5), (1, 7, 1, 1), (4, 2, 3, 6))


def reference():
    """Load the reference solution's class and return an instance."""
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]()


def pads(seed: int, batch: int, num_prefix: int, num_fast: int):
    g = torch.Generator().manual_seed(seed)
    prefix_pad = torch.rand(batch, num_prefix, generator=g) < 0.7
    prefix_pad[:, 0] = True
    fast_pad = torch.rand(batch, num_fast, generator=g) < 0.7
    return prefix_pad, fast_pad


def qkv(seed: int, batch: int, heads: int, total: int, dim: int):
    g = torch.Generator().manual_seed(seed + 1000)
    return [
        torch.randn(batch, heads, total, dim, generator=g, dtype=torch.float64).requires_grad_(True)
        for _ in range(3)
    ]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mask_and_positions_match_fineart(seed, layout):
    batch, num_prefix, num_fast, num_actions = layout
    prefix_pad, fast_pad = pads(seed, batch, num_prefix, num_fast)
    mask, pos = reference().build_mask(prefix_pad, fast_pad, num_actions)
    want_mask, want_pos = lerobot_fineart_mask_and_positions(prefix_pad, fast_pad, num_actions)
    assert torch.equal(mask, want_mask)
    assert torch.equal(pos, want_pos.long())


@pytest.mark.parametrize("num_fast", (0, 1, 4))
def test_edge_layouts_match_fineart(num_fast):
    # No FAST tokens (stock pi0.5 layout), a fully padded FAST row, and a single action.
    prefix_pad = torch.tensor([[True, True, False], [True, False, True]])
    fast_pad = torch.zeros(2, num_fast, dtype=torch.bool)
    if num_fast:
        fast_pad[1] = True
    for num_actions in (1, 4):
        mask, pos = reference().build_mask(prefix_pad, fast_pad, num_actions)
        want_mask, want_pos = lerobot_fineart_mask_and_positions(prefix_pad, fast_pad, num_actions)
        assert torch.equal(mask, want_mask)
        assert torch.equal(pos, want_pos.long())


def _losses(out, valid, num_backbone, seed):
    """Three seeded losses over valid query rows: backbone rows, action rows, both."""
    g = torch.Generator().manual_seed(seed + 2000)
    weights = torch.randn(out.shape, generator=g, dtype=torch.float64) * valid[:, None, :, None]
    rows = torch.arange(out.shape[2])
    backbone = (rows < num_backbone)[None, None, :, None]
    return {
        "backbone": (out * weights * backbone).sum(),
        "action": (out * weights * ~backbone).sum(),
        "both": (out * weights).sum(),
    }


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_attend_values_and_gradients_match_pi05(seed, layout):
    batch, num_prefix, num_fast, num_actions = layout
    prefix_pad, fast_pad = pads(seed, batch, num_prefix, num_fast)
    mask, _ = lerobot_fineart_mask_and_positions(prefix_pad, fast_pad, num_actions)
    num_backbone = num_prefix + num_fast
    total = num_backbone + num_actions
    valid = mask.any(dim=-1)
    q, k, v = qkv(seed, batch, 2, total, 4)

    out = reference().attend(q, k, v, mask, num_backbone)
    want = lerobot_ki_attention(q, k, v, mask, num_backbone)
    rows = valid[:, None, :, None].expand_as(out)
    assert torch.allclose(out[rows], want[rows], atol=1e-12)
    # Padding rows: upstream's additive -2.38e38 and the reference's finfo.min fill both
    # give a uniform softmax in float64, so even the ungraded rows agree.
    assert torch.allclose(out, want, atol=1e-12)

    got_losses = _losses(out, valid, num_backbone, seed)
    want_losses = _losses(want, valid, num_backbone, seed)
    for name in got_losses:
        got = torch.autograd.grad(got_losses[name], [q, k, v], retain_graph=True)
        expected = torch.autograd.grad(want_losses[name], [q, k, v], retain_graph=True)
        for which, a, b in zip("qkv", got, expected):
            assert torch.allclose(a, b, atol=1e-10), (name, which)


@pytest.mark.parametrize("num_backbone", (0, 3, 7))
def test_attend_split_edges_match_pi05(num_backbone):
    # Upstream with knowledge_insulation off is plain attention; with prefix_len 0 or N the
    # insulated path must coincide with it too.
    mask = torch.rand(2, 7, 7, generator=torch.Generator().manual_seed(num_backbone)) < 0.6
    mask |= torch.eye(7, dtype=torch.bool)
    q, k, v = qkv(num_backbone, 2, 3, 7, 5)
    out = reference().attend(q, k, v, mask, num_backbone)
    want = lerobot_ki_attention(q, k, v, mask, num_backbone)
    assert torch.allclose(out, want, atol=1e-12)
    got = torch.autograd.grad(out.pow(2).sum(), [q, k, v])
    expected = torch.autograd.grad(want.pow(2).sum(), [q, k, v])
    for a, b in zip(got, expected):
        assert torch.allclose(a, b, atol=1e-10)
    if num_backbone in (0, 7):
        plain = lerobot_ki_attention(q, k, v, mask, num_backbone, knowledge_insulation=False)
        assert torch.allclose(out, plain, atol=1e-12)
