"""Mutation gate for the multi-part jepa_block_masks_and_loss exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.parts_gate import assert_first_fails_in_part, mutant
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "jepa_block_masks_and_loss"
REFERENCE = get_task(TASK_ID)["solution"]

# (name, first part that must fail, [(text in the reference, replacement), ...]).
MUTATIONS = [
    # Part 1: masks.
    ("target is the intersection of blocks", 1, [
        ("            covered = torch.zeros((T, H, W), dtype=torch.bool)\n",
         "            covered = torch.ones((T, H, W), dtype=torch.bool)\n"),
        ("                covered[t0:t0 + t, top:top + h, left:left + w] = True\n",
         "                one = torch.zeros((T, H, W), dtype=torch.bool)\n"
         "                one[t0:t0 + t, top:top + h, left:left + w] = True\n"
         "                covered &= one\n"),
    ]),
    # Research section 3.2 W3: not truncating to the batch minimum, so the batch cannot be stacked.
    ("no truncation to the batch minimum", 1, [
        ("        ctx_idx = torch.stack([c[:k_ctx] for c in contexts])\n        tgt_idx = torch.stack([x[:k_tgt] for x in targets])\n",
         "        ctx_idx = torch.stack(contexts)\n        tgt_idx = torch.stack(targets)\n"),
    ]),
    ("truncation keeps the highest indices", 1, [
        ("        ctx_idx = torch.stack([c[:k_ctx] for c in contexts])\n        tgt_idx = torch.stack([x[:k_tgt] for x in targets])\n",
         "        ctx_idx = torch.stack([c[len(c) - k_ctx:] for c in contexts])\n"
         "        tgt_idx = torch.stack([x[len(x) - k_tgt:] for x in targets])\n"),
    ]),
    ("flat index with x and y swapped", 1, [
        ("            covered = covered.flatten()  # row-major: index = t*H*W + y*W + x\n",
         "            covered = covered.transpose(1, 2).flatten()  # row-major: index = t*H*W + y*W + x\n"),
    ]),
    ("block end inclusive", 1, [
        ("covered[t0:t0 + t, top:top + h, left:left + w] = True",
         "covered[t0:t0 + t + 1, top:top + h + 1, left:left + w + 1] = True"),
    ]),
    ("max_keep ignored", 1, [
        ("        if max_keep is not None:\n            k_ctx = min(k_ctx, max_keep)\n", ""),
    ]),
    ("max_keep caps the target too", 1, [
        ("        k_tgt = min(len(x) for x in targets)\n",
         "        k_tgt = min(len(x) for x in targets)\n"
         "        if max_keep is not None:\n            k_tgt = min(k_tgt, max_keep)\n"),
    ]),
    ("empty context returned instead of raised", 1, [
        ("            if context.numel() == 0:\n                raise ValueError(\"the blocks cover the whole grid, so the context is empty\")\n", ""),
    ]),
    ("blocks outside the grid clipped silently", 1, [
        ("                if min(t, h, w) < 1 or min(t0, top, left) < 0 or t0 + t > T or top + h > H or left + w > W:\n",
         "                if False:\n"),
    ]),
    # Part 2: loss.
    # Research section 3.2 W3: no layer norm on the targets.
    ("no layer norm on targets", 2, [
        ("            h = F.layer_norm(target_feats, (D,))  # eps 1e-5, no affine\n",
         "            h = target_feats.clone()  # eps 1e-5, no affine\n"),
    ]),
    ("layer norm over tokens", 2, [
        ("            h = F.layer_norm(target_feats, (D,))  # eps 1e-5, no affine\n",
         "            h = F.layer_norm(target_feats.transpose(1, 2), (target_feats.shape[1],)).transpose(1, 2)\n"),
    ]),
    ("no division by loss_exp", 2, [
        ("torch.mean(torch.abs(z - h_masked) ** loss_exp) / loss_exp", "torch.mean(torch.abs(z - h_masked) ** loss_exp)"),
    ]),
    ("loss_exp ignored", 2, [
        ("torch.mean(torch.abs(z - h_masked) ** loss_exp) / loss_exp", "torch.mean(torch.abs(z - h_masked))"),
    ]),
    ("masks pooled over tokens", 2, [
        ("        loss = 0\n        for z, idx in zip(preds, indices):\n"
         "            h_masked = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, D))\n"
         "            loss = loss + torch.mean(torch.abs(z - h_masked) ** loss_exp) / loss_exp\n"
         "        return loss / len(preds)  # each mask counts once, whatever its size\n",
         "        zs = torch.cat(preds, dim=1)\n"
         "        hs = torch.cat([torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, D)) for idx in indices], dim=1)\n"
         "        return torch.mean(torch.abs(zs - hs) ** loss_exp) / loss_exp\n"),
    ]),
    ("gradient reaches the targets", 2, [
        ("        with torch.no_grad():\n            h = F.layer_norm(target_feats, (D,))  # eps 1e-5, no affine\n",
         "        h = F.layer_norm(target_feats, (D,))  # eps 1e-5, no affine\n"),
    ]),
    ("first sample's indices for every row", 2, [
        ("            h_masked = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, D))\n",
         "            h_masked = h[:, idx[0]]\n"),
    ]),
    # Part 3: EMA target.
    # Research section 3.2 W3: EMA with the weights swapped.
    ("EMA weights swapped", 3, [
        ("                target.mul_(m).add_(online, alpha=1 - m)\n",
         "                target.mul_(1 - m).add_(online, alpha=m)\n"),
    ]),
    ("EMA rebinds instead of updating in place", 3, [
        ("            for target, online in zip(target_params, online_params):\n"
         "                target.mul_(m).add_(online, alpha=1 - m)\n",
         "            for i, (target, online) in enumerate(zip(target_params, online_params)):\n"
         "                target_params[i] = m * target + (1 - m) * online\n"),
    ]),
    ("EMA writes into the online tensors", 3, [
        ("                target.mul_(m).add_(online, alpha=1 - m)\n",
         "                target.copy_(online.mul_(1 - m).add_(target, alpha=m))\n"),
    ]),
    ("EMA outside no_grad", 3, [
        ("        with torch.no_grad():\n            for target, online in zip(target_params, online_params):\n"
         "                target.mul_(m).add_(online, alpha=1 - m)\n",
         "        for target, online in zip(target_params, online_params):\n"
         "            target.mul_(m).add_(online, alpha=1 - m)\n"),
    ]),
    ("EMA pairs stop at the shorter list", 3, [
        ("        if len(target_params) != len(online_params):\n"
         "            raise ValueError(\"target and online parameter lists differ in length\")\n", ""),
    ]),
    ("momentum schedule off by one step", 3, [
        ("return float(start + step * (end - start) / total_steps)",
         "return float(start + step * (end - start) / (total_steps + 1))"),
    ]),
    ("momentum schedule runs backwards", 3, [
        ("return float(start + step * (end - start) / total_steps)",
         "return float(end + step * (start - end) / total_steps)"),
    ]),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(
        TASK_ID, [Mutation(name, mutant(REFERENCE, edits)) for name, _, edits in MUTATIONS], require_unshown=True,
    )
    assert set(rejected) == {name for name, _, _ in MUTATIONS}


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_mutation_targets_are_unique():
    for name, _, edits in MUTATIONS:
        for old, _ in edits:
            assert REFERENCE.count(old) == 1, f"{name}: edit target not unique: {old[:60]!r}"


def test_mutation_names_are_unique():
    names = [name for name, _, _ in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=set(TASKS))
    assert not any(key.endswith("_zh") for key in task)
    assert set(task["advisory_prerequisites"]) <= set(TASKS)
    assert len(task["interview_questions"]) == 7
