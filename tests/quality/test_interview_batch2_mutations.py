"""Mutation gates and metadata contracts for the RL, LLM efficiency and DL exercise batch."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

ROOT = Path(__file__).resolve().parents[2]

# task id -> learning path it belongs to
NEW_TASKS = {
    "rloo_advantage": "rl-posttraining",
    "loss_aggregation_modes": "rl-posttraining",
    "truncated_importance_sampling": "rl-posttraining",
    "simpo_ipo_loss": "alignment-agents",
    "pass_at_k": "rl-posttraining",
    "packed_sequence_mask": "training-inference-efficiency",
    "chunked_cross_entropy": "training-inference-efficiency",
    "logits_processors": "inference-distributed",
    "manual_backprop_mlp": "ml-fundamentals",
    "muon_newton_schulz": "ml-fundamentals",
}

# (name, text in the reference solution, replacement). Each replacement is one realistic bug.
MUTATIONS = {
    "rloo_advantage": [
        ("baseline includes self", "(grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)",
         "grouped.mean(dim=1, keepdim=True)"),
        ("divides by k instead of k - 1", "/ (group_size - 1)", "/ group_size"),
        ("standardizes like GRPO", "return (grouped - baseline).reshape(-1)",
         "return ((grouped - baseline) / (grouped.std(dim=1, keepdim=True) + 1e-6)).reshape(-1)"),
        ("groups by stride", "grouped = rewards.reshape(-1, group_size)\n    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)\n    return (grouped - baseline).reshape(-1)",
         "grouped = rewards.reshape(group_size, -1).T\n    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)\n    return (grouped - baseline).T.reshape(-1)"),
        ("accepts group size one", "group_size < 2", "group_size < 1"),
    ],
    "loss_aggregation_modes": [
        ("multiplies by the mask", "masked = torch.where(mask, per_token_loss, torch.zeros_like(per_token_loss))",
         "masked = per_token_loss * mask"),
        ("seq mean over all rows", "valid = seq_len > 0", "valid = seq_len >= 0"),
        ("token-mean averages sequence means", "return seq_sum.sum() / seq_len.sum()", "return (seq_sum / seq_len.clamp(min=1)).mean()"),
        ("norm divides by own length", "per_seq = seq_sum[valid] / mask.shape[1]", "per_seq = seq_sum[valid] / seq_len[valid]"),
        ("token-sum divides by length", "per_seq = seq_sum[valid]\n", "per_seq = seq_sum[valid] / seq_len[valid]\n"),
        ("accepts empty masks", "if not bool(mask.any()):", "if False:"),
    ],
    "truncated_importance_sampling": [
        ("no truncation", "torch.clamp(delta, max=math.log(cap))", "delta"),
        ("reversed ratio", "train_logprobs.detach() - rollout_logprobs.detach()", "rollout_logprobs.detach() - train_logprobs.detach()"),
        ("not detached", "train_logprobs.detach() - rollout_logprobs.detach()", "train_logprobs - rollout_logprobs"),
        ("sequence mean instead of sum", "delta = delta.sum(dim=1, keepdim=True).expand_as(delta)",
         "delta = (delta.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)).expand_as(delta)"),
        ("masked positions keep weights", "return torch.where(mask, weights, torch.zeros_like(weights))", "return weights"),
        ("multiplies by the mask", "delta = torch.where(mask, train_logprobs.detach() - rollout_logprobs.detach(), torch.zeros_like(train_logprobs))",
         "delta = (train_logprobs.detach() - rollout_logprobs.detach()) * mask"),
    ],
    "simpo_ipo_loss": [
        ("SimPO without length normalization", "chosen_logps / chosen_lengths - rejected_logps / rejected_lengths",
         "chosen_logps - rejected_logps"),
        ("gamma sign flipped", "rejected_logps / rejected_lengths) - gamma", "rejected_logps / rejected_lengths) + gamma"),
        ("unstable log sigmoid", "return F.softplus(-margin).mean()", "return -torch.log(torch.sigmoid(margin)).mean()"),
        ("IPO target 1 / beta", "((gap - 1 / (2 * beta)) ** 2)", "((gap - 1 / beta) ** 2)"),
        ("IPO ignores the reference", "gap = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)",
         "gap = chosen_logps - rejected_logps"),
        ("IPO uses absolute error", "((gap - 1 / (2 * beta)) ** 2)", "(gap - 1 / (2 * beta)).abs()"),
        ("accepts zero lengths", "if bool((chosen_lengths <= 0).any()) or bool((rejected_lengths <= 0).any()):", "if False:"),
    ],
    "pass_at_k": [
        ("biased plug-in estimate", "        miss = 1.0\n        for i in range(n - c + 1, n + 1):\n            miss *= 1.0 - k / i\n",
         "        miss = (1.0 - c / n) ** k\n"),
        ("off-by-one product range", "range(n - c + 1, n + 1)", "range(n - c, n)"),
        ("float binomials", "        miss = 1.0\n        for i in range(n - c + 1, n + 1):\n            miss *= 1.0 - k / i\n",
         "        miss = float(math.comb(n - c, k)) / float(math.comb(n, k))\n"),
        ("sums instead of averaging", "return total / len(num_samples)", "return total"),
        ("accepts c above n", "if n < k or c < 0 or c > n:", "if n < k or c < 0:"),
    ],
    "packed_sequence_mask": [
        ("plain causal mask", "mask = (doc_id[:, None] == doc_id[None, :]) & causal", "mask = causal"),
        ("block mask without causality", "mask = (doc_id[:, None] == doc_id[None, :]) & causal", "mask = doc_id[:, None] == doc_id[None, :]"),
        ("continuous position ids", "position_ids[:used] = torch.arange(used) - starts", "position_ids[:used] = torch.arange(used)"),
        ("padding rows are empty", "pad_ids = torch.arange(len(seq_lens), len(seq_lens) + total_len - used)",
         "pad_ids = torch.full((total_len - used,), -1)\n    causal_pad = None"),
        ("cu_seqlens without leading zero", "return mask, position_ids, cu_seqlens.to(torch.int32)", "return mask, position_ids, cu_seqlens[1:].to(torch.int32)"),
        ("accepts overflowing rows", "or sum(seq_lens) > total_len", ""),
    ],
    "chunked_cross_entropy": [
        ("no recomputation", "total = total + checkpoint(chunk_loss, hidden[start:end], weight, targets[start:end], use_reentrant=False)",
         "total = total + chunk_loss(hidden[start:end], weight, targets[start:end])"),
        ("averages chunk means", "total = total + checkpoint(chunk_loss, hidden[start:end], weight, targets[start:end], use_reentrant=False)\n    return total / valid",
         "total = total + checkpoint(chunk_loss, hidden[start:end], weight, targets[start:end], use_reentrant=False) / max(1, int((targets[start:end] != ignore_index).sum()))\n    return total / ((hidden.shape[0] + chunk_size - 1) // chunk_size)"),
        ("divides by all tokens", "return total / valid", "return total / targets.numel()"),
        ("drops the last partial chunk", "range(0, hidden.shape[0], chunk_size)", "range(0, hidden.shape[0] - chunk_size + 1, chunk_size)"),
        ("ignores ignore_index", "F.cross_entropy(h @ w.T, t, ignore_index=ignore_index, reduction=\"sum\")",
         "F.cross_entropy(h @ w.T, t.clamp(min=0), reduction=\"sum\")"),
        ("accepts no valid tokens", "if valid == 0:", "if False:"),
    ],
    "logits_processors": [
        ("penalty ignores sign", "torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)", "seen / repetition_penalty"),
        ("top-p before temperature", "    out = out / temperature\n    if top_k:", "    if top_k:"),
        ("top-p drops the crossing token", "before = torch.cumsum(probs, dim=-1) - probs", "before = torch.cumsum(probs, dim=-1)"),
        ("top-k drops ties", "out = out.masked_fill(out < kth, float(\"-inf\"))",
         "out = torch.full_like(out, float(\"-inf\")).scatter(-1, torch.topk(out, top_k, dim=-1).indices, torch.topk(out, top_k, dim=-1).values)"),
        ("modifies logits in place", "out = logits.clone()", "out = logits"),
        ("penalizes repeated ids repeatedly", "idx = torch.tensor(sorted(set(ids)), dtype=torch.long)\n            seen = out[b, idx]\n            out[b, idx] = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)",
         "for i in ids:\n                seen = out[b, i]\n                out[b, i] = seen / repetition_penalty if seen > 0 else seen * repetition_penalty"),
    ],
    "manual_backprop_mlp": [
        ("ReLU derivative one at zero", "dz1 = da1 * (z1 > 0).to(da1.dtype)", "dz1 = da1 * (z1 >= 0).to(da1.dtype)"),
        ("forgets the batch mean", "    dlogits /= N\n", ""),
        ("skips the ReLU mask", "dz1 = da1 * (z1 > 0).to(da1.dtype)", "dz1 = da1"),
        ("bias gradient averaged", "db1 = dz1.sum(dim=0)", "db1 = dz1.mean(dim=0)"),
        ("unstable softmax", "shifted = logits - logits.max(dim=1, keepdim=True).values", "shifted = logits"),
        ("uses autograd", "    z1 = x @ W1.T + b1\n",
         "    W1 = W1.clone().requires_grad_()\n    (x @ W1.T).sum().backward()\n    W1 = W1.detach()\n    z1 = x @ W1.T + b1\n"),
        ("accepts bad labels", "if bool((y < 0).any()) or bool((y >= C).any()):", "if False:"),
    ],
    "muon_newton_schulz": [
        ("no Nesterov look-ahead", "G = (1 - beta) * grad + beta * momentum if nesterov else momentum.clone()", "G = momentum.clone()"),
        ("momentum without dampening", "momentum.mul_(beta).add_(grad, alpha=1 - beta)", "momentum.mul_(beta).add_(grad)"),
        ("no shape scaling", "return X * math.sqrt(max(1.0, rows / cols))", "return X"),
        ("cubic iteration", "B = b * A + c * A @ A\n        X = a * X + B @ X", "X = 1.5 * X - 0.5 * A @ X"),
        ("normalizes by the max entry", "X = G / (G.norm() + eps)", "X = G / (G.abs().max() + eps)"),
        ("modifies grad", "momentum.mul_(beta).add_(grad, alpha=1 - beta)", "momentum.mul_(beta).add_(grad, alpha=1 - beta)\n    grad.mul_(1.0)\n    grad.add_(momentum, alpha=1e-3)"),
        ("accepts beta of one", "if not 0 <= beta < 1 or ns_steps < 1:", "if ns_steps < 1:"),
    ],
}

_SEED_TUPLE = re.compile(r"for seed in \(([\d, ]+)\)")
_MANUAL_SEED = re.compile(r"manual_seed\((\d+)\)")


def _seeded_task_variant(task_id: str, repeat: int) -> dict:
    """Shift every seed in the test cases so a mutation cannot pass by luck."""
    task = copy.deepcopy(get_task(task_id))
    changed = 0
    for case in task["tests"]:
        def shift_tuple(match):
            nonlocal changed
            changed += 1
            seeds = ", ".join(str(int(s) + 100 * repeat) for s in match.group(1).split(","))
            return f"for seed in ({seeds},)"

        def shift_manual(match):
            nonlocal changed
            changed += 1
            return f"manual_seed({int(match.group(1)) + 100 * repeat})"

        case["code"] = _MANUAL_SEED.sub(shift_manual, _SEED_TUPLE.sub(shift_tuple, case["code"]))
    assert changed, f"{task_id} needs a seeded unshown oracle"
    return task


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("task_id,targets", MUTATIONS.items())
def test_mutations_rejected_across_distinct_seeds(task_id, targets, repeat):
    original = get_task(task_id)["solution"]
    mutations = []
    for name, old, new in targets:
        assert old in original, f"mutation target drifted: {task_id}/{name}"
        mutations.append(Mutation(name, original.replace(old, new, 1)))
    rejected = assert_mutations_rejected(
        task_id, mutations, require_unshown=True,
        task_override=_seeded_task_variant(task_id, repeat),
    )
    assert set(rejected) == {name for name, _, _ in targets}


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_task_metadata_is_valid_and_english_only(task_id):
    task = get_task(task_id)
    validate_task(task_id, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert task["interview_questions"]
    assert task.get("sources"), "every new exercise needs pinned provenance"


def test_paths_and_starters_list_every_new_task():
    paths = {p["id"]: p for p in json.loads((ROOT / "web/src/lib/paths.json").read_text())["paths"]}
    for task_id, path_id in NEW_TASKS.items():
        assert task_id in paths[path_id]["problems"], (task_id, path_id)
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert all(task_id in starters for task_id in NEW_TASKS)
