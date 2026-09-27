"""Mutation gates and metadata contracts for the ML fundamentals path."""

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

NEW_TASK_IDS = (
    "binary_cross_entropy",
    "weight_regularization",
    "sgd_momentum",
    "precision_recall_f1",
    "roc_auc",
)

# (name, text in the reference solution, replacement). Each replacement is one realistic bug.
MUTATIONS = {
    "binary_cross_entropy": [
        ("sigmoid then log", "per_element = weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)",
         "p = torch.sigmoid(logits)\n    per_element = -(weight * targets * torch.log(p) + (1 - targets) * torch.log(1 - p))"),
        ("pos_weight on the negative term", "weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)",
         "targets * F.softplus(-logits) + weight * (1 - targets) * F.softplus(logits)"),
        ("sums instead of averaging", "return per_element.mean()", "return per_element.sum()"),
        ("swaps softplus signs", "weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)",
         "weight * targets * F.softplus(logits) + (1 - targets) * F.softplus(-logits)"),
        ("detaches the loss", "return per_element.mean()", "return per_element.mean().detach()"),
        ("calls the built-in", "per_element = weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)\n    return per_element.mean()",
         "return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.as_tensor(weight, dtype=logits.dtype))"),
        ("broadcasts mismatched shapes", "if logits.shape != targets.shape:", "if False:"),
    ],
    "weight_regularization": [
        ("penalizes biases", "if weight.ndim < 2:", "if weight.ndim < 1:"),
        ("drops the one-half", "0.5 * l2 * (weight * weight).sum()", "l2 * (weight * weight).sum()"),
        ("averages instead of summing", "l1 * weight.abs().sum()", "l1 * weight.abs().mean()"),
        ("uses the L2 norm instead of its square", "(weight * weight).sum()", "weight.norm()"),
        ("detaches the L1 term", "l1 * weight.abs().sum()", "l1 * weight.detach().abs().sum()"),
        ("consumes the generator twice", "    loss = data_loss\n", "    _count = len(list(named_parameters))\n    loss = data_loss\n"),
        ("accepts negative strengths", "if l1 < 0 or l2 < 0:", "if False:"),
    ],
    "sgd_momentum": [
        ("dampens the first buffer", "buf = d.clone()", "buf = (1 - self.momentum) * d"),
        ("ignores Nesterov", "d = d + self.momentum * buf if self.nesterov else buf", "d = buf"),
        ("ignores weight decay", "d = d + self.weight_decay * p", "d = d"),
        ("decouples weight decay", "d = d + self.weight_decay * p", "p.mul_(1 - self.lr * self.weight_decay)"),
        ("rebinds instead of updating in place", "p.add_(d, alpha=-self.lr)", "self.params[index] = p - self.lr * d"),
        ("updates parameters without gradients", "if p.grad is None:\n                continue", "if p.grad is None:\n                p.grad = torch.zeros_like(p)"),
        ("accepts Nesterov without momentum", "if nesterov and momentum == 0:", "if False:"),
    ],
    "precision_recall_f1": [
        ("takes classes from targets only", "torch.unique(torch.cat([preds, targets]))", "torch.unique(targets)"),
        ("includes classes that never appear", "torch.unique(torch.cat([preds, targets])).tolist()",
         "list(range(int(torch.cat([preds, targets]).max()) + 1))"),
        ("macro F1 from macro precision and recall", "    return {\"precision\"",
         "    if average != \"micro\":\n        f1 = divide(2 * precision * recall, precision + recall)\n    return {\"precision\""),
        ("weights by predicted count", "int(true_c.sum())", "int(pred_c.sum())"),
        ("zero division gives one", "return a / b if b else 0.0", "return a / b if b else 1.0"),
        ("micro takes the averaged path", "if average == \"micro\":", "if False:"),
        ("skips input validation", "if preds.ndim != 1 or targets.ndim != 1 or preds.shape != targets.shape or preds.numel() == 0:", "if False:"),
    ],
    "roc_auc": [
        ("ordinal ranks for ties", "ranks = torch.repeat_interleave(average_rank, counts)",
         "ranks = torch.arange(1, sorted_scores.numel() + 1, dtype=torch.float64)"),
        ("ties get the lowest rank", "average_rank = (ends - counts + 1 + ends).double() / 2", "average_rank = (ends - counts + 1).double()"),
        ("reversed orientation", "return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)",
         "return 1 - (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)"),
        ("no single-class check", "if n_pos == 0 or n_neg == 0:", "if False:"),
        ("no label check", "if bool(((labels != 0) & (labels != 1)).any()):", "if False:"),
    ],
}

_SEED_TUPLE = re.compile(r"for seed in \(([\d, ]+)\)")
_MANUAL_SEED = re.compile(r"manual_seed\((\d+)\)")


def _seeded_task_variant(task_id: str, repeat: int) -> dict:
    """Shift every seed in the unshown cases so a mutation cannot pass by luck."""
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


@pytest.mark.parametrize("task_id", NEW_TASK_IDS)
def test_task_metadata_is_valid_and_english_only(task_id):
    task = get_task(task_id)
    validate_task(task_id, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert task["interview_questions"]


def test_ml_fundamentals_path_lists_every_new_task():
    paths = {p["id"]: p for p in json.loads((ROOT / "web/src/lib/paths.json").read_text())["paths"]}
    path = paths["ml-fundamentals"]
    assert set(NEW_TASK_IDS) <= set(path["problems"])
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert all(task_id in starters for task_id in NEW_TASK_IDS)
