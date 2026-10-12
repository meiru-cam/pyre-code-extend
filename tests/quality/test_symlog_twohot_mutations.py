"""Mutation gate for the symlog_twohot exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "symlog_twohot"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """Apply exact-substring edits to the reference; each old string must occur once."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"edit target not unique: {old!r}"
        code = code.replace(old, new)
    return code


MUTATIONS = [
    # Research section 3.2 W1: two-hot in raw space instead of symlog space.
    Mutation("twohot_in_raw_space", edit(("    x = symlog(y)\n", "    x = y\n"))),
    # Research section 3.2 W1: weight_below and weight_above swapped.
    Mutation("swapped_weights", edit((
        "    weight_below = dist_to_above / total\n    weight_above = dist_to_below / total\n",
        "    weight_below = dist_to_below / total\n    weight_above = dist_to_above / total\n",
    ))),
    # Research section 3.2 W1: plain left-to-right sum, non-zero for uniform logits.
    Mutation("plain_sum_mean", edit((
        "    return symexp(total)\n",
        "    return symexp(weighted.sum(-1))\n",
    ))),
    # The NM512 port's symlog and symexp: log(|x| + 1) and exp(|x|) - 1.
    Mutation("symlog_without_log1p", edit((
        "torch.sign(x) * torch.log1p(torch.abs(x))",
        "torch.sign(x) * torch.log(torch.abs(x) + 1.0)",
    ))),
    Mutation("symexp_without_expm1", edit((
        "torch.sign(x) * torch.expm1(torch.abs(x))",
        "torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)",
    ))),
    Mutation("symexp_missing_minus_one", edit((
        "torch.sign(x) * torch.expm1(torch.abs(x))",
        "torch.sign(x) * torch.exp(torch.abs(x))",
    ))),
    Mutation("mean_stays_in_symlog_space", edit(("    return symexp(total)\n", "    return total\n"))),
    Mutation("below_uses_strict_less", edit((
        "    below = (bins <= x[..., None]).sum(-1) - 1\n",
        "    below = (bins < x[..., None]).sum(-1) - 1\n",
    ))),
    Mutation("no_clip_above", edit(("    above = above.clamp(0, K - 1)\n", ""))),
    Mutation("edge_split_without_equal_case", edit((
        "    equal = below == above  # outside the range: both indices sit on the edge bin\n",
        "    equal = torch.zeros_like(below, dtype=torch.bool)\n",
    ))),
    Mutation("target_not_detached", edit((
        "twohot_target(y.detach().to(logits.dtype), bins)",
        "twohot_target(y.to(logits.dtype), bins)",
    ))),
    Mutation("loss_averaged_over_batch", edit((
        "    return -(target * torch.log_softmax(logits, -1)).sum(-1)\n",
        "    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()\n",
    ))),
    Mutation("loss_on_probabilities", edit((
        "    return -(target * torch.log_softmax(logits, -1)).sum(-1)\n",
        "    return -(target * torch.softmax(logits, -1)).sum(-1)\n",
    ))),
    # danijar's even-K construction: two halves that both end at zero.
    Mutation("even_k_duplicate_zero_bin", edit((
        "    lower = torch.linspace(-limit, limit, K, dtype=torch.float32)[: K // 2]\n"
        "    middle = torch.zeros(K % 2, dtype=torch.float32)\n",
        "    lower = torch.linspace(-limit, 0.0, (K + 1) // 2, dtype=torch.float32)\n"
        "    lower, middle = (lower[:-1], lower[-1:]) if K % 2 else (lower, torch.zeros(0))\n"
        "    if K % 2 == 0:\n"
        "        return torch.cat([lower, -lower.flip(0)])\n",
    ))),
    Mutation("mean_pairs_bin_with_neighbour", edit((
        "    pairs = weighted[..., :half].flip(-1) + weighted[..., K - half:]\n",
        "    pairs = weighted[..., :half] + weighted[..., K - half:]\n",
    ))),
    Mutation("odd_k_middle_dropped", edit((
        "        total = weighted[..., half] + total\n",
        "        total = total\n",
    ))),
    Mutation("bins_not_cast_to_logits_dtype", edit((
        "    bins = bins.to(logits.dtype)\n    K = bins.shape[0]\n    half",
        "    K = bins.shape[0]\n    half",
    ))),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {mutation.name for mutation in MUTATIONS}


def test_mutation_names_are_unique():
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=set(TASKS))
    assert not any(key.endswith("_zh") for key in task)
    assert set(task["advisory_prerequisites"]) <= set(TASKS)
    assert len(task["interview_questions"]) == 7
