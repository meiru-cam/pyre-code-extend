"""Mutation gate for the knowledge_insulation_attention exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "knowledge_insulation_attention"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """Apply exact-substring edits to the reference; each old string must occur once."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"edit target not unique: {old!r}"
        code = code.replace(old, new)
    return code


_ACTION_BLOCK = "        allowed = allowed | (is_action[:, None] & is_action[None, :])\n"
_FAST_BLOCK = "        allowed = allowed | (is_fast[:, None] & is_fast[None, :] & (key <= query))\n"
_K_INSULATED = "        k_insulated = torch.cat([k[:, :, :nb].detach(), k[:, :, nb:]], dim=2)\n"
_V_INSULATED = "        v_insulated = torch.cat([v[:, :, :nb].detach(), v[:, :, nb:]], dim=2)\n"
_BACKBONE_OUT = "        backbone_out = masked_attention(q[:, :, :nb], k, v, mask[:, :nb])\n"

MUTATIONS = [
    # KI App. B / FineART's att_2d_masks[:, fast_end:, fast_start:fast_end] = False is skipped:
    # the cumulative prefix-LM rule alone lets the action block read the FAST tokens.
    Mutation("action_rows_see_fast", edit((
        _ACTION_BLOCK,
        _ACTION_BLOCK + "        allowed = allowed | (is_action[:, None] & is_fast[None, :])\n",
    ))),
    # HybridVLA-style: autoregressive tokens attend to the continuous action inputs.
    Mutation("fast_rows_see_actions", edit((
        _ACTION_BLOCK,
        _ACTION_BLOCK + "        allowed = allowed | (is_fast[:, None] & is_action[None, :])\n",
    ))),
    Mutation("fast_bidirectional", edit((
        _FAST_BLOCK,
        "        allowed = allowed | (is_fast[:, None] & is_fast[None, :])\n",
    ))),
    Mutation("actions_causal", edit((
        _ACTION_BLOCK,
        "        allowed = allowed | (is_action[:, None] & is_action[None, :] & (key <= query))\n",
    ))),
    Mutation("prefix_sees_fast", edit((
        _ACTION_BLOCK,
        _ACTION_BLOCK + "        allowed = allowed | (is_prefix[:, None] & is_fast[None, :])\n",
    ))),
    Mutation("ignores_fast_padding", edit((
        "        pad = torch.cat([prefix_pad, fast_pad, actions_valid], dim=1)\n",
        "        pad = torch.cat([prefix_pad, torch.ones_like(fast_pad), actions_valid], dim=1)\n",
    ))),
    # Naive cumsum over the whole sequence: actions continue counting after FAST.
    Mutation("action_positions_after_fast", edit((
        "        first_action = prefix_pad.long().sum(dim=1, keepdim=True)\n",
        "        first_action = pad[:, :num_backbone].long().sum(dim=1, keepdim=True)\n",
    ))),
    # Paper eq. 6 stops the gradient through V_b too; detaching only the keys leaks through values.
    Mutation("detach_keys_only", edit((
        _V_INSULATED,
        "        v_insulated = v\n",
    ))),
    Mutation("detach_values_only", edit((
        _K_INSULATED,
        "        k_insulated = k\n",
    ))),
    # The joint-training baseline: no stop-gradient at all.
    Mutation("no_stop_gradient", edit(
        (_K_INSULATED, "        k_insulated = k\n"),
        (_V_INSULATED, "        v_insulated = v\n"),
    )),
    # Stop-gradient also applied on the backbone path: the FAST loss can no longer train K_b, V_b.
    Mutation("detach_on_backbone_path_too", edit((
        _BACKBONE_OUT,
        "        backbone_out = masked_attention(q[:, :, :nb], torch.cat([k[:, :, :nb].detach(), k[:, :, nb:]], dim=2), "
        "torch.cat([v[:, :, :nb].detach(), v[:, :, nb:]], dim=2), mask[:, :nb])\n",
    ))),
    # Detaching the whole key/value tensors on the action path also cuts the expert's own K_a, V_a.
    Mutation("detach_all_keys_for_actions", edit(
        (_K_INSULATED, "        k_insulated = k.detach()\n"),
        (_V_INSULATED, "        v_insulated = v.detach()\n"),
    )),
    Mutation("detach_action_queries", edit((
        "        action_out = masked_attention(q[:, :, nb:], k_insulated, v_insulated, mask[:, nb:])\n",
        "        action_out = masked_attention(q[:, :, nb:].detach(), k_insulated, v_insulated, mask[:, nb:])\n",
    ))),
    # Split at the prefix only (FAST K/V left attached): wrong split point for attend's contract.
    Mutation("split_one_short", edit(
        (_K_INSULATED, "        k_insulated = torch.cat([k[:, :, :max(nb - 1, 0)].detach(), k[:, :, max(nb - 1, 0):]], dim=2)\n"),
        (_V_INSULATED, "        v_insulated = torch.cat([v[:, :, :max(nb - 1, 0)].detach(), v[:, :, max(nb - 1, 0):]], dim=2)\n"),
    )),
    Mutation("neg_inf_fill", edit((
        "torch.finfo(scores.dtype).min)",
        "float(\"-inf\"))",
    ))),
    Mutation("missing_scale", edit((
        "        scale = 1.0 / math.sqrt(dim)\n",
        "        scale = 1.0\n",
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
