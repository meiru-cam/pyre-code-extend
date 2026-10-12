"""Mutation gate for the oft_parallel_action_decoding exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "oft_parallel_action_decoding"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """Apply exact-substring edits to the reference; each old string must occur once."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"edit target not unique: {old!r}"
        code = code.replace(old, new)
    return code


BIAS_LINE = "        return bias.masked_fill(padded_key, torch.finfo(dtype).min)\n"

MUTATIONS = [
    # Keep the causal mask OpenVLA uses for autoregressive decoding.
    Mutation("causal_mask_kept", edit((
        BIAS_LINE,
        "        future = torch.ones(seq, seq, dtype=torch.bool, device=attention_mask.device).triu(1)\n"
        "        return bias.masked_fill(padded_key | future, torch.finfo(dtype).min)\n",
    ))),
    # pi0-style prefix-LM: the prompt stays causal, only the action slots and stop see everything.
    Mutation("prefix_lm_mask", edit((
        "        hidden, logits = decoder(embeds, self.attention_bias(mask, embeds.dtype))\n",
        "        bias = self.attention_bias(mask, embeds.dtype)\n"
        "        first_slot = 1 + num_patches + num_prompt_tokens\n"
        "        seq = embeds.shape[1]\n"
        "        rows = torch.arange(seq)[:, None]\n"
        "        cols = torch.arange(seq)[None, :]\n"
        "        hide = (rows < first_slot) & (cols >= first_slot)\n"
        "        bias = bias.masked_fill(hide, torch.finfo(embeds.dtype).min)\n"
        "        hidden, logits = decoder(embeds, bias)\n",
    ))),
    # Mask padding query rows too, not only padding keys.
    Mutation("padding_rows_masked", edit((
        "        padded_key = (attention_mask == 0)[:, None, None, :].expand(batch, 1, seq, seq)\n",
        "        pad = attention_mask == 0\n"
        "        padded_key = pad[:, None, None, :] | pad[:, None, :, None]\n",
    ))),
    # Read the slots' own outputs instead of their left neighbors'.
    Mutation("inference_readout_not_shifted", edit((
        "        start = num_patches + num_prompt_tokens  # one position before the first slot\n",
        "        start = num_patches + num_prompt_tokens + 1\n",
    ))),
    Mutation("training_readout_not_shifted", edit((
        "        text_hidden = last_hidden[:, num_patches:-1]\n",
        "        text_hidden = last_hidden[:, num_patches + 1:]\n",
    ))),
    # Feed the placeholder embeddings instead of empty (zero) slots.
    Mutation("slots_not_zeroed", edit((
        "        x = x * ~(current | following).unsqueeze(-1)  # empty action slots\n",
        "",
    ))),
    Mutation("stop_token_zeroed", edit((
        "        x = x * ~(current | following).unsqueeze(-1)  # empty action slots\n",
        "        x = x * ~(current | following | (labels == self.STOP_ID)).unsqueeze(-1)\n",
    ))),
    Mutation("patches_before_bos", edit((
        "        embeds = torch.cat([x[:, :1], patch_embeds, x[:, 1:]], dim=1)\n",
        "        embeds = torch.cat([patch_embeds, x], dim=1)\n",
    ))),
    # Group the head input by action dimension instead of by time step.
    Mutation("head_grouped_by_dimension", edit((
        "            grouped = hidden[:, read].reshape(batch, self.chunk_len, -1)\n",
        "            h = hidden[:, read]\n"
        "            grouped = h.reshape(batch, self.action_dim, self.chunk_len, -1).transpose(1, 2).reshape(batch, self.chunk_len, -1)\n",
    ))),
    Mutation("bin_index_without_minus_one", edit((
        "        index = torch.clamp(self.vocab_size - tokens - 1, 0, self.n_bins - 2)\n",
        "        index = torch.clamp(self.vocab_size - tokens, 0, self.n_bins - 2)\n",
    ))),
    Mutation("empty_token_always_appended", edit((
        "        if not bool((input_ids[:, -1] == self.EMPTY_ID).all()):\n",
        "        if True:\n",
    ))),
    Mutation("prompt_count_includes_bos", edit((
        "        num_prompt_tokens = input_ids.shape[1] - 1\n",
        "        num_prompt_tokens = input_ids.shape[1]\n",
    ))),
    # Count only action labels, which differs when a non-action label precedes the actions.
    Mutation("count_only_action_labels", edit((
        "        count = (labels != self.IGNORE_INDEX).long().cumsum(dim=1)\n        is_action = labels > self.action_begin\n",
        "        is_action = labels > self.action_begin\n        count = is_action.long().cumsum(dim=1)\n",
    ))),
    Mutation("prompt_padding_ignored", edit((
        "        mask = torch.cat([attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1)\n",
        "        mask = torch.ones(batch, embeds.shape[1], dtype=attention_mask.dtype)\n",
    ))),
    Mutation("bias_padding_uses_minus_1e4", edit((
        BIAS_LINE,
        "        return bias.masked_fill(padded_key, -1e4)\n",
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
