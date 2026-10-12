"""Mutation gate for the Coconut continuous-thought forward exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "coconut_latent_forward"

FILL_LINE = "        inputs_embeds = inputs_embeds.index_put((rows, cols), hidden[rows, cols - 1])\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


# Every latent is filled from the first pass, while earlier latents still hold the placeholder embedding.
FILLS_ALL_IN_ONE_PASS = '''import torch


def coconut_forward(model, embed, input_ids, latent_id):
    if not torch.is_tensor(input_ids) or input_ids.dim() != 2 or input_ids.is_floating_point():
        raise ValueError("bad input_ids")
    B, T = input_ids.shape
    latent_lists = [torch.nonzero(input_ids[b] == latent_id).flatten().tolist() for b in range(B)]
    if any(p and p[0] == 0 for p in latent_lists):
        raise ValueError("latent at 0")
    n_latents = max((len(p) for p in latent_lists), default=0)
    inputs_embeds = embed(input_ids)
    for pass_idx in range(n_latents):
        _, hidden = model(inputs_embeds)
        if pass_idx == 0:
            pairs = [(b, t) for b, p in enumerate(latent_lists) for t in p]
            rows = torch.tensor([b for b, _ in pairs])
            cols = torch.tensor([t for _, t in pairs])
            inputs_embeds = inputs_embeds.index_put((rows, cols), hidden[rows, cols - 1])
    return model(inputs_embeds)[0]
'''


# Writes into the embedding tensor in place after the model has consumed it.
IN_PLACE_WRITE = '''import torch


def coconut_forward(model, embed, input_ids, latent_id):
    if not torch.is_tensor(input_ids) or input_ids.dim() != 2 or input_ids.is_floating_point():
        raise ValueError("bad input_ids")
    B, T = input_ids.shape
    latent_lists = [torch.nonzero(input_ids[b] == latent_id).flatten().tolist() for b in range(B)]
    if any(p and p[0] == 0 for p in latent_lists):
        raise ValueError("latent at 0")
    n_latents = max((len(p) for p in latent_lists), default=0)
    inputs_embeds = embed(input_ids)
    for pass_idx in range(n_latents):
        _, hidden = model(inputs_embeds)
        for b, p in enumerate(latent_lists):
            if len(p) > pass_idx:
                inputs_embeds[b, p[pass_idx]] = hidden[b, p[pass_idx] - 1]
    return model(inputs_embeds)[0]
'''


MUTATIONS = [
    _mutation(
        "feeds the hidden state at the latent's own position",
        (FILL_LINE, "        inputs_embeds = inputs_embeds.index_put((rows, cols), hidden[rows, cols])\n"),
    ),
    Mutation("fills every latent in one pass", FILLS_ALL_IN_ONE_PASS),
    _mutation(
        "detaches the fed-back hidden state",
        (FILL_LINE, "        inputs_embeds = inputs_embeds.index_put((rows, cols), hidden[rows, cols - 1].detach())\n"),
    ),
    _mutation(
        "skips the final pass",
        ("    logits, _ = model(inputs_embeds)\n    return logits\n",
         "    return last_logits if n_latents else model(inputs_embeds)[0]\n"),
        ("        _, hidden = model(inputs_embeds)\n", "        last_logits, hidden = model(inputs_embeds)\n"),
    ),
    _mutation(
        "runs one extra pass",
        ("    logits, _ = model(inputs_embeds)\n", "    model(inputs_embeds)\n    logits, _ = model(inputs_embeds)\n"),
    ),
    _mutation(
        "passes counted by the batch minimum",
        ("    n_latents = max(", "    n_latents = min("),
    ),
    _mutation(
        "reuses the first item's latent positions for every item",
        ("pairs = [(b, positions[pass_idx]) for b, positions in enumerate(latent_lists) if len(positions) > pass_idx]",
         "pairs = [(b, latent_lists[0][pass_idx]) for b in range(B) if len(latent_lists[0]) > pass_idx]"),
    ),
    _mutation(
        "feeds the input embedding before the latent instead of the hidden state",
        (FILL_LINE, "        inputs_embeds = inputs_embeds.index_put((rows, cols), inputs_embeds[rows, cols - 1])\n"),
    ),
    _mutation(
        "fills latents last to first",
        ("for b in range(B)]\n    if any(", "for b in range(B)]\n    latent_lists = [p[::-1] for p in latent_lists]\n    if any("),
        ("positions and positions[0] == 0", "positions and positions[-1] == 0"),
    ),
    Mutation("writes into the consumed embeddings in place", IN_PLACE_WRITE),
    _mutation(
        "accepts a latent at position 0 (reads position -1 like upstream)",
        ('        raise ValueError("a latent token cannot be at position 0")', "        pass"),
    ),
]


def test_coconut_latent_forward_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_coconut_latent_forward_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
