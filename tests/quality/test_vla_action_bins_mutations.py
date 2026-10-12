"""Mutation gate for the vla_action_bins exercise (OpenVLA and RT-1 action binning)."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.parts_gate import mutant

TASK_ID = "vla_action_bins"

# (name, [(text in the reference, replacement), ...]).
EDITS = [
    ("decode_to_bin_edges", [("n = self.centers[i]", "n = self.edges[i]")]),
    ("no_top_index_clip", [
        ("torch.clamp(self.vocab_size - token_ids - 1, 0, self.n_bins - 2)",
         "torch.clamp(self.vocab_size - token_ids - 1, min=0)"),
    ]),
    ("decode_forgets_minus_one", [
        ("torch.clamp(self.vocab_size - token_ids - 1, 0, self.n_bins - 2)",
         "torch.clamp(self.vocab_size - token_ids, 0, self.n_bins - 2)"),
    ]),
    ("batch_min_max_instead_of_quantiles", [
        ("low, high = self.low.to(actions.dtype), self.high.to(actions.dtype)",
         "flat = actions.reshape(-1, actions.shape[-1])\n"
         "        low, high = flat.amin(0), flat.amax(0)"),
    ]),
    ("skips_constant_dim_zeroing", [("return torch.where(low == high, torch.zeros_like(n), n)", "return n")]),
    ("no_clip_after_normalize", [
        ("n = torch.clamp(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1)",
         "n = 2 * (actions - low) / (high - low + 1e-8) - 1"),
    ]),
    ("drops_epsilon", [("(high - low + 1e-8)", "(high - low)")]),
    ("digitize_closed_on_the_right", [("right=True", "right=False")]),
    ("ascending_ids_at_tail", [
        ("return self.vocab_size - bins", "return self.vocab_size - self.n_bins - 1 + bins"),
        ("torch.clamp(self.vocab_size - token_ids - 1, 0, self.n_bins - 2)",
         "torch.clamp(token_ids - (self.vocab_size - self.n_bins), 0, self.n_bins - 2)"),
    ]),
    ("unnormalize_without_half", [
        ("return 0.5 * (n + 1) * (self.high - self.low) + self.low", "return n * (self.high - self.low) + self.low"),
    ]),
    ("rt1_rounds_instead_of_floor", [
        ("return ((n + 1) / 2 * (self.n_bins - 1)).long()", "return ((n + 1) / 2 * (self.n_bins - 1)).round().long()"),
    ]),
    ("rt1_scales_by_n_bins", [
        ("return ((n + 1) / 2 * (self.n_bins - 1)).long()",
         "return ((n + 1) / 2 * self.n_bins).long().clamp(max=self.n_bins - 1)"),
    ]),
    ("rt1_decode_clamps_ids", [
        ("return token_ids.to(self.low.dtype) / (self.n_bins - 1)",
         "return token_ids.clamp(0, self.n_bins - 1).to(self.low.dtype) / (self.n_bins - 1)"),
    ]),
    ("rt1_decodes_to_centers", [
        ("return token_ids.to(self.low.dtype) / (self.n_bins - 1)",
         "return (token_ids.to(self.low.dtype) + 0.5) / self.n_bins"),
    ]),
    ("rt1_uses_vocab_tail", [
        ("return ((n + 1) / 2 * (self.n_bins - 1)).long()",
         "return self.vocab_size - 1 - ((n + 1) / 2 * (self.n_bins - 1)).long()"),
        ("return token_ids.to(self.low.dtype) / (self.n_bins - 1)",
         "return (self.vocab_size - 1 - token_ids).to(self.low.dtype) / (self.n_bins - 1)"),
    ]),
]


def mutations() -> list[Mutation]:
    original = get_task(TASK_ID)["solution"]
    return [Mutation(name, mutant(original, edits)) for name, edits in EDITS]


@pytest.mark.parametrize("_repeat", range(3))
def test_vla_action_bins_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, mutations(), require_unshown=True)
    assert set(rejected) == {name for name, _ in EDITS}


def test_vla_action_bins_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert sum(t.get("visibility", "visible") == "visible" for t in task["tests"]) <= 3
