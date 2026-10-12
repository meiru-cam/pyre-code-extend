"""Mutation gate and metadata contract for the fsq_quantizer exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "fsq_quantizer"


def _edit(name: str, old: str, new: str) -> Mutation:
    reference = get_task(TASK_ID)["solution"]
    assert reference.count(old) == 1, f"{name}: edit target must occur exactly once"
    return Mutation(name, reference.replace(old, new, 1))


def _mutations() -> list[Mutation]:
    return [
        _edit("no_even_offset",
              "offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.dtype)",
              "offset = torch.zeros_like(half_l)"),
        _edit("offset_without_atanh_shift",
              "shift = torch.atanh(offset / half_l)",
              "shift = torch.zeros_like(offset)"),
        _edit("paper_listing_tan_and_one_minus_eps",
              "half_l = (levels - 1) * (1 + self.eps) / 2\n"
              "        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.dtype)\n"
              "        shift = torch.atanh(offset / half_l)",
              "half_l = (levels - 1) * (1 - self.eps) / 2\n"
              "        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.dtype)\n"
              "        shift = torch.tan(offset / half_l)"),
        _edit("eps_dropped",
              "half_l = (levels - 1) * (1 + self.eps) / 2",
              "half_l = (levels - 1) / 2"),
        _edit("normalize_by_l_minus_one_over_two",
              "return rounded / (self._levels // 2).to(z.dtype)",
              "return rounded / ((self._levels - 1) / 2).to(z.dtype)"),
        _edit("round_without_straight_through",
              "rounded = bounded + (bounded.round() - bounded).detach()",
              "rounded = bounded.round()"),
        _edit("straight_through_skips_bound",
              "rounded = bounded + (bounded.round() - bounded).detach()",
              "rounded = z + (bounded.round() - z).detach()"),
        _edit("reversed_basis_first_channel_most_significant",
              "for L in levels:\n            basis.append(running)",
              "for L in reversed(levels):\n            basis.insert(0, running)"),
        _edit("decode_without_modulo",
              "digits = (indices.unsqueeze(-1) // self._basis) % self._levels",
              "digits = indices.unsqueeze(-1) // self._basis"),
        _edit("index_from_signed_codes",
              "digits = (zhat.detach() * half_width + half_width).round().long()",
              "digits = (zhat.detach() * half_width).round().long()"),
        _edit("clamp_instead_of_tanh",
              "return torch.tanh(z + shift) * half_l - offset",
              "return torch.clamp(z + shift, -1.0, 1.0) * half_l - offset"),
        _edit("no_validation",
              "        if z.shape[-1:] != (len(self.levels),):\n"
              "            raise ValueError(f\"last dimension must be {len(self.levels)}\")\n",
              "        pass\n"),
    ]


def test_metadata_is_valid_and_english_only():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, set(TASKS))
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert all(q.get("hint") for q in task["interview_questions"])


def test_code_sources_are_pinned():
    for source in get_task(TASK_ID)["sources"]:
        if source["kind"] == "code":
            assert source["commit"] == "3584ae752ce8ebdbe06a420bf60d7513c0e878cc"
            assert source["license"] == "Apache-2.0"


@pytest.mark.parametrize("_repeat", range(3))
def test_reference_passes_and_mutations_are_rejected(_repeat):
    mutations = _mutations()
    assert len(mutations) >= 6
    rejected = assert_mutations_rejected(TASK_ID, mutations, require_unshown=True)
    assert set(rejected) == {m.name for m in mutations}
