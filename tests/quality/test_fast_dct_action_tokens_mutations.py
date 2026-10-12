"""Mutation gate and metadata contract for the fast_dct_action_tokens exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "fast_dct_action_tokens"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """The reference with exact-substring edits; fails loudly if a target drifts."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"mutation target not unique or missing: {old!r}"
        code = code.replace(old, new)
    return code


ENCODE_DCT = "coeff = _dct_matrix(T) @ chunk.to(torch.float64)"
ENCODE_ROUND = "ints = torch.round(coeff * scale).reshape(*chunk.shape[:-2], T * D)"
ENCODE_SHIFT = "return torch.clamp(ints - min_token, min=0).long()"
DECODE_COEFF = "coeff = (tokens.to(torch.float64) + min_token).reshape(*tokens.shape[:-1], T, D) / scale"
DECODE_IDCT = "return _dct_matrix(T).T @ coeff"

FAST_MUTATIONS = [
    # Research doc section 3.1, V2.
    Mutation("dct_over_action_dims", edit((ENCODE_DCT, "coeff = chunk.to(torch.float64) @ _dct_matrix(chunk.shape[-1]).T"))),
    Mutation("column_major_flatten", edit((ENCODE_ROUND, "ints = torch.round(coeff * scale).transpose(-1, -2).reshape(*chunk.shape[:-2], T * D)"))),
    Mutation("unnormalized_dct", edit(
        ("C[0] *= math.sqrt(1.0 / T)", "C[0] *= 2.0"),
        ("C[1:] *= math.sqrt(2.0 / T)", "C[1:] *= 2.0"),
    )),
    Mutation("decode_forgets_scale", edit((DECODE_COEFF, DECODE_COEFF.removesuffix(" / scale")))),
    # Further plausible slips.
    Mutation("same_weight_on_dc_row", edit(("C[0] *= math.sqrt(1.0 / T)", "C[0] *= math.sqrt(2.0 / T)"))),
    Mutation("no_clamp", edit((ENCODE_SHIFT, "return (ints - min_token).long()"))),
    Mutation("adds_min_token_on_encode", edit((ENCODE_SHIFT, "return torch.clamp(ints + min_token, min=0).long()"))),
    Mutation("truncate_instead_of_round", edit((ENCODE_ROUND, "ints = (coeff * scale).trunc().reshape(*chunk.shape[:-2], T * D)"))),
    Mutation("dct_in_input_dtype", edit((ENCODE_DCT, "coeff = _dct_matrix(T).to(chunk.dtype) @ chunk"))),
    Mutation("decode_with_forward_dct", edit((DECODE_IDCT, "return _dct_matrix(T) @ coeff"))),
    Mutation("decode_forgets_min_token", edit((DECODE_COEFF, DECODE_COEFF.replace(" + min_token", "")))),
    Mutation("decode_skips_length_check", edit((
        "if tokens.ndim not in (1, 2) or tokens.shape[-1] != T * D:", "if tokens.ndim not in (1, 2):",
    ))),
    Mutation("encode_accepts_any_rank", edit((
        "if chunk.ndim not in (2, 3):", "if chunk.ndim == 0:",
    ))),
]


def test_fast_dct_action_tokens_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=None)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert "scipy" not in task["solution"]
    assert all("scipy" not in test["code"] for test in task["tests"])


def test_fast_dct_action_tokens_code_source_is_pinned():
    (code,) = [source for source in get_task(TASK_ID)["sources"] if source["kind"] == "code"]
    assert code["commit"] == "ec4d7aa71691cac0b8bed6942be45684db2110f4"
    assert code["commit"] in code["url"]
    assert code["license"] == "Apache-2.0"


@pytest.mark.parametrize("_repeat", range(3))
def test_fast_dct_action_tokens_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected(TASK_ID, FAST_MUTATIONS, require_unshown=True)
    ) == {m.name for m in FAST_MUTATIONS}
