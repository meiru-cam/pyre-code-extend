"""Mutation gate for the multi-part crowd labels code-reading exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "crowd_code_reading"

MUTATIONS = [
    ("one layer counted", 1, [("\"net_macs\": N * (D * H + H * C),", "\"net_macs\": N * D * H,")]),
    ("outputs of the last layer only", 1, [("\"net_outputs\": N * (H + C),", "\"net_outputs\": N * C,")]),
    ("loss counted per class", 1, [("\"nll_macs\": P * C * C,", "\"nll_macs\": P * C,")]),
    ("loop counts observed pairs", 1, [("\"loop_iterations\": N * A,", "\"loop_iterations\": P,")]),
    ("dense product without classes", 1, [("\"dense_elements\": N * A * C,", "\"dense_elements\": N * A,")]),
    ("whole matrix averaged", 2, [("return M.diagonal(dim1=-2, dim2=-1).mean(dim=-1)", "return M.mean(dim=(-2, -1))")]),
    ("most reliable picked", 2, [("return int(torch.argmin(r))", "return int(torch.argmax(r))")]),
    ("tensor returned", 2, [("return int(torch.argmin(r))", "return torch.argmin(r)")]),
    ("last index on ties", 2, [("return int(torch.argmin(r))", "return len(r) - 1 - int(torch.argmin(r.flip(0)))")]),
    ("missing labels wrap around", 3, [("i, a = (Y >= 0).nonzero(as_tuple=True)", "i, a = (Y >= -1).nonzero(as_tuple=True)")]),
    ("transposed confusion", 3, [("M[a, :, lab]", "M[a, lab, :]")]),
    ("epsilon dropped", 3, [("-torch.log(q + 1e-8).sum()", "-torch.log(q).sum()")]),
    ("mean instead of sum", 3, [("-torch.log(q + 1e-8).sum()", "-torch.log(q + 1e-8).mean()")]),
    ("loop over samples", 3, [("    i, a = (Y >= 0).nonzero(as_tuple=True)  # observed pairs only: O(P * C) memory, not O(N * A * C)\n    lab = Y[i, a]\n    q = torch.einsum(\"pc,pc->p\", p[i], M[a, :, lab])\n    return -torch.log(q + 1e-8).sum()\n",
                               "    total = p.new_zeros(())\n    for i in range(Y.shape[0]):\n        a = (Y[i] >= 0).nonzero(as_tuple=True)[0]\n        q = torch.einsum(\"c,pc->p\", p[i], M[a, :, Y[i, a]])\n        total = total - torch.log(q + 1e-8).sum()\n    return total\n")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
